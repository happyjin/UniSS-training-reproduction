#!/usr/bin/env python3
"""Single Python entrypoint for the Anyscale training Job.

Two phases run on the head/driver pod, both talking to the same already-up
Anyscale Ray cluster:

  Phase 1 — worker prepare
    Re-stage /home/ray/default/custom (overlayfs ephemeral) on every alive
    non-head worker via NodeAffinitySchedulingStrategy(soft=False). Non-zero
    rc on any failure aborts the job before swift sft burns cycles.

  Phase 2 — swift dispatch
    Pin swift sft driver to workers[0] as a ray actor (num_cpus=2,
    num_gpus=0); ms-swift's --use_ray spawns the actual GPU training tasks
    on workers. On rc=0, archive final checkpoint + tensorboard runs to S3.
    On rc!=0, dispatch a log-collection task to every alive node.

Why dispatch instead of running on head:
    Anyscale head pod has no GPU. transformers.TrainingArguments(bf16=True)
    runs torch.cuda.is_bf16_supported() in the driver process and raises
    before any --use_ray task ships. Pinning the driver to a worker
    sidesteps this without changing the Ray + DeepSpeed topology.

ENV SOURCING (post 2026-06 reorg):
    job.yaml.env_vars carries static cluster/framework vars (NCCL/EFA/Wandb
    etc.) plus task identity (MODEL_PATH/MODEL_TYPE). Path vars with derive
    logic (SOURCE_ROOT etc.) are loaded by source env.paths.sh on the cluster.
    The two-hop env whitelist below decides what survives each ray hop.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# =============================================================================
# Env-forwarding whitelist — single source of truth for both hops.
# Hop 1: head process → ray actor (collect_outer, includes RAY_* prefix
#        because actors stay inside ray's lifecycle).
# Hop 2: ray actor → swift bash subprocess (collect_inner, strips RAY_*
#        prefix to avoid replicating ray-runtime-internal vars into
#        swift's _create_workers RuntimeEnv. Re-includes a curated set of
#        legitimate user-set RAY_* keys, plus PATH/LD_LIBRARY_PATH/HOME
#        and other system essentials swift's bash needs.).
# =============================================================================
# Blocklist approach (replaces the old 4 whitelists — 75+ items → 15 items).
# All training-relevant env vars are already set by entrypoint.sh (which sources
# anyscale/env.sh), so we only need to filter out shell-internal and Ray-internal
# vars. No per-variable maintenance required as new training vars are added.
#
# Two tiers:
#   _INNER_BLOCKLIST  — shell-internal vars stripped from BOTH hops.
#   _OUTER_BLOCKLIST  — system-essential vars (PATH/HOME/PWD/SHELL) stripped from
#                       Hop 1 (runtime_env) ONLY, so the ray worker's native PATH
#                       is preserved. These MUST pass through Hop 2 (collect_inner)
#                       because subprocess.run(env=...) completely replaces the
#                       environment — without PATH/HOME the subprocess can't find
#                       commands like `swift`.
#
# RAY_* allowlist:
#   collect_outer() 和 collect_inner() 的 blanket `not k.startswith("RAY_")`
#   过滤会把所有 RAY_* 变量扔掉，但其中一些是训练必需的、非 Ray 内部变量。
#   以下列表中的变量允许通过两跳过滤：
#     - RAY_DEVICE_GROUPS / RAY_WORLD_SIZE / RAY_NPROC_PER_NODE：
#       由 derive_ray_device_groups() 从 COMPUTE_CONFIG_PATH 派生，
#       ms-swift 的 --device_groups 参数读取这个 JSON 来分配 GPU ranks。
#     - RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES / RAY_DEDUP_LOGS：
#       job.yaml env_vars 中设置的 Ray runtime 调优开关。
_RAY_ALLOWLIST = frozenset({
    "RAY_DEVICE_GROUPS",
    "RAY_WORLD_SIZE",
    "RAY_NPROC_PER_NODE",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_DEDUP_LOGS",
})

_INNER_BLOCKLIST = frozenset({
    "SHLVL", "_", "OLDPWD",
    "LS_COLORS", "HOSTNAME", "HOSTTYPE", "MACHTYPE", "OSTYPE",
    "PS1", "PS2", "IFS",
})
_OUTER_BLOCKLIST = frozenset({
    "HOME", "PATH", "PWD", "SHELL",
})

# Torchrun / torchelastic residue keys that must be stripped before swift subprocess.
_TORCHRUN_KEYS = frozenset({
    "NNODES", "NODE_RANK", "NPROC_PER_NODE",
    "MASTER_ADDR", "MASTER_PORT",
    "WORLD_SIZE", "RANK", "LOCAL_RANK",
    "GROUP_RANK", "GROUP_WORLD_SIZE",
    "LOCAL_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
})
_TORCHRUN_PREFIXES = ("TORCHELASTIC_", "PET_")

REQUIRED_ENV_VARS = ("TRAIN_SCRIPT",)


def collect_outer() -> dict:
    """Hop 1 filter: head os.environ → ray actor's runtime_env env_vars.
    Forward everything except shell-internal, system-essential, and Ray-internal
    keys. System-essential vars (PATH/HOME/PWD/SHELL) are excluded here so
    runtime_env doesn't override the ray worker's native values.
    See also _OUTER_BLOCKLIST.

    RAY_* vars in _RAY_ALLOWLIST (e.g. RAY_DEVICE_GROUPS derived from compute
    config) are explicitly allowed through; all other RAY_* vars are stripped."""
    return {k: v for k, v in os.environ.items()
            if k not in _INNER_BLOCKLIST
            and k not in _OUTER_BLOCKLIST
            and (not k.startswith("RAY_") or k in _RAY_ALLOWLIST)}


def collect_inner() -> dict:
    """Hop 2 filter: ray actor's os.environ → swift bash subprocess env.
    Also strips torchrun/elastic residue keys.

    RAY_* vars in _RAY_ALLOWLIST (e.g. RAY_DEVICE_GROUPS) are explicitly
    allowed through so the swift subprocess can read them (common_args.sh
    passes RAY_DEVICE_GROUPS as --device_groups to the swift CLI)."""
    return {k: v for k, v in os.environ.items()
            if k not in _INNER_BLOCKLIST
            and k not in _TORCHRUN_KEYS
            and not any(k.startswith(p) for p in _TORCHRUN_PREFIXES)
            and (not k.startswith("RAY_") or k in _RAY_ALLOWLIST)}


def assert_required_env() -> None:
    """Fail fast before ray.init() if critical env vars are missing.

    Also normalizes TRAIN_SCRIPT ↔ TRAIN_ENTRYPOINT: the production baseline
    (prodjob_inbveyburwxwubg5eziplp21hl) uses TRAIN_ENTRYPOINT as the env var
    name; v4.3's submit_job.sh uses TRAIN_SCRIPT. Accept either; mirror the
    value into the other so the rest of the code (and the production
    naming) both work.
    """
    if not os.environ.get("TRAIN_SCRIPT") and os.environ.get("TRAIN_ENTRYPOINT"):
        os.environ["TRAIN_SCRIPT"] = os.environ["TRAIN_ENTRYPOINT"]
    if os.environ.get("TRAIN_SCRIPT") and not os.environ.get("TRAIN_ENTRYPOINT"):
        os.environ["TRAIN_ENTRYPOINT"] = os.environ["TRAIN_SCRIPT"]
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        msg = (f"required env vars missing: {missing}. "
               f"Example: TRAIN_SCRIPT="
               f"pipelines/sft/normal_translation/swift")
        print(f"[dispatch-train] FATAL: {msg}", file=sys.stderr, flush=True)
        sys.exit(2)


def derive_ray_device_groups(workers: list) -> dict:
    """If USE_RAY is true and RAY_DEVICE_GROUPS is unset, derive it from the
    actual alive Ray cluster — NOT from the local compute config YAML.

    The compute config is declarative (what was requested). The Ray cluster
    is reality (what Anyscale actually provisioned). Using the config when
    fewer nodes joined causes DeepSpeed/NCCL to wait for ranks that never
    appear → deadlock.

    prodjob_hzbwwdvdlsmekm1pd74yrjzjjw is the canonical example: compute
    config said 3 workers (24 GPU), but Anyscale only allocated 2 (16 GPU).
    DeepSpeed tried to init 24-rank communicator, 8 ranks never showed up,
    all 16 hung on the first NCCL collective.

    Returns the env vars to inject (RAY_DEVICE_GROUPS / RAY_WORLD_SIZE /
    RAY_NPROC_PER_NODE).
    """
    def _warn(msg: str) -> dict:
        print(f"[dispatch-train] WARN: {msg}", flush=True)
        return derived

    derived: dict = {}
    if os.environ.get("USE_RAY", "").lower() not in ("true", "1"):
        return derived
    if os.environ.get("RAY_DEVICE_GROUPS"):
        derived["RAY_DEVICE_GROUPS"] = os.environ["RAY_DEVICE_GROUPS"]
        return derived

    if not workers:
        return _warn("no alive workers — skipping RAY_DEVICE_GROUPS derivation")

    # Query actual GPU resources from the live Ray cluster.
    # ray.cluster_resources() returns totals across all nodes (including
    # head, but head has 0 GPU so it's a no-op). This is the ground truth
    # of what Anyscale actually provisioned — no race conditions, no stale
    # YAML, no mismatch between declared and actual topology.
    cluster_resources = ray.cluster_resources()
    total_gpus = int(cluster_resources.get("GPU", 0))
    num_workers = len(workers)

    if total_gpus == 0:
        return _warn("ray.cluster_resources() reports 0 GPUs — cluster may still be booting")

    if total_gpus % num_workers != 0:
        return _warn(
            f"GPU count {total_gpus} not evenly divisible by {num_workers} "
            f"workers — heterogeneous cluster? skipping RAY_DEVICE_GROUPS derivation"
        )

    nproc_i = total_gpus // num_workers
    world_i = total_gpus

    device_groups = {
        "nproc_per_node": nproc_i,
        "default": {"device": "GPU", "ranks": list(range(world_i)), "workers": ["default"]},
    }
    derived["RAY_DEVICE_GROUPS"] = json.dumps(device_groups, separators=(",", ":"))
    derived["RAY_WORLD_SIZE"] = str(world_i)
    derived["RAY_NPROC_PER_NODE"] = str(nproc_i)
    print(f"[dispatch-train] derived RAY_DEVICE_GROUPS for {world_i} GPUs "
          f"({nproc_i}/node, {num_workers} workers from live cluster)", flush=True)
    return derived


# =============================================================================
# Phase 1 — worker prepare (was anyscale/train/train_prepare.py +
#           anyscale/train/train_worker_setup.sh; both deleted, logic inlined)
# =============================================================================
def run_prepare(workers: list, env_to_forward: dict) -> int:
    """Re-stage the swift plugin tree on every worker node, in parallel.

    Why this phase exists at all: ms-swift requires
    ``PLUGIN_PATH=/home/ray/default/custom/plugin.py`` to load our custom
    qwen-omnispeak ops. ``/home/ray/default/`` lives on the pod-ephemeral
    overlayfs and gets wiped between pod incarnations; the ETL MARKER on the
    persistent NVMe (hostPath) does NOT cover it, so a Job retry on the same
    EC2 host can short-circuit ETL while custom/ is missing → swift dies with
    ``AssertionError: py_dir: /home/ray/default/custom`` (observed on
    prodjob_nwunwge7acjbf67u6bfha2xid4 on 2026-06-05). The fix is a cheap
    (~60 KB) ``cp -r`` on every training Job start.

    What this phase does NOT do: anything with manifest jsonls, S3 sync, or
    on-disk path rewriting. Those are ETL-pipeline concerns
    (anyscale/data/job_etl.yaml). If a training Job starts and the manifest
    doesn't exist at the path the task script declares, that's a contract
    violation between ETL and training — fail fast and tell the user to run
    the ETL Job, don't paper it over with shell hacks.

    Returns 0 if every worker re-stages custom/ successfully, non-zero
    otherwise. Caller (main) is expected to abort the job on failure before
    phase 2 starts.

    Assumes ``ray.init()`` has already been called by ``main()``; does NOT
    re-initialize ray.
    """
    if not workers:
        print("[train-prepare] no worker nodes — head-only run, nothing to dispatch.")
        return 0

    @ray.remote(num_cpus=2, num_gpus=0, max_retries=0)
    def prepare_on_worker():
        import shutil
        host = socket.gethostname()

        # Ray's runtime_env replicates the project working_dir to every
        # actor's cwd; ``custom/`` therefore lives at ``${cwd}/custom`` on
        # this worker. REPO_ROOT points to NVMe for models/data — code
        # (including custom/) is uploaded via working_dir to cwd, NOT REPO_ROOT.
        cwd = os.getcwd()
        for candidate in (os.path.join(cwd, "custom"),
                          os.path.join(os.environ.get("REPO_ROOT", ""), "custom")):
            if os.path.isdir(candidate):
                src = candidate
                break
        else:
            return host, 127, "", f"FATAL: source custom/ not found at cwd={cwd} or REPO_ROOT"

        dst = "/home/ray/default/custom"
        os.makedirs("/home/ray/default", exist_ok=True)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        msg = f"staged {src} → {dst}"
        print(f"[train-prepare][{host}] {msg}", flush=True)
        return host, 0, msg, ""

    refs = []
    for node in workers:
        strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        refs.append(
            prepare_on_worker.options(
                scheduling_strategy=strategy,
                runtime_env={"env_vars": env_to_forward},
            ).remote()
        )

    print(f"[train-prepare] waiting for {len(refs)} worker preps …")
    t0 = time.time()
    results = ray.get(refs)
    elapsed = time.time() - t0
    print(f"[train-prepare] all returned in {elapsed:.1f}s")

    failures = 0
    for host, rc, stdout, stderr in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"[train-prepare][{host}] rc={rc} ({status})")
        if rc != 0:
            if stdout:
                print(f"[train-prepare][{host}] STDOUT:\n{stdout}")
            if stderr:
                print(f"[train-prepare][{host}] STDERR:\n{stderr}")
            failures += 1

    if failures:
        print(f"[train-prepare] {failures}/{len(results)} workers FAILED setup", file=sys.stderr)
        return 1
    print(f"[train-prepare] all {len(results)} workers completed setup")
    return 0


# =============================================================================
# Phase 2 — swift dispatch helpers
# =============================================================================
def archive_checkpoint_to_s3(chosen_id: str) -> None:
    """Sync the latest checkpoint + tensorboard runs from worker NVMe to S3.

    Pinned to the same worker[0] that did the training so OUTPUT_ROOT is local.
    Skips silently if FINAL_CHECKPOINT_S3_PREFIX or OUTPUT_ROOT is unset.
    Caller invokes only on training rc=0.
    """
    s3_prefix = os.environ.get("FINAL_CHECKPOINT_S3_PREFIX", "").rstrip("/")
    output_root = os.environ.get("OUTPUT_ROOT", "")
    # Identity segments so multi-user / multi-job runs don't collide in S3.
    # fall back to sane defaults for local/dev or manual submits.
    def _slug(v: str, default: str) -> str:
        v = "".join(c if (c.isalnum() or c in "._-") else "-" for c in (v or "").strip()).strip("-")
        return v or default
    submit_user = _slug(os.environ.get("SUBMIT_USER", ""), "unknown-user")
    submit_job = _slug(os.environ.get("SUBMIT_JOB_NAME", ""), "unknown-job")
    if not s3_prefix:
        print("[dispatch-train] FINAL_CHECKPOINT_S3_PREFIX unset; skipping S3 archival", flush=True)
        return
    if not output_root:
        print("[dispatch-train] OUTPUT_ROOT unset; cannot locate final ckpt", flush=True)
        return

    print(f"[dispatch-train] archiving final checkpoint to {s3_prefix}/ …", flush=True)
    archive_env = {k: v for k, v in os.environ.items()
                   if k.startswith("AWS_") or k in ("PATH", "HOME")}

    @ray.remote(num_cpus=2, num_gpus=0, max_retries=0)
    def _archive():
        import glob
        host = socket.gethostname()
        # Find the most recent <task>/v*-<timestamp>/ directory under OUTPUT_ROOT.
        candidates = []
        for task_dir in glob.glob(f"{output_root}/*"):
            if not os.path.isdir(task_dir):
                continue
            for run_dir in glob.glob(f"{task_dir}/v*-*"):
                if os.path.isdir(run_dir):
                    candidates.append(run_dir)
        if not candidates:
            return host, 1, f"no v*-* run dir under {output_root}"
        run_dir = max(candidates, key=os.path.getmtime)
        run_name = os.path.basename(run_dir)
        task_name = os.path.basename(os.path.dirname(run_dir))
        print(f"[archive][{host}] selected run: {task_name}/{run_name} ({run_dir})", flush=True)

        # Find latest checkpoint-N (final state).
        ckpts = sorted(
            [d for d in glob.glob(f"{run_dir}/checkpoint-*") if os.path.isdir(d)],
            key=lambda p: int(os.path.basename(p).split("-")[1]),
        )
        if not ckpts:
            return host, 1, f"no checkpoint-* under {run_dir}"
        final_ckpt = ckpts[-1]
        ckpt_name = os.path.basename(final_ckpt)
        # <prefix>/<user>/<job>/<task>/<run> — keeps different people / jobs /
        # tasks isolated so a shared OUTPUT_ROOT or s3 prefix won't collide.
        s3_run_uri = f"{s3_prefix}/{submit_user}/{submit_job}/{task_name}/{run_name}"
        print(f"[archive][{host}] final ckpt: {ckpt_name} (of {len(ckpts)}) → {s3_run_uri}", flush=True)

        # 1) Final checkpoint (full)
        rc_a = subprocess.call([
            "aws", "s3", "sync", final_ckpt, f"{s3_run_uri}/{ckpt_name}/",
            "--only-show-errors",
        ])
        if rc_a != 0:
            return host, rc_a, f"checkpoint sync failed rc={rc_a}"

        # 2) Top-level run files + tensorboard runs/
        for top_file in ["args.json", "logging.jsonl", "val_dataset.jsonl"]:
            src = os.path.join(run_dir, top_file)
            if os.path.isfile(src):
                subprocess.call([
                    "aws", "s3", "cp", src, f"{s3_run_uri}/{top_file}",
                    "--only-show-errors",
                ])
        runs_dir = os.path.join(run_dir, "runs")
        if os.path.isdir(runs_dir):
            subprocess.call([
                "aws", "s3", "sync", runs_dir, f"{s3_run_uri}/runs/",
                "--only-show-errors",
            ])

        return host, 0, s3_run_uri

    ref = _archive.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=chosen_id, soft=False),
        runtime_env={"env_vars": archive_env},
    ).remote()
    try:
        a_host, a_rc, a_msg = ray.get(ref, timeout=1800)
        prefix = "✅ archive complete" if a_rc == 0 else f"⚠️  archive failed rc={a_rc}"
        print(f"[dispatch-train] {prefix}: {a_msg}", flush=True)
    except Exception as e:
        print(f"[dispatch-train] ⚠️  archive task crashed: {e}", flush=True)


def collect_failure_logs() -> None:
    """Dispatch a log-collection ray task to every alive node and print to stdout.

    Used after a swift failure to surface python-core-worker-*.err and .out
    from all nodes, not just worker[0] (the swift driver only ran on
    worker[0]; ray actors that died on other nodes have their tracebacks
    only in /tmp/ray/session_*/logs/ on those nodes).
    """
    print("[dispatch-train] dispatching log-collection to all alive nodes …", flush=True)

    @ray.remote(num_cpus=1, num_gpus=0, max_retries=0)
    def _collect():
        import glob
        host = socket.gethostname()
        out = [f"=== node {host} ==="]
        log_dirs = sorted(
            glob.glob("/tmp/ray/session_*/logs"),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )[:1]
        for log_dir in log_dirs:
            # Find non-empty worker stderr files (the empty ones are
            # successful actors; the non-empty ones probably crashed).
            err_files = sorted(
                [f for f in glob.glob(os.path.join(
                    log_dir, "python-core-worker-*.err"))
                 if os.path.getsize(f) > 0],
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )[:5]
            for f in err_files:
                out.append(f"\n--- {f} (size={os.path.getsize(f)}, "
                           f"last 60 lines) ---")
                try:
                    with open(f) as fp:
                        lines = fp.readlines()[-60:]
                        out.append("".join(lines))
                except Exception as e:
                    out.append(f"  (read failed: {e})")
            # Also worker .out (sometimes errors print to stdout)
            out_files = sorted(
                [f for f in glob.glob(os.path.join(
                    log_dir, "python-core-worker-*.out"))
                 if os.path.getsize(f) > 0],
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )[:3]
            for f in out_files:
                out.append(f"\n--- {f} (size={os.path.getsize(f)}, "
                           f"last 40 lines) ---")
                try:
                    with open(f) as fp:
                        lines = fp.readlines()[-40:]
                        out.append("".join(lines))
                except Exception as e:
                    out.append(f"  (read failed: {e})")
        return host, "\n".join(out)

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    refs = [_collect.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=n["NodeID"], soft=False)).remote()
            for n in nodes]
    for r in refs:
        try:
            host, dump = ray.get(r, timeout=60)
            print(f"\n{'#' * 60}\n# logs from {host}\n{'#' * 60}\n{dump}", flush=True)
        except Exception as e:
            print(f"[dispatch-train] log collect failed: {e}", flush=True)


# =============================================================================
# Phase 2 — swift dispatch (driver actor)
# =============================================================================
def run_train_dispatch(workers: list, env_to_forward: dict) -> int:
    """Pin swift driver to workers[0] as a ray actor and run training.

    On rc=0, archives final checkpoint to S3 (if FINAL_CHECKPOINT_S3_PREFIX
    is set). On rc!=0, dispatches a log-collection task to every alive
    node so the per-node python-core-worker-*.err files surface in the
    head log stream.

    Returns the swift subprocess return code.
    """
    chosen = workers[0]
    chosen_id = chosen["NodeID"]
    chosen_ip = chosen.get("NodeManagerAddress", "?")
    print(f"[dispatch-train] driver will run on worker {chosen_id[:12]} ({chosen_ip})")
    print(f"[dispatch-train] forwarding RAY_DEVICE_GROUPS={env_to_forward.get('RAY_DEVICE_GROUPS', '<unset>')[:200]}", flush=True)

    # num_cpus=2 so the driver doesn't compete with the worker's training
    # tasks for too many cores. num_gpus=0 — the driver doesn't need a GPU;
    # ms-swift's --use_ray reserves GPUs via its own ray.remote calls.
    @ray.remote(num_cpus=2, num_gpus=0, max_retries=0)
    def run_train():
        host = socket.gethostname()
        cwd = os.getcwd()
        script = os.environ.get("TRAIN_SCRIPT")
        if not script:
            msg = ("TRAIN_SCRIPT env var is required. "
                   "Example: pipelines/sft/normal_translation/swift")
            print(f"[dispatch-train][{host}] FATAL: {msg}", flush=True)
            return host, 2, msg

        # 直接调用 TRAIN_SCRIPT（已在 job.yaml 中指定完整路径）。
        # 原 pipelines/train.sh 统一入口已移除，各任务脚本（如
        # pipelines/bridge_alignment/train_joint.sh）自包含路径解析逻辑。
        train_sh = os.path.join(cwd, script)
        if not os.path.isfile(train_sh):
            return host, 3, (
                f"TRAIN_SCRIPT not found at {train_sh} (cwd={cwd}). "
                f"TRAIN_SCRIPT={script} must be a valid path relative to "
                f"the repo root."
            )
        print(f"[dispatch-train][{host}] launching driver: bash {train_sh} "
              f"(TRAIN_SCRIPT={script}, cwd={cwd})", flush=True)

        # Build the CLEAN env for the swift subprocess (hop 2). See header
        # for why this second filter is required.
        clean_env = collect_inner()

        # Defense-in-depth: strip torchrun/torchelastic markers before swift sft.
        # If swift sees NNODES/NPROC_PER_NODE etc. set, it triggers use_torchrun()
        # which plants TORCHELASTIC_USE_AGENT_STORE=True via LocalElasticAgent;
        # swift's _create_workers then replicates that into all training actors,
        # forcing rank-0 into client mode → all ranks hang on TCPStore connect.
        torchrun_trigger_keys = ('NNODES', 'NODE_RANK', 'NPROC_PER_NODE', 'MASTER_ADDR', 'MASTER_PORT')
        torchelastic_residue_keys = (
            'GROUP_RANK', 'GROUP_WORLD_SIZE', 'LOCAL_WORLD_SIZE',
            'ROLE_RANK', 'ROLE_WORLD_SIZE', 'ROLE_NAME',
            'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
        )
        torchelastic_residue_prefixes = ('TORCHELASTIC_', 'PET_')
        for k in list(clean_env):
            if (k in torchrun_trigger_keys or k in torchelastic_residue_keys
                    or any(k.startswith(p) for p in torchelastic_residue_prefixes)):
                clean_env.pop(k, None)

        # Prepend cuda lib so aws-ofi-nccl can dlopen libcudart.so.13 — anyscale's
        # runtime sometimes appends /usr/local/nvidia/lib{,64} ahead of ours.
        cuda_lib = "/usr/local/cuda/lib"
        ld = clean_env.get("LD_LIBRARY_PATH", "")
        if cuda_lib not in ld.split(":"):
            clean_env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{ld}".rstrip(":")

        # Stripped count for visibility
        stripped = len(os.environ) - len(clean_env)
        print(f"[dispatch-train][{host}] swift env: kept {len(clean_env)} "
              f"vars, stripped {stripped} (mostly ray-internal)",
              flush=True)

        result = subprocess.run(
            ["bash", train_sh],
            env=clean_env,
        )
        if result.returncode != 0:
            print(f"[dispatch-train] swift exited rc={result.returncode}", flush=True)
        return host, result.returncode, ""

    ref = run_train.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=chosen_id, soft=False),
        runtime_env={"env_vars": env_to_forward},
    ).remote()

    print("[dispatch-train] driver dispatched, waiting for completion …")
    t0 = time.time()
    host, rc, msg = ray.get(ref)
    elapsed = time.time() - t0
    status = "OK" if rc == 0 else "FAIL"
    print(f"[dispatch-train][{host}] rc={rc} ({status}) elapsed={elapsed:.1f}s")
    if msg:
        print(f"[dispatch-train][{host}] msg: {msg}")

    # post-processing
    if rc == 0:
        archive_checkpoint_to_s3(chosen_id)
    else:
        collect_failure_logs()

    return rc


# =============================================================================
# Top-level orchestrator
# =============================================================================
def main() -> int:
    # Fail-fast before ray.init() — TRAIN_SCRIPT must be set or we'll just
    # waste cluster startup time.
    assert_required_env()

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    head_node_id = ray.get_runtime_context().get_node_id()
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    workers = [n for n in nodes if n["NodeID"] != head_node_id]
    print(f"[dispatch-train] head={head_node_id[:12]}, "
          f"workers={len(workers)}/{len(nodes)} alive nodes")

    if not workers:
        print("[dispatch-train] ERROR: no worker nodes available", file=sys.stderr)
        return 1

    # Derive RAY_DEVICE_GROUPS from the live Ray cluster (actual GPU count ×
    # actual worker count), NOT from the local compute config YAML.
    # Must happen BEFORE collect_outer() so the derived RAY_* vars are picked up.
    ray_env = derive_ray_device_groups(workers)
    for k, v in ray_env.items():
        os.environ[k] = v

    # Hop 1 filter: head os.environ → ray actor's runtime_env env_vars.
    # Used by both phase 1 (worker prepare) and phase 2 (swift driver).
    env_to_forward = collect_outer()

    # ===== Phase 1: worker prepare (was train_prepare.py) =====
    print("[dispatch-train] === phase 1: worker prepare ===")
    rc = run_prepare(workers, env_to_forward)
    if rc != 0:
        print(f"[dispatch-train] FATAL: worker prepare failed (rc={rc})",
              file=sys.stderr)
        return rc

    # ===== Phase 2: dispatch swift training (original dispatch_train body) =====
    print("[dispatch-train] === phase 2: dispatch swift training ===")
    return run_train_dispatch(workers, env_to_forward)


if __name__ == "__main__":
    sys.exit(main())
