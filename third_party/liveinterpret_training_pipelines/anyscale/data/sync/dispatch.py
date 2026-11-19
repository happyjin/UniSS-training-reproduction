#!/usr/bin/env python3
"""通用同步 dispatch。

- 涉及 NVMe (/local/) → 分发到每个 worker 并行执行（每节点独立 NVMe）
- 仅涉及共享存储 (S3 / FSx / EFS) → head 节点单次执行

ENV (来自 job.yaml env_vars, 透传):
    多组 (推荐):  SYNC_SOURCES="SRC1|DST1 SRC2|DST2 ..."  (空格分隔, 每组 SRC|DST)
    单组 (兼容):  SYNC_SOURCE / SYNC_TARGET
    公共:         SYNC_REGION / SYNC_MARKER / SYNC_OPTS
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

SCRIPT = "anyscale/data/sync/sync_worker.sh"


def is_nvme(path: str) -> bool:
    """NVMe 路径判定：/local/ 为 hostPath 挂载的每节点独立 NVMe。"""
    return path.startswith("/local/") or path.startswith("/local")


def parse_pairs() -> list[tuple[str, str]]:
    """解析同步源→目标对。优先 SYNC_SOURCES (多组)，否则回退 SYNC_SOURCE/SYNC_TARGET。"""
    raw = os.environ.get("SYNC_SOURCES", "").strip()
    if raw:
        pairs: list[tuple[str, str]] = []
        for item in raw.split():
            if "|" not in item:
                print("[sync-dispatch] FATAL: SYNC_SOURCES entries must be 'SRC|DST'",
                      file=sys.stderr)
                sys.exit(2)
            src, dst = item.split("|", 1)
            pairs.append((src, dst))
        return pairs

    src = os.environ.get("SYNC_SOURCE", "")
    dst = os.environ.get("SYNC_TARGET", "")
    if not src or not dst:
        print("[sync-dispatch] FATAL: set SYNC_SOURCES or "
              "(SYNC_SOURCE and SYNC_TARGET)", file=sys.stderr)
        sys.exit(2)
    return [(src, dst)]


def run_sync_local(src: str, dst: str) -> tuple[str, int, str]:
    """在当前节点（head）本地执行 sync_worker.sh。"""
    host = socket.gethostname()
    script = os.path.join(os.getcwd(), SCRIPT)
    env = dict(os.environ)
    env["SYNC_SOURCE"] = src
    env["SYNC_TARGET"] = dst

    print(f"[sync-dispatch][{host}] {src} -> {dst} ...", flush=True)
    t0 = time.time()

    result = subprocess.run(
        ["bash", script],
        capture_output=True, text=True,
        env=env,
    )

    elapsed = time.time() - t0
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    if result.returncode == 0:
        return host, 0, f"done in {elapsed:.1f}s"
    else:
        return host, result.returncode, (
            result.stderr[-500:] if result.stderr else "unknown error"
        )


def run_sync_on_workers(workers: list[dict], pairs: list[tuple[str, str]]) -> int:
    """通过 Ray NodeAffinity 分发 sync_worker.sh 到每个 worker。"""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=2, num_gpus=0, max_retries=0)
    def _run(pairs: list[tuple[str, str]]) -> list[tuple[str, int, str]]:
        return [run_sync_local(s, d) for (s, d) in pairs]

    refs = []
    for node in workers:
        strategy = NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"], soft=False
        )
        refs.append(
            _run.options(scheduling_strategy=strategy).remote(pairs)
        )

    print(f"[sync-dispatch] dispatching to {len(refs)} workers ...", flush=True)
    t0 = time.time()
    results = ray.get(refs)
    elapsed = time.time() - t0
    print(f"[sync-dispatch] all workers returned in {elapsed:.1f}s", flush=True)

    failures = 0
    for worker_results in results:
        for host, rc, msg in worker_results:
            status = "OK" if rc == 0 else "FAIL"
            print(f"[sync-dispatch][{host}] rc={rc} ({status}): {msg}", flush=True)
            if rc != 0:
                failures += 1

    return 1 if failures else 0


def main() -> int:
    pairs = parse_pairs()
    if not pairs:
        print("[sync-dispatch] FATAL: no sync pairs resolved", file=sys.stderr)
        return 2

    needs_workers = any(is_nvme(s) or is_nvme(d) for s, d in pairs)

    for s, d in pairs:
        print(f"[sync-dispatch] pair: {s} -> {d}")
    print(f"[sync-dispatch] nvme involved: {needs_workers} "
          f"→ {'workers' if needs_workers else 'head-only'}")

    # --- head-only 路径 ---
    if not needs_workers:
        failures = 0
        for s, d in pairs:
            host, rc, msg = run_sync_local(s, d)
            status = "OK" if rc == 0 else "FAIL"
            print(f"[sync-dispatch][{host}] rc={rc} ({status}): {msg}", flush=True)
            if rc != 0:
                failures += 1
        return 1 if failures else 0

    # --- worker 分发路径 ---
    import ray
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    head_node_id = ray.get_runtime_context().get_node_id()
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    workers = [n for n in nodes if n["NodeID"] != head_node_id]

    print(f"[sync-dispatch] head={head_node_id[:12]}, "
          f"workers={len(workers)}/{len(nodes)} alive nodes")

    if not workers:
        print("[sync-dispatch] ERROR: no worker nodes available", file=sys.stderr)
        return 1

    for w in workers:
        print(f"[sync-dispatch]   worker: {w['NodeID'][:12]}  "
              f"({w.get('NodeManagerAddress', '?')})",
              flush=True)

    return run_sync_on_workers(workers, pairs)


if __name__ == "__main__":
    sys.exit(main())
