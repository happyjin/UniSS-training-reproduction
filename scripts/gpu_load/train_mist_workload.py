#!/usr/bin/env python3
"""MIST/Thinker training load with real batches and optimizer updates, no checkpoints.

One independent replica per selected GPU; this is not a synchronized experiment.
Existing model weights and training data are read only. Only stdout/stderr are used
for output. See train_no_checkpoint.zh-CN.md for launch and completion-gate examples.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
MIB = 1024**2
BASE = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged"
ROWS = "data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl"


KEEPER_SCRIPT = "experiments/mist_w3po_v1/scripts/gpu_keeper.py"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def devices(value):
    try:
        result = [int(s) for s in value.split(",")]
    except ValueError as ex:
        raise argparse.ArgumentTypeError("devices must be physical indices, e.g. 0,1") from ex
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("device indices must be unique and nonnegative")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", required=True, type=devices, help="physical nvidia-smi indices")
    p.add_argument("--base", type=Path, default=Path(BASE))
    p.add_argument("--rows", type=Path, default=Path(ROWS))
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--steps", type=int, default=0, help="successful updates per GPU; 0 runs until stopped")
    p.add_argument("--target-memory-percent", type=float, default=95)
    p.add_argument("--min-free-mib", type=int, default=8192)
    p.add_argument("--max-used-mib", type=int, default=132000, help="memory target cap; 0 disables this cap")
    p.add_argument("--memory-chunk-mib", type=int, default=512)
    p.add_argument("--memory-window-steps", type=int, default=64, help="recent updates used for peak-memory calibration")
    p.add_argument("--warmup-steps", type=int, default=8, help="measure training peaks before filling memory")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--wait-idle", action="store_true", help="wait instead of exiting when GPUs are busy")
    p.add_argument("--tolerate-keeper", action="store_true",
                   help="opt-in: ignore our own root-owned W3PO reservation keepers (" + KEEPER_SCRIPT + ") on the "
                        "selected GPUs, at start and while running (their memory does not count as busy)")
    p.add_argument("--idle-seconds", type=float, default=60, help="all selected GPUs must remain idle this long")
    p.add_argument("--idle-memory-mib", type=int, default=1000)
    p.add_argument("--poll-seconds", type=float, default=5)
    p.add_argument("--wait-for-file", type=Path, help="require an explicit task-completion marker before starting")
    p.add_argument("--dry-run", action="store_true", help="validate paths and show configuration without CUDA")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    for key in ("micro_batch_size", "grad_accum", "lora_rank", "memory_chunk_mib", "memory_window_steps", "warmup_steps", "log_interval"):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("num_workers", "steps", "max_used_mib", "idle_memory_mib", "seed"):
        if getattr(a, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be nonnegative")
    for key in ("lr", "lora_alpha", "poll_seconds"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not math.isfinite(a.idle_seconds) or a.idle_seconds < 0:
        p.error("--idle-seconds must be finite and nonnegative")
    if not 0 < a.target_memory_percent < 100:
        p.error("--target-memory-percent must be in (0, 100)")
    if a.min_free_mib < 1024:
        p.error("--min-free-mib must be at least 1024")
    if a.worker is not None and a.worker not in a.devices:
        p.error("worker must belong to --devices")
    for key in ("base", "rows", "wait_for_file"):
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, (REPO / value).resolve())
    return a


def validate_assets(a):
    for path in (a.base / "config.json", a.base / "sd_proj.pt", a.rows):
        if not path.is_file():
            raise RuntimeError(f"missing local input: {path}")
    if not any(a.base.glob("*.safetensors")) and not any(a.base.glob("pytorch_model*.bin")):
        raise RuntimeError(f"no local model weights in {a.base}")


def smi(query):
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return [[v.strip() for v in row] for row in csv.reader(result.stdout.splitlines()) if row]


def is_keeper(pid):
    """a root-owned W3PO reservation keeper (only consulted with --tolerate-keeper)."""
    try:
        if os.stat(f"/proc/{pid}").st_uid != 0:
            return False
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return KEEPER_SCRIPT in fh.read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return False


def gpu_snapshot(a=None):
    cards = {}
    for i, uuid, total, used in smi("--query-gpu=index,uuid,memory.total,memory.used"):
        cards[int(i)] = {"uuid": uuid, "total": int(total), "used": int(used), "pids": set()}
    by_uuid = {c["uuid"]: c for c in cards.values()}
    if a is not None and getattr(a, "tolerate_keeper", False):
        for uuid, pid, used in smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
            if uuid in by_uuid:
                if is_keeper(int(pid)):
                    by_uuid[uuid]["used"] -= int(used) if used.isdigit() else 0
                else:
                    by_uuid[uuid]["pids"].add(int(pid))
        return cards
    for uuid, pid in smi("--query-compute-apps=gpu_uuid,pid"):
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].add(int(pid))
    return cards


def busy_cards(a, cards):
    missing = set(a.devices) - cards.keys()
    if missing:
        raise RuntimeError(f"GPU indices not found: {sorted(missing)}")
    return [g for g in a.devices if cards[g]["pids"] or cards[g]["used"] > a.idle_memory_mib]


def wait_until_ready(a):
    idle_since = None
    next_log = 0.0
    while True:
        now = time.monotonic()
        if a.wait_for_file is not None and not a.wait_for_file.is_file():
            idle_since = None
            message = f"waiting for completion marker: {a.wait_for_file}"
        else:
            cards = gpu_snapshot(a)
            busy = busy_cards(a, cards)
            if busy:
                idle_since = None
                message = f"GPUs busy: {[(g, cards[g]['used'], sorted(cards[g]['pids'])) for g in busy]}"
                if not a.wait_idle:
                    raise RuntimeError(message + "; use --wait-idle to wait")
            else:
                idle_since = now if idle_since is None else idle_since
                if now - idle_since >= a.idle_seconds:
                    return cards
                message = f"all selected GPUs idle; checking stability for {a.idle_seconds:g}s"
        if now >= next_log:
            log(message)
            next_log = now + 60
        time.sleep(a.poll_seconds)


class MemoryFill:
    """Keep padding outside the measured activation peak; back off after a training OOM."""

    def __init__(self, torch, a):
        self.torch, self.args = torch, a
        self.buffers = []
        self.peak_training = 0
        self.completed = 0
        self.backoff = 0
        self.peaks = deque(maxlen=a.memory_window_steps)

    @property
    def held(self):
        return sum(t.numel() for t in self.buffers)

    def record_peak(self):
        self.peaks.append(self.torch.cuda.max_memory_allocated() - self.held)
        self.peak_training = max(self.peaks)

    def release_after_oom(self):
        self.record_peak()
        self.buffers.clear()
        self.backoff += 4096 * MIB
        self.completed = 0
        self.torch.cuda.empty_cache()

    def after_step(self):
        t, a = self.torch, self.args
        self.record_peak()
        self.completed += 1
        if self.completed < a.warmup_steps:
            return
        if self.completed % 128 == 0 and self.backoff:
            self.backoff = max(0, self.backoff - 512 * MIB)
        free, total = t.cuda.mem_get_info()
        # CUDA context/library allocations are outside PyTorch's allocator.
        outside = max(0, total - free - t.cuda.memory_reserved())
        target = min(int(total * a.target_memory_percent / 100), total - a.min_free_mib * MIB)
        if a.max_used_mib:
            target = min(target, a.max_used_mib * MIB)
        wanted = max(0, target - outside - self.peak_training - self.backoff)
        held = self.held
        if abs(wanted - held) < a.memory_chunk_mib * MIB:
            return
        while self.buffers and held > wanted:
            held -= self.buffers[-1].numel()
            self.buffers.pop()
        # Release cached activation blocks before allocating padding. Otherwise
        # padding could consume that cache and make the next backward pass OOM.
        t.cuda.empty_cache()
        while wanted - held >= MIB:
            size = min(wanted - held, a.memory_chunk_mib * MIB)
            try:
                self.buffers.append(t.empty(size, dtype=t.uint8, device="cuda"))
                self.buffers[-1].zero_()
            except t.cuda.OutOfMemoryError:
                self.backoff += 4096 * MIB
                t.cuda.empty_cache()
                log("padding allocation reached available memory; reducing subsequent target")
                break
            held += size
        log(f"memory calibration: training_peak={self.peak_training / MIB:.0f} MiB "
            f"padding={self.held / MIB:.0f} MiB target<={target / MIB:.0f} MiB")


def repeat_batches(loader):
    while True:
        yield from loader


def training_batches(torch, dataset, a):
    options = {"multiprocessing_context": "spawn", "persistent_workers": True} if a.num_workers else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.micro_batch_size, shuffle=True,
                                        collate_fn=dataset.collate_fn, num_workers=a.num_workers,
                                        pin_memory=True, **options)
    return repeat_batches(loader)


def backward_update(torch, model, batches, accumulation, device="cuda"):
    # Keep CUDA batches/graphs inside this frame so an OOM unwinds their references
    # before the caller releases padding and retries. Optimizer updates happen later.
    metrics = None
    for _ in range(accumulation):
        batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(**batch)
            loss = output[0] / accumulation
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        values = output.detach().float().cpu()
        metrics = values if metrics is None else metrics + values
        del batch, output, loss
    return metrics / accumulation


def train_worker(a):
    import torch
    from megatron.core.transformer.transformer_config import TransformerConfig
    from experiments.uniss_qwen3_omni_v1.training.pretrain_thinker_gstar import GStarRows
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import METRIC_NAMES, ThinkerGStarModel

    torch.cuda.set_device(0)  # parent pins this worker to one physical UUID
    torch.manual_seed(a.seed + a.worker)
    log(f"GPU {a.worker}: loading local Thinker; checkpoint saving disabled; independent training replica")
    config = TransformerConfig(num_layers=24, hidden_size=896, num_attention_heads=14, num_query_groups=2, bf16=True)
    model = ThinkerGStarModel(config, qwen3_path=str(a.base), lora_rank=a.lora_rank,
                             lora_alpha=a.lora_alpha, lora_from=0, recompute=True, init_merged=True).cuda().train()
    model.thinker.config.use_cache = False
    model.thinker.model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0, foreach=False)
    with a.rows.open() as source:
        n_rows = sum(1 for line in source if line.strip())
    if not n_rows:
        raise RuntimeError("training rows are empty")
    dataset = GStarRows(str(a.rows), n_rows, a.seed + a.worker, str(a.base), perturb_p=0.2, oracle=False)
    batches = training_batches(torch, dataset, a)
    fill = MemoryFill(torch, a)
    step, retries, started = 0, 0, time.monotonic()
    log(f"GPU {a.worker}: {json.dumps(model.trainable_report())}; rows={n_rows}; "
        f"micro_batch={a.micro_batch_size}; accumulation={a.grad_accum}; no checkpoints or exports")
    while not a.steps or step < a.steps:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        retry = False
        resize = False
        try:
            metrics = backward_update(torch, model, batches, a.grad_accum)
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True, foreach=False)
        except torch.cuda.OutOfMemoryError:
            resize = not fill.buffers or retries >= 8
            if resize and a.micro_batch_size == 1:
                raise RuntimeError("training OOM at micro-batch-size=1 with no recoverable padding") from None
            retry = True
        if retry:
            # The exception traceback has been discarded here, including its graphs.
            optimizer.zero_grad(set_to_none=True)
            fill.release_after_oom()
            retries += 1
            if resize:
                effective = a.micro_batch_size * a.grad_accum
                a.micro_batch_size = max(1, a.micro_batch_size // 2)
                a.grad_accum = math.ceil(effective / a.micro_batch_size)
                batches.close()
                batches = training_batches(torch, dataset, a)
                retries = 0
                log(f"GPU {a.worker}: AUTO_TUNE micro_batch={a.micro_batch_size} accumulation={a.grad_accum}")
            log(f"GPU {a.worker}: released padding after OOM; retrying a new batch ({retries}/8)")
            continue
        # Do not retry a partially completed optimizer update if it fails.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step += 1
        fill.after_step()
        if step == 1 or step % a.log_interval == 0 or step == a.steps:
            free, total = torch.cuda.mem_get_info()
            metric_text = " ".join(f"{n}={v:.4f}" for n, v in zip(METRIC_NAMES, metrics.tolist()))
            log(f"GPU {a.worker}: step={step} {metric_text} "
                f"memory={(total-free)/MIB:.0f}/{total/MIB:.0f} MiB "
                f"padding={fill.held/MIB:.0f} MiB seconds/update={(time.monotonic()-started)/step:.2f}")
    log(f"GPU {a.worker}: finished {step} updates; no checkpoint saved")


def stop_workers(workers):
    # Each worker starts its own process group, including its DataLoader children.
    for process in workers.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in workers.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def supervise(a):
    workers = {}
    try:
        log(f"training-load supervisor PID={os.getpid()}; GPUs={a.devices}; no checkpoints")
        cards = wait_until_ready(a)
        # A fresh check immediately before launching; idle samples are not a reservation.
        if busy_cards(a, gpu_snapshot(a)):
            raise RuntimeError("GPU became busy before launch; retry later")
        for gpu in a.devices:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[gpu]["uuid"])
            workers[gpu] = subprocess.Popen(
                [sys.executable, "-B", "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker", str(gpu)],
                env=env, cwd=REPO, start_new_session=True,
            )
            log(f"GPU {gpu}: started training-load PID {workers[gpu].pid}")
        while True:
            codes = {gpu: p.poll() for gpu, p in workers.items()}
            failed = {gpu: code for gpu, code in codes.items() if code not in (None, 0)}
            if failed:
                raise RuntimeError(f"worker failure: {failed}")
            if all(code == 0 for code in codes.values()):
                return 0
            snapshot = gpu_snapshot(a)
            for gpu, process in workers.items():
                if codes[gpu] is not None:
                    continue
                others = snapshot[gpu]["pids"] - {process.pid}
                if others:
                    raise RuntimeError(f"GPU {gpu}: another GPU process appeared: {sorted(others)}; stopping this load")
                if snapshot[gpu]["total"] - snapshot[gpu]["used"] < 1024:
                    raise RuntimeError(f"GPU {gpu}: less than 1 GiB free; stopping this load")
            time.sleep(a.poll_seconds)
    finally:
        stop_workers(workers)


def main():
    sys.dont_write_bytecode = True
    a = parse_args()
    validate_assets(a)
    if a.dry_run:
        log("DRY RUN: no CUDA initialization, model loading, subprocesses, or output files")
        print(json.dumps({**vars(a), "mode": "independent MIST training load", "checkpoint_saving": False},
                         default=str, indent=2, ensure_ascii=False))
        return 0
    os.chdir(REPO)  # rows contain repository-relative audio/posterior paths
    sys.path[:0] = [str(REPO), str(REPO / "third_party/Megatron-LM")]
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path[:2] + [os.environ.get("PYTHONPATH", "")])
    for key, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"}.items():
        os.environ[key] = value
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if a.worker is not None:
        train_worker(a)
        return 0

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        return supervise(a)
    except KeyboardInterrupt:
        log("stopped; worker processes released; no checkpoint saved")
        return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        log(f"ERROR: {error}")
        sys.exit(1)
