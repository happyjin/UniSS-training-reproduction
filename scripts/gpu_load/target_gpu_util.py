#!/usr/bin/env python3
"""Generate a controlled CUDA compute load on one or more GPUs.

The program alternates tensor-core matrix multiplication and sleep in fixed
cycles.  It is intended for temporary diagnostics where a visible, bounded
GPU utilization is useful.  Stop it with SIGINT/SIGTERM; no model or dataset
files are touched.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerConfig:
    device: int
    target_util: float
    cycle_seconds: float
    matrix_size: int
    dtype: str
    sync_every: int
    log_interval: float


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one CUDA device is required")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("CUDA device list contains duplicates")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1,2,3,4,5,6,7"))
    parser.add_argument(
        "--target-util",
        type=float,
        default=60.0,
        help="target compute duty cycle in percent (default: 60)",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=float,
        default=1.0,
        help="compute/sleep control period in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=16384,
        help="square tensor-core matrix dimension (default: 16384)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=1,
        help="CUDA synchronize after this many GEMMs (default: 1)",
    )
    parser.add_argument("--log-interval", type=float, default=10.0)
    args = parser.parse_args()

    if not 0.0 < args.target_util <= 100.0:
        parser.error("--target-util must be in (0, 100]")
    if args.cycle_seconds <= 0.0:
        parser.error("--cycle-seconds must be positive")
    if args.matrix_size <= 0:
        parser.error("--matrix-size must be positive")
    if args.sync_every <= 0:
        parser.error("--sync-every must be positive")
    if args.log_interval <= 0.0:
        parser.error("--log-interval must be positive")
    return args


def worker_main(config: WorkerConfig, stop_event: mp.synchronize.Event) -> None:
    # Import after spawning so the parent never creates a CUDA context.
    import torch

    torch.cuda.set_device(config.device)
    if config.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested CUDA device {config.device}, but only "
            f"{torch.cuda.device_count()} devices are visible"
        )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.dtype]

    # Three matrices use about 1.5 GiB at the default BF16 16384 setting.
    a = torch.randn((config.matrix_size, config.matrix_size), device=config.device, dtype=dtype)
    b = torch.randn((config.matrix_size, config.matrix_size), device=config.device, dtype=dtype)
    out = torch.empty_like(a)

    # Warm up cuBLAS and allocate any workspaces before duty-cycle accounting.
    for _ in range(3):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize(config.device)

    active_seconds = config.cycle_seconds * config.target_util / 100.0
    next_log = time.monotonic() + config.log_interval
    cycles = 0
    gemms = 0
    print(
        f"[gpu {config.device}] ready: target={config.target_util:.1f}% "
        f"cycle={config.cycle_seconds:.3f}s matrix={config.matrix_size} "
        f"dtype={config.dtype}",
        flush=True,
    )

    while not stop_event.is_set():
        cycle_start = time.monotonic()
        active_deadline = cycle_start + active_seconds
        cycle_deadline = cycle_start + config.cycle_seconds

        queued = 0
        while not stop_event.is_set() and time.monotonic() < active_deadline:
            torch.mm(a, b, out=out)
            queued += 1
            gemms += 1
            if queued >= config.sync_every:
                torch.cuda.synchronize(config.device)
                queued = 0
        if queued:
            torch.cuda.synchronize(config.device)

        remaining = cycle_deadline - time.monotonic()
        if remaining > 0:
            stop_event.wait(remaining)
        cycles += 1

        now = time.monotonic()
        if now >= next_log:
            allocated_gib = torch.cuda.memory_allocated(config.device) / (1024**3)
            print(
                f"[gpu {config.device}] alive: cycles={cycles} gemms={gemms} "
                f"allocated={allocated_gib:.2f} GiB",
                flush=True,
            )
            next_log = now + config.log_interval

    torch.cuda.synchronize(config.device)
    print(f"[gpu {config.device}] stopped cleanly", flush=True)


def main() -> int:
    args = parse_args()
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    def request_stop(signum: int, _frame: object) -> None:
        print(f"parent received signal {signum}; stopping workers", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    configs = [
        WorkerConfig(
            device=device,
            target_util=args.target_util,
            cycle_seconds=args.cycle_seconds,
            matrix_size=args.matrix_size,
            dtype=args.dtype,
            sync_every=args.sync_every,
            log_interval=args.log_interval,
        )
        for device in args.devices
    ]
    workers = [
        ctx.Process(target=worker_main, args=(config, stop_event), name=f"gpu-load-{config.device}")
        for config in configs
    ]

    print(
        f"starting controlled GPU load: devices={args.devices}, "
        f"target_util={args.target_util:.1f}%",
        flush=True,
    )
    for process in workers:
        process.start()

    exit_code = 0
    try:
        while any(process.is_alive() for process in workers):
            for process in workers:
                if process.exitcode not in (None, 0):
                    print(
                        f"worker {process.name} exited with code {process.exitcode}; "
                        "stopping all workers",
                        file=sys.stderr,
                        flush=True,
                    )
                    exit_code = 1
                    stop_event.set()
            time.sleep(0.25)
    finally:
        stop_event.set()
        for process in workers:
            process.join(timeout=10.0)
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if any(process.exitcode not in (0, None) for process in workers):
            exit_code = 1

    print("all GPU load workers stopped", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
