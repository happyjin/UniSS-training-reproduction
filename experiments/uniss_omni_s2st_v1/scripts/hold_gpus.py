"""Hold the GPUs so nothing else claims them between experiments.

This computes nothing anyone wants. It reserves memory and keeps a small
matmul running so the cards read as busy to whoever is looking, which on
a shared box is the difference between finding them free later and not.

    python -m experiments.uniss_omni_s2st_v1.scripts.hold_gpus \
        --fraction 0.85 --gpus 0,1,2,3,4,5,6,7

Stop it with ``pkill -f hold_gpus``. It also exits on SIGINT/SIGTERM and
releases everything, and ``--minutes`` gives it a deadline so a forgotten
holder does not sit on eight cards indefinitely.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time

import torch

STOP = threading.Event()


def hold(index: int, fraction: float, side: int, idle: float) -> None:
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    # Leave the slack out of the reservation rather than discovering it is
    # missing: cuBLAS workspaces and the context itself are not free.
    want = int(free * fraction)
    block = 256 * 1024 * 1024
    blocks = []
    try:
        while sum(b.numel() for b in blocks) * 1 < want:
            blocks.append(torch.empty(block, dtype=torch.uint8, device=device))
    except torch.cuda.OutOfMemoryError:
        pass
    held = sum(b.numel() for b in blocks) / 1024**3
    a = torch.randn(side, side, device=device, dtype=torch.bfloat16)
    b = torch.randn(side, side, device=device, dtype=torch.bfloat16)
    print(f"  GPU{index}: 占用 {held:.1f} GiB / 总 {total/1024**3:.1f} GiB", flush=True)
    while not STOP.is_set():
        (a @ b).sum().item()
        if idle:
            time.sleep(idle)
    del blocks, a, b
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default=None, help="comma separated; default every visible one")
    ap.add_argument("--fraction", type=float, default=0.85,
                    help="of *free* memory to reserve on each card")
    ap.add_argument("--side", type=int, default=8192, help="matmul side; sets the load")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="seconds to sleep between matmuls; 0 keeps the card hot")
    ap.add_argument("--minutes", type=float, default=0.0, help="0 means until stopped")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA devices")
    indices = ([int(x) for x in args.gpus.split(",")] if args.gpus
               else list(range(torch.cuda.device_count())))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())

    print(f"占用 {len(indices)} 张卡，fraction={args.fraction}，"
          f"{'无限期' if not args.minutes else f'{args.minutes:.0f} 分钟'}；"
          f"停止: pkill -f hold_gpus", flush=True)
    threads = [threading.Thread(target=hold, args=(i, args.fraction, args.side, args.idle),
                                daemon=True) for i in indices]
    for t in threads:
        t.start()
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    try:
        while not STOP.is_set():
            if deadline and time.time() > deadline:
                STOP.set()
            time.sleep(2)
    except KeyboardInterrupt:
        STOP.set()
    for t in threads:
        t.join(timeout=30)
    print("已释放全部 GPU", flush=True)


if __name__ == "__main__":
    main()
