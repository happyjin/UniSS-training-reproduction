"""Reservation keeper for one of OUR cards (user order 2026-11-06 20:35 UTC): keeps the card's real free memory below
CAP_MIB (80000 MiB) with plain CUDA blocks, so an external scheduler that needs >= 88000-90000 MiB free never sees
the card as available -- not even during our job transitions. It never starves anyone:
  * allocates only while free >= FILL_ABOVE_MIB (78000 MiB) and free stays >= the room promised to a starting job
    + SAFE_MIB, one BLOCK_MIB block at a time;
  * releases blocks whenever free < SAFE_MIB (10 GiB) + one block, until it is above again or nothing is held;
  * honours the queue's request file <dir>/card<k>.req (JSON, written by gpuq_w3.py):
      {"release_to_mib": X, "release_until": t1, "pending_mib": P, "pending_since": t0, "pending_until": t2}
    while now < t1 it releases blocks until free >= X and allocates nothing (room for our job before its start);
    while t0 <= now < t2 a job of ours that needs P MiB is starting: the keeper keeps
    free >= (P - what the other processes on the card have grown since t0) + SAFE_MIB, i.e. it leaves the job its
    room but still tops up everything else, so the card stays below the cap during the job's start-up too.
State (rewritten every tick, atomically): <dir>/card<k>.state = {"pid", "card", "held_mib", "free_mib", "time", ...}.
Stop: kill <pid> (SIGTERM; blocks are freed when the process exits).

    CUDA_VISIBLE_DEVICES=<k> setsid nohup python gpu_keeper.py <dir> <k> >> <dir>/card<k>.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

CAP_MIB = 80000
FILL_ABOVE_MIB = 78000
SAFE_MIB = 10240
BLOCK_MIB = 2048
TICK = 1.0
MIB = 1 << 20


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def main():
    d, card = Path(sys.argv[1]), int(sys.argv[2])
    d.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(card), "set CUDA_VISIBLE_DEVICES to the card"
    torch.cuda.set_device(0)
    blocks: list[torch.Tensor] = []
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    req_f, st_f, tmp_f = d / f"card{card}.req", d / f"card{card}.state", d / f".card{card}.state.tmp"
    log(f"keeper card {card} pid {os.getpid()} cap {CAP_MIB} MiB free")
    last_msg = ""
    base = {"since": None, "others": 0.0}   # other processes' usage when the current pending job started

    def mem():
        f, t = torch.cuda.mem_get_info()
        f /= MIB
        others = t / MIB - f - len(blocks) * BLOCK_MIB   # everything that is not our padding
        return f, others

    def release_one():
        blocks.pop()
        torch.cuda.empty_cache()

    while not stop["flag"]:
        now = time.time()
        req = {}
        try:
            req = json.loads(req_f.read_text())
        except Exception:
            pass
        rel_to = float(req.get("release_to_mib", 0)) if now < float(req.get("release_until", 0)) else 0.0
        free, others = mem()
        pending = 0.0
        t0 = float(req.get("pending_since", 0))
        if t0 and t0 <= now < float(req.get("pending_until", 0)):
            if base["since"] != t0:
                base.update(since=t0, others=others)
            P = float(req.get("pending_mib", 0))
            pending = max(0.0, min(P, P - (others - base["others"])))
        # 1) safety: never let our padding starve anyone; leave a starting job of ours its room
        while blocks and free < max(SAFE_MIB + BLOCK_MIB, pending + SAFE_MIB):
            release_one()
            free, others = mem()
        # 2) the queue asks for room for one of our jobs
        while blocks and rel_to > 0 and free < rel_to:
            release_one()
            free, others = mem()
        # 3) keep the card below the cap
        if rel_to <= 0:
            while free >= FILL_ABOVE_MIB and free - BLOCK_MIB >= pending + SAFE_MIB:
                try:
                    t = torch.empty(BLOCK_MIB * MIB, dtype=torch.uint8, device="cuda")
                    t.fill_(0)
                    blocks.append(t)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                free, others = mem()
        held = len(blocks) * BLOCK_MIB
        tmp_f.write_text(json.dumps({"pid": os.getpid(), "card": card, "held_mib": held, "free_mib": round(free),
                                     "time": now, "pending_mib": round(pending), "release_to_mib": rel_to}))
        os.replace(tmp_f, st_f)
        msg = f"held {held} MiB" + (" (release request)" if rel_to else "") + (" (job starting)" if pending else "")
        if msg != last_msg:
            log(f"{msg}, free {free:.0f} MiB")
            last_msg = msg
        time.sleep(TICK)
    blocks.clear()
    torch.cuda.empty_cache()
    log("SIGTERM: blocks freed, exiting")


if __name__ == "__main__":
    main()
