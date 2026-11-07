"""Yield guard (first for kenxsong's cards 4-6; since 2026-11-07 all 8 cards, other users' jobs appear anywhere): our decodes / talkers may share them, but if his processes grow and a
card's real free memory stays below MIN_FREE_GB for two checks in a row, SIGTERM OUR most recently started GPU process
on that card (root-owned, one of ours: online_decode / si_infer / asr_transcribe / train_w3po), so he never OOMs. Our
queue jobs are retried and decodes resume from their earlier windows. Never touches anyone else's processes.
Log: <qdir>/yield_guard.log.

    setsid nohup python yield_guard.py <qdir> >> <qdir>/yield_guard.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

CARDS = (0, 1, 2, 3, 4, 5, 6, 7)   # 2026-11-07: other users' root jobs (zezhongjin) appear on any card
MIN_FREE_GB = 4.0
TICK = 5
OURS = ("experiments.mist_w3po_v1.evaluation.online_decode", "evaluation.si_infer", "evaluation.asr_transcribe",
        "experiments.mist_w3po_v1.rl.train_w3po", "experiments.mist_w3po_v1.rl.precompute_anchors")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def smi(q):
    out = subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"], capture_output=True,
                         text=True).stdout
    return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def main():
    low = {c: 0 for c in CARDS}
    log(f"yield guard up, pid {os.getpid()}, cards {CARDS}, min free {MIN_FREE_GB} GB")
    while True:
        gpus = {int(i): (u, float(t) - float(m)) for i, u, m, t in smi("gpu=index,uuid,memory.used,memory.total")}
        for c in CARDS:
            uuid, free_mib = gpus[c]
            low[c] = low[c] + 1 if free_mib / 1024 * 1.048576 < MIN_FREE_GB else 0
            if low[c] < 2:
                continue
            cands = []
            for u, pid in smi("compute-apps=gpu_uuid,pid"):
                if u != uuid:
                    continue
                try:
                    st = os.stat(f"/proc/{pid}")
                    args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
                except OSError:
                    continue
                if st.st_uid == 0 and any(o in args for o in OURS):
                    # an RL trainer (checkpoints every 10 steps) only if nothing cheaper of ours is on the card
                    cands.append((0 if "rl.train_w3po" in args else 1, st.st_ctime, int(pid), args[:200]))
            if cands:
                _, _, pid, args = max(cands)
                log(f"card {c}: {free_mib:.0f} MiB free twice -> SIGTERM our pid {pid}: {args}")
                os.kill(pid, signal.SIGTERM)
                low[c] = -6   # give it 30 s to exit before acting again
            else:
                log(f"card {c}: {free_mib:.0f} MiB free, none of our processes there")
                low[c] = -60
        time.sleep(TICK)


if __name__ == "__main__":
    main()
