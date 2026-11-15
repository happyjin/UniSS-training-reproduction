#!/usr/bin/env python
"""v11 P0-2: one card reserved for the AliMeeting Eval text decodes through the shared pending ledger (fair share).

Why: the 64 GiB decodes of logs/mist_w3po_v1/joint/alieval/{b,m,m4} waited > 3 h because many 24 GiB talker starts of
other orchestrators keep every card's free memory below 64 GiB (each start is a 300 s pending ledger entry).
How: every 120 s, while one of b, m, m4 (in that order) has fewer than 8 started decode shards, append a pending
ledger entry {"owner": "joint_<row>", "id": "aev_reserve_<t>", "card": X, "mib": 69632, "state": "pending"}: every
OTHER owner (finalpost_dev_orch.free_plan) subtracts 68 GiB on card X, so no new job of theirs starts there; the row's
own orchestrator (same owner) ignores it and takes the card once its current jobs end. X = the card with the least
memory in use at start (sticky). Entries expire after 300 s (LEDGER_TTL_S), so the reservation ends by itself <= 5 min
after this process exits. Exits when all 24 decode shards have started. Nothing else is touched.

    python alieval_reserve.py [--rows b,m,m4] [--exclude <card>...]   (detached; log logs/mist_w3po_v1/joint/alieval/reserve*.log)
A second instance (--rows m,m4 --exclude <first card>) reserves a second card for the later rows.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalpost_dev_orch as fo  # noqa: E402

REPO = Path(fo.REPO)
RUN = REPO / "logs/mist_w3po_v1/joint/alieval"
LEDGER = str(REPO / "logs/mist_w3po_v1/locks/pending.jsonl")
NAMES = {"b": "jaev_b_joint_v8h6k_20261113T073658Z_s06000",
         "m": "jaev_m_joint_v2p_20261113T120415Z_s03500_wbzm1",
         "m4": "jaev_m4_joint_v2p_20261113T120415Z_s04000_wbzm1"}


def started(r: str) -> int:
    st = RUN / r / "state"
    return sum(1 for k in range(8) if any(st.glob(f"{NAMES[r]}__dec{k}.a*.pid")))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="b,m,m4")
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    a = ap.parse_args()
    rows = a.rows.split(",")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    use = {int(c): float(m) for c, m in (l.split(",") for l in out.strip().splitlines()) if int(c) not in a.exclude}
    X = min(use, key=use.get)
    log(f"reserving card {X} (used {use[X]:.0f} MiB at start) for the alieval decodes")
    while True:
        todo = [r for r in rows if started(r) < 8]
        if not todo:
            log(f"all decode shards of {rows} started; exit (entries expire within 300 s)")
            return
        r = todo[0]
        t = time.time()
        fo.ledger_append(LEDGER, {"owner": f"joint_{r}", "id": f"aev_reserve_{int(t)}", "card": X, "mib": 69632,
                                  "vllm": False, "state": "pending", "pid": 0})
        log(f"pending entry for joint_{r} on card {X}; started b/m/m4 = {started('b')}/{started('m')}/{started('m4')}")
        time.sleep(120)


if __name__ == "__main__":
    main()
