"""Pick cards for one more Omni engine (auto_launch.sh), by real nvidia-smi use (coordinator, 2026-11-12 ~20:40 Prague):
a thinker card with >= FREE_T GiB free and another card with >= FREE_K GiB free (each ~5 GiB margin included), or one card
with >= FREE_ONE GiB free for all three stages. Free = total - used + memory of the guard's placeholders
(train_mist_workload.py, which the launcher stops by PID) - others' pending starts in the shared ledger
(logs/mist_w3po_v1/locks/pending.jsonl, younger than 300 s, not yet visible). Cards in --exclude (my engines still
starting) are skipped. Prints "T:K" or nothing.

    python3 pick_cards.py [--exclude 0 4]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

LEDGER = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/mist_w3po_v1/locks/pending.jsonl"
OWNER = "r5_qabc_omnitalker"


def smi(q: str, kind: str = "gpu") -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={q}", "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, timeout=60, check=True).stdout
    return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def ledger_pending() -> list[dict]:
    pend = {}
    try:
        lines = open(LEDGER, encoding="utf-8").read().splitlines()
    except OSError:
        return []
    for l in lines:
        try:
            r = json.loads(l)
        except ValueError:
            continue
        k = (r.get("owner"), r.get("id"))
        if r.get("state") == "pending":
            pend[k] = r
        elif r.get("state") == "visible":
            pend.pop(k, None)
    t = time.time()
    return [r for r in pend.values() if t - float(r.get("t", 0)) < 300 and r.get("owner") != OWNER]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--free-t", type=float, default=93.0)
    ap.add_argument("--free-k", type=float, default=39.0)
    ap.add_argument("--free-one", type=float, default=119.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    gpus = smi("index,uuid,memory.total,memory.used")
    by_uuid = {u: int(i) for i, u, _, _ in gpus}
    free = {int(i): (float(t) - float(u)) / 1024 for i, _, t, u in gpus}
    for uuid, pid, mem in smi("gpu_uuid,pid,used_memory", "compute-apps"):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "scripts/gpu_load/train_mist_workload.py" in cmd and uuid in by_uuid:
            free[by_uuid[uuid]] += float(mem) / 1024
    for r in ledger_pending():
        if r.get("card") in free:
            free[r["card"]] -= float(r.get("mib", 0)) / 1024
    cand = {c: f for c, f in free.items() if c not in a.exclude}
    if a.verbose:
        print("free GiB", {c: round(f, 1) for c, f in sorted(free.items())}, flush=True)
    one = [c for c, f in cand.items() if f >= a.free_one]
    if one:
        c = max(one, key=lambda c: cand[c])
        print(f"{c}:{c}")
        return
    ts = sorted([c for c, f in cand.items() if f >= a.free_t], key=lambda c: -cand[c])
    for t in ts:
        ks = sorted([c for c, f in cand.items() if c != t and f >= a.free_k], key=lambda c: -cand[c])
        if ks:
            print(f"{t}:{ks[0]}")
            return


if __name__ == "__main__":
    main()
