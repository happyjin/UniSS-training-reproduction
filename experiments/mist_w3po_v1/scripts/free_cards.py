"""Free cards for an exclusive (training) job: optionally wait until a job has an exit code, then for every running
queue job whose id starts with one of --prefix and whose cards (state/<id>.started) intersect --cards:
move its job file from <qdir>/jobs to <qdir>/held (so the queue does not retry it) and SIGTERM the processes of its
wrapper's descendants by PID (wrapper pid from state/<id>.pid; descendants read from /proc; never this process).
The stopped jobs are meant to be requeued under new ids (their tools resume from their outputs).

    python free_cards.py --qdir Q --cards 4,5,6,7 --prefix anchors_main_ [--after-rc dev_pilot50_ctx0__asr] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def children() -> dict:
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                pp = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(pp, []).append(int(d))
    return kids


def tree(root: int, kids: dict) -> list[int]:
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--after-rc", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = Path(a.qdir)
    cards = {int(c) for c in a.cards.split(",")}
    if a.after_rc and not a.dry_run:
        while not (q / "state" / f"{a.after_rc}.rc").exists():
            time.sleep(20)
        print(f"{time.strftime('%H:%M:%S')} {a.after_rc} finished", flush=True)
    kids = children()
    me = os.getpid()
    for f in sorted((q / "jobs").glob("*.json")):
        try:
            jid = json.loads(f.read_text())["id"]
        except Exception:
            continue
        if not any(jid.startswith(p) for p in a.prefix):
            continue
        st = q / "state"
        if (st / f"{jid}.rc").exists() or not (st / f"{jid}.started").exists():
            continue
        try:
            jc = set(json.loads((st / f"{jid}.started").read_text()).get("cards", []))
            wp = int((st / f"{jid}.pid").read_text().strip())
        except Exception:
            continue
        if not jc & cards:
            continue
        pids = [p for p in tree(wp, kids) if p not in (me, wp)]   # the wrapper stays and writes the job's rc
        print(f"{jid} on cards {sorted(jc)}: wrapper {wp}, processes {pids}", flush=True)
        if a.dry_run:
            continue
        os.replace(f, q / "held" / f.name)
        for p in reversed(pids):      # deepest first
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
    print("FREE_CARDS_DONE", flush=True)


if __name__ == "__main__":
    main()
