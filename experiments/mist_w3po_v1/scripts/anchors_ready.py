"""Wait until the precomputed FLAD anchors cover every window of RL steps [--from-step, --steps) (the trainer's stream,
rl/precompute_anchors.window_stream), reading the part_*.jsonl of several cache dirs (e.g. one run from step 12 and a
stacked second-half run from step 31 into another dir). Then stop the precompute processes that are still running
on these dirs (they would only redo windows the other dir has): their queue job files are first moved from
<qdir>/jobs to <qdir>/held (so the queue does not retry them), then each process is sent SIGTERM by its PID (read
from /proc, never this process). Exit 0 when covered.

    python anchors_ready.py --qdir <qdir> --dirs d1,d2 --from-step 12 --steps 50 --world 8 --ctx-max 0 \
        --jobs anchors_p12_,anchors_p31_ [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"


def done_ids(dirs: list[str]) -> set:
    ids = set()
    for d in dirs:
        for f in Path(d).glob("part_*.jsonl"):
            with open(f, "rb") as fh:
                data = fh.read()
            for l in data[: data.rfind(b"\n") + 1].splitlines():
                try:
                    ids.add(json.loads(l)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def precompute_pids(dirs: list[str]) -> list[int]:
    me = os.getpid()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cl = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="ignore") for a in cl]
        if "experiments.mist_w3po_v1.rl.precompute_anchors" in args and "--out" in args:
            o = args[args.index("--out") + 1]
            if any(os.path.normpath(o) == os.path.normpath(x) for x in dirs):
                out.append(int(d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--from-step", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--windows-per-step", type=int, default=64)
    ap.add_argument("--ctx-max", type=int, default=0)
    ap.add_argument("--jobs", default="", help="comma list of job-id prefixes whose files go to held/ when covered")
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from experiments.mist_w3po_v1.rl.precompute_anchors import window_stream
    ns = argparse.Namespace(windows="eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl",
                            max_dur_ms=20500, seed=20261107, windows_per_step=a.windows_per_step, oversample=1.0,
                            world=a.world, from_step=a.from_step, steps=a.steps)
    need = {w["id"] for w in window_stream(ns)}
    dirs = a.dirs.split(",")
    while True:
        miss = need - done_ids(dirs)
        print(f"{time.strftime('%H:%M:%S')} {len(need) - len(miss)}/{len(need)} windows covered", flush=True)
        if not miss or a.dry_run:
            break
        time.sleep(a.poll)
    if a.dry_run:
        print("dry run: would hold jobs", a.jobs, "and stop", precompute_pids(dirs))
        return
    q = Path(a.qdir)
    pre = [p for p in a.jobs.split(",") if p]
    for f in sorted((q / "jobs").glob("*.json")):
        try:
            jid = json.loads(f.read_text())["id"]
        except Exception:
            continue
        if any(jid.startswith(p) for p in pre) and not (q / "state" / f"{jid}.rc").exists():
            (q / "held").mkdir(exist_ok=True)
            os.replace(f, q / "held" / f.name)
            print(f"held {f.name}", flush=True)
    for pid in precompute_pids(dirs):
        print(f"SIGTERM precompute pid {pid}", flush=True)
        os.kill(pid, signal.SIGTERM)
    print("ANCHORS_READY", flush=True)


if __name__ == "__main__":
    main()
