"""Re-queue a failed / stopped gpuq_w3 job and every job downstream of it under new ids (ids get SUFFIX; the commands and
outputs stay the same, so a resumable command continues). Done jobs are kept as dependencies. New job files are written
next to the originals with the suffix in the file name; nothing is overwritten.

    python requeue.py <qdir> <job id> [<job id> ...] [--suffix r1] [--retries N] [--set key=json ...]
--set applies to the re-queued root jobs only (e.g. --set 'only=[4,5,6]').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qdir")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--suffix", default="r1")
    ap.add_argument("--retries", type=int, default=-1)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    q = Path(a.qdir)
    files = {}
    for f in sorted((q / "jobs").glob("*.json")):
        j = json.loads(f.read_text())
        files[j["id"]] = (f, j)

    def done(jid):
        rc = q / "state" / f"{jid}.rc"
        return rc.exists() and rc.read_text().strip() == "0"

    redo = set(a.ids)
    changed = True
    while changed:   # downstream closure (not done jobs only)
        changed = False
        for jid, (_, j) in files.items():
            if jid not in redo and not done(jid) and any(d in redo for d in j.get("deps", [])):
                redo.add(jid)
                changed = True
    new = lambda jid: f"{jid}_{a.suffix}"  # noqa: E731
    for jid in sorted(redo, key=lambda x: files[x][0].name):
        f, j = files[jid]
        assert not done(jid) or jid in a.ids, jid
        k = dict(j)
        k["id"] = new(jid)
        k["deps"] = [new(d) if d in redo else d for d in j.get("deps", [])]
        if jid in a.ids:
            if a.retries >= 0:
                k["retries"] = a.retries
            for s in a.set:
                key, val = s.split("=", 1)
                k[key] = json.loads(val)
        out = f.with_name(f"{f.stem}_{a.suffix}.json")
        with open(out, "x") as fh:
            json.dump(k, fh, indent=1)
        print(f"{jid} -> {k['id']}  ({out.name}) deps {k['deps']}")


if __name__ == "__main__":
    main()
