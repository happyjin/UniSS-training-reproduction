"""Paper v6: finish decode shards whose worker was stopped (by the memory watchdog, by PID). For a system directory,
the windows not yet in any hyp_s*.jsonl are written, per stopped shard k, to <dir>/rest_s<k>.windows.jsonl, and a
gstar_multi_v6 run file is written whose runs decode them into hyp_s<k>r<n>.jsonl (new files; nothing is
overwritten). A shard counts as stopped if its log has no DONE line and no worker of it is alive (--force-dead to skip
the liveness check). Prints the run file and the number of remaining windows.

    python3 reports/gstar_si_v6/make_rest.py <system dir> <decode windows file> <thinker key> <module> <argv-json> \
        [--batch 4] [--workers 2]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_v6_jobs import T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("windows")
    ap.add_argument("thinker")
    ap.add_argument("module")
    ap.add_argument("extra", help="json list of extra argv (policy, window, post-key, dir, ...)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    d = Path(a.dir)
    done = set()
    for f in d.glob("hyp_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except ValueError:
                pass   # a line cut by the stop
    alive = subprocess.run(["bash", "-c", f"ps -eo args | grep -c '[-]-out {d}/hyp_s'"], capture_output=True, text=True).stdout.strip()
    if alive not in ("", "0"):
        sys.exit(f"{alive} decoder(s) of {d} still running")
    W = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
    rest = [w for w in W if w["id"] not in done]
    if not rest:
        print("nothing left")
        return
    n = 1
    while (d / f"rest{n}.windows.jsonl").exists():
        n += 1
    wf = d / f"rest{n}.windows.jsonl"
    with open(wf, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in rest)
    runs = [{"module": a.module,
             "argv": ["--thinker", T[a.thinker], "--windows", str(wf), "--out", str(d / f"hyp_s{k}r{n}.jsonl"), "--shard", str(k),
                      "--num-shards", str(a.workers), "--batch", str(a.batch)] + json.loads(a.extra),
             "log": str(d / f"infer_s{k}r{n}.log")} for k in range(a.workers)]
    rf = d / f"rest{n}.runs.json"
    with open(rf, "x") as fh:
        json.dump(runs, fh, indent=1)
    print(rf, len(rest))


if __name__ == "__main__":
    main()
