"""Block until one concrete event, then print ONE json line and exit.

Events (first one wins):
  daemon_dead  : a --daemon PID is gone
  pid_gone     : a --pid (e.g. a torchrun) is gone
  drift        : a --drift run hits the RL drift-stop rule (see below)
  file         : a --file path exists
  timeout      : --timeout-min elapsed (hard cap 50)

Drift rule (handoff 2026-11-09 §1.2), from step --drift-from (10) on, any of
three conditions true for --drift-k (5) consecutive steps:
  C1 segs_per_traj > 1.10 x dia(same step)
  C2 spurious_per_traj > dia(same step)  or  rep_mean > 0.0037
  C3 kl > 0.05
--drift NAME:METRICS_JSONL:DIA_METRICS_JSONL  (repeatable)
Read-only: never stops anything itself.
"""
import argparse
import json
import os
import sys
import time


def alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = f"/proc/{int(pid)}/stat"
    try:
        with open(st) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def load(p):
    out = {}
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    out[int(d["step"])] = d
    except OSError:
        pass
    return out


def drift_state(m, dia, start=10, k=5):
    runs = [0, 0, 0]
    last = None
    for s in sorted(m):
        if s < start:
            continue
        d, r = m[s], dia.get(s)
        if r is None:
            break
        c = [
            d["segs_per_traj"] > 1.10 * r["segs_per_traj"],
            d["spurious_per_traj"] > r["spurious_per_traj"] or d.get("rep_mean", 0.0) > 0.0037,
            d["kl"] > 0.05,
        ]
        runs = [x + 1 if ci else 0 for x, ci in zip(runs, c)]
        last = s
        if max(runs) >= k:
            return True, s, runs
    return False, last, runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="append", default=[])
    ap.add_argument("--pid", action="append", default=[])
    ap.add_argument("--drift", action="append", default=[])
    ap.add_argument("--file", action="append", default=[])
    ap.add_argument("--drift-from", type=int, default=10)
    ap.add_argument("--drift-k", type=int, default=5)
    ap.add_argument("--timeout-min", type=float, default=50.0)
    ap.add_argument("--poll", type=float, default=30.0)
    a = ap.parse_args(argv)
    t_end = time.time() + 60.0 * min(a.timeout_min, 50.0)
    while True:
        ev = None
        for p in a.daemon:
            if not alive(p):
                ev = {"event": "daemon_dead", "pid": p}
                break
        if ev is None:
            for p in a.pid:
                if not alive(p):
                    ev = {"event": "pid_gone", "pid": p}
                    break
        if ev is None:
            for spec in a.drift:
                name, mp, dp = spec.split(":")
                hit, s, runs = drift_state(load(mp), load(dp), a.drift_from, a.drift_k)
                if hit:
                    ev = {"event": "drift", "run": name, "step": s, "runs": runs}
                    break
        if ev is None:
            for p in a.file:
                if os.path.exists(p):
                    ev = {"event": "file", "path": p}
                    break
        if ev is None and time.time() >= t_end:
            ev = {"event": "timeout"}
        if ev is not None:
            info = {}
            for spec in a.drift:
                name, mp, dp = spec.split(":")
                m = load(mp)
                _, s, runs = drift_state(m, load(dp), a.drift_from, a.drift_k)
                info[name] = {"last_step": max(m) if m else None, "runs": runs}
            ev["drift_info"] = info
            ev["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(json.dumps(ev))
            sys.stdout.flush()
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
