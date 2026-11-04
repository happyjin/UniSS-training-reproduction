"""Paper v8 B6 (prereg_v8 amendment 3): wall-clock latency of one whole meeting per set from the batch-1 timed decodes
(gstar_si_stream_online_v6s --time-reads: per window, "read_ms" = measured time of every read).

  schedule   read r of a window is due at the window's start + its read time, clamped to the window's opening
             (open_detect_ms) and, for the last read, to the moment its end is known (close_detect_ms), as in
             listener_latency_v8.py
  queue      one decoder stream per meeting: start = max(due, previous end), end = start + read_ms
  RTF        sum of read_ms / meeting span (first due .. last due + 1 s); since the reads ran beside other jobs on
             the same cards, it is an UPPER bound on the single-stream RTF
  text lag   wall-clock commit of a segment (end of the read that wrote it) minus its source end; shares > 5 / 10 / 30 s;
             also the lag without computation (commit = due), and the queue delay of the reads
  failures   windows of the meeting without output (crash / timeout), reads slower than the 1 s cadence
  co-tenancy (optional --cotenancy b6/cotenancy.jsonl) per card: share of samples with another job's process and the
             mean memory of other processes
Writes <out> (json; refuses to overwrite).

    python reports/gstar_si_v8/b6_wallclock.py --b6 eval_outputs/.../gstar_v8_20261103/b6 --out <b6>/b6_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CADENCE_MS = 1000


def read_due(w, t_ms):
    x = max(t_ms, w["open_detect_ms"])
    if t_ms >= w["dur_ms"]:
        x = max(x, w["close_detect_ms"])
    return w["start_s"] + x / 1000


def analyse(windows, hyps, period_ms=CADENCE_MS):
    """windows: the meeting's online windows; hyps: {id: hyp with read_ms}. Returns the summary dict of one run."""
    reads, segs = [], []
    missing = 0
    for w in windows:
        h = hyps.get(w["id"])
        if h is None or "read_ms" not in h:
            missing += 1
            continue
        R = len(h["read_ms"])
        ts = [min((r + 1) * period_ms, w["dur_ms"]) for r in range(R)]
        idx = [len(reads) + r for r in range(R)]
        for r in range(R):
            reads.append({"due": read_due(w, ts[r]), "ms": h["read_ms"][r], "win": w["id"], "t": ts[r]})
        for s in h["segments"]:
            if "write_ms" not in s:
                continue
            r = next((k for k, t in enumerate(ts) if t >= s["write_ms"]), R - 1)
            segs.append({"read": idx[r], "src_end": w["start_s"] + max(s["end_s"], s["start_s"]),
                         "due": read_due(w, s["write_ms"])})
    order = sorted(range(len(reads)), key=lambda i: (reads[i]["due"], i))
    prev = -math.inf
    for i in order:
        st = max(reads[i]["due"], prev)
        reads[i]["end"] = st + reads[i]["ms"] / 1000
        reads[i]["wait"] = st - reads[i]["due"]
        prev = reads[i]["end"]
    res = {"windows": len(windows), "windows_missing": missing, "reads": len(reads), "segments": len(segs)}
    if not reads:
        return res
    span = max(r["due"] for r in reads) - min(r["due"] for r in reads) + 1.0
    ms = np.array([r["ms"] for r in reads], float)
    res["rtf"] = round(float(ms.sum() / 1000 / span), 3)
    res["read_ms_p50"] = round(float(np.median(ms)), 1)
    res["read_ms_p90"] = round(float(np.percentile(ms, 90)), 1)
    res["reads_over_cadence"] = round(float((ms > period_ms).mean()), 4)
    wq = np.array([r["wait"] for r in reads])
    res["queue_wait_p50"] = round(float(np.median(wq)), 3)
    res["queue_wait_max"] = round(float(wq.max()), 3)
    if segs:
        lag = np.array([reads[s["read"]]["end"] - s["src_end"] for s in segs])
        lag0 = np.array([s["due"] - s["src_end"] for s in segs])
        res["text_lag_wall"] = {"p50": round(float(np.median(lag)), 3), "p90": round(float(np.percentile(lag, 90)), 3),
                                **{f"late{t}": round(float((lag > t).mean()), 4) for t in (5, 10, 30)}}
        res["text_lag_nocompute_p50"] = round(float(np.median(lag0)), 3)
    return res


def cotenancy(path):
    by = defaultdict(lambda: {"samples": set(), "foreign": set(), "foreign_mib": []})
    for ln in open(path):
        r = json.loads(ln)
        g = by[r["gpu"]]
        g["samples"].add(r["t"])
        if not r["ours"]:
            g["foreign"].add(r["t"])
            g["foreign_mib"].append(r["mib"])
    return {str(k): {"samples": len(v["samples"]), "share_with_foreign": round(len(v["foreign"]) / max(len(v["samples"]), 1), 3),
                     "foreign_mib_mean": round(float(np.mean(v["foreign_mib"])), 0) if v["foreign_mib"] else 0}
            for k, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b6", required=True)
    ap.add_argument("--cotenancy", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    b6 = Path(a.b6)
    res = {"note": "reads ran beside other jobs on the same cards: RTF is an upper bound on single-stream RTF "
                   "(prereg_v8 amendment 3)", "runs": {}}
    for wf in sorted(b6.glob("*_windows.jsonl")):
        st, m = wf.name.split("_")[:2]
        W = [json.loads(ln) for ln in open(wf)]
        for tag in ("S2dl", "O2_LA2"):
            hf = b6 / f"{st}_{m}_{tag}_hyp.jsonl"
            if not hf.exists():
                res["runs"][f"{st}_{tag}"] = {"meeting": W[0]["meeting"] if W else m, "status": "not run"}
                continue
            H = {}
            for ln in open(hf):
                try:
                    h = json.loads(ln)
                    H[h["id"]] = h
                except json.JSONDecodeError:
                    pass
            r = analyse(W, H)
            r["meeting"] = W[0]["meeting"] if W else m
            r["status"] = "complete" if r["windows_missing"] == 0 else "partial (stopped, prereg_v8 amendment 4)"
            res["runs"][f"{st}_{tag}"] = r
    if a.cotenancy and Path(a.cotenancy).exists():
        res["cotenancy"] = cotenancy(a.cotenancy)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
