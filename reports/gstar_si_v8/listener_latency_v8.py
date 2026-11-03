"""Paper v8 A4 (review M05 / re-review M07): listener-side lag of whole-meeting online runs with the reads QUEUED.

As reports/gstar_si_v6/listener_latency.py (one interpreter channel, or one channel per speaker; synthesis at the
Talker's measured single-stream rate; play = max(ready, channel free [+0.2 s on a speaker change])), except:
  queue     a meeting is decoded by ONE decoder stream: its reads happen at their scheduled times t_r (window start +
            read time, clamped to open/close detection as before) but a read starts only when the previous one has
            finished:  start_r = max(t_r, finish_{r-1}),  finish_r = start_r + c  (c = --read-ms, measured median of
            one read); a block is committed at finish of its read. --no-queue reproduces v6/v7 (commit = t_r + c).
  RTF       sum of read computation / meeting length; read_delay = finish_r - (t_r + c) (p50 / p90 / max)
  late      share of units heard more than 5 / 10 / 30 s after their source segment ended
  region    a block is "ov" when two or more annotated speakers talk during >= 20% of its source span
  start/ATD OpenSTBench-style, per window: Start = first playback onset - source start of the window's first block;
            ATD = mean over output units of (time the unit is heard - source time it translates), units spread
            uniformly over their block's playback and source span (our definition; not OpenSTBench's own alignment)
  censored  streams not synthesised (more than 400 blocks, talker_in/DROPPED.txt) are reported as never heard: their
            units count with an infinite lag in the "with failures" percentiles
Writes <out> (json, refuses to overwrite).

    python reports/gstar_si_v8/listener_latency_v8.py --dir en2zh --windows <online windows> --hyp <hyp.jsonl> \
        --talker <speech dir> --read-ms 1008 --tok-rate 25 --lang cmn [--layout per-speaker] [--no-queue] --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units  # noqa: E402
from meeting_score import universe  # noqa: E402


def wq(x, w, q):
    if not len(x):
        return None
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    v = float(x[min(np.searchsorted(c, q), len(x) - 1)])
    return v if math.isfinite(v) else None


def two_talk_share(a, b, uts):
    """share of [a, b] during which two or more annotated speakers talk."""
    if b <= a:
        return 0.0
    ev = []
    for u in uts:
        x0, x1 = max(a, u[2]), min(b, u[3])
        if x1 > x0:
            ev += [(x0, 1), (x1, -1)]
    ev.sort()
    n, t0, tot = 0, None, 0.0
    for t, d in ev:
        if n >= 2 and t0 is not None:
            tot += t - t0
        n += d
        t0 = t
    return tot / (b - a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, choices=["en2zh", "zh2en", "ali"])
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--read-ms", type=float, required=True)
    ap.add_argument("--tok-rate", type=float, required=True)
    ap.add_argument("--lang", choices=["cmn", "eng"], required=True)
    ap.add_argument("--gap", type=float, default=0.2)
    ap.add_argument("--layout", choices=["single", "per-speaker"], default="single")
    ap.add_argument("--no-queue", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    c = a.read_ms / 1000
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    H = {h["id"]: h for h in map(json.loads, open(a.hyp, encoding="utf-8")) if h["id"] in W}
    U = universe(a.dir, {W[i]["meeting"] for i in H})
    talker = Path(a.talker)
    prep = {r["id"]: r["blocks"] for r in map(json.loads, open(talker / "talker_in/blocks.jsonl", encoding="utf-8"))}
    dropped = set()
    for f in (talker / "talker_in/DROPPED.txt", talker / "DROPPED_ids.txt"):
        if f.exists():
            dropped |= {ln.split(":")[0].strip() for ln in open(f) if ln.strip()}
    # read times per meeting: every read of every window (1 s period), clamped as the commits are
    reads = defaultdict(set)

    def read_time(w, wm):
        x = max(wm, w["open_detect_ms"])
        if wm >= w["dur_ms"]:
            x = max(x, w["close_detect_ms"])
        return round(w["start_s"] + x / 1000, 3)

    for wid, h in H.items():
        w = W[wid]
        for k in range(1, int(math.ceil(w["dur_ms"] / 1000)) + 1):
            reads[w["meeting"]].add(read_time(w, min(k * 1000, w["dur_ms"])))
        for s in h["segments"]:
            if "write_ms" in s:
                reads[w["meeting"]].add(read_time(w, s["write_ms"]))
    fin, rtf, rdelay = {}, {}, []
    for m, ts in reads.items():
        prev = -1e18
        for t in sorted(ts):
            st = t if a.no_queue else max(t, prev)
            prev = st + c
            fin[(m, t)] = prev
            rdelay.append(prev - (t + c))
        span = max(ts) - min(ts) + 1.0
        rtf[m] = len(ts) * c / span
    by_m = defaultdict(list)
    cens_units = 0
    for r in map(json.loads, open(talker / "talker/results.jsonl", encoding="utf-8")):
        wid, k = r["id"].rsplit("__S", 1)
        w = W.get(wid)
        if w is None or r.get("error") or wid not in H:
            continue
        for b, src in zip(r["blocks"], prep[r["id"]]):
            commit = fin[(w["meeting"], read_time(w, src["write_ms"]))]
            sa = w["start_s"] + (src.get("start_s") or 0.0)
            sb = w["start_s"] + src.get("end_s", w["dur_ms"] / 1000)
            by_m[w["meeting"]].append({"spk": int(k), "win": wid, "commit": commit,
                                       "ready": commit + b.get("codes", 0) / a.tok_rate, "dur": b["dur_ms"] / 1000,
                                       "src_a": sa, "src_b": sb, "n": max(1, len(units(b["text"], a.lang))),
                                       "reg": "ov" if two_talk_share(sa, sb, U[w["meeting"]]) >= 0.2 else "1spk"})
    for sid in dropped:
        wid = sid.rsplit("__S", 1)[0]
        for bl in prep.get(sid, []):
            cens_units += max(1, len(units(bl.get("text", ""), a.lang)))
    comp, wts, reg, onset = [], [], [], []
    backlog, growth = [], []
    starts, atd = [], []
    first = {}
    for m, bs in by_m.items():
        bs.sort(key=lambda b: (b["ready"], b["src_a"]))
        free, last = defaultdict(float), None
        bl_m = []
        for b in bs:
            ch = b["spk"] if a.layout == "per-speaker" else 0
            bl_m.append((b["commit"], max(0.0, free[ch] - b["commit"])))
            gap = a.gap if (a.layout == "single" and last not in (None, b["spk"])) else 0.0
            st = max(b["ready"], free[ch] + gap)
            b["play_a"], b["play_b"] = st, st + b["dur"]
            free[ch], last = b["play_b"], b["spk"]
            comp.append(b["play_b"] - b["src_b"])
            onset.append(b["play_a"] - b["src_a"])
            wts.append(b["n"])
            reg.append(b["reg"])
            n = b["n"]
            for j in range(n):
                heard = b["play_a"] + (j + 0.5) * b["dur"] / n
                srct = b["src_a"] + (j + 0.5) * (b["src_b"] - b["src_a"]) / n
                atd.append(heard - srct)
            f = first.get(b["win"])
            if f is None or b["play_a"] < f[0]:
                first[b["win"]] = (b["play_a"], min(b["src_a"], f[1] if f else 1e18))
        backlog += [x for _, x in bl_m]
        t = np.array([x for x, _ in bl_m])
        q = np.array([y for _, y in bl_m])
        if len(t) > 2 and t.max() > t.min():
            growth.append(float(np.polyfit(t / 60, q, 1)[0]))
    starts = [p - s for p, s in first.values()]
    comp, wts, reg = np.array(comp), np.array(wts, dtype=float), np.array(reg)
    res = {"dir": a.dir, "layout": a.layout, "queue": not a.no_queue, "read_ms": a.read_ms, "tok_rate": a.tok_rate,
           "blocks": int(len(comp)), "units": int(wts.sum()),
           "rtf_mean": round(float(np.mean(list(rtf.values()))), 3), "rtf_max": round(float(max(rtf.values())), 3),
           "read_delay_p50": round(float(np.median(rdelay)), 3), "read_delay_p90": round(float(np.percentile(rdelay, 90)), 3),
           "read_delay_max": round(float(np.max(rdelay)), 3)}
    for name, sel in (("all", np.ones(len(comp), bool)), ("ov", reg == "ov"), ("1spk", reg == "1spk")):
        x, w = comp[sel], wts[sel]
        res[f"comp_{name}"] = {"p50": wq(x, w, 0.5), "p90": wq(x, w, 0.9), "units": int(w.sum()),
                               **{f"late{s}": round(float(w[x > s].sum() / max(w.sum(), 1)), 4) for s in (5, 10, 30)}}
    # with failures: censored units never heard (infinite lag)
    xc = np.concatenate([comp, np.full(1, np.inf)]) if cens_units else comp
    wc = np.concatenate([wts, np.array([cens_units], float)]) if cens_units else wts
    res["censored_streams"], res["censored_units"] = len(dropped), int(cens_units)
    res["comp_with_failures"] = {"p50": wq(xc, wc, 0.5), "p90": wq(xc, wc, 0.9),
                                 **{f"late{s}": round(float(wc[xc > s].sum() / max(wc.sum(), 1)), 4) for s in (5, 10, 30)}}
    res["onset_p50"] = wq(np.array(onset), wts, 0.5)
    res["start_mean"] = round(float(np.mean(starts)), 3) if starts else None
    res["start_p50"] = round(float(np.median(starts)), 3) if starts else None
    res["atd_mean"] = round(float(np.mean(atd)), 3) if atd else None
    res["backlog_mean"] = round(float(np.mean(backlog)), 3)
    res["backlog_max"] = round(float(np.max(backlog)), 3)
    res["backlog_growth_s_per_min"] = round(float(np.median(growth)), 4) if growth else None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
