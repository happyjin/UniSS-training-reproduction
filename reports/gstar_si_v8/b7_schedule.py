"""Paper v8 B7 (prereg_v8 amendment 1): single-channel playback scheduling, replaying the committed and synthesised
blocks of a whole-meeting online run (reads queued as in listener_latency_v8.py; nothing is re-synthesised, so the
speech of every block that is played is unchanged).

  a  as is (v7): every block on one channel, in ready order, 0.2 s gap on a speaker change
  b  short interjections to captions: a block of <= 1 s source inside another speaker's speech (two speakers talk
     during >= 50% of its source span) is shown as a caption, not played
  c  stale skip: a block whose channel is already more than 10 s behind its source end when it is ready is shown as
     a caption, not played
  d  b + c
  e  one channel per speaker mixed into one stream (overlap allowed)
Reported: completion lag of the played units (p50 / p90, late > 5 / 10 / 30 s), units delivered as audio (coverage),
blocks and units turned into captions, backlog max, time from the meeting's last source speech to its last audio.

    python reports/gstar_si_v8/b7_schedule.py --dir en2zh --windows <online windows> --hyp <hyp.jsonl> \
        --talker <speech dir> --read-ms 1008 --tok-rate 161.8 --lang cmn --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listener_latency_v8 as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("--dir", "--windows", "--hyp", "--talker", "--lang", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--read-ms", type=float, required=True)
    ap.add_argument("--tok-rate", type=float, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    c = a.read_ms / 1000
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    H = {h["id"]: h for h in map(json.loads, open(a.hyp, encoding="utf-8")) if h["id"] in W}
    U = L.universe(a.dir, {W[i]["meeting"] for i in H})
    talker = Path(a.talker)
    prep = {r["id"]: r["blocks"] for r in map(json.loads, open(talker / "talker_in/blocks.jsonl", encoding="utf-8"))}

    def read_time(w, wm):
        x = max(wm, w["open_detect_ms"])
        if wm >= w["dur_ms"]:
            x = max(x, w["close_detect_ms"])
        return round(w["start_s"] + x / 1000, 3)

    reads = defaultdict(set)
    for wid, h in H.items():
        w = W[wid]
        for k in range(1, int(math.ceil(w["dur_ms"] / 1000)) + 1):
            reads[w["meeting"]].add(read_time(w, min(k * 1000, w["dur_ms"])))
        for s in h["segments"]:
            if "write_ms" in s:
                reads[w["meeting"]].add(read_time(w, s["write_ms"]))
    fin = {}
    for m, ts in reads.items():
        prev = -1e18
        for t in sorted(ts):
            prev = max(t, prev) + c
            fin[(m, t)] = prev
    by_m = defaultdict(list)
    for r in map(json.loads, open(talker / "talker/results.jsonl", encoding="utf-8")):
        wid, k = r["id"].rsplit("__S", 1)
        w = W.get(wid)
        if w is None or r.get("error") or wid not in H:
            continue
        for b, src in zip(r["blocks"], prep[r["id"]]):
            commit = fin[(w["meeting"], read_time(w, src["write_ms"]))]
            sa = w["start_s"] + (src.get("start_s") or 0.0)
            sb = w["start_s"] + src.get("end_s", w["dur_ms"] / 1000)
            by_m[w["meeting"]].append({"spk": int(k), "ready": commit + b.get("codes", 0) / a.tok_rate,
                                       "dur": b["dur_ms"] / 1000, "src_a": sa, "src_b": sb,
                                       "n": max(1, len(L.units(b["text"], a.lang))),
                                       "inter": (sb - sa) <= 1.0 and L.two_talk_share(sa, sb, U[w["meeting"]]) >= 0.5})
    res = {"dir": a.dir, "read_ms": a.read_ms}
    for pol in "abcde":
        comp, wts, cap_b, cap_u, tot_u, backlog, tail = [], [], 0, 0, 0, [], []
        for m, bs in by_m.items():
            bs = sorted(bs, key=lambda b: (b["ready"], b["src_a"]))
            free, last = defaultdict(float), None
            for b in bs:
                tot_u += b["n"]
                ch = b["spk"] if pol == "e" else 0
                if pol in "bd" and b["inter"]:
                    cap_b, cap_u = cap_b + 1, cap_u + b["n"]
                    continue
                gap = 0.2 if (pol != "e" and last not in (None, b["spk"])) else 0.0
                st = max(b["ready"], free[ch] + gap)
                if pol in "cd" and st - b["src_b"] > 10.0:
                    cap_b, cap_u = cap_b + 1, cap_u + b["n"]
                    continue
                backlog.append(max(0.0, free[ch] - b["ready"]))
                free[ch], last = st + b["dur"], b["spk"]
                comp.append(st + b["dur"] - b["src_b"])
                wts.append(b["n"])
            if free:
                tail.append(max(free.values()) - max(b["src_b"] for b in bs))
        x, w = np.array(comp), np.array(wts, float)
        res[pol] = {"p50": L.wq(x, w, 0.5), "p90": L.wq(x, w, 0.9),
                    **{f"late{s}": round(float(w[x > s].sum() / max(w.sum(), 1)), 4) for s in (5, 10, 30)},
                    "coverage_audio": round(float(w.sum() / max(tot_u, 1)), 4), "caption_blocks": cap_b,
                    "caption_units": cap_u, "backlog_max": round(float(max(backlog)), 2) if backlog else 0.0,
                    "tail_after_meeting_s_p50": round(float(np.median(tail)), 2) if tail else None}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
