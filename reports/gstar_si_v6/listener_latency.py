"""Paper v6, review M03 (prereg_v6 §1): what a listener hears, and when, over whole meetings.

Every committed block of an online run (build_online_windows.py windows, decoded read by read) is placed on ONE
interpreter channel per meeting, kept for the whole meeting:
  commit      the block's read time, clamped to the window's opening (open_detect_ms); a last-read block moved to the
              moment the window's end is known (close_detect_ms); plus the measured computation of a read (--read-ms,
              median single-stream time of one read, from the --time-reads runs)
  ready       commit + synthesis: the block's codec tokens / --tok-rate (measured single-stream Talker rate)
  play        max(ready, end of the previous block on the channel + 0.2 s if the speaker changes)   (Eq. 18);
              --layout per-speaker gives every speaker slot its own channel instead (overlapping speech stays
              overlapped, as in the meeting; review T17), so blocks wait only for the same speaker's previous block
Durations are the Talker's own (si_infer results). Reported over blocks (weighted by their units):
  completion lag   play end - end of the block's source segment (window start + the segment's predicted end)
  onset lag        play start - start of the block's source segment
  backlog          channel still busy when a block is committed (s), its mean / p95 / max, and its growth over the meeting
  speech share     seconds of translation speech per second of meeting
Writes <out> (json).

    python reports/gstar_si_v6/listener_latency.py --windows <online windows> --hyp <hyp.jsonl> --talker <speech dir> \
        --read-ms 240 --tok-rate 180 --lang cmn --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units  # noqa: E402


def wq(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--talker", required=True, help="speech dir with talker_in/blocks.jsonl and talker/results.jsonl")
    ap.add_argument("--read-ms", type=float, required=True)
    ap.add_argument("--tok-rate", type=float, required=True)
    ap.add_argument("--lang", choices=["cmn", "eng"], required=True)
    ap.add_argument("--gap", type=float, default=0.2)
    ap.add_argument("--layout", choices=["single", "per-speaker"], default="single")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    H = {h["id"]: h for h in map(json.loads, open(a.hyp, encoding="utf-8"))}
    prep = {r["id"]: r["blocks"] for r in map(json.loads, open(Path(a.talker) / "talker_in/blocks.jsonl", encoding="utf-8"))}
    by_m = defaultdict(list)
    n_res = 0
    for r in map(json.loads, open(Path(a.talker) / "talker/results.jsonl", encoding="utf-8")):
        wid, k = r["id"].rsplit("__S", 1)
        w = W.get(wid)
        if w is None or r.get("error"):
            continue
        n_res += 1
        for b, src in zip(r["blocks"], prep[r["id"]]):
            wm = src["write_ms"]
            c = max(wm, w["open_detect_ms"])
            if wm >= w["dur_ms"]:
                c = max(c, w["close_detect_ms"])
            commit = w["start_s"] + c / 1000 + a.read_ms / 1000
            by_m[w["meeting"]].append({"spk": int(k), "commit": commit, "ready": commit + b.get("codes", 0) / a.tok_rate,
                                       "dur": b["dur_ms"] / 1000, "src_a": w["start_s"] + (src.get("start_s") or 0.0),
                                       "src_b": w["start_s"] + src.get("end_s", w["dur_ms"] / 1000),
                                       "n": max(1, len(units(b["text"], a.lang)))})
    comp, onset, wts, backlog, growth, share = [], [], [], [], [], []
    per_meeting = {}
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
        backlog += [x for _, x in bl_m]
        t = np.array([x for x, _ in bl_m])
        q = np.array([y for _, y in bl_m])
        if len(t) > 2 and t.max() > t.min():
            growth.append(float(np.polyfit(t / 60, q, 1)[0]))   # s of backlog per minute of meeting
        span = max(b["play_b"] for b in bs) - min(b["src_a"] for b in bs)
        share.append(sum(b["dur"] for b in bs) / max(span, 1e-9))
        lags = [b["play_b"] - b["src_b"] for b in bs]
        per_meeting[m] = {"blocks": len(bs), "completion_p50": round(float(np.median(lags)), 3),
                          "backlog_max": round(float(q.max()) if len(q) else 0.0, 2)}
    res = {"layout": a.layout, "streams": n_res, "blocks": len(comp), "read_ms": a.read_ms, "tok_rate": a.tok_rate,
           "comp_p50": round(wq(comp, wts, 0.5), 3), "comp_p90": round(wq(comp, wts, 0.9), 3),
           "comp_p95": round(wq(comp, wts, 0.95), 3), "comp_mean": round(float(np.average(comp, weights=wts)), 3),
           "onset_p50": round(wq(onset, wts, 0.5), 3), "onset_p90": round(wq(onset, wts, 0.9), 3),
           "onset_p95": round(wq(onset, wts, 0.95), 3),
           "backlog_mean": round(float(np.mean(backlog)), 3), "backlog_p95": round(float(np.percentile(backlog, 95)), 3),
           "backlog_max": round(float(np.max(backlog)), 3),
           "backlog_growth_s_per_min": round(float(np.median(growth)), 4) if growth else None,
           "speech_share": round(float(np.mean(share)), 3), "per_meeting": per_meeting}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_meeting"}))


if __name__ == "__main__":
    main()
