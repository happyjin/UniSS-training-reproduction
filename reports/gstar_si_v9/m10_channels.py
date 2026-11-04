"""Paper v9 M10: what one channel per speaker means for a listener. Replays MIST-Simul's whole-meeting blocks as
listener_latency_v8.py (reads queued at the measured median read time, Talker durations, one channel per output slot)
and reports per meeting: the median completion lag of each speaker's channel and of the SLOWEST speaker, the share of
playback time with two or more channels sounding at once, and (from B7 policy b) the share of units shown as captions.
Writes <out> (refuses to overwrite).

    python reports/gstar_si_v9/m10_channels.py --out eval_outputs/uniss_qwen3_omni_v1/gstar_v9_20261104/m10.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units  # noqa: E402

EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
V6L = EV / "gstar_v6_20261031/latency"
ON = REPO / "data/online_v6_20261031"


def run(st):
    cfg = json.load(open(V6L / f"{st}_S2dl.json"))
    c, tok = cfg["read_ms"] / 1000, cfg["tok_rate"]
    lang = "cmn" if st == "en2zh" else "eng"
    W = {w["id"]: w for w in map(json.loads, open(ON / f"windows_test_online_{st}.jsonl"))}
    H = {h["id"]: h for h in map(json.loads, open(EV / f"gstar_v7_20261101/on_{st}_S2dl_full/hyp.jsonl")) if h["id"] in W}
    talker = EV / f"gstar_v7_speech_20261101/on_{st}_S2dl_full2"
    prep = {r["id"]: r["blocks"] for r in map(json.loads, open(talker / "talker_in/blocks.jsonl"))}

    def due(w, wm):
        x = max(wm, w["open_detect_ms"])
        if wm >= w["dur_ms"]:
            x = max(x, w["close_detect_ms"])
        return round(w["start_s"] + x / 1000, 3)

    reads = defaultdict(set)
    for wid, h in H.items():
        w = W[wid]
        for k in range(1, int(math.ceil(w["dur_ms"] / 1000)) + 1):
            reads[w["meeting"]].add(due(w, min(k * 1000, w["dur_ms"])))
        for s in h["segments"]:
            if "write_ms" in s:
                reads[w["meeting"]].add(due(w, s["write_ms"]))
    fin = {}
    for m, ts in reads.items():
        prev = -1e18
        for t in sorted(ts):
            prev = max(t, prev) + c
            fin[(m, t)] = prev
    by = defaultdict(list)
    for r in map(json.loads, open(talker / "talker/results.jsonl")):
        wid, k = r["id"].rsplit("__S", 1)
        w = W.get(wid)
        if w is None or r.get("error") or wid not in H:
            continue
        for b, src in zip(r["blocks"], prep[r["id"]]):
            cm = fin[(w["meeting"], due(w, src["write_ms"]))]
            by[w["meeting"]].append({"spk": int(k), "ready": cm + b.get("codes", 0) / tok, "dur": b["dur_ms"] / 1000,
                                     "src_b": w["start_s"] + src.get("end_s", w["dur_ms"] / 1000),
                                     "n": max(1, len(units(b["text"], lang)))})
    per = {}
    for m, bs in by.items():
        free = defaultdict(float)
        lags = defaultdict(list)
        iv = []
        for b in sorted(bs, key=lambda b: (b["ready"], b["src_b"])):
            st_ = max(b["ready"], free[b["spk"]])
            free[b["spk"]] = st_ + b["dur"]
            lags[b["spk"]] += [st_ + b["dur"] - b["src_b"]] * b["n"]
            iv.append((st_, st_ + b["dur"]))
        ev = sorted([(a, 1) for a, _ in iv] + [(e, -1) for _, e in iv])
        on, t0, any_t, multi_t = 0, None, 0.0, 0.0
        for t, d in ev:
            if t0 is not None:
                any_t += (t - t0) * (on >= 1)
                multi_t += (t - t0) * (on >= 2)
            on += d
            t0 = t
        p50 = {k: float(np.median(v)) for k, v in lags.items() if len(v) >= 5}
        per[m] = {"speakers": len(p50), "slowest_p50": round(max(p50.values()), 2) if p50 else None,
                  "fastest_p50": round(min(p50.values()), 2) if p50 else None,
                  "simultaneous_share": round(multi_t / max(any_t, 1e-9), 4)}
    sl = [v["slowest_p50"] for v in per.values() if v["slowest_p50"] is not None]
    sim = [v["simultaneous_share"] for v in per.values()]
    b7 = json.load(open(EV / f"gstar_v8_20261103/b7/{st}_S2dl.json"))
    return {"meetings": len(per), "slowest_p50_median": round(float(np.median(sl)), 2),
            "slowest_p50_max": round(float(np.max(sl)), 2), "simultaneous_share_mean": round(float(np.mean(sim)), 4),
            "caption_share_policy_b": round(1 - b7["b"]["coverage_audio"], 4), "per_meeting": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {st: run(st) for st in ("en2zh", "zh2en", "ali")}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_meeting"} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
