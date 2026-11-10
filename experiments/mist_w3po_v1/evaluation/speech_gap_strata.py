"""Is the talker's fidelity loss in multi-speaker windows explained by fragmentation? (CPU; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md §2.) From speech_gap_decomp streams.jsonl files (one per TTS seed): Whisper WER against
the talker input, micro-averaged, by streams-in-window (1 / 2 / 3+) within strata of words per segment and of stream
length, by forced blocks, and by speech seconds per word; plus a logistic-free linear check (least squares of per-stream
WER on the factors, weighted by words). Writes <out> (json; refuses to overwrite).

    python -m experiments.mist_w3po_v1.evaluation.speech_gap_strata --streams <a.streams.jsonl> [...] --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wer(rows):
    n = sum(r["n"] for r in rows)
    return {"streams": len(rows), "words": n, "wer": round(sum(r["S"] + r["D"] + r["I"] for r in rows) / max(n, 1), 4),
            "del": round(sum(r["D"] for r in rows) / max(n, 1), 4)} if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    R = [json.loads(l) for f in args.streams for l in open(f, encoding="utf-8")]
    R = [r for r in R if r["n"] > 0]
    ns = lambda r: "1" if r["nstreams"] == 1 else ("2" if r["nstreams"] == 2 else "3+")  # noqa: E731
    wps = [("<=2", 0, 2), ("2-5", 2, 5), ("5-10", 5, 10), (">10", 10, 1e9)]
    ln = [("1-4", 0, 4), ("5-16", 4, 16), ("17-32", 16, 32), (">32", 32, 1e9)]
    res = {"streams": len(R), "by_nstreams_x_words_per_seg": {}, "by_nstreams_x_len": {}, "by_nstreams_noforced_x_wps": {}}
    for name, lo, hi in wps:
        res["by_nstreams_x_words_per_seg"][name] = {s: wer([r for r in R if lo < r["wps"] <= hi and ns(r) == s]) for s in ("1", "2", "3+")}
        res["by_nstreams_noforced_x_wps"][name] = {s: wer([r for r in R if lo < r["wps"] <= hi and ns(r) == s and not r["forced"]])
                                                   for s in ("1", "2", "3+")}
    for name, lo, hi in ln:
        res["by_nstreams_x_len"][name] = {s: wer([r for r in R if lo < r["n"] <= hi and ns(r) == s]) for s in ("1", "2", "3+")}
    res["by_ov_x_wps"] = {name: {o: wer([r for r in R if lo < r["wps"] <= hi and (r["multi_share"] >= 0.2) == (o == "ov")])
                                 for o in ("1spk", "ov")} for name, lo, hi in wps}
    res["forced_by_nstreams"] = {s: round(np.mean([r["forced"] for r in R if ns(r) == s]), 4) for s in ("1", "2", "3+")}
    res["spt_by_nstreams"] = {s: round(float(np.median([r["spt"] for r in R if ns(r) == s])), 4) for s in ("1", "2", "3+")}
    # weighted least squares: stream WER ~ 1 + log(words per segment) + log(n) + [2 streams] + [3+ streams] + forced + ov
    X = np.array([[1.0, np.log(max(r["wps"], 1e-3)), np.log(r["n"]), float(r["nstreams"] == 2), float(r["nstreams"] >= 3),
                   float(r["forced"]), float(r["multi_share"] >= 0.2)] for r in R])
    y = np.array([(r["S"] + r["D"] + r["I"]) / r["n"] for r in R])
    w = np.sqrt(np.array([r["n"] for r in R], dtype=float))
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    res["wls"] = dict(zip(["const", "log_words_per_seg", "log_len", "nstreams_2", "nstreams_3plus", "forced", "ov_stream"],
                          [round(float(b), 4) for b in beta]))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
