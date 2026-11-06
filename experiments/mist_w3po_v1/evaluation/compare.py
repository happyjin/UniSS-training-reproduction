"""Paired meeting-level bootstrap of two score files of reports/gstar_si_v8/overlap_score.py (same meetings):
tcpBLEU all / ov / 1spk from the per-meeting BLEU statistics, 10k resamples of meetings; prints the point difference
B - A, the 95 % interval and P(B - A <= 0) (one-sided, for "B not worse / better than A").

    python -m experiments.mist_w3po_v1.evaluation.compare A.json B.json [--n 10000] [--out json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def bleu_of(stats: np.ndarray, eng: bool = True) -> float:
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize="13a" if eng else "zh")._compute_score_from_stats(list(stats)).score


def compare(a: dict, b: dict, regs=("all", "ov", "1spk"), n: int = 10000, seed: int = 0) -> dict:
    eng = a.get("dir", "zh2en") != "en2zh"
    ms = sorted(set(a["per_meeting"]) & set(b["per_meeting"]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ms), size=(n, len(ms)))
    out = {"meetings": len(ms)}
    for reg in regs:
        A = np.array([a["per_meeting"][m][f"{reg}_stats"] or [0] * 10 for m in ms], dtype=np.float64)
        B = np.array([b["per_meeting"][m][f"{reg}_stats"] or [0] * 10 for m in ms], dtype=np.float64)
        pa, pb = bleu_of(A.sum(0), eng), bleu_of(B.sum(0), eng)
        d = np.array([bleu_of(B[i].sum(0), eng) - bleu_of(A[i].sum(0), eng) for i in idx])
        out[reg] = {"A": round(pa, 2), "B": round(pb, 2), "diff": round(pb - pa, 2),
                    "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                    "p_le0": round(float((d <= 0).mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    r = compare(json.load(open(args.a)), json.load(open(args.b)), n=args.n)
    r["A_file"], r["B_file"] = args.a, args.b
    print(json.dumps(r, indent=1))
    if args.out:
        with open(args.out, "x") as fh:
            json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()
