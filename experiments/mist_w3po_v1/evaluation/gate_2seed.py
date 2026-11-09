"""Stage gate with the speech item measured over several TTS draws (LOG 2026-11-09 03:09 UTC; the rule of gate.py is
unchanged: point >= ref - 0.2 and paired meeting bootstrap P(diff <= 0) < 0.95 on text all / text ov / speech all,
reference = the better of the refs per item). Speech: per meeting, the BLEU sufficient statistics of the draws are
averaged before compare(), so the bootstrap stays paired (same meetings resampled for ref and cand, same seeds).
Text items come from the first decode dir (all draws share the hypothesis).

    python -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand <seed1 dir> <seed2 dir> \
        --ref <v8 ctx0 seed1> <v8 ctx0 seed2> --ref <v8 ctx2 seed1> <v8 ctx2 seed2> [--out gate_2seed.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.compare import compare
from experiments.mist_w3po_v1.evaluation.gate import P_MAX, TOL


def pooled(scores: list) -> dict:
    """mean over draws of every per-meeting *_stats vector (meetings common to all draws)."""
    ms = sorted(set.intersection(*[set(s["per_meeting"]) for s in scores]))
    out = {"dir": scores[0].get("dir", "zh2en"), "per_meeting": {}}
    for m in ms:
        keys = [k for k in scores[0]["per_meeting"][m] if k.endswith("_stats")]
        out["per_meeting"][m] = {k: (np.mean([np.asarray(s["per_meeting"][m][k] or [0] * 10, dtype=np.float64)
                                              for s in scores], 0).tolist()) for k in keys}
    return out


def gate2(refs: list, cand: list, n: int = 10000) -> dict:
    """refs: [[ref seed dirs...], ...]; cand: [cand seed dirs...] (same seed order everywhere)."""
    ct = json.load(open(Path(cand[0]) / "score_text.json"))
    cs = pooled([json.load(open(Path(d) / "score_asr.json")) for d in cand])
    items, info = {}, {}
    for key, reg in (("text_all", "all"), ("text_ov", "ov"), ("speech_all", "all")):
        best = None
        for r in refs:
            if key.startswith("text"):
                ra, cj = json.load(open(Path(r[0]) / "score_text.json")), ct
            else:
                ra, cj = pooled([json.load(open(Path(d) / "score_asr.json")) for d in r]), cs
            res = compare(ra, cj, regs=(reg,), n=n)[reg]
            res["ref"] = [str(x) for x in r]
            info[f"{key}@{Path(r[0]).name}"] = res
            if best is None or res["A"] > best["A"]:
                best = res
        best = dict(best)
        best["pass"] = bool(best["diff"] >= -TOL and best["p_le0"] < P_MAX)
        items[key] = best
    return {"refs": [[str(x) for x in r] for r in refs], "cand": [str(x) for x in cand], "draws": len(cand),
            "items": items, "per_ref": info, "pass": all(v["pass"] for v in items.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", action="append", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert all(len(r) == len(a.cand) for r in a.ref), "every system needs the same draws"
    r = gate2(a.ref, a.cand, a.n)
    print(json.dumps({k: {"diff": v["diff"], "p_le0": v["p_le0"], "pass": v["pass"]} for k, v in r["items"].items()}))
    if a.out:
        with open(a.out, "x") as fh:
            json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()
