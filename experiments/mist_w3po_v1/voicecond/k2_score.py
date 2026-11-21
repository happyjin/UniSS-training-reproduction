"""K2 step 2 scoring (reports/mist_w3po_v1/robust_k2_w2map_prereg.zh-CN.md): point estimates (no bootstrap) on S1 dev for
B (wave2_S1/talker_<d>_s20261021), V1 (wave3_combos/V/v1) and W2 (k2/w2), all T1c, seed 20261021, same blocks/shards.
Cells: direction x {a, b, b_noxt} x {noisy, clean}, items common to the three arms (wave3_v_score.py cell definition).
Per arm: ASR-BLEU (corpus, speech-only ASR), SQUIM PESQ median, TitaNet similarity median (to the clean source), D2 mean;
plus W2 - V1 paired median of per-unit mean differences (pesq, sim; units = stream, as wave3_v_score.py).
Rule (W2 vs V1, arm medians / corpus BLEU, cells {a, b_noxt} x {en2zh, zh2en}): noisy sim > V1; clean sim >= V1;
ASR-BLEU >= V1 - 0.3 and PESQ >= V1 - 0.1 in both conditions. Refuses an existing --out.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.k2_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, Bleu, jl, laal_defs  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
V = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V"
K2 = REPO / "eval_outputs/mist_w3po_v1/k2"
SEED = "20261021"
PATHS = {"B": (lambda d: S1 / f"talker_{d}_s{SEED}", V / "metrics", "B"),
         "v1": (lambda d: V / "v1" / f"talker_{d}", V / "metrics", "v1"),
         "w2": (lambda d: K2 / "w2" / f"talker_{d}", K2 / "metrics", "w2")}


def load(arm, d, LD):
    rf, md, tag = PATHS[arm]
    run = rf(d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(md / f"{tag}_{d}_squim.jsonl")}
    sim = {r["id"]: r.get("sim") for r in jl(md / f"{tag}_{d}_spksim.jsonl")}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i), "d2": sd["D2"] / 1000 if sd else None}
    return per


def med(P, items, k):
    v = [P[i][k] for i in items if P[i][k] is not None]
    return round(float(np.median(v)), 4) if v else None


def paired_unit_median(X, Y, items, meta, k):
    by = {}
    for i in items:
        if X[i][k] is not None and Y[i][k] is not None:
            by.setdefault(meta[i]["stream"], []).append(X[i][k] - Y[i][k])
    return round(float(np.median([np.mean(v) for v in by.values()])), 4) if by else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed": SEED, "estimates": "point (no bootstrap)", "cells": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in PATHS}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                bl = Bleu(LANG[d])
                cell = {"n_items": len(items)}
                for arm, X in P.items():
                    st = bl.stats([X[i]["asr"] for i in items], [ref[i] for i in items]).sum(0)
                    d2 = [X[i]["d2"] for i in items if X[i]["d2"] is not None]
                    cell[arm] = {"asr_bleu": round(bl.score(st), 3), "pesq_median": med(X, items, "pesq"),
                                 "sim_median": med(X, items, "sim"), "d2_mean": round(float(np.mean(d2)), 4) if d2 else None}
                cell["w2_minus_v1_paired"] = {k: paired_unit_median(P["w2"], P["v1"], items, meta, k) for k in ("pesq", "sim")}
                res["cells"][f"{d}/{set_}/{cond}"] = cell
        for set_ in ("a", "b", "b_noxt"):
            n, c = res["cells"][f"{d}/{set_}/noisy"], res["cells"][f"{d}/{set_}/clean"]
            g = {"noisy_sim_gt_v1": n["w2"]["sim_median"] > n["v1"]["sim_median"],
                 "clean_sim_ge_v1": c["w2"]["sim_median"] >= c["v1"]["sim_median"],
                 "asr_ok": all(x["w2"]["asr_bleu"] >= x["v1"]["asr_bleu"] - 0.3 for x in (n, c)),
                 "pesq_ok": all(x["w2"]["pesq_median"] >= x["v1"]["pesq_median"] - 0.1 for x in (n, c))}
            g["pass"] = all(g.values())
            res["rule"][f"{d}/{set_}"] = g
    res["rule"]["K2_overall"] = all(res["rule"][f"{d}/{s}"]["pass"] for d in ("en2zh", "zh2en") for s in ("a", "b_noxt"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
