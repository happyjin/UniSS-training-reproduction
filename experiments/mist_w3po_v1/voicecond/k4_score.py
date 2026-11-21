"""Robust K4 scoring (reports/mist_w3po_v1/robust_k4_tokencls_prereg.zh-CN.md section 4; copy of oracle_voice_score.py):
S1 dev, point estimates. Arms per direction d: B, V1, K3, oracle (existing outputs) and K4 = k4/talker_<d> (metrics
k4/metrics/k4_<d>_*), each compared with B by wave3_v_score.compare (one resample, CI dropped); K4 also against V1.
Recovered fraction of B's noise loss per cell: (arm noisy sim median - B noisy) / (B clean - B noisy).
Judged cells: en2zh / zh2en x {a, b_noxt}. K4 holds iff (1) mean recovered(K4) >= mean recovered(V1) + 0.20, (2) mean clean
sim median K4 >= V1, (3) ASR-BLEU K4 >= V1 - 0.3 in every judged cell x {noisy, clean}. {a, b} means reported only.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.k4_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.wave3_v_score as W  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, jl, laal_defs  # noqa: E402

W.NB = 1
K3 = REPO / "eval_outputs/mist_w3po_v1/k3"
OR = REPO / "eval_outputs/mist_w3po_v1/oracle_voice"
K4 = REPO / "eval_outputs/mist_w3po_v1/k4"
ARMS = ("v1", "k3", "oracle", "k4")
JUDGED = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b_noxt")]
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        "v1": lambda d: (W.V / "v1" / f"talker_{d}", W.V / "metrics" / f"v1_{d}"),
        "k3": lambda d: (K3 / f"talker_{d}", K3 / "metrics" / f"k3_{d}"),
        "oracle": lambda d: (OR / f"talker_{d}", OR / "metrics" / f"oracle_{d}"),
        "k4": lambda d: (K4 / f"talker_{d}", K4 / "metrics" / f"k4_{d}")}


def load(arm, d, LD):
    run, mp = RUNS[arm](d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(Path(f"{mp}_squim.jsonl"))}
    sim = {r["id"]: r.get("sim") for r in jl(Path(f"{mp}_spksim.jsonl"))}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def strip(c):
    o = {"n_items": c["n_items"], "n_units": c["n_units"]}
    for k in ("asr", "pesq", "sim", "d1", "d2"):
        o[k] = {kk: v for kk, v in c[k].items() if kk != "ci95"}
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed": W.SEED, "estimates": "point (no bootstrap)", "cells": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in RUNS}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                cell = {f"{arm}_vs_B": strip(W.compare(meta, ref, items, P[arm], P["B"], LANG[d], "stream"))
                        for arm in ARMS}
                cell["k4_vs_v1"] = strip(W.compare(meta, ref, items, P["k4"], P["v1"], LANG[d], "stream"))
                res["cells"][f"{d}/{set_}/{cond}"] = cell
        for set_ in ("a", "b", "b_noxt"):
            n = res["cells"][f"{d}/{set_}/noisy"]
            c = res["cells"][f"{d}/{set_}/clean"]
            bn, bc = n["oracle_vs_B"]["sim"]["ref_median"], c["oracle_vs_B"]["sim"]["ref_median"]
            res["rule"][f"{d}/{set_}"] = {
                "B_noisy": bn, "B_clean": bc, "loss": bc - bn,
                **{f"recovered_{arm}": (n[f"{arm}_vs_B"]["sim"]["arm_median"] - bn) / (bc - bn) for arm in ARMS},
                **{f"clean_sim_{arm}": c[f"{arm}_vs_B"]["sim"]["arm_median"] for arm in ARMS},
                **{f"bleu_{cond}_{arm}": res["cells"][f"{d}/{set_}/{cond}"][f"{arm}_vs_B"]["asr"]["arm"]
                   for cond in ("noisy", "clean") for arm in ARMS}}
    R = res["rule"]
    mean = lambda cells, k: sum(R[c][k] for c in cells) / len(cells)  # noqa: E731
    ab = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b")]
    summ = {"judged_cells": JUDGED,
            **{f"mean_recovered_{arm}": mean(JUDGED, f"recovered_{arm}") for arm in ARMS},
            **{f"mean_clean_sim_{arm}": mean(JUDGED, f"clean_sim_{arm}") for arm in ARMS},
            **{f"ab_mean_recovered_{arm}": mean(ab, f"recovered_{arm}") for arm in ARMS}}
    r1 = summ["mean_recovered_k4"] >= summ["mean_recovered_v1"] + 0.20
    r2 = summ["mean_clean_sim_k4"] >= summ["mean_clean_sim_v1"]
    bad = [f"{c}/{cond}" for c in JUDGED for cond in ("noisy", "clean")
           if R[c][f"bleu_{cond}_k4"] < R[c][f"bleu_{cond}_v1"] - 0.3]
    summ.update({"rule1_recovered": r1, "rule2_clean_sim": r2, "rule3_bleu_fail_cells": bad, "rule3_bleu": not bad,
                 "verdict": "K4 holds" if (r1 and r2 and not bad) else "K4 does not hold"})
    R["summary"] = summ
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
