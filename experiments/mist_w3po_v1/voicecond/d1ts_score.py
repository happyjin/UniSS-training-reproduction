"""Robust D1 Talker-sensitivity scoring (reports/mist_w3po_v1/robust_d1_talker_sensitivity_prereg.zh-CN.md section 3): S1 dev
CLEAN items, point estimates. Arms per direction d: C0 / C10 / C45 = d1/talker_<arm>_<d> (metrics d1/metrics/<arm>_<d>_*),
B = wave2_S1/talker_<d>_s20261021 (metrics wave3_combos/V/metrics/B_<d>_*), restricted to the clean items. Cell = direction x
set (a, b judged; b_noxt reported). Drop_X = C0 sim median - X sim median; reference gap G = oracle noisy sim median - V1 noisy
sim median (oracle_voice/summary.json). Arm "large" if drop >= G/2 in >= 3/4 judged cells, "small" if drop < 0.005 in >= 3/4,
else "intermediate". Verdict: C10 or C45 large -> token_bottleneck; both small -> dvector_offset; else intermediate.
Sanity: C0 vs B paired median sim diff within +-0.003 in every judged cell.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.d1ts_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.wave3_v_score as W  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, jl  # noqa: E402

W.NB = 1
D1 = REPO / "eval_outputs/mist_w3po_v1/d1"
OR = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/summary.json"
ARMS = ("C0", "C10", "C45")
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        **{a: (lambda d, a=a: (D1 / f"talker_{a}_{d}", D1 / "metrics" / f"{a}_{d}")) for a in ARMS}}
JUDGED = [(d, s) for d in ("en2zh", "zh2en") for s in ("a", "b")]


def load(arm, d):
    run, mp = RUNS[arm](d)
    recs = {r["id"] for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(Path(f"{mp}_squim.jsonl"))}
    sim = {r["id"]: r.get("sim") for r in jl(Path(f"{mp}_spksim.jsonl"))}
    return {i: {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i), "d1": None, "d2": None} for i in recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    orc = json.load(open(OR, encoding="utf-8"))
    res = {"seed": W.SEED, "estimates": "point (no bootstrap)", "tokens": json.load(open(D1 / "tokens" / "tokens_stats.json")),
           "cells": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d) for arm in RUNS}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == "clean"
                           and (set_ != "b_noxt" or not meta[i]["ovl"]))
            c = {}
            for arm, base in (("C0", "B"), ("C10", "C0"), ("C45", "C0")):
                x = W.compare(meta, ref, items, P[arm], P[base], LANG[d], "stream")
                c[f"{arm}_vs_{base}"] = {"n_items": x["n_items"], "n_units": x["n_units"],
                                         **{k: {kk: v for kk, v in x[k].items() if kk != "ci95"} for k in ("asr", "pesq", "sim")}}
            n = orc["cells"][f"{d}/{set_}/noisy"]
            G = n["oracle_vs_B"]["sim"]["arm_median"] - n["v1_vs_B"]["sim"]["arm_median"]
            Gk3 = n["oracle_vs_B"]["sim"]["arm_median"] - n["k3_vs_B"]["sim"]["arm_median"]
            sim0 = c["C10_vs_C0"]["sim"]["ref_median"]
            r = {"n": len(items), "B_sim": c["C0_vs_B"]["sim"]["ref_median"], "C0_sim": sim0,
                 "C0_minus_B_paired": c["C0_vs_B"]["sim"]["paired_median_diff"], "gap_oracle_minus_v1_noisy": G,
                 "gap_oracle_minus_k3_noisy": Gk3}
            for arm in ("C10", "C45"):
                s = c[f"{arm}_vs_C0"]["sim"]
                r[arm] = {"sim": s["arm_median"], "drop": sim0 - s["arm_median"], "drop_paired": -s["paired_median_diff"],
                          "drop_over_gap": (sim0 - s["arm_median"]) / G if G else None}
            res["cells"][f"{d}/{set_}"] = c
            res["rule"][f"{d}/{set_}"] = r
    cls = {}
    for arm in ("C10", "C45"):
        big = sum(res["rule"][f"{d}/{s}"][arm]["drop"] >= res["rule"][f"{d}/{s}"]["gap_oracle_minus_v1_noisy"] / 2
                  for d, s in JUDGED)
        small = sum(res["rule"][f"{d}/{s}"][arm]["drop"] < 0.005 for d, s in JUDGED)
        cls[arm] = "large" if big >= 3 else "small" if small >= 3 else "intermediate"
    res["rule"]["class"] = cls
    res["rule"]["sanity_C0_vs_B_ok"] = all(abs(res["rule"][f"{d}/{s}"]["C0_minus_B_paired"]) <= 0.003 for d, s in JUDGED)
    res["rule"]["verdict"] = ("token_bottleneck" if "large" in cls.values() else
                              "dvector_offset" if all(v == "small" for v in cls.values()) else "intermediate")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
