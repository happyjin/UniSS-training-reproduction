"""Robust K3 scoring (reports/mist_w3po_v1/robust_k3_discrim_prereg.zh-CN.md section 5): S1 dev, point estimates only.
Runs per direction d: B = wave2_S1/talker_<d>_s20261021, V1 = wave3_combos/V/v1/talker_<d> (metrics wave3_combos/V/
metrics/{B,v1}_<d>_*), K3 = k3/talker_<d> (metrics k3/metrics/k3_<d>_*). Point estimates computed by wave3_v_score.compare
(same units and definitions; its bootstrap is cut to one resample and the CI fields are dropped).
Rule (K3 vs V1, every direction x set {a, b}): noisy sim paired median diff > 0; clean sim diff >= 0; ASR-BLEU (noisy and
clean) >= V1 - 0.3; PESQ paired median diff (noisy and clean) >= -0.1. b_noxt reported only.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.k3_score --out <json>
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
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        "v1": lambda d: (W.V / "v1" / f"talker_{d}", W.V / "metrics" / f"v1_{d}"),
        "k3": lambda d: (K3 / f"talker_{d}", K3 / "metrics" / f"k3_{d}")}


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
                res["cells"][f"{d}/{set_}/{cond}"] = {
                    "k3_vs_v1": strip(W.compare(meta, ref, items, P["k3"], P["v1"], LANG[d], "stream")),
                    "k3_vs_B": strip(W.compare(meta, ref, items, P["k3"], P["B"], LANG[d], "stream")),
                    "v1_vs_B": strip(W.compare(meta, ref, items, P["v1"], P["B"], LANG[d], "stream"))}
        for set_ in ("a", "b", "b_noxt"):
            n = res["cells"][f"{d}/{set_}/noisy"]["k3_vs_v1"]
            c = res["cells"][f"{d}/{set_}/clean"]["k3_vs_v1"]
            g = {"noisy_sim": n["sim"]["paired_median_diff"] > 0, "clean_sim": c["sim"]["paired_median_diff"] >= 0,
                 "asr": n["asr"]["diff"] >= -0.3 and c["asr"]["diff"] >= -0.3,
                 "pesq": n["pesq"]["paired_median_diff"] >= -0.1 and c["pesq"]["paired_median_diff"] >= -0.1}
            g["pass"] = all(g.values())
            res["rule"][f"{d}/{set_}"] = g
    res["rule"]["K3_overall"] = all(res["rule"][f"{d}/{s}"]["pass"] for d in ("en2zh", "zh2en") for s in ("a", "b"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
