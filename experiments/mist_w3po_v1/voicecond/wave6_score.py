"""Robust wave 6 scoring (reports/mist_w3po_v1/robust_wave6_disen_prereg.zh-CN.md, incl. 变更：统一判定标准).

Runs per direction d (same S1 blocks, 4 shards, seed 20261021, voice ids = wave3 V1 C-token table):
  B     wave2_S1/talker_<d>_s20261021                  (T1c, raw voice ids; S1's baseline)
  v1    wave3_combos/V/v1/talker_<d>                     (T1c, C ids)
  tft   wave6_disen/eval/tft/talker_<d>                  (plain LoRA control, C ids)
  disen wave6_disen/eval/disen/talker_<d>                (disentangled Talker, c_id only, C ids)
Metrics as wave3_v_score.compare (ASR-BLEU from unit-summed statistics, SQUIM PESQ and TitaNet similarity paired medians,
D1/D2 item means; 2000 unit resamples, seed 0); cells d x {a, b, b_noxt} x {noisy, clean}.

PRIMARY (unified rule + 变更 2, all of: both directions, cells a and b_noxt), arm vs B; D2 LAAL is auxiliary (reported only):
  ASR-BLEU noisy and clean: diff >= -0.3 and CI high >= 0;   (aux: D2 noisy and clean diff <= +0.10 s);
  noisy PESQ: paired median diff > 0 and CI low > 0.
SECONDARY (gates 2-5 of the prereg, noisy, cells a and b_noxt): 2 d2(disen - tft) <= 0 and d2(disen - v1) <= +0.02;
  3 pesq >= -0.05 vs tft and v1; 4 asr >= -0.5 and CI high >= 0 vs tft and v1; 5 sim >= -0.02 vs v1.
Gate 1 (duration variability): durtest/<model>_<cond>_<d>/results.jsonl, model in {t1c, tft, disen}, cond in
  {raw, c, v2, cclean}: per item CV = std/mean of the code counts over the 4 conditions; paired disen - tft (and - t1c):
  relative median change = median(diff) / median(CV_tft) <= -0.5 and CI high of median(diff) < 0 (2000 item resamples).

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.wave6_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, jl, laal_defs  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
V = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V"
W6 = REPO / "eval_outputs/mist_w3po_v1/wave6_disen"
SEED = "20261021"
NB = 2000


def run_dir(arm, d):
    if arm == "B":
        return S1 / f"talker_{d}_s{SEED}"
    if arm == "v1":
        return V / "v1" / f"talker_{d}"
    return W6 / "eval" / arm / f"talker_{d}"


def metrics_dir(arm):
    return V / "metrics" if arm in ("B", "v1") else W6 / "metrics"


def load(arm, d, LD):
    run, M = run_dir(arm, d), metrics_dir(arm)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(M / f"{arm}_{d}_squim.jsonl")}
    sim = {r["id"]: r.get("sim") for r in jl(M / f"{arm}_{d}_spksim.jsonl")}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def primary(noisy, clean):
    g = {"asr_noisy": noisy["asr"]["diff"] >= -0.3 and noisy["asr"]["ci95"][1] >= 0,
         "asr_clean": clean["asr"]["diff"] >= -0.3 and clean["asr"]["ci95"][1] >= 0,
         "pesq_noisy": noisy["pesq"]["paired_median_diff"] > 0 and noisy["pesq"]["ci95"][0] > 0}
    g["pass"] = all(g.values())
    # 变更 2: D2 LAAL is auxiliary only (reported, not part of the pass)
    g["aux_d2_noisy_le_0.10"] = noisy["d2"]["diff"] <= 0.10
    g["aux_d2_clean_le_0.10"] = clean["d2"]["diff"] <= 0.10
    return g


def duration_gate():
    out = {}
    for d in ("en2zh", "zh2en"):
        codes = {}
        for model in ("t1c", "tft", "disen"):
            per = {}
            for cond in ("raw", "c", "v2", "cclean"):
                p = W6 / "durtest" / f"{model}_{cond}_{d}" / "results.jsonl"
                if not p.exists():
                    return {"missing": str(p)}
                for r in jl(p):
                    if r.get("codes"):
                        per.setdefault(r["id"], {})[cond] = float(r["codes"])
            codes[model] = {i: np.std(list(v.values())) / np.mean(list(v.values())) for i, v in per.items() if len(v) == 4}
        ids = sorted(set.intersection(*[set(c) for c in codes.values()]))
        cv = {m: np.array([codes[m][i] for i in ids]) for m in codes}
        res = {"n": len(ids), "median_cv": {m: round(float(np.median(v)), 4) for m, v in cv.items()},
               "mean_cv": {m: round(float(np.mean(v)), 4) for m, v in cv.items()}}
        rng = np.random.default_rng(0)
        for ref in ("tft", "t1c"):
            dv = cv["disen"] - cv[ref]
            med = float(np.median(dv))
            boots = [float(np.median(dv[rng.integers(0, len(dv), len(dv))])) for _ in range(NB)]
            rel = med / max(float(np.median(cv[ref])), 1e-9)
            res[f"disen_vs_{ref}"] = {"median_diff": round(med, 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                      round(float(np.percentile(boots, 97.5)), 4)], "relative": round(rel, 3)}
        r = res["disen_vs_tft"]
        res["gate1"] = r["relative"] <= -0.5 and r["ci95"][1] < 0
        out[d] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed": SEED, "cells": {}, "primary": {}, "secondary": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in ("B", "v1", "tft", "disen")}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                cell = {}
                for x, y in (("disen", "B"), ("tft", "B"), ("v1", "B"), ("disen", "tft"), ("disen", "v1"), ("tft", "v1")):
                    cell[f"{x}_vs_{y}"] = compare(meta, ref, items, P[x], P[y], LANG[d], "stream")
                res["cells"][f"{d}/{set_}/{cond}"] = cell
        for set_ in ("a", "b", "b_noxt"):
            C = lambda c, k: res["cells"][f"{d}/{set_}/{c}"][k]  # noqa: E731
            for arm in ("disen", "tft", "v1"):
                res["primary"][f"{arm}/{d}/{set_}"] = primary(C("noisy", f"{arm}_vs_B"), C("clean", f"{arm}_vs_B"))
            n = lambda k: C("noisy", k)  # noqa: E731
            res["secondary"][f"{d}/{set_}"] = {
                "g2_d2": n("disen_vs_tft")["d2"]["diff"] <= 0 and n("disen_vs_v1")["d2"]["diff"] <= 0.02,
                "g3_pesq": n("disen_vs_tft")["pesq"]["paired_median_diff"] >= -0.05
                and n("disen_vs_v1")["pesq"]["paired_median_diff"] >= -0.05,
                "g4_asr": all(n(k)["asr"]["diff"] >= -0.5 and n(k)["asr"]["ci95"][1] >= 0 for k in ("disen_vs_tft", "disen_vs_v1")),
                "g5_sim": n("disen_vs_v1")["sim"]["paired_median_diff"] >= -0.02}
    cells = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b_noxt")]
    for arm in ("disen", "tft", "v1"):
        res["primary"][f"{arm}_overall"] = all(res["primary"][f"{arm}/{c}"]["pass"] for c in cells)
    res["secondary"]["gate1_duration"] = duration_gate()
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"primary": res["primary"], "secondary": res["secondary"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
