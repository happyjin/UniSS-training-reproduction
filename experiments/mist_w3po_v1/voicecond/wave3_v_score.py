"""Robust wave 3 V group scoring and gates (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md, V 组 + 变更 1:
one TTS seed 20261021, direct pairing).

Runs per direction d: B = wave2_S1/talker_<d>_s20261021 (S1's baseline), v1/v2/v3 = wave3_combos/V/<arm>/talker_<d>; all
with the same blocks, shards and seed. Metrics per item: ASR-BLEU (run/asr_speech.jsonl, paper speech-only protocol),
SQUIM PESQ and TitaNet similarity to the clean source (<metrics>/<arm>_<d>_{squim,spksim}.jsonl), LAAL D1 / D2
(laal_defs.sentence_defs of results.jsonl).

Cells: direction x set {a, b, b_noxt (b turns without crosstalk)} x condition {noisy, clean}. Units: a = sentence,
b = single-speaker stream. Paired arm - ref, 2000 bootstrap resamples of units (seed 0):
  pesq, sim : per unit mean of item differences -> median over units (+ CI of the median)
  asr       : corpus BLEU difference from unit-summed sufficient statistics
  d1, d2    : item-mean difference (unit-weighted resampling)
Gates (vs B): noisy  pesq med > 0 and CI low > 0; asr diff >= -0.5 and CI high >= 0; sim med >= -0.02; d2 diff <= +0.10 s
              clean  pesq med >= -0.05; asr CI high >= 0
Verdict cells (fixed before any number): V1 = {a, b_noxt} x {en2zh, zh2en}; V2 / V3 = b_noxt zh2en (prereg: main
criterion), the other cells reported. Also v2 / v3 vs v1 (pesq, sim, all metrics reported, no gate).

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.wave3_v_score --out <json> [--dirs en2zh,zh2en]
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
SEED = "20261021"
NB = 2000
ARMS = ("v1", "v2", "v3")


def run_dir(arm, d):
    return S1 / f"talker_{d}_s{SEED}" if arm == "B" else V / arm / f"talker_{d}"


def load(arm, d, LD):
    run = run_dir(arm, d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(V / "metrics" / f"{arm}_{d}_squim.jsonl")}
    sim = {r["id"]: r.get("sim") for r in jl(V / "metrics" / f"{arm}_{d}_spksim.jsonl")}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def compare(meta, ref, items, X, Y, lang, unit_key):
    """arm X vs reference Y on items (list of ids); returns stats dict"""
    bl = Bleu(lang)
    units = sorted({meta[i][unit_key] for i in items})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i][unit_key]] for i in items])
    nU = len(units)
    out = {"n_items": len(items), "n_units": nU}
    # BLEU sufficient stats per unit
    def ustats(P):
        st = bl.stats([P[i]["asr"] for i in items], [ref[i] for i in items])
        U = np.zeros((nU, st.shape[1]), np.int64)
        np.add.at(U, pu, st)
        return U
    UX, UY = ustats(X), ustats(Y)
    # per item differences
    def diffs(k):
        dv = np.full(len(items), np.nan)
        xv, yv = np.full(len(items), np.nan), np.full(len(items), np.nan)
        for n, i in enumerate(items):
            a, b = X[i][k], Y[i][k]
            if a is not None:
                xv[n] = a
            if b is not None:
                yv[n] = b
            if a is not None and b is not None:
                dv[n] = a - b
        return dv, xv, yv
    def unit_mean(v):
        s, c = np.zeros(nU), np.zeros(nU)
        ok = ~np.isnan(v)
        np.add.at(s, pu[ok], v[ok])
        np.add.at(c, pu[ok], 1)
        return s, c
    rng = np.random.default_rng(0)
    W = [np.ones(nU)] + [np.bincount(rng.integers(0, nU, nU), minlength=nU).astype(float) for _ in range(NB)]
    res = {}
    # asr bleu
    vals = []
    for w in W:
        vals.append(bl.score((UX * w[:, None]).sum(0)) - bl.score((UY * w[:, None]).sum(0)))
    res["asr"] = {"arm": round(bl.score(UX.sum(0)), 3), "ref": round(bl.score(UY.sum(0)), 3), "diff": round(vals[0], 3),
                  "ci95": [round(float(np.percentile(vals[1:], 2.5)), 3), round(float(np.percentile(vals[1:], 97.5)), 3)]}
    for k in ("pesq", "sim"):
        dv, xv, yv = diffs(k)
        s, c = unit_mean(dv)
        ok = c > 0
        um = np.where(ok, s / np.maximum(c, 1), np.nan)
        idx = np.nonzero(ok)[0]
        med0 = float(np.median(um[idx])) if len(idx) else float("nan")
        boots = []
        for _ in range(NB):
            pick = rng.choice(idx, len(idx), replace=True)
            boots.append(float(np.median(um[pick])))
        res[k] = {"arm_median": round(float(np.nanmedian(xv)), 4), "ref_median": round(float(np.nanmedian(yv)), 4),
                  "paired_median_diff": round(med0, 4),
                  "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                  "n_units": int(len(idx))}
    for k in ("d1", "d2"):
        dv, xv, yv = diffs(k)
        s, c = unit_mean(dv)
        vals = [float((s * w).sum() / max((c * w).sum(), 1e-9)) for w in W]
        res[k] = {"arm_mean": round(float(np.nanmean(xv)), 4), "ref_mean": round(float(np.nanmean(yv)), 4),
                  "diff": round(vals[0], 4),
                  "ci95": [round(float(np.percentile(vals[1:], 2.5)), 4), round(float(np.percentile(vals[1:], 97.5)), 4)]}
    out.update(res)
    return out


def gates(noisy, clean):
    g = {"noisy_pesq": noisy["pesq"]["paired_median_diff"] > 0 and noisy["pesq"]["ci95"][0] > 0,
         "noisy_asr": noisy["asr"]["diff"] >= -0.5 and noisy["asr"]["ci95"][1] >= 0,
         "noisy_sim": noisy["sim"]["paired_median_diff"] >= -0.02,
         "noisy_d2": noisy["d2"]["diff"] <= 0.10,
         "clean_pesq": clean["pesq"]["paired_median_diff"] >= -0.05,
         "clean_asr": clean["asr"]["ci95"][1] >= 0}
    g["pass"] = all(g.values())
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default="en2zh,zh2en")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed": SEED, "bootstrap": f"{NB} resamples of units, seed 0", "cells": {}, "verdict": {}}
    for d in a.dirs.split(","):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in ("B", *ARMS)}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                cell = {}
                for arm in ARMS:
                    cell[f"{arm}_vs_B"] = compare(meta, ref, items, P[arm], P["B"], LANG[d], "stream")
                for arm in ("v2", "v3"):
                    cell[f"{arm}_vs_v1"] = compare(meta, ref, items, P[arm], P["v1"], LANG[d], "stream")
                res["cells"][f"{d}/{set_}/{cond}"] = cell
        for set_ in ("a", "b", "b_noxt"):
            for arm in ARMS:
                res["verdict"][f"{arm}/{d}/{set_}"] = gates(res["cells"][f"{d}/{set_}/noisy"][f"{arm}_vs_B"],
                                                           res["cells"][f"{d}/{set_}/clean"][f"{arm}_vs_B"])
    V_ = res["verdict"]
    dirs = a.dirs.split(",")
    res["verdict"]["V1_overall"] = all(V_.get(f"v1/{d}/{s}", {"pass": False})["pass"] for d in dirs for s in ("a", "b_noxt"))
    for arm in ("v2", "v3"):
        res["verdict"][f"{arm.upper()}_overall"] = V_.get(f"{arm}/zh2en/b_noxt", {"pass": False})["pass"]
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
