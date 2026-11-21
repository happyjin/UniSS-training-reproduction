"""Robust B confirmation scoring (reports/mist_w3po_v1/robust_b_abs_prereg.zh-CN.md, section "确认运行").

Arms B / V1 / AbS on S1 dev, two TTS seeds:
  seed 1 (20261021, existing): B = wave2_S1/talker_<d>_s20261021, V1 = wave3_combos/V/v1/talker_<d>, AbS = b_abs/talker_abs_<d>;
         TitaNet / SQUIM of B, V1 in wave3_combos/V/metrics, ECAPA2 in a_sv/metrics; AbS metrics in b_abs/metrics.
  seed 2 (20261022): b_abs_seed2/talker_<arm>_<d>, metrics b_abs_seed2/metrics/{<arm>_<d>_{squim,spksim},ecapa2_<arm>_<d>_spksim}.
Per seed and cell (direction x {a, b, b_noxt} x {noisy, clean}): wave3_v_score.compare (unit = stream, 2000 resamples,
seed 0) for AbS - V1 and AbS - B with TitaNet and ECAPA2 similarity -> ASR-BLEU diff + CI, similarity paired diff + CI,
per-arm ASR-BLEU / PESQ median / similarity median. Recovered fraction per seed with that seed's own B.
Two-seed mean = mean of the two seeds' cell values. Reported only: pooled two-seed ASR-BLEU AbS - V1 CI (per unit, the
two seeds' sufficient statistics summed). Reading fixed in the prereg (section 4).
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.b_abs_seed2_score --out <json> [--seed1-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.wave3_v_score as W  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, Bleu, jl  # noqa: E402

A = REPO / "eval_outputs/mist_w3po_v1/a_sv"
BA = REPO / "eval_outputs/mist_w3po_v1/b_abs"
B2 = REPO / "eval_outputs/mist_w3po_v1/b_abs_seed2"
ARMS = ("B", "v1", "abs")
EVALS = ("titanet", "ecapa2")
MAIN = [f"{d}/{s}/{c}" for d in ("en2zh", "zh2en") for s in ("a", "b") for c in ("noisy", "clean")]
FAILED_S1 = ["zh2en/a/noisy", "zh2en/b/clean", "en2zh/b/noisy"]


def paths(seed, arm, d):
    """-> run dir, squim, titanet spksim, ecapa2 spksim"""
    if seed == 2:
        m = B2 / "metrics"
        return B2 / f"talker_{arm}_{d}", m / f"{arm}_{d}_squim.jsonl", m / f"{arm}_{d}_spksim.jsonl", \
            m / f"ecapa2_{arm}_{d}_spksim.jsonl"
    if arm == "abs":
        m = BA / "metrics"
        return BA / f"talker_abs_{d}", m / f"abs_{d}_squim.jsonl", m / f"abs_{d}_spksim.jsonl", m / f"ecapa2_abs_{d}_spksim.jsonl"
    run = W.S1 / f"talker_{d}_s{W.SEED}" if arm == "B" else W.V / "v1" / f"talker_{d}"
    return run, W.V / "metrics" / f"{arm}_{d}_squim.jsonl", W.V / "metrics" / f"{arm}_{d}_spksim.jsonl", \
        A / "metrics" / f"ecapa2_{arm}_{d}_spksim.jsonl"


def load(seed, arm, d):
    run, sq, ti, ec = paths(seed, arm, d)
    ids = [r["id"] for r in jl(run / "results.jsonl")]
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    pq = {r["id"]: r.get("pesq") for r in jl(sq)}
    out = {}
    for ev, p in (("titanet", ti), ("ecapa2", ec)):
        sim = {r["id"]: r.get("sim") for r in jl(p)}
        out[ev] = {i: {"asr": asr.get(i, ""), "pesq": pq.get(i), "sim": sim.get(i), "d1": None, "d2": None} for i in ids}
    return out


def pooled_asr(meta, ref, items, Xs, Ys, lang):
    """corpus BLEU diff of (X seed1 + X seed2) - (Y seed1 + Y seed2), unit stats summed, 2000 unit resamples (seed 0)"""
    bl = Bleu(lang)
    units = sorted({meta[i]["stream"] for i in items})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i]["stream"]] for i in items])
    nU = len(units)

    def ustats(P):
        st = bl.stats([P[i]["asr"] for i in items], [ref[i] for i in items])
        U = np.zeros((nU, st.shape[1]), np.int64)
        np.add.at(U, pu, st)
        return U
    UX = sum(ustats(P) for P in Xs)
    UY = sum(ustats(P) for P in Ys)
    rng = np.random.default_rng(0)
    vals = [bl.score(UX.sum(0)) - bl.score(UY.sum(0))]
    for _ in range(W.NB):
        w = np.bincount(rng.integers(0, nU, nU), minlength=nU).astype(float)
        vals.append(bl.score((UX * w[:, None]).sum(0)) - bl.score((UY * w[:, None]).sum(0)))
    return {"diff": round(vals[0], 3), "ci95": [round(float(np.percentile(vals[1:], 2.5)), 3),
                                                 round(float(np.percentile(vals[1:], 97.5)), 3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed1-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seeds = (1,) if a.seed1_only else (1, 2)
    res = {"seeds": {1: 20261021, 2: 20261022}, "bootstrap": f"{W.NB} resamples of streams, seed 0", "per_seed": {},
           "rec": {}, "mean2": {}, "pooled_asr_abs_minus_v1": {}, "n_common": {}}
    for s in seeds:
        res["per_seed"][s] = {}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {s: {arm: load(s, arm, d) for arm in ARMS} for s in seeds}
        common = set.intersection(*[set(P[s][arm]["titanet"]) for s in seeds for arm in ARMS])
        res["n_common"][d] = len(common)
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                key = f"{d}/{set_}/{cond}"
                for s in seeds:
                    c = {"n_items": len(items)}
                    for ev in EVALS:
                        X = P[s]["abs"][ev]
                        cv = W.compare(meta, ref, items, X, P[s]["v1"][ev], LANG[d], "stream")
                        cb = W.compare(meta, ref, items, X, P[s]["B"][ev], LANG[d], "stream")
                        c["n_units"] = cv["n_units"]
                        c[f"sim_{ev}"] = {"abs": cv["sim"]["arm_median"], "v1": cv["sim"]["ref_median"],
                                          "B": cb["sim"]["ref_median"],
                                          "abs_minus_v1": {k: cv["sim"][k] for k in ("paired_median_diff", "ci95")},
                                          "abs_minus_B": {k: cb["sim"][k] for k in ("paired_median_diff", "ci95")}}
                        if ev == "titanet":
                            c["asr"] = {"abs": cv["asr"]["arm"], "v1": cv["asr"]["ref"], "B": cb["asr"]["ref"],
                                        "abs_minus_v1": {k: cv["asr"][k] for k in ("diff", "ci95")},
                                        "abs_minus_B": {k: cb["asr"][k] for k in ("diff", "ci95")}}
                            c["pesq"] = {"abs": cv["pesq"]["arm_median"], "v1": cv["pesq"]["ref_median"],
                                         "B": cb["pesq"]["ref_median"]}
                    res["per_seed"][s][key] = c
                if len(seeds) == 2:
                    res["pooled_asr_abs_minus_v1"][key] = pooled_asr(
                        meta, ref, items, [P[s]["abs"]["titanet"] for s in seeds], [P[s]["v1"]["titanet"] for s in seeds],
                        LANG[d])
                print(key, "done", flush=True)
    # recovered fractions per seed (own B)
    for s in seeds:
        res["rec"][s] = {}
        for d in ("en2zh", "zh2en"):
            for set_ in ("a", "b", "b_noxt"):
                n, c = res["per_seed"][s][f"{d}/{set_}/noisy"], res["per_seed"][s][f"{d}/{set_}/clean"]
                for ev in EVALS:
                    bn, bc = n[f"sim_{ev}"]["B"], c[f"sim_{ev}"]["B"]
                    res["rec"][s][f"{ev}/{d}/{set_}"] = {"B_loss": bc - bn, **{arm: (n[f"sim_{ev}"][arm] - bn) / (bc - bn)
                                                                               for arm in ("v1", "abs")}}
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    for s in seeds:
        res["rec"][s]["mean4"] = {ev: {arm: mean([res["rec"][s][f"{ev}/{d}/{x}"][arm] for d in ("en2zh", "zh2en")
                                                  for x in ("a", "b")]) for arm in ("v1", "abs")} for ev in EVALS}
    if len(seeds) == 2:
        for key in res["per_seed"][1]:
            res["mean2"][key] = {m: {arm: mean([res["per_seed"][s][key][m][arm] for s in seeds]) for arm in ARMS}
                                 for m in ("asr", "pesq", "sim_titanet", "sim_ecapa2")}
        res["mean2"]["rec"] = {k: {arm: mean([res["rec"][s][k][arm] for s in seeds]) for arm in ("v1", "abs")}
                               for k in res["rec"][1] if k != "mean4"}
        res["mean2"]["rec_mean4"] = {ev: {arm: mean([res["rec"][s]["mean4"][ev][arm] for s in seeds])
                                          for arm in ("v1", "abs")} for ev in EVALS}
        gap = {k: res["mean2"][k]["asr"]["abs"] - res["mean2"][k]["asr"]["v1"] for k in MAIN}
        ci_has0 = {k: res["per_seed"][1][k]["asr"]["abs_minus_v1"]["ci95"][0] <= 0
                   <= res["per_seed"][1][k]["asr"]["abs_minus_v1"]["ci95"][1] for k in FAILED_S1}
        c1, c2 = all(v >= -0.3 for v in gap.values()), all(ci_has0.values())
        res["reading"] = {"mean2_asr_abs_minus_v1": gap, "cond1_mean2_ge_v1_minus_0.3": c1,
                          "seed1_ci_includes_0_failed_cells": ci_has0, "cond2": c2,
                          "verdict": "seed-1 ASR shortfall not confirmed" if c1 and c2 else "ASR drop confirmed"}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res.get("reading", {}), indent=1))


if __name__ == "__main__":
    main()
