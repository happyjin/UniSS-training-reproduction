"""Robust K5 scoring (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md section 5; copy of oracle_voice_score.py):
S1 dev, point estimates. Arms per direction d: B, V1, oracle (existing outputs), K5-dec / K5-lm / K5-full
(k5/talker_<arm>_<d>, metrics k5/metrics/<arm>_<d>_*). Recovered fraction of B's noise loss per cell (direction x set):
(arm noisy sim median - B noisy) / (B clean - B noisy). Judged cells en2zh / zh2en x {a, b}; b_noxt reported only.
Rule (K5-full, all four): (1) mean recovered over the 4 cells >= V1 mean + 0.20; (2) mean clean sim median over the 4
cells >= V1's; (3) ASR-BLEU >= V1 - 0.3 in every judged cell x {noisy, clean}; (4) PESQ median >= V1 - 0.1 likewise.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.k5_score --out <json>
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
K5 = REPO / "eval_outputs/mist_w3po_v1/k5"
OR = REPO / "eval_outputs/mist_w3po_v1/oracle_voice"
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        "v1": lambda d: (W.V / "v1" / f"talker_{d}", W.V / "metrics" / f"v1_{d}"),
        "oracle": lambda d: (OR / f"talker_{d}", OR / "metrics" / f"oracle_{d}"),
        **{arm: (lambda d, arm=arm: (K5 / f"talker_{arm}_{d}", K5 / "metrics" / f"{arm}_{d}"))
           for arm in ("k5dec", "k5lm", "k5full")}}
ARMS = ("v1", "oracle", "k5dec", "k5lm", "k5full")
JUDGED = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b")]


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
        res[f"n_common_{d}"] = len(common)
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                res["cells"][f"{d}/{set_}/{cond}"] = {f"{arm}_vs_B": strip(W.compare(meta, ref, items, P[arm], P["B"],
                                                                                 LANG[d], "stream")) for arm in ARMS}
        for set_ in ("a", "b", "b_noxt"):
            n, c = res["cells"][f"{d}/{set_}/noisy"], res["cells"][f"{d}/{set_}/clean"]
            bn, bc = n["v1_vs_B"]["sim"]["ref_median"], c["v1_vs_B"]["sim"]["ref_median"]
            res["rule"][f"{d}/{set_}"] = {"B_noisy": bn, "B_clean": bc, "loss": bc - bn,
                                          **{f"recovered_{arm}": (n[f"{arm}_vs_B"]["sim"]["arm_median"] - bn) / (bc - bn)
                                             for arm in ARMS}}
    C, Rr = res["cells"], res["rule"]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    rec = {arm: mean([Rr[c][f"recovered_{arm}"] for c in JUDGED]) for arm in ARMS}
    clean = {arm: mean([C[f"{c}/clean"][f"{arm}_vs_B"]["sim"]["arm_median"] for c in JUDGED]) for arm in ARMS}
    checks = {}
    for arm in ("k5dec", "k5lm", "k5full"):
        asr_ok = {f"{c}/{k}": C[f"{c}/{k}"][f"{arm}_vs_B"]["asr"]["arm"] - C[f"{c}/{k}"]["v1_vs_B"]["asr"]["arm"]
                  for c in JUDGED for k in ("noisy", "clean")}
        pesq_ok = {f"{c}/{k}": C[f"{c}/{k}"][f"{arm}_vs_B"]["pesq"]["arm_median"] - C[f"{c}/{k}"]["v1_vs_B"]["pesq"]["arm_median"]
                   for c in JUDGED for k in ("noisy", "clean")}
        checks[arm] = {"rec_mean": rec[arm], "rec_mean_v1": rec["v1"], "r1": rec[arm] >= rec["v1"] + 0.20,
                       "clean_mean": clean[arm], "clean_mean_v1": clean["v1"], "r2": clean[arm] >= clean["v1"],
                       "asr_minus_v1": asr_ok, "r3": all(v >= -0.3 for v in asr_ok.values()),
                       "pesq_minus_v1": pesq_ok, "r4": all(v >= -0.1 for v in pesq_ok.values())}
        checks[arm]["all"] = all(checks[arm][k] for k in ("r1", "r2", "r3", "r4"))
    res["rule"]["rec_mean"] = rec
    res["rule"]["clean_mean"] = clean
    res["rule"]["checks"] = checks
    res["rule"]["verdict_k5full"] = "established" if checks["k5full"]["all"] else "not_established"
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
