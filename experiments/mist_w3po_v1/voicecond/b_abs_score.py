"""Robust B scoring (reports/mist_w3po_v1/robust_b_abs_prereg.zh-CN.md section 5; copy of a_sv_score.py): S1 dev, point
estimates. Arms per direction d: B, V1, oracle, K5-full (existing outputs, TitaNet / SQUIM metrics, ECAPA2 metrics in
a_sv/metrics) and AbS (b_abs/talker_abs_<d>, metrics in b_abs/metrics). ECAPA2 is the primary judge, TitaNet (the search
utility's encoder) is reported only. Recovered fraction of B's noise loss per cell and evaluator:
(arm noisy sim median - B noisy) / (B clean - B noisy). Judged cells en2zh / zh2en x {a, b}; b_noxt reported.
Rule (arm abs): (1) ECAPA2 mean recovered over the 4 cells >= V1 mean + 0.15; (2) ECAPA2 mean clean sim median over the
4 cells >= V1's; (3) ASR-BLEU >= V1 - 0.3 in every judged cell x {noisy, clean}; (4) PESQ median >= V1 - 0.1 likewise.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.b_abs_score --out <json>
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
A = REPO / "eval_outputs/mist_w3po_v1/a_sv"
BA = REPO / "eval_outputs/mist_w3po_v1/b_abs"
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        "v1": lambda d: (W.V / "v1" / f"talker_{d}", W.V / "metrics" / f"v1_{d}"),
        "oracle": lambda d: (OR / f"talker_{d}", OR / "metrics" / f"oracle_{d}"),
        "k5full": lambda d: (K5 / f"talker_k5full_{d}", K5 / "metrics" / f"k5full_{d}"),
        "abs": lambda d: (BA / f"talker_abs_{d}", BA / "metrics" / f"abs_{d}")}
ARMS = ("v1", "oracle", "k5full", "abs")
EVALS = ("titanet", "ecapa2")
JUDGED = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b")]


def load(arm, d, LD, ev):
    run, mp = RUNS[arm](d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(Path(f"{mp}_squim.jsonl"))}
    sp = Path(f"{mp}_spksim.jsonl") if ev == "titanet" else \
        (BA if arm == "abs" else A) / "metrics" / f"ecapa2_{arm}_{d}_spksim.jsonl"
    sim = {r["id"]: r.get("sim") for r in jl(sp)}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def strip(c, sim_only=False):
    o = {"n_items": c["n_items"], "n_units": c["n_units"]}
    for k in (("sim",) if sim_only else ("asr", "pesq", "sim", "d1", "d2")):
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
    res = {"seed": W.SEED, "estimates": "point (no bootstrap)", "cells": {}, "cells_ecapa2": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {ev: {arm: load(arm, d, LD, ev) for arm in RUNS} for ev in EVALS}
        common = set.intersection(*[set(p) for p in P["titanet"].values()])
        res[f"n_common_{d}"] = len(common)
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                key = f"{d}/{set_}/{cond}"
                res["cells"][key] = {f"{arm}_vs_B": strip(W.compare(meta, ref, items, P["titanet"][arm], P["titanet"]["B"],
                                                                    LANG[d], "stream")) for arm in ARMS}
                res["cells_ecapa2"][key] = {f"{arm}_vs_B": strip(W.compare(meta, ref, items, P["ecapa2"][arm],
                                                                           P["ecapa2"]["B"], LANG[d], "stream"), True)
                                            for arm in ARMS}
        for set_ in ("a", "b", "b_noxt"):
            for ev, CC in (("titanet", res["cells"]), ("ecapa2", res["cells_ecapa2"])):
                n, c = CC[f"{d}/{set_}/noisy"], CC[f"{d}/{set_}/clean"]
                bn, bc = n["v1_vs_B"]["sim"]["ref_median"], c["v1_vs_B"]["sim"]["ref_median"]
                res["rule"][f"{ev}/{d}/{set_}"] = {"B_noisy": bn, "B_clean": bc, "loss": bc - bn,
                                                   **{f"recovered_{arm}": (n[f"{arm}_vs_B"]["sim"]["arm_median"] - bn) / (bc - bn)
                                                      for arm in ARMS}}
    C, Rr = res["cells"], res["rule"]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    rec = {ev: {arm: mean([Rr[f"{ev}/{c}"][f"recovered_{arm}"] for c in JUDGED]) for arm in ARMS} for ev in EVALS}
    clean = {ev: {arm: mean([(C if ev == "titanet" else res["cells_ecapa2"])[f"{c}/clean"][f"{arm}_vs_B"]["sim"]["arm_median"]
                             for c in JUDGED]) for arm in ARMS} for ev in EVALS}
    checks = {}
    for arm in ("abs", "k5full"):
        asr_ok = {f"{c}/{k}": C[f"{c}/{k}"][f"{arm}_vs_B"]["asr"]["arm"] - C[f"{c}/{k}"]["v1_vs_B"]["asr"]["arm"]
                  for c in JUDGED for k in ("noisy", "clean")}
        pesq_ok = {f"{c}/{k}": C[f"{c}/{k}"][f"{arm}_vs_B"]["pesq"]["arm_median"] - C[f"{c}/{k}"]["v1_vs_B"]["pesq"]["arm_median"]
                   for c in JUDGED for k in ("noisy", "clean")}
        pesq_or = {f"{c}/{k}": C[f"{c}/{k}"][f"{arm}_vs_B"]["pesq"]["arm_median"] - C[f"{c}/{k}"]["oracle_vs_B"]["pesq"]["arm_median"]
                   for c in JUDGED for k in ("noisy", "clean")}
        ck = {"rec_mean": {ev: rec[ev][arm] for ev in EVALS}, "rec_mean_v1": {ev: rec[ev]["v1"] for ev in EVALS},
              "r1": rec["ecapa2"][arm] >= rec["ecapa2"]["v1"] + 0.15,
              "clean_mean": {ev: clean[ev][arm] for ev in EVALS}, "clean_mean_v1": {ev: clean[ev]["v1"] for ev in EVALS},
              "r2": clean["ecapa2"][arm] >= clean["ecapa2"]["v1"],
              "titanet_r1_reported": rec["titanet"][arm] >= rec["titanet"]["v1"] + 0.15,
              "titanet_r2_reported": clean["titanet"][arm] >= clean["titanet"]["v1"],
              "asr_minus_v1": asr_ok, "r3": all(v >= -0.3 for v in asr_ok.values()),
              "pesq_minus_v1": pesq_ok, "r4": all(v >= -0.1 for v in pesq_ok.values()), "pesq_minus_oracle": pesq_or}
        ck["all"] = ck["r1"] and ck["r2"] and ck["r3"] and ck["r4"]
        checks[arm] = ck
    res["rule"]["rec_mean"] = rec
    res["rule"]["clean_mean"] = clean
    res["rule"]["checks"] = checks
    res["rule"]["verdict_abs"] = "established" if checks["abs"]["all"] else "not_established"
    pc = BA / "precheck.json"
    if pc.exists():
        res["latency_s"] = {d: v["latency_s"] for d, v in json.load(open(pc))["dirs"].items()}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: res["rule"][k] for k in ("rec_mean", "clean_mean", "verdict_abs")}, indent=1))
    print(json.dumps({arm: {k: v for k, v in ck.items() if k.startswith("r") or k == "all"} for arm, ck in checks.items()}, indent=1))


if __name__ == "__main__":
    main()
