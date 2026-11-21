"""Robust oracle-voice scoring (reports/mist_w3po_v1/robust_oracle_voice_prereg.zh-CN.md section 4): S1 dev, point estimates.
Arms per direction d: B = wave2_S1/talker_<d>_s20261021, V1 = wave3_combos/V/v1/talker_<d>, K3 = k3/talker_<d>, oracle =
oracle_voice/talker_<d> (metrics oracle_voice/metrics/oracle_<d>_*). Point estimates by wave3_v_score.compare (one resample,
CI dropped), every arm against B. Recovered fraction of B's noise loss (cell = direction x set): (arm noisy sim median - B
noisy sim median) / (B clean sim median - B noisy sim median). Rule: oracle >= 0.7 in all 4 judged cells (a, b) -> loss is
in the global code; < 0.7 in >= 3/4 -> another channel; else mixed. b_noxt reported only.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.oracle_voice_score --out <json>
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
RUNS = {"B": lambda d: (W.S1 / f"talker_{d}_s{W.SEED}", W.V / "metrics" / f"B_{d}"),
        "v1": lambda d: (W.V / "v1" / f"talker_{d}", W.V / "metrics" / f"v1_{d}"),
        "k3": lambda d: (K3 / f"talker_{d}", K3 / "metrics" / f"k3_{d}"),
        "oracle": lambda d: (OR / f"talker_{d}", OR / "metrics" / f"oracle_{d}")}


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
                res["cells"][f"{d}/{set_}/{cond}"] = {f"{arm}_vs_B": strip(W.compare(meta, ref, items, P[arm], P["B"],
                                                                                 LANG[d], "stream"))
                                                       for arm in ("v1", "k3", "oracle")}
        for set_ in ("a", "b", "b_noxt"):
            n = res["cells"][f"{d}/{set_}/noisy"]
            c = res["cells"][f"{d}/{set_}/clean"]
            bn, bc = n["oracle_vs_B"]["sim"]["ref_median"], c["oracle_vs_B"]["sim"]["ref_median"]
            res["rule"][f"{d}/{set_}"] = {"B_noisy": bn, "B_clean": bc, "loss": bc - bn,
                                          **{f"recovered_{arm}": (n[f"{arm}_vs_B"]["sim"]["arm_median"] - bn) / (bc - bn)
                                             for arm in ("v1", "k3", "oracle")}}
    rec = [res["rule"][f"{d}/{s}"]["recovered_oracle"] for d in ("en2zh", "zh2en") for s in ("a", "b")]
    res["rule"]["verdict"] = ("global_code" if all(r >= 0.7 for r in rec) else
                              "other_channel" if sum(r < 0.7 for r in rec) >= 3 else "mixed")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()
