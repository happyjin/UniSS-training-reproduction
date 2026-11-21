"""Experiment A evaluator check (after the precheck stop): recovered fraction of B's noise loss for the EXISTING arms
(V1, K5-full, oracle) under TitaNet-large (existing metrics) and ECAPA2 (a_sv/metrics/ecapa2_<arm>_<d>_spksim.jsonl),
per cell = direction x set {a, b, b_noxt}; median sim over items present in all arms (as k5_score / a_sv_score).
    python -m experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_existing --out <json>   (any python with numpy)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
TIT = {"B": E / "wave3_combos/V/metrics/B_{d}_spksim.jsonl", "v1": E / "wave3_combos/V/metrics/v1_{d}_spksim.jsonl",
       "k5full": E / "k5/metrics/k5full_{d}_spksim.jsonl", "oracle": E / "oracle_voice/metrics/oracle_{d}_spksim.jsonl"}
JUDGED = [f"{d}/{s}" for d in ("en2zh", "zh2en") for s in ("a", "b")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    res = {}
    for ev in ("titanet", "ecapa2"):
        for d in ("en2zh", "zh2en"):
            meta = {m["id"]: m for m in jl(E / "wave2_S1/data" / f"meta_{d}.jsonl")}
            S = {}
            for arm in TIT:
                p = Path(str(TIT[arm]).format(d=d)) if ev == "titanet" else E / "a_sv/metrics" / f"ecapa2_{arm}_{d}_spksim.jsonl"
                S[arm] = {r["id"]: r["sim"] for r in jl(p) if r["sim"] is not None}
            common = set.intersection(*[set(v) for v in S.values()])
            for s in ("a", "b", "b_noxt"):
                med = {}
                for cond in ("noisy", "clean"):
                    ids = [i for i in common if meta[i]["set"] == s[0] and meta[i]["cond"] == cond
                           and (s != "b_noxt" or not meta[i]["ovl"])]
                    med[cond] = {arm: float(np.median([S[arm][i] for i in ids])) for arm in S}
                bn, bc = med["noisy"]["B"], med["clean"]["B"]
                res[f"{ev}/{d}/{s}"] = {"median": med, "loss": bc - bn,
                                        "recovered": {arm: (med["noisy"][arm] - bn) / (bc - bn) for arm in ("v1", "k5full", "oracle")}}
        res[f"{ev}/mean4"] = {arm: float(np.mean([res[f"{ev}/{c}"]["recovered"][arm] for c in JUDGED]))
                              for arm in ("v1", "k5full", "oracle")}
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        if "mean4" in k:
            print(k, {x: round(y, 3) for x, y in v.items()})
        else:
            print(k, "loss %.3f" % v["loss"], {x: round(y, 2) for x, y in v["recovered"].items()},
                  "clean", {x: round(y, 3) for x, y in v["median"]["clean"].items()})


if __name__ == "__main__":
    main()
