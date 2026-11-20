"""Robust wave 3, V1 second seed and the V1 / V2 choice (robust_wave3_single_prereg.zh-CN.md, "V1 第二种子（选型用）").

Pooled two seeds (20261021 + 1), each arm vs B on the same seeds (wave3_v2s1_score.compare): v1 + v1_s1 vs B + B_s1 and
v2 + v2_s1 vs B + B_s1, all cells; seed 1 alone v1_s1 vs B_s1. Choice on the 4 main cells {a, b_noxt} x {en2zh, zh2en},
noisy: (1) count of cells with ASR-BLEU diff >= -0.3 and CI high >= 0; (2) zh2en b_noxt pooled ASR diff; (3) mean PESQ
paired median diff over the 4 cells. LAAL / similarity reported only.

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.wave3_v1s1_select --out <json>
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
from experiments.mist_w3po_v1.voicecond.wave3_v2s1_score import compare  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave3_v_score import S1, load  # noqa: E402

MAIN = ["en2zh/a/noisy", "en2zh/b_noxt/noisy", "zh2en/a/noisy", "zh2en/b_noxt/noisy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"v1_seed1": {}, "v1_pooled": {}, "v2_pooled": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in ("B", "B_s1", "v1", "v1_s1", "v2", "v2_s1")}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                k = f"{d}/{set_}/{cond}"
                res["v1_seed1"][k] = compare(meta, ref, items, [P["v1_s1"]], [P["B_s1"]], LANG[d])
                res["v1_pooled"][k] = compare(meta, ref, items, [P["v1"], P["v1_s1"]], [P["B"], P["B_s1"]], LANG[d])
                res["v2_pooled"][k] = compare(meta, ref, items, [P["v2"], P["v2_s1"]], [P["B"], P["B_s1"]], LANG[d])
    crit = {}
    for arm in ("v1", "v2"):
        R = res[f"{arm}_pooled"]
        ok = {c: R[c]["asr"]["diff"] >= -0.3 and R[c]["asr"]["ci95"][1] >= 0 for c in MAIN}
        crit[arm] = {"asr_ok_cells": ok, "n_ok": sum(ok.values()), "zh2en_b_noxt_asr": R["zh2en/b_noxt/noisy"]["asr"]["diff"],
                     "mean_pesq": float(np.mean([R[c]["pesq"]["paired_median_diff"] for c in MAIN]))}
    c1, c2 = crit["v1"], crit["v2"]
    if c1["n_ok"] != c2["n_ok"]:
        win, by = ("v1" if c1["n_ok"] > c2["n_ok"] else "v2"), "1: non-degraded ASR cells"
    elif c1["zh2en_b_noxt_asr"] != c2["zh2en_b_noxt_asr"]:
        win, by = ("v1" if c1["zh2en_b_noxt_asr"] > c2["zh2en_b_noxt_asr"] else "v2"), "2: zh2en b_noxt ASR"
    else:
        win, by = ("v1" if c1["mean_pesq"] >= c2["mean_pesq"] else "v2"), "3: mean PESQ"
    res["choice"] = {"criteria": crit, "winner": win, "decided_by": by}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["choice"], indent=1))


if __name__ == "__main__":
    main()
