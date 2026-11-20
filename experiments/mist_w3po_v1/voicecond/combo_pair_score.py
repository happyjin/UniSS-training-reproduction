"""Pair-only scoring for Thinker-level control arms (robust_combo_plan_20261120 section 4.2): X vs Y on every
direction x {a, b, b_noxt} x {noisy, clean} cell, with combo_score's loaders / statistics unchanged (ASR-BLEU, text BLEU,
D1 / D2, unit bootstrap seed 0). Equal-latency verdict (pre-written, section 4.2): cross-view information holds iff
  (1) >= 1 noisy cell with text diff CI low > 0, (2) those cells have D1 diff <= +0.10 s,
  (3) no cell (a / b, both conditions) with text or ASR diff < -0.3 and CI high < 0.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.combo_pair_score \
        --arm cvla_base=<root> --arm ctrl_budget_base=<root> --pair cvla_base_b:ctrl_budget_base_b --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.combo_score import DIRS, LANG, S1, clean, compare, jl, laal_defs, load, systems, text_compare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    S = systems(dict(x.split("=", 1) for x in a.arm))
    x, y = a.pair.split(":")
    LD = laal_defs()
    res, imp, deg = {"pair": f"{x}_vs_{y}", "cells": {}}, [], []
    for d in DIRS:
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        PX, PY = load(S[x], d, LD), load(S[y], d, LD)
        common = set(PX) & set(PY)
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                c = compare(meta, ref, items, PX, PY, LANG[d], "stream")
                c["text"] = text_compare(meta, ref, items, PX, PY, LANG[d], "stream")
                key = f"{d}/{set_}/{cond}"
                res["cells"][key] = {k: c[k] for k in ("asr", "text", "d1", "d2") if k in c}
                t = c["text"]
                if cond == "noisy" and t and t["ci95"][0] > 0:
                    imp.append([key, t["diff"], t["ci95"], c["d1"]["diff"], c["d1"]["diff"] <= 0.10])
                if set_ in ("a", "b"):
                    for k in ("text", "asr"):
                        v = c[k]
                        if v and v["diff"] < -0.3 and v["ci95"][1] < 0:
                            deg.append([key, k, v["diff"], v["ci95"]])
    res["verdict"] = {"noisy_text_improvements_[cell,d,ci,d1,d1_ok]": imp, "degradations": deg,
                      "cross_view_information": any(r[4] for r in imp) and not deg}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(res["verdict"], indent=1))
    for k, v in res["cells"].items():
        print(k, "text", v["text"] and (v["text"]["diff"], v["text"]["ci95"]), "asr", (v["asr"]["diff"], v["asr"]["ci95"]),
              "d1", (v["d1"]["diff"], v["d1"]["ci95"]))


if __name__ == "__main__":
    main()
