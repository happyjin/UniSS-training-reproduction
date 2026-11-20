"""Robust line, FINAL TEST scoring (reports/mist_w3po_v1/robust_test_prereg_20261120.zh-CN.md section 6, written before
any test number). Systems from eval_outputs/mist_w3po_v1/test_robust/<ARM>/{b,v1}/talker_<d> (test_orch.sh): B_b, EW_b,
EW_v1, C_b, C_v1 (+ CE_b / CE_v1 if that root exists). Cells: direction x {realsi/clean, realsi/noisy, obsi/clean,
obsi/id2, obsi/ood1, ood3_en | ood3_ali | ood3_ami}; OBSI en2zh main = without pair e031_0004 (an "_all" cell keeps it).
Statistics: wave3_v_score.compare (ASR-BLEU unit-summed sufficient statistics, PESQ / similarity paired medians, D1 / D2)
+ combo_score.text_compare; 2000 resamples, seed 0; unit = meta "stream" (RealSI: the sentence; aux "realsi_rec" CI by
recording). Primary rules (6.4): (1) C_v1 vs B_b ASR-BLEU diff >= -0.3 and CI high >= 0 in every main cell (ood3_ami and
_all excluded); (2) some noisy cell (realsi/noisy, obsi/id2, obsi/ood1) with ASR-BLEU CI low > 0. Reported: C_v1 vs EW_v1,
EW_v1 vs B_b, C_v1 vs C_b, CE_v1 vs C_v1 / B_b when present.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.test_score --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, compare, jl, laal_defs, load, text_compare

REPO = Path(__file__).resolve().parents[3]
T = REPO / "eval_outputs/mist_w3po_v1/test_robust"
DIRS = ("en2zh", "zh2en")
EXCL = {"en2zh": {"e031_0004"}, "zh2en": set()}
NOISY = ("realsi/noisy", "obsi/id2", "obsi/ood1")


def sysd(arm, run):
    r = T / arm
    return {"talker": lambda d: r / run / f"talker_{d}", "metrics": lambda d, k: r / "metrics" / f"{run}_{d}_{k}.jsonl",
            "thinker": lambda d: r / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    S = {"B_b": sysd("B", "b"), "EW_b": sysd("EW", "b"), "EW_v1": sysd("EW", "v1"), "C_b": sysd("C", "b"), "C_v1": sysd("C", "v1")}
    pairs = [("C_v1", "B_b"), ("C_v1", "EW_v1"), ("EW_v1", "B_b"), ("C_v1", "C_b"), ("EW_v1", "EW_b")]
    if (T / "CE" / "DONE").exists():
        S.update({"CE_b": sysd("CE", "b"), "CE_v1": sysd("CE", "v1")})
        pairs += [("CE_v1", "C_v1"), ("CE_v1", "B_b")]
    for k in S:
        assert (T / k.split("_")[0] / "DONE").exists(), f"{k}: arm not done"
    LD = laal_defs()
    res = {"systems": {k: str(v["talker"]("en2zh")) for k, v in S.items()}, "cells": {}, "abs": {}}
    r1, r2 = [], []
    for d in DIRS:
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P = {s: load(S[s], d, LD) for s in S}
        common = set.intersection(*[set(p) for p in P.values()])
        assert len(common) == len(meta), (d, len(common), len(meta))
        cells = sorted({f"{m['set']}/{m['cond']}" if m["set"] in ("realsi", "obsi") else m["set"] for m in meta.values()})
        if EXCL[d]:
            cells.append("obsi/clean_all"); cells.append("obsi/id2_all"); cells.append("obsi/ood1_all")
        for cell in cells:
            full = cell.endswith("_all")
            key0 = cell[:-4] if full else cell

            def inc(m):
                c = f"{m['set']}/{m['cond']}" if m["set"] in ("realsi", "obsi") else m["set"]
                return c == key0 and (full or m["set"] != "obsi" or m["pair"] not in EXCL[d])
            items = sorted(i for i in common if inc(meta[i]))
            cres = {}
            for x, y in pairs:
                c = compare(meta, ref, items, P[x], P[y], LANG[d], "stream")
                c["text"] = text_compare(meta, ref, items, P[x], P[y], LANG[d], "stream")
                if key0.startswith("realsi"):
                    cc = compare(meta, ref, items, P[x], P[y], LANG[d], "rec")
                    c["asr_ci95_by_recording"] = cc["asr"]["ci95"]
                cres[f"{x}_vs_{y}"] = c
            k = f"{d}/{cell}"
            res["cells"][k] = cres
            m = cres["C_v1_vs_B_b"]["asr"]
            if not full and key0 != "ood3_ami":
                if not (m["diff"] >= -0.3 and m["ci95"][1] >= 0):
                    r1.append([k, m["diff"], m["ci95"]])
            if not full and key0 in NOISY and m["ci95"][0] > 0:
                r2.append([k, m["diff"], m["ci95"]])
    res["verdict"] = {"rule1_degradations": r1, "rule1_PASS": not r1, "rule2_noisy_improvements": r2, "rule2_PASS": bool(r2),
                      "full_system_holds_on_test": not r1 and bool(r2)}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(res["verdict"], indent=1))
    for k, v in res["cells"].items():
        c = v["C_v1_vs_B_b"]
        print(k, "C_v1", c["asr"]["arm"], "B", c["asr"]["ref"], "d", c["asr"]["diff"], c["asr"]["ci95"],
              "| vs EW_v1", v["C_v1_vs_EW_v1"]["asr"]["diff"], v["C_v1_vs_EW_v1"]["asr"]["ci95"], "| d1", c["d1"]["diff"])


if __name__ == "__main__":
    main()
