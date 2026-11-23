"""sst_hsmm_v1 P2 dev gate scoring (reports/sst_hsmm_v1/prereg.zh-CN.md 3.5 + change 2).
NEW = eval_outputs/sst_hsmm_v1/p2 (HSMM Thinker blocks -> T1c, source voice), CTRL = wave17/eval_enc_wise05 (LA2 blocks ->
T1c, source voice). Both played with s0_apply --policy dip (rows in p2/play/{new,ctrl}_<d>.jsonl): D1 / D2 / NP_in / NC /
PT_in / EndOffset from the player. ASR-BLEU from asr_speech.jsonl via wave3_v_score.compare (unit = stream, 2000 resamples,
seed 0); playout metrics: stream-bootstrap mean difference (s_score.boot). Pooled per direction (sets a/b x clean/noisy)
= the gate; per cell = diagnostics. GO iff both directions: ASR diff >= -0.5, D2 diff <= -0.2 s, NP_in rel <= -30%.
--root p2b --rules pareto (prereg change 6, P2'): NEW = <root>; GO iff both directions: ASR diff >= -0.5, D2 diff <= +0.05 s,
NP_in rel <= -25%.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.p2_score --out eval_outputs/sst_hsmm_v1/p2/summary.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, jl, text_compare
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare
from experiments.voice_onset_v1.s_score import boot

REPO = Path(__file__).resolve().parents[2]
R = REPO / "eval_outputs/sst_hsmm_v1/p2"
CTRL = REPO / "eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
DIRS = ("en2zh", "zh2en")
PM = ("NP_in", "PT_in", "PT_all", "NC", "n_pause", "EndOffset")
SYS = {"new": (R, R / "play"), "ctrl": (CTRL, R / "play")}
RULES = {"orig": (-0.5, -0.2, -0.30), "pareto": (-0.5, 0.05, -0.25)}     # asr diff >=, d2 diff <=, NP_in rel <=


def load(name, d):
    root, play = SYS[name]
    run = root / f"talker_{d}_s20261021"
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    tr = {r["id"]: r["translation"] for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    per = {}
    for r in jl(play / f"{name}_{d}.jsonl"):
        if r.get("D2") is None:
            continue
        i = r["id"]
        per[i] = {"asr": asr.get(i, ""), "pesq": None, "sim": None, "text": tr.get(i), "d1": r["D1"] / 1000,
                  "d2": r["D2"] / 1000, **{k: (r.get(k) if r.get(k) is not None else 0.0) for k in PM}}
    return per


def pair(meta, ref, items, X, Y, lang):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = compare(meta, ref, items, X, Y, lang, "stream")
    c.pop("pesq", None)
    c.pop("sim", None)
    c["text_bleu"] = text_compare(meta, ref, items, X, Y, lang, "stream")
    for q in PM:
        b = boot(meta, items, {i: X[i][q] for i in items}, {i: Y[i][q] for i in items})
        c[q] = {"arm": float(b["arm"]), "ref": float(b["ref"]), "diff": float(b["diff"]), "ci95": [float(v) for v in b["ci95"]]}
        c[q]["rel"] = c[q]["diff"] / max(c[q]["ref"], 1e-9)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="p2", help="NEW run dir under eval_outputs/sst_hsmm_v1 (p2b for P2')")
    ap.add_argument("--rules", choices=tuple(RULES), default="orig")
    a = ap.parse_args()
    root = REPO / "eval_outputs/sst_hsmm_v1" / a.root
    SYS["new"], SYS["ctrl"] = (root, root / "play"), (CTRL, root / "play")
    ta, td, tn = RULES[a.rules]
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"pooled": {}, "cells": {}, "n": {}}
    verdict = {}
    for d in DIRS:
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(DATA / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(DATA / f"items_{d}.jsonl")}
        X, Y = load("new", d), load("ctrl", d)
        items = sorted(set(X) & set(Y) & set(meta))
        res["n"][d] = {"items": len(meta), "new": len(X), "ctrl": len(Y), "common": len(items)}
        c = pair(meta, ref, items, X, Y, lang)
        res["pooled"][d] = c
        for cell in sorted({f"{meta[i]['set']}/{meta[i]['cond']}" for i in items}):
            it = [i for i in items if f"{meta[i]['set']}/{meta[i]['cond']}" == cell]
            res["cells"][f"{d}/{cell}"] = pair(meta, ref, it, X, Y, lang)
        g = {"asr": bool(c["asr"]["diff"] >= ta), "d2": bool(c["d2"]["diff"] <= td), "np_in": bool(c["NP_in"]["rel"] <= tn)}
        verdict[d] = {**g, "asr_diff": c["asr"]["diff"], "d2_diff": c["d2"]["diff"], "np_in_rel": round(c["NP_in"]["rel"], 4),
                      "nc_rel": round(c["NC"]["rel"], 4), "pass": bool(all(g.values()))}
    verdict["GO"] = all(verdict[d]["pass"] for d in DIRS)
    verdict["rules"] = {"name": a.rules, "asr_ge": ta, "d2_le": td, "np_in_rel_le": tn}
    res["verdict"] = verdict
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(clean(verdict), indent=1))


if __name__ == "__main__":
    main()
