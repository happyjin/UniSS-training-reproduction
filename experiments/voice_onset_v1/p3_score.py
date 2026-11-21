"""voice_onset_v1 P3 scoring (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md section 4, written before any test number).
Systems: B_b, UE_b, UE_v1, UE_abs (existing), UE_mbr, UE_hmo. Cells: direction x {realsi/clean, realsi/noisy, obsi/clean,
obsi/id2, obsi/ood1}; OBSI en2zh without pair e031_0004; en2zh RealSI clean without rsc__en2zh-02-health_003 in ALL
systems (user, post hoc). Per pair: wave3_v_score.compare (ASR-BLEU stream bootstrap 2000 seed 0, PESQ / similarity paired
medians, D1 / D2 = segment LAAL / speech LAAL), ECAPA2 + TitaNet similarity, TER vs the Thinker text, runaway-free ASR-BLEU.
Rules: (1) UE_hmo - UE_abs ECAPA2 CI low > 0 in >= 3 of 6 noisy cells, no main cell CI high < 0; (2) ASR-BLEU UE_hmo - UE_abs
CI low >= -1.0 in all 10 cells; (3) report: mean D2 diff > +0.5 s flagged.
    qwen3omni PYTHONPATH=. python -m experiments.voice_onset_v1.p3_score --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, jl, laal_defs, load
from experiments.mist_w3po_v1.voicecond.test_abs_score import edit, toks
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare

REPO = Path(__file__).resolve().parents[2]
T = REPO / "eval_outputs/mist_w3po_v1/test_robust"
DIRS = ("en2zh", "zh2en")
CELLS = ("realsi/clean", "realsi/noisy", "obsi/clean", "obsi/id2", "obsi/ood1")
NOISY = {"realsi/noisy": "realsi/clean", "obsi/id2": "obsi/clean", "obsi/ood1": "obsi/clean"}
DROP = {"rsc__en2zh-02-health_003"}
# name: (root, run, ecapa2 file)
SYS = {"B_b": ("B", "b", "UE_abs/metrics/ecapa2_B_b_{d}_spksim.jsonl"),
       "UE_b": ("UE", "b", "UE_abs/metrics/ecapa2_UE_b_{d}_spksim.jsonl"),
       "UE_v1": ("UE", "v1", "UE_abs/metrics/ecapa2_UE_v1_{d}_spksim.jsonl"),
       "UE_abs": ("UE_abs", "abs", "UE_abs/metrics/ecapa2_abs_{d}_spksim.jsonl"),
       "UE_mbr": ("UE_mbr", "mbr", "UE_mbr/metrics/ecapa2_mbr_{d}_spksim.jsonl"),
       "UE_hmo": ("UE_hmo", "hmo", "UE_hmo/metrics/ecapa2_hmo_{d}_spksim.jsonl")}
PAIRS = [("UE_hmo", "UE_abs"), ("UE_mbr", "UE_abs"), ("UE_hmo", "UE_mbr"), ("UE_hmo", "B_b"), ("UE_abs", "B_b"),
         ("UE_v1", "B_b"), ("UE_b", "B_b")]


# prereg change 18:45 CET: all deployed onsets j = 0 on the test -> UE_hmo == UE_mbr (one arm run); alias when no hmo run
if not (T / "UE_hmo" / "hmo").exists():
    SYS["UE_hmo"] = SYS["UE_mbr"]
    PAIRS.remove(("UE_hmo", "UE_mbr"))


def sysd(name, ev):
    root, run, ef = SYS[name]
    r = T / root
    th = T / ("B" if root == "B" else "UE")

    def met(d, k):
        if k == "spksim" and ev == "ecapa2":
            return T / ef.format(d=d)
        return r / "metrics" / f"{run}_{d}_{k}.jsonl"
    return {"talker": lambda d: r / run / f"talker_{d}", "metrics": met,
            "thinker": lambda d: th / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"cells": {}, "recovery": {}, "medians": {}, "means": {}, "dropped": sorted(DROP)}
    r1_pos, r1_neg, r2_bad, r3 = [], [], [], []
    for d in DIRS:
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P = {ev: {s: load(sysd(s, ev), d, LD) for s in SYS} for ev in ("titanet", "ecapa2")}
        thk = {r["id"]: r["translation"] for r in jl(T / "UE" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        tkB = {r["id"]: r["translation"] for r in jl(T / "B" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        common = set.intersection(*[set(p) for p in P["titanet"].values()]) - DROP
        ter = {}
        for s in SYS:
            tx = tkB if s == "B_b" else thk
            ter[s] = {}
            for i in common:
                t = toks(tx.get(i, ""), lang); h = toks(P["titanet"][s][i]["asr"], lang)
                ter[s][i] = (edit(h, t), len(t), len(h) > 3 * len(t) + 10)
        med = {}
        for cell in CELLS:
            items = sorted(i for i in common if f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in EXCL[d]))
            k = f"{d}/{cell}"
            cres = {"n_items": len(items)}
            for s in SYS:
                e = sum(ter[s][i][0] for i in items); n = sum(ter[s][i][1] for i in items)
                cres[f"TER_{s}"] = round(100 * e / max(n, 1), 2)
                cres[f"runaway_{s}"] = sum(ter[s][i][2] for i in items)
                for ev in ("titanet", "ecapa2"):
                    v = [P[ev][s][i]["sim"] for i in items if P[ev][s][i]["sim"] is not None]
                    med[(d, cell, s, ev)] = float(np.median(v))
                    res["medians"][f"{k}/{s}/{ev}"] = round(med[(d, cell, s, ev)], 4)
                for q in ("pesq",):
                    v = [P["titanet"][s][i][q] for i in items if P["titanet"][s][i][q] is not None]
                    res["medians"][f"{k}/{s}/pesq"] = round(float(np.median(v)), 3)
                for q in ("d1", "d2"):
                    v = [P["titanet"][s][i][q] for i in items if P["titanet"][s][i][q] is not None]
                    res["means"][f"{k}/{s}/{q}"] = round(float(np.mean(v)), 3)
            for x, y in PAIRS:
                c = compare(meta, ref, items, P["titanet"][x], P["titanet"][y], lang, "stream")
                ce = compare(meta, ref, items, P["ecapa2"][x], P["ecapa2"][y], lang, "stream")
                c["sim_ecapa2"] = ce["sim"]; c["sim_titanet"] = c.pop("sim")
                keep = [i for i in items if not (ter[x][i][2] or ter[y][i][2])]
                cn = compare(meta, ref, keep, P["titanet"][x], P["titanet"][y], lang, "stream")
                c["asr_no_runaway"] = {**cn["asr"], "n_items": len(keep)}
                cres[f"{x}_vs_{y}"] = c
            res["cells"][k] = cres
            m = cres["UE_hmo_vs_UE_abs"]
            if cell in NOISY and m["sim_ecapa2"]["ci95"][0] > 0:
                r1_pos.append(k)
            if m["sim_ecapa2"]["ci95"][1] < 0:
                r1_neg.append(k)
            if m["asr"]["ci95"][0] < -1.0:
                r2_bad.append([k, m["asr"]["diff"], m["asr"]["ci95"]])
            if m["d2"]["diff"] > 0.5:
                r3.append([k, m["d2"]["diff"]])
        for cell, cc in NOISY.items():
            for ev in ("titanet", "ecapa2"):
                L = med[(d, cc, "B_b", ev)] - med[(d, cell, "B_b", ev)]
                res["recovery"][f"{d}/{cell}/{ev}"] = {"B_loss": round(L, 4), **{
                    s: (round((med[(d, cell, s, ev)] - med[(d, cell, "B_b", ev)]) / L, 3) if L > 0 else None)
                    for s in SYS if s != "B_b"}}
    res["verdict"] = {"rule1_noisy_cells_ecapa2_ci_low_gt0": r1_pos, "rule1_cells_ci_high_lt0": r1_neg,
                      "rule1_PASS": len(r1_pos) >= 3 and not r1_neg,
                      "rule2_cells_asr_ci_low_below_-1": r2_bad, "rule2_PASS": not r2_bad,
                      "hmo_holds": len(r1_pos) >= 3 and not r1_neg and not r2_bad, "rule3_D2_flags": r3}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
