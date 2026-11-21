"""Robust test, AbS voice scoring (reports/mist_w3po_v1/robust_test_prereg_20261120.zh-CN.md amendment 3, written before
any AbS test number). Systems: B_b (test_robust/B/b), UE_b (UE/b), UE_v1 (UE/v1), UE_abs (UE_abs/abs). Cells: direction x
{realsi/clean, realsi/noisy, obsi/clean, obsi/id2, obsi/ood1}; OBSI en2zh without pair e031_0004.
Per pair (UE_abs vs UE_v1 = main, UE_abs vs B_b, UE_v1 vs B_b): wave3_v_score.compare (ASR-BLEU unit-summed sufficient
statistics, stream bootstrap 2000, seed 0; PESQ / similarity paired medians; D1 / D2), with TitaNet and with ECAPA2
similarity; Talker TER vs the Thinker text; ASR-BLEU without runaway items (ASR tokens > 3 x Thinker tokens + 10, union of
the two systems). Identity recovery = (arm noisy median - B_b noisy) / (B_b clean - B_b noisy) per evaluator.
Rules: (1) >= 4 of the 6 noisy cells with ECAPA2 UE_abs - UE_v1 CI low > 0 and no main cell with CI high < 0;
(2) every main cell ASR-BLEU UE_abs - UE_v1 CI low >= -1.0. Reported: UE_abs vs B_b by section 6.4 rules.
    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.test_abs_score --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, jl, laal_defs, load
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare
from evaluation.text_metrics import normalize_for_bleu

REPO = Path(__file__).resolve().parents[3]
T = REPO / "eval_outputs/mist_w3po_v1/test_robust"
A = T / "UE_abs"
DIRS = ("en2zh", "zh2en")
CELLS = ("realsi/clean", "realsi/noisy", "obsi/clean", "obsi/id2", "obsi/ood1")
NOISY = {"realsi/noisy": "realsi/clean", "obsi/id2": "obsi/clean", "obsi/ood1": "obsi/clean"}
SYS = {"B_b": ("B", "b"), "UE_b": ("UE", "b"), "UE_v1": ("UE", "v1"), "UE_abs": ("UE_abs", "abs")}
PAIRS = [("UE_abs", "UE_v1"), ("UE_abs", "B_b"), ("UE_v1", "B_b")]


def sysd(name, ev):
    arm, run = SYS[name]
    r = T / arm
    th = T / ("B" if arm == "B" else "UE")

    def met(d, k):
        if k == "spksim" and ev == "ecapa2":
            return A / "metrics" / f"ecapa2_{run if arm == 'UE_abs' else name}_{d}_spksim.jsonl"
        return r / "metrics" / f"{run}_{d}_{k}.jsonl"
    return {"talker": lambda d: r / run / f"talker_{d}", "metrics": met,
            "thinker": lambda d: th / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"}


def toks(x, lang):
    x = normalize_for_bleu(x or "", lang)
    return [c for c in x if not c.isspace()] if lang == "cmn" else x.split()


def edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"cells": {}, "recovery": {}, "search_latency": {}}
    r1_pos, r1_neg, r2_bad, b_r1, b_r2 = [], [], [], [], []
    for d in DIRS:
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P = {ev: {s: load(sysd(s, ev), d, LD) for s in SYS} for ev in ("titanet", "ecapa2")}
        thk = {r["id"]: r["translation"] for r in jl(T / "UE" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        tkB = {r["id"]: r["translation"] for r in jl(T / "B" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        common = set.intersection(*[set(p) for p in P["titanet"].values()])
        st = [json.loads(l) for k in range(4) for l in open(A / "search" / f"search_{d}_{k}.jsonl") if l.strip()]
        ts = np.array([r["t_search"] for r in st])
        res["search_latency"][d] = {"median_s": round(float(np.median(ts)), 3), "p90_s": round(float(np.percentile(ts, 90)), 3),
                                    "mean_s": round(float(ts.mean()), 3), "n": len(ts)}
        # Talker TER + runaway flags per system
        ter = {}
        for s in SYS:
            tx = tkB if s == "B_b" else thk
            ter[s] = {}
            for i in common:
                t = toks(tx.get(i, ""), lang)
                h = toks(P["titanet"][s][i]["asr"], lang)
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
            for x, y in PAIRS:
                c = compare(meta, ref, items, P["titanet"][x], P["titanet"][y], lang, "stream")
                ce = compare(meta, ref, items, P["ecapa2"][x], P["ecapa2"][y], lang, "stream")
                c["sim_ecapa2"] = ce["sim"]
                c["sim_titanet"] = c.pop("sim")
                keep = [i for i in items if not (ter[x][i][2] or ter[y][i][2])]
                cn = compare(meta, ref, keep, P["titanet"][x], P["titanet"][y], lang, "stream")
                c["asr_no_runaway"] = {**cn["asr"], "n_items": len(keep)}
                cres[f"{x}_vs_{y}"] = c
            res["cells"][k] = cres
            m = cres["UE_abs_vs_UE_v1"]
            if cell in NOISY and m["sim_ecapa2"]["ci95"][0] > 0:
                r1_pos.append(k)
            if m["sim_ecapa2"]["ci95"][1] < 0:
                r1_neg.append(k)
            if m["asr"]["ci95"][0] < -1.0:
                r2_bad.append([k, m["asr"]["diff"], m["asr"]["ci95"]])
            mb = cres["UE_abs_vs_B_b"]["asr"]
            if not (mb["diff"] >= -0.3 and mb["ci95"][1] >= 0):
                b_r1.append([k, mb["diff"], mb["ci95"]])
            if cell in NOISY and mb["ci95"][0] > 0:
                b_r2.append([k, mb["diff"], mb["ci95"]])
        for cell, cc in NOISY.items():
            for ev in ("titanet", "ecapa2"):
                L = med[(d, cc, "B_b", ev)] - med[(d, cell, "B_b", ev)]
                res["recovery"][f"{d}/{cell}/{ev}"] = {
                    "B_loss": round(L, 4),
                    **{s: round((med[(d, cell, s, ev)] - med[(d, cell, "B_b", ev)]) / L, 3) if L > 0 else None
                       for s in ("UE_b", "UE_v1", "UE_abs")}}
        res["medians"] = res.get("medians", {})
        for (dd, cell, s, ev), v in med.items():
            res["medians"][f"{dd}/{cell}/{s}/{ev}"] = round(v, 4)
    res["verdict"] = {
        "rule1_noisy_cells_ecapa2_ci_low_gt0": r1_pos, "rule1_cells_ecapa2_ci_high_lt0": r1_neg,
        "rule1_PASS": len(r1_pos) >= 4 and not r1_neg,
        "rule2_cells_asr_ci_low_below_-1": r2_bad, "rule2_PASS": not r2_bad,
        "abs_holds_on_test": len(r1_pos) >= 4 and not r1_neg and not r2_bad,
        "report_vs_B_b_rule6.4_1_degradations": b_r1, "report_vs_B_b_rule6.4_2_noisy_improvements": b_r2}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
