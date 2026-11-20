"""Robust line, full-system combination scoring (reports/mist_w3po_v1/robust_combo_plan_20261120.zh-CN.md, section 6).

Generalises wave14_combo_score.py / wave5_m_combo_score.py to any set of arms. Systems:
  B      = S1 baseline (raw LA2 Thinker + raw voice + T1c)            wave2_S1/talker_<d>_s20261021, metrics wave3 V B_*
  ew_v1  = wave-5 ERM-WiSE full system                                wave5/M/combo/ew_v1
  <arm>_v1 / <arm>_b for every --arm name=root:
         a combo root (combo_orch.sh: <root>/v1/talker_<d>, <root>/b/talker_<d>, <root>/metrics/{v1,b}_<d>_{squim,spksim})
         or a wave eval root (<root>/talker_<d>_s20261021 -> <arm>_b; PESQ / similarity absent = not reported).
         Thinker text from <root>/thinker_<d>/blocks_<d>_w1.jsonl.
Cells, units and statistics exactly as wave3_v_score.compare (direction x {a, b, b_noxt} x {noisy, clean}; ASR-BLEU from
unit-summed sufficient statistics, 2000 unit bootstrap resamples, seed 0); text BLEU of the Thinker output with the same
unit bootstrap (aux).
Primary rule (the wave-5 full-system rule, unchanged) for <primary>_v1 vs B:
  (1) ASR-BLEU not degraded in every direction x {a, b} x condition: diff >= -0.3 and CI high >= 0;
  (2) noisy PESQ improves (paired median > 0, CI low > 0) in every direction x {a, b_noxt}.
Reported with the same no-degradation check: <primary>_v1 vs ew_v1. Ablation table: <primary>_v1 vs <x>_v1 for every
--ablate x (component contribution = full - full-without-component), <primary>_v1 vs <primary>_b (- V1 voice).
Extra pairs: --pair X:Y (system names), e.g. cvla_base_b:ctrl_budget_base_b (CV-LA vs its equal-latency control).

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.combo_score --arm full=eval_outputs/mist_w3po_v1/combo/full \
        --arm noN=... --ablate noN --ablate noC --primary full [--pair X:Y] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, Bleu, jl, laal_defs  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave3_v_score import NB, S1, SEED, V, compare  # noqa: E402

W5 = REPO / "eval_outputs/mist_w3po_v1/wave5/M/combo"
DIRS = ("en2zh", "zh2en")


def systems(arms: dict) -> dict:
    """name -> {talker: f(d), metrics: f(d, kind) or None, thinker: f(d) or None}"""
    S = {"B": {"talker": lambda d: S1 / f"talker_{d}_s{SEED}", "metrics": lambda d, k: V / "metrics" / f"B_{d}_{k}.jsonl",
               "thinker": lambda d: S1 / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"},
         "ew_v1": {"talker": lambda d: W5 / "ew_v1" / f"talker_{d}", "metrics": lambda d, k: W5 / "metrics" / f"ew_v1_{d}_{k}.jsonl",
                   "thinker": lambda d: REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"}}
    for name, root in arms.items():
        root = (REPO / root).resolve() if not Path(root).is_absolute() else Path(root)
        th = (lambda r: lambda d: r / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")(root)
        if (root / "v1").is_dir() or (root / "b").is_dir():
            for run in ("v1", "b"):
                if (root / run / f"talker_{DIRS[0]}").exists():
                    S[f"{name}_{run}"] = {"talker": (lambda r, u: lambda d: r / u / f"talker_{d}")(root, run),
                                          "metrics": (lambda r, u: lambda d, k: r / "metrics" / f"{u}_{d}_{k}.jsonl")(root, run),
                                          "thinker": th}
        elif (root / f"talker_{DIRS[0]}_s{SEED}").exists():
            S[f"{name}_b"] = {"talker": (lambda r: lambda d: r / f"talker_{d}_s{SEED}")(root), "metrics": None, "thinker": th}
        else:
            raise SystemExit(f"arm {name}: no talker runs under {root}")
    return S


def load(sysd: dict, d: str, LD) -> dict:
    run = sysd["talker"](d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    met = {}
    for k, f in (("squim", "pesq"), ("spksim", "sim")):
        p = sysd["metrics"](d, k) if sysd["metrics"] else None
        met[f] = {r["id"]: r.get(f) for r in jl(p)} if p and p.exists() else {}
    tr = {}
    if sysd["thinker"] and sysd["thinker"](d).exists():
        tr = {r["id"]: r["translation"] for r in jl(sysd["thinker"](d))}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": met["pesq"].get(i), "sim": met["sim"].get(i), "text": tr.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def text_compare(meta, ref, items, X, Y, lang, unit_key):
    """Thinker text BLEU X - Y, same unit bootstrap as compare's ASR-BLEU (seed 0)"""
    if any(X[i]["text"] is None or Y[i]["text"] is None for i in items):
        return None
    bl = Bleu(lang)
    units = sorted({meta[i][unit_key] for i in items})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i][unit_key]] for i in items])
    nU = len(units)

    def ustats(P):
        st = bl.stats([P[i]["text"] for i in items], [ref[i] for i in items])
        U = np.zeros((nU, st.shape[1]), np.int64)
        np.add.at(U, pu, st)
        return U
    UX, UY = ustats(X), ustats(Y)
    rng = np.random.default_rng(0)
    W = [np.ones(nU)] + [np.bincount(rng.integers(0, nU, nU), minlength=nU).astype(float) for _ in range(NB)]
    vals = [bl.score((UX * w[:, None]).sum(0)) - bl.score((UY * w[:, None]).sum(0)) for w in W]
    return {"arm": round(bl.score(UX.sum(0)), 3), "ref": round(bl.score(UY.sum(0)), 3), "diff": round(vals[0], 3),
            "ci95": [round(float(np.percentile(vals[1:], 2.5)), 3), round(float(np.percentile(vals[1:], 97.5)), 3)]}


def clean(x):
    """NaN (PESQ / similarity of systems without those metrics) -> null, so the json stays standard"""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], help="name=root")
    ap.add_argument("--primary", default="full")
    ap.add_argument("--ablate", action="append", default=[])
    ap.add_argument("--pair", action="append", default=[], help="X:Y system names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    arms = dict(x.split("=", 1) for x in a.arm)
    S = systems(arms)
    P0 = f"{a.primary}_v1"
    pairs = [(P0, "B"), (P0, "ew_v1")] + [(P0, f"{x}_v1") for x in a.ablate] + [tuple(p.split(":")) for p in a.pair]
    if f"{a.primary}_b" in S:
        pairs.append((P0, f"{a.primary}_b"))
    for x, y in pairs:
        if x not in S or y not in S:
            raise SystemExit(f"pair {x}:{y}: unknown system (have {sorted(S)})")
    LD = laal_defs()
    res = {"seed": SEED, "systems": {k: str(v["talker"]("en2zh")) for k, v in S.items()}, "pairs": [f"{x}_vs_{y}" for x, y in pairs],
           "cells": {}}
    fails, pesq_ok, vs_erm = [], [], []
    for d in DIRS:
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        used = sorted({s for p in pairs for s in p})
        Pd = {s: load(S[s], d, LD) for s in used}
        common = set.intersection(*[set(p) for p in Pd.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                cell = {}
                for x, y in pairs:
                    c = compare(meta, ref, items, Pd[x], Pd[y], LANG[d], "stream")
                    c["text"] = text_compare(meta, ref, items, Pd[x], Pd[y], LANG[d], "stream")
                    cell[f"{x}_vs_{y}"] = c
                key = f"{d}/{set_}/{cond}"
                res["cells"][key] = cell
                m, e = cell[f"{P0}_vs_B"], cell[f"{P0}_vs_ew_v1"]
                if set_ in ("a", "b"):
                    if not (m["asr"]["diff"] >= -0.3 and m["asr"]["ci95"][1] >= 0):
                        fails.append([key, m["asr"]["diff"], m["asr"]["ci95"]])
                    if not (e["asr"]["diff"] >= -0.3 and e["asr"]["ci95"][1] >= 0):
                        vs_erm.append([key, e["asr"]["diff"], e["asr"]["ci95"]])
                if cond == "noisy" and set_ in ("a", "b_noxt"):
                    pesq_ok.append([f"{d}/{set_}", m["pesq"]["paired_median_diff"], m["pesq"]["ci95"],
                                    m["pesq"]["paired_median_diff"] > 0 and m["pesq"]["ci95"][0] > 0])
    abl = {}
    for x, y in pairs[2:]:
        k = f"{x}_vs_{y}"
        abl[k] = {c: {"asr": v[k]["asr"]["diff"], "asr_ci": v[k]["asr"]["ci95"],
                      "text": (v[k]["text"] or {}).get("diff"), "text_ci": (v[k]["text"] or {}).get("ci95"),
                      "d1": v[k]["d1"]["diff"], "d2": v[k]["d2"]["diff"]} for c, v in res["cells"].items()}
    res["ablation_table"] = abl
    res["verdict"] = {"asr_no_degradation_failures": fails, "noisy_pesq": pesq_ok,
                      "PASS": not fails and all(p[3] for p in pesq_ok),
                      "reported_vs_erm_wise_asr_failures": vs_erm}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
