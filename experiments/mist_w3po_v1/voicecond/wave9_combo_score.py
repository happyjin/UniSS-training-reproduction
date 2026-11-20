"""Robust wave 9 anchor, full-system combination scoring (reports/mist_w3po_v1/robust_wave9_anchor_prereg.zh-CN.md): a copy of
wave8_aul_combo_score.py with the anchor arms an_b (A WiSE-FT 0.5 Thinker + raw voice, wave9/combo/an_b) and an_v1 (A
WiSE + V1 voice, the full system, wave9/combo/an_v1) added. Primary rule (as wave 5) for an_v1 vs B; an_v1 vs ew_v1
(the wave-5 ERM-WiSE full system) and an_b vs ew_b reported with the same no-degradation check, no gate.
The original wave-5 docstring follows.

Robust wave 5, full-system combination scoring (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md, "完整系统组合").

Systems per direction (one TTS seed 20261021, S1 talker flags, 4 shards):
  B     = S1 baseline (raw LA2 Thinker + raw voice + T1c)            wave2_S1/talker_<d>_s20261021
  v1    = raw LA2 Thinker + V1 voice (wave 3)                         wave3_combos/V/v1/talker_<d>
  ew_b  = ERM WiSE-FT 0.5 Thinker + raw voice                         wave5/M/combo/ew_b/talker_<d>
  ew_v1 = ERM WiSE-FT 0.5 Thinker + V1 voice  (the full system)       wave5/M/combo/ew_v1/talker_<d>
Cells, units and statistics exactly as wave3_v_score.compare (direction x {a, b, b_noxt} x {noisy, clean}).
Primary rule for ew_v1 vs B (unified rule + change 2; LAAL auxiliary, no gate):
  (1) ASR-BLEU not degraded in every direction x {a, b} x condition: diff >= -0.3 and CI high >= 0;
  (2) noisy PESQ improves (paired median > 0, CI low > 0) in every direction x {a, b_noxt};
  similarity (median >= -0.02) and D2 LAAL reported.
Reported, no gate: ew_b vs B (Thinker only), ew_v1 vs ew_b (voice on top of the Thinker), ew_v1 vs v1.

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.wave5_m_combo_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, jl, laal_defs  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave3_v_score import S1, SEED, V, compare  # noqa: E402

C = REPO / "eval_outputs/mist_w3po_v1/wave5/M/combo"
A = REPO / "eval_outputs/mist_w3po_v1/wave9/combo"
RUN = {"B": lambda d: S1 / f"talker_{d}_s{SEED}", "v1": lambda d: V / "v1" / f"talker_{d}",
       "ew_b": lambda d: C / "ew_b" / f"talker_{d}", "ew_v1": lambda d: C / "ew_v1" / f"talker_{d}",
       "an_b": lambda d: A / "an_b" / f"talker_{d}", "an_v1": lambda d: A / "an_v1" / f"talker_{d}"}
MET = {"B": V / "metrics", "v1": V / "metrics", "ew_b": C / "metrics", "ew_v1": C / "metrics",
       "an_b": A / "metrics", "an_v1": A / "metrics"}
PAIRS = (("an_v1", "B"), ("an_b", "B"), ("an_v1", "an_b"), ("an_v1", "ew_v1"), ("an_b", "ew_b"), ("ew_v1", "B"))


def load(arm, d, LD):
    run = RUN[arm](d)
    recs = {r["id"]: r for r in jl(run / "results.jsonl")}
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    sq = {r["id"]: r.get("pesq") for r in jl(MET[arm] / f"{arm}_{d}_squim.jsonl")}
    sim = {r["id"]: r.get("sim") for r in jl(MET[arm] / f"{arm}_{d}_spksim.jsonl")}
    per = {}
    for i, r in recs.items():
        sd = LD.sentence_defs(r)
        per[i] = {"asr": asr.get(i, ""), "pesq": sq.get(i), "sim": sim.get(i),
                  "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None}
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed": SEED, "cells": {}}
    fails, pesq_ok, vs_erm = [], [], []
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in RUN}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                cell = {f"{x}_vs_{y}": compare(meta, ref, items, P[x], P[y], LANG[d], "stream") for x, y in PAIRS}
                res["cells"][f"{d}/{set_}/{cond}"] = cell
                m = cell["an_v1_vs_B"]
                for x, y in (("an_v1", "ew_v1"), ("an_b", "ew_b")):
                    e = cell[f"{x}_vs_{y}"]
                    if set_ in ("a", "b") and not (e["asr"]["diff"] >= -0.3 and e["asr"]["ci95"][1] >= 0):
                        vs_erm.append([f"{x}_vs_{y}", f"{d}/{set_}/{cond}", e["asr"]["diff"], e["asr"]["ci95"]])
                if set_ in ("a", "b") and not (m["asr"]["diff"] >= -0.3 and m["asr"]["ci95"][1] >= 0):
                    fails.append([f"{d}/{set_}/{cond}", m["asr"]["diff"], m["asr"]["ci95"]])
                if cond == "noisy" and set_ in ("a", "b_noxt"):
                    pesq_ok.append([f"{d}/{set_}", m["pesq"]["paired_median_diff"], m["pesq"]["ci95"],
                                    m["pesq"]["paired_median_diff"] > 0 and m["pesq"]["ci95"][0] > 0])
    res["verdict"] = {"asr_no_degradation_failures": fails, "noisy_pesq": pesq_ok,
                      "PASS": not fails and all(p[3] for p in pesq_ok),
                      "reported_vs_erm_wise_asr_failures": vs_erm}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
