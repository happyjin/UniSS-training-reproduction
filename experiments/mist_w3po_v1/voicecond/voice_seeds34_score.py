"""Voice seeds 3-4 measurement scoring (reports/mist_w3po_v1/robust_voice_seeds34_prereg.zh-CN.md sections 3-4).

Arms B / V1 / AbS / AbS-C on S1 dev, four TTS seeds:
  seeds 1, 2 (20261021, 20261022): existing outputs, paths as abs_c_score.paths
  seeds 3, 4 (20261023, 20261024): voice_seeds34/talker_<arm>_<d>_s<seed>, metrics voice_seeds34/metrics/
                                   {<arm>_<d>_s<seed>_{squim,spksim}, ecapa2_<arm>_<d>_s<seed>_spksim}.jsonl
Items = intersection over 4 seeds x 4 arms. Per seed / cell / arm: corpus ASR-BLEU (Bleu stats summed), PESQ and similarity
item medians (= wave3_v_score.compare arm medians). 4-seed mean and sample SD (n-1). Recovered fraction per seed with that
seed's own B. Pooled 4-seed paired bootstrap of ASR-BLEU M - V1 (b_abs_seed2_score.pooled_asr: unit = stream, unit stats
of all seeds summed, 2000 resamples, seed 0). Reading (section 4): M non-inferior on ASR iff in all 8 main cells
mean4(M - V1) >= -0.3 or pooled CI low >= -0.5.
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.voice_seeds34_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.wave3_v_score as W  # noqa: E402
from experiments.mist_w3po_v1.voicecond.abs_c_score import paths as paths12  # noqa: E402
from experiments.mist_w3po_v1.voicecond.b_abs_seed2_score import pooled_asr  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, Bleu, jl  # noqa: E402

VS = REPO / "eval_outputs/mist_w3po_v1/voice_seeds34"
SEEDS = {1: 20261021, 2: 20261022, 3: 20261023, 4: 20261024}
ARMS = ("B", "v1", "abs", "absc")
METHODS = ("abs", "absc")
EVALS = ("titanet", "ecapa2")
MAIN = [f"{d}/{s}/{c}" for d in ("en2zh", "zh2en") for s in ("a", "b") for c in ("noisy", "clean")]


def paths(seed, arm, d):
    if seed in (1, 2):
        return paths12(seed, arm, d)
    t, m = f"{arm}_{d}_s{SEEDS[seed]}", VS / "metrics"
    return VS / f"talker_{t}", m / f"{t}_squim.jsonl", m / f"{t}_spksim.jsonl", m / f"ecapa2_{t}_spksim.jsonl"


def load(seed, arm, d):
    run, sq, ti, ec = paths(seed, arm, d)
    ids = [r["id"] for r in jl(run / "results.jsonl")]
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    pq = {r["id"]: r.get("pesq") for r in jl(sq)}
    s_ti = {r["id"]: r.get("sim") for r in jl(ti)}
    s_ec = {r["id"]: r.get("sim") for r in jl(ec)}
    return {i: {"asr": asr.get(i, ""), "pesq": pq.get(i), "titanet": s_ti.get(i), "ecapa2": s_ec.get(i)} for i in ids}


def med(P, items, k):
    v = [P[i][k] for i in items if P[i][k] is not None]
    return float(np.median(v)) if v else float("nan")


def ms(xs):
    return {"mean": sum(xs) / len(xs), "sd": st.stdev(xs), "per_seed": xs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seeds = (1, 2, 3, 4)
    res = {"seeds": SEEDS, "bootstrap": f"pooled 4 seeds, {W.NB} resamples of streams, seed 0", "per_seed": {s: {} for s in seeds},
           "cell": {}, "rec": {}, "pooled_asr": {}, "n_common": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {s: {arm: load(s, arm, d) for arm in ARMS} for s in seeds}
        common = set.intersection(*[set(P[s][arm]) for s in seeds for arm in ARMS])
        res["n_common"][d] = len(common)
        bl = Bleu(LANG[d])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                key = f"{d}/{set_}/{cond}"
                refs = [ref[i] for i in items]
                for s in seeds:
                    res["per_seed"][s][key] = {arm: {
                        "asr": bl.score(bl.stats([P[s][arm][i]["asr"] for i in items], refs)),
                        "pesq": med(P[s][arm], items, "pesq"),
                        "sim_titanet": med(P[s][arm], items, "titanet"),
                        "sim_ecapa2": med(P[s][arm], items, "ecapa2")} for arm in ARMS}
                c = {"n_items": len(items), "n_units": len({meta[i]["stream"] for i in items})}
                for m in ("asr", "pesq", "sim_titanet", "sim_ecapa2"):
                    c[m] = {arm: ms([res["per_seed"][s][key][arm][m] for s in seeds]) for arm in ARMS}
                    for M in METHODS:
                        c[m][f"{M}_minus_v1"] = ms([res["per_seed"][s][key][M][m] - res["per_seed"][s][key]["v1"][m]
                                                    for s in seeds])
                res["cell"][key] = c
                pa = lambda M: {i: {"asr": P_[i]["asr"]} for i in items for P_ in [M]}  # noqa: E731
                res["pooled_asr"][key] = {f"{M}_minus_v1": pooled_asr(
                    meta, ref, items, [pa(P[s][M]) for s in seeds], [pa(P[s]["v1"]) for s in seeds], LANG[d])
                    for M in METHODS}
                print(key, "done", flush=True)
    # recovered fraction per seed (own B), per direction x set
    for ev in EVALS:
        for d in ("en2zh", "zh2en"):
            for set_ in ("a", "b", "b_noxt"):
                per = {arm: [] for arm in ("v1", "abs", "absc")}
                for s in seeds:
                    n, c = res["per_seed"][s][f"{d}/{set_}/noisy"], res["per_seed"][s][f"{d}/{set_}/clean"]
                    bn, bc = n["B"][f"sim_{ev}"], c["B"][f"sim_{ev}"]
                    for arm in per:
                        per[arm].append((n[arm][f"sim_{ev}"] - bn) / (bc - bn))
                res["rec"][f"{ev}/{d}/{set_}"] = {arm: ms(v) for arm, v in per.items()}
        res["rec"][f"{ev}/mean4"] = {arm: ms([sum(res["rec"][f"{ev}/{d}/{x}"][arm]["per_seed"][k]
                                                  for d in ("en2zh", "zh2en") for x in ("a", "b")) / 4 for k in range(4)])
                                     for arm in ("v1", "abs", "absc")}
    reading = {}
    for M in METHODS:
        cells = {}
        for k in MAIN:
            mdiff = res["cell"][k]["asr"][f"{M}_minus_v1"]["mean"]
            lo = res["pooled_asr"][k][f"{M}_minus_v1"]["ci95"][0]
            cells[k] = {"mean4_diff": mdiff, "pooled_ci_low": lo, "pass": mdiff >= -0.3 or lo >= -0.5}
        ok = all(v["pass"] for v in cells.values())
        reading[M] = {"cells": cells, "fail": [k for k, v in cells.items() if not v["pass"]],
                      "verdict": "non-inferior on ASR" if ok else "ASR non-inferiority not supported"}
    res["reading"] = reading
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({M: {"verdict": v["verdict"], "fail": v["fail"]} for M, v in reading.items()}, indent=1))


if __name__ == "__main__":
    main()
