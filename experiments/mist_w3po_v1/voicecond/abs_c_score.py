"""AbS-C Talker scoring (reports/mist_w3po_v1/robust_abs_c_prereg.zh-CN.md sections 4-5).

Arms B / V1 / AbS (existing outputs, paths as b_abs_seed2_score.paths) and AbS-C on S1 dev, two TTS seeds:
  AbS-C seed s: abs_c/talker_absc_<d>_s<seed>, metrics abs_c/metrics/{absc_<d>_s<seed>_{squim,spksim},
                ecapa2_absc_<d>_s<seed>_spksim}.jsonl
Per seed and cell (direction x {a, b, b_noxt} x {noisy, clean}): wave3_v_score.compare (unit = stream, 2000 resamples,
seed 0) of AbS-C against each other arm -> per-arm corpus ASR-BLEU / PESQ median / similarity median (TitaNet, ECAPA2)
and per-seed AbS-C - V1 / AbS-C - AbS CIs. Recovered fraction per seed with that seed's own B. Two-seed mean = mean of
the two seeds' cell values. Rule (section 5) on two-seed means; reported only: AbS-C vs AbS reading, pooled two-seed
ASR-BLEU CIs (per unit, both seeds' sufficient statistics summed).
    qwen3omni python -m experiments.mist_w3po_v1.voicecond.abs_c_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.wave3_v_score as W  # noqa: E402
from experiments.mist_w3po_v1.voicecond.b_abs_seed2_score import paths as paths_old, pooled_asr  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, jl  # noqa: E402

AC = REPO / "eval_outputs/mist_w3po_v1/abs_c"
SEEDS = {1: 20261021, 2: 20261022}
ARMS = ("B", "v1", "abs", "absc")
EVALS = ("titanet", "ecapa2")
MAIN = [f"{d}/{s}/{c}" for d in ("en2zh", "zh2en") for s in ("a", "b") for c in ("noisy", "clean")]
CONSISTENT = ["en2zh/b/noisy", "zh2en/a/noisy"]     # AbS two-seed consistent ASR drops


def paths(seed, arm, d):
    if arm == "absc":
        m, t = AC / "metrics", f"{d}_s{SEEDS[seed]}"
        return AC / f"talker_absc_{t}", m / f"absc_{t}_squim.jsonl", m / f"absc_{t}_spksim.jsonl", \
            m / f"ecapa2_absc_{t}_spksim.jsonl"
    return paths_old(seed, arm, d)


def load(seed, arm, d):
    run, sq, ti, ec = paths(seed, arm, d)
    ids = [r["id"] for r in jl(run / "results.jsonl")]
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    pq = {r["id"]: r.get("pesq") for r in jl(sq)}
    out = {}
    for ev, p in (("titanet", ti), ("ecapa2", ec)):
        sim = {r["id"]: r.get("sim") for r in jl(p)}
        out[ev] = {i: {"asr": asr.get(i, ""), "pesq": pq.get(i), "sim": sim.get(i), "d1": None, "d2": None} for i in ids}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seeds = (1, 2)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    res = {"seeds": SEEDS, "bootstrap": f"{W.NB} resamples of streams, seed 0", "per_seed": {1: {}, 2: {}},
           "rec": {}, "mean2": {}, "pooled_asr": {}, "n_common": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(W.S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(W.S1 / "data" / f"items_{d}.jsonl")}
        P = {s: {arm: load(s, arm, d) for arm in ARMS} for s in seeds}
        common = set.intersection(*[set(P[s][arm]["titanet"]) for s in seeds for arm in ARMS])
        res["n_common"][d] = len(common)
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                key = f"{d}/{set_}/{cond}"
                for s in seeds:
                    c = {"n_items": len(items), "asr": {}, "pesq": {}, "sim_titanet": {}, "sim_ecapa2": {}}
                    for ev in EVALS:
                        X = P[s]["absc"][ev]
                        for arm in ("B", "v1", "abs"):
                            cv = W.compare(meta, ref, items, X, P[s][arm][ev], LANG[d], "stream")
                            c["n_units"] = cv["n_units"]
                            c[f"sim_{ev}"]["absc"] = cv["sim"]["arm_median"]
                            c[f"sim_{ev}"][arm] = cv["sim"]["ref_median"]
                            c[f"sim_{ev}"][f"absc_minus_{arm}"] = {k: cv["sim"][k] for k in ("paired_median_diff", "ci95")}
                            if ev == "titanet":
                                c["asr"]["absc"], c["asr"][arm] = cv["asr"]["arm"], cv["asr"]["ref"]
                                c["asr"][f"absc_minus_{arm}"] = {k: cv["asr"][k] for k in ("diff", "ci95")}
                                c["pesq"]["absc"], c["pesq"][arm] = cv["pesq"]["arm_median"], cv["pesq"]["ref_median"]
                    res["per_seed"][s][key] = c
                res["pooled_asr"][key] = {f"absc_minus_{arm}": pooled_asr(
                    meta, ref, items, [P[s]["absc"]["titanet"] for s in seeds], [P[s][arm]["titanet"] for s in seeds],
                    LANG[d]) for arm in ("v1", "abs")}
                print(key, "done", flush=True)
    for s in seeds:
        res["rec"][s] = {}
        for d in ("en2zh", "zh2en"):
            for set_ in ("a", "b", "b_noxt"):
                n, c = res["per_seed"][s][f"{d}/{set_}/noisy"], res["per_seed"][s][f"{d}/{set_}/clean"]
                for ev in EVALS:
                    bn, bc = n[f"sim_{ev}"]["B"], c[f"sim_{ev}"]["B"]
                    res["rec"][s][f"{ev}/{d}/{set_}"] = {"B_loss": bc - bn, **{
                        arm: (n[f"sim_{ev}"][arm] - bn) / (bc - bn) for arm in ("v1", "abs", "absc")}}
        res["rec"][s]["mean4"] = {ev: {arm: mean([res["rec"][s][f"{ev}/{d}/{x}"][arm] for d in ("en2zh", "zh2en")
                                                  for x in ("a", "b")]) for arm in ("v1", "abs", "absc")} for ev in EVALS}
    for key in res["per_seed"][1]:
        res["mean2"][key] = {m: {arm: mean([res["per_seed"][s][key][m][arm] for s in seeds]) for arm in ARMS}
                             for m in ("asr", "pesq", "sim_titanet", "sim_ecapa2")}
    res["mean2"]["rec"] = {k: {arm: mean([res["rec"][s][k][arm] for s in seeds]) for arm in ("v1", "abs", "absc")}
                           for k in res["rec"][1] if k != "mean4"}
    res["mean2"]["rec_mean4"] = {ev: {arm: mean([res["rec"][s]["mean4"][ev][arm] for s in seeds])
                                      for arm in ("v1", "abs", "absc")} for ev in EVALS}
    M = res["mean2"]
    r4 = M["rec_mean4"]["ecapa2"]
    clean = {arm: mean([M[f"{d}/{x}/clean"]["sim_ecapa2"][arm] for d in ("en2zh", "zh2en") for x in ("a", "b")])
             for arm in ARMS}
    asr_gap = {k: M[k]["asr"]["absc"] - M[k]["asr"]["v1"] for k in MAIN}
    pesq_gap = {k: M[k]["pesq"]["absc"] - M[k]["pesq"]["v1"] for k in MAIN}
    rules = {
        "1_ecapa2_rec_mean4_ge_v1_plus_15": {"absc": r4["absc"], "v1": r4["v1"], "pass": r4["absc"] >= r4["v1"] + 0.15},
        "2_clean_ecapa2_mean4_ge_v1": {**clean, "pass": clean["absc"] >= clean["v1"]},
        "3_asr_ge_v1_minus_0.3_all8": {"gap": asr_gap, "pass": all(v >= -0.3 for v in asr_gap.values())},
        "4_pesq_ge_v1_minus_0.1_all8": {"gap": pesq_gap, "pass": all(v >= -0.1 for v in pesq_gap.values())},
    }
    rules["n_pass"] = sum(v["pass"] for v in rules.values() if isinstance(v, dict))
    rules["verdict"] = "established" if rules["n_pass"] == 4 else "not_established"
    res["rule"] = rules
    res["vs_abs"] = {
        "identity_kept": {"absc": r4["absc"], "abs": r4["abs"], "pass": r4["absc"] >= r4["abs"] - 0.10},
        "asr_recovered": {"gap_vs_v1": {k: asr_gap[k] for k in CONSISTENT},
                          "abs_gap_vs_v1": {k: M[k]["asr"]["abs"] - M[k]["asr"]["v1"] for k in CONSISTENT},
                          "pass": all(asr_gap[k] >= -0.3 for k in CONSISTENT)},
        "asr_absc_minus_abs_mean2": {k: M[k]["asr"]["absc"] - M[k]["asr"]["abs"] for k in MAIN},
        "pesq_absc_minus_abs_mean2": {k: M[k]["pesq"]["absc"] - M[k]["pesq"]["abs"] for k in MAIN},
    }
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"rule": rules, "vs_abs": res["vs_abs"]}, indent=1))


if __name__ == "__main__":
    main()
