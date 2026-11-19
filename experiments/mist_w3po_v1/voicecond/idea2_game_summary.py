"""Idea 2 M1b esports demo summary (prereg v2 §G; plan §7.2f acceptance bar): paired speech metrics vs baseline B
(idea1_speech_metrics compare output), speaker-blind ASR-BLEU (game_blind_bleu; baseline B computed here if missing),
speech LAAL (unit speech_laal_online, all units, from the orchestrator's latency files), token-table statistics.

    python -m experiments.mist_w3po_v1.voicecond.idea2_game_summary [--tag shrink]      (conda_envs/uniss-eval python)
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/game_summary_<tag>.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
OUT = E / "voicecond_idea2"
MET = OUT / "metrics"


def bleu(d, run):
    p = MET / f"blind_bleu_{run}.json"
    if not p.exists():
        subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.game_blind_bleu", "--dir", d, "--hyp",
                        str(E / "decode" / run / "hyp_asr.jsonl"), "--out", str(p)], check=True, cwd=REPO)
    return json.load(open(p))["corpus"]


def laal(run, base):
    p = E / "crf/t_eval" / base / f"lat_{run}.json"
    return json.load(open(p))["unit"]["speech_laal_online"]["all"]["mean"]


def spksim(tag, n_boot=10000):
    """prereg v2.1: TitaNet cosine to the whole-excerpt solo-frame HDemucs-vocals reference of the stream's slot"""
    SP = OUT / "spksim"
    arms = {"B": "", tag: f"_{tag}", "V": "_vocals", "ZNE_Rt": "_zne_Rt"}
    load = lambda run: ({r["id"]: r["sim"] for r in map(json.loads, open(SP / f"{run}.jsonl"))}
                        if (SP / f"{run}.jsonl").exists() else None)
    res = {"per_run_median": {}, "missing_runs": [], "delta_vs_B": {}}
    deltas = {}       # (dir, id) -> seed-averaged delta tag - B
    noobs = {}
    for d in ("zh2en", "en2zh"):
        tok = {t["id"]: t for t in map(json.loads, open(OUT / "tokens" / f"game_{d}_{tag}.jsonl"))}
        sims = {}
        for arm, sfx in arms.items():
            for sname, ssfx in (("s1", ""), ("s2", "_tts2")):
                run = f"game_b_{d}__t1c{sfx}{ssfx}"
                v = load(run)
                if v is None:
                    res["missing_runs"].append(run)
                    continue
                sims[(arm, sname)] = v
                res["per_run_median"][run] = float(np.median(list(v.values())))
        for arm in arms:
            if arm == "B":
                continue
            for sname in ("s1", "s2"):
                if (arm, sname) not in sims or ("B", sname) not in sims:
                    continue
                A, X = sims[("B", sname)], sims[(arm, sname)]
                ids = sorted(set(A) & set(X))
                dd = np.array([X[i] - A[i] for i in ids])
                res["delta_vs_B"][f"{d}_{arm}_{sname}"] = {"n": len(ids), "median": float(np.median(dd)),
                                                          "mean": float(dd.mean()), "share_pos": float((dd > 0).mean())}
        if all((k, s) in sims for k in ("B", tag) for s in ("s1", "s2")):
            ids = sorted(set.intersection(*[set(sims[(k, s)]) for k in ("B", tag) for s in ("s1", "s2")]))
            for i in ids:
                deltas[(d, i)] = 0.5 * ((sims[(tag, "s1")][i] - sims[("B", "s1")][i]) + (sims[(tag, "s2")][i] - sims[("B", "s2")][i]))
                noobs[(d, i)] = tok[i]["n_win"] == 0 if i in tok else False
    rng = np.random.default_rng(0)

    def stat(keys):
        v = np.array([deltas[k] for k in keys])
        if len(v) == 0:
            return None
        bs = np.median(v[rng.integers(0, len(v), (n_boot, len(v)))], axis=1)
        return {"n": len(v), "median": float(np.median(v)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "mean": float(v.mean())}
    allk = sorted(deltas)
    res["gate"] = {"all": stat(allk), "prior_mean_streams": stat([k for k in allk if noobs[k]]),
                   "other_streams": stat([k for k in allk if not noobs[k]]),
                   "zh2en": stat([k for k in allk if k[0] == "zh2en"]), "en2zh": stat([k for k in allk if k[0] == "en2zh"])}
    res["gate"]["pass"] = bool(res["gate"]["all"] and res["gate"]["all"]["median"] >= -0.02)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="shrink")
    a = ap.parse_args()
    out = OUT / f"game_summary_{a.tag}.json"
    if out.exists():
        sys.exit(f"refuse to overwrite {out}")
    cmp = json.load(open(MET / f"compare_game_{a.tag}_vs_B.json"))
    S = {"tag": a.tag, "dirs": {}}
    for d in ("zh2en", "en2zh"):
        r = {"seeds": {}}
        for s, sfx in (("s1", ""), ("s2", "_tts2")):
            c = cmp[f"{d}_{s}"]
            nb, nn = f"game_b_{d}__t1c{sfx}", f"game_b_{d}__t1c_{a.tag}{sfx}"
            r["seeds"][s] = {
                "pesq_median_diff": c["pesq"]["median_diff"], "pesq_median_ci": c["pesq"]["median_ci"],
                "pesq_A_median": c["pesq"]["A_median"], "pesq_B_median": c["pesq"]["B_median"],
                "contrast_median_diff": c["contrast"]["median_diff"], "contrast_median_ci": c["contrast"]["median_ci"],
                "flatness_median_diff": c["flatness"]["median_diff"], "n": c["pesq"]["n"],
                "bleu_B": bleu(d, nb), "bleu_new": bleu(d, nn),
                "laal_B": laal(nb, f"game_b_{d}__t1c"), "laal_new": laal(nn, f"game_b_{d}__t1c_{a.tag}")}
        sd = r["seeds"]
        r["pesq_median_diff_mean2"] = float(np.mean([sd[s]["pesq_median_diff"] for s in sd]))
        r["contrast_median_diff_mean2"] = float(np.mean([sd[s]["contrast_median_diff"] for s in sd]))
        r["bleu_diff_mean2"] = float(np.mean([sd[s]["bleu_new"] - sd[s]["bleu_B"] for s in sd]))
        r["laal_diff_mean2"] = float(np.mean([sd[s]["laal_new"] - sd[s]["laal_B"] for s in sd]))
        r["accept"] = {"pesq_ge_0.5": r["pesq_median_diff_mean2"] >= 0.5,
                       "contrast_ge_1.5dB": r["contrast_median_diff_mean2"] >= 1.5,
                       "bleu_not_lower": r["bleu_diff_mean2"] >= 0}
        tok = [json.loads(l) for l in open(OUT / "tokens" / f"game_{d}_{a.tag}.jsonl")]
        r["tokens"] = {"n": len(tok), "no_obs": sum(t["n_win"] == 0 for t in tok),
                       "n_win_median": float(np.median([t["n_win"] for t in tok])),
                       "ham_vs_V_median": float(np.median([sum(x != y for x, y in zip(t["ids"], t["ids_V"])) for t in tok]))}
        S["dirs"][d] = r
    S["spksim"] = spksim(a.tag)
    out.write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
