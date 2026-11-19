"""Idea 2 prereg v3 §D: test summaries for the frozen shrink+Vfb against baseline B (strict gates, honest reporting).

bench  OpenBroadcast-SI v1 (bench_eval metrics.json of obsi_b_* and obsi_i2_*): per (cond, dir) two-seed means of
       tcpASR-BLEU, blind ASR-BLEU, chunk speech LAAL, EndOffset; per-stream paired differences (seed-averaged per stream)
       of PESQ, contrast, flatness, SI-SDR and WavLM-SV similarity (median, bootstrap 95 % CI, Wilcoxon); Holm over the
       PESQ family; gates: tcpASR-BLEU not lower, LAAL not larger, median similarity delta >= -0.02.
ood3   AMI / AliMeeting subsets (idea2_ood3_run.sh outputs): tcpBLEU_all of score_asr.json, chunk LAAL, per-item paired
       PESQ / contrast / flatness (idea1_speech_metrics compare) and TitaNet similarity (seed-averaged per item).
    python -m experiments.mist_w3po_v1.voicecond.idea2_test_summary bench|ood3      (conda_envs/uniss-eval python)
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/test_summary_<which>.json (a new _v<k> file if one exists).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
OUT = E / "voicecond_idea2"
VB = E / "voicecond_bench"


def boot_median(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.median(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    if len(d) < 5:
        return None
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    return {"n": int(len(d)), "median": float(np.median(d)), "ci": boot_median(d), "mean": float(d.mean()), "p": p,
            "share_pos": float((d > 0).mean())}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [None] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


def out_path(which):
    p = OUT / f"test_summary_{which}.json"
    k = 2
    while p.exists():
        p, k = OUT / f"test_summary_{which}_v{k}.json", k + 1
    return p


def bench():
    S, fam = {}, []
    for c in ("id2", "ood1", "clean"):
        for d in ("zh2en", "en2zh"):
            fb, fn = VB / f"obsi_b_{c}_{d}__t1c/metrics.json", VB / f"obsi_i2_{c}_{d}__t1c/metrics.json"
            if not (fb.exists() and fn.exists()):
                S[f"{c}_{d}"] = {"status": "missing", "B": fb.exists(), "new": fn.exists()}
                continue
            B, N = json.load(open(fb)), json.load(open(fn))
            r = {"status": "ok"}
            for k in ("tcpASR_BLEU", "blind_ASR_BLEU", "speech_LAAL_chunk", "EndOffset_chunk", "PESQ_median", "contrast_median",
                      "flatness_median", "SIM_mean"):
                b, n = B["summary"][k], N["summary"][k]
                r[k] = {"B": b, "new": n, "diff_mean2": float(np.mean(n) - np.mean(b))}
            psB, psN = B["per_stream"], N["per_stream"]
            sB, sN = list(psB), list(psN)

            def stream_mean(ps, runs, key):
                out = {}
                for run in runs:
                    for m, v in ps[run].items():
                        if v.get(key) is not None:
                            out.setdefault(m, []).append(v[key])
                return {m: float(np.mean(v)) for m, v in out.items()}
            for key in ("pesq", "contrast", "flatness", "sisdr", "sim"):
                a, bb = stream_mean(psB, sB, key), stream_mean(psN, sN, key)
                ms = sorted(set(a) & set(bb))
                r[f"paired_{key}"] = paired([a[m] for m in ms], [bb[m] for m in ms])
            tok = OUT / "tokens" / f"obsi_{c}_{d}.jsonl"
            if tok.exists():
                rows = [json.loads(l) for l in open(tok)]
                r["fallback_share"] = float(np.mean([x["fallback"] for x in rows]))
            r["gates"] = {"tcpASR_BLEU_not_lower": r["tcpASR_BLEU"]["diff_mean2"] >= 0,
                          "LAAL_not_larger": r["speech_LAAL_chunk"]["diff_mean2"] <= 0,
                          "identity_ge_-0.02": bool(r["paired_sim"] and r["paired_sim"]["median"] >= -0.02)}
            S[f"{c}_{d}"] = r
            if r["paired_pesq"]:
                fam.append((f"{c}_{d}", r["paired_pesq"]["p"]))
    if fam:
        for (k, _), h in zip(fam, holm([p for _, p in fam])):
            S[k]["paired_pesq"]["holm_p"] = h
    return S


def ood3():
    S = {}
    D = OUT / "ood3"
    for ss in ("ami", "en", "ali"):
        nb, nn = f"idea2_ood3_b_{ss}__t1c", f"idea2_ood3_sv_{ss}__t1c"
        if not (D / f"compare_{ss}.json").exists():
            S[ss] = {"status": "missing"}
            continue
        r = {"status": "ok"}
        bleu, lat = {}, {}
        for arm, n in (("B", nb), ("new", nn)):
            bleu[arm] = [json.load(open(E / "decode" / f"{n}{sfx}" / "score_asr.json"))["tcpBLEU_all"] for sfx in ("", "_tts2")]
            lat[arm] = [json.load(open(D / f"lat_chunk_{n}{sfx}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
                        for sfx in ("", "_tts2")]
        r["tcpASR_BLEU"] = {**bleu, "diff_mean2": float(np.mean(bleu["new"]) - np.mean(bleu["B"]))}
        r["speech_LAAL_chunk"] = {**lat, "diff_mean2": float(np.mean(lat["new"]) - np.mean(lat["B"]))}
        cmp = json.load(open(D / f"compare_{ss}.json"))
        r["speech"] = {s: {k: (cmp[f"{ss}_{s}"][k] or {}).get("median_diff") for k in ("pesq", "contrast", "flatness", "sisdr")}
                       | {"pesq_ci": cmp[f"{ss}_{s}"]["pesq"]["median_ci"], "n": cmp[f"{ss}_{s}"]["pesq"]["n"]}
                       for s in ("s1", "s2")}
        sp = OUT / "spksim"
        load = lambda n: {x["id"]: x["sim"] for x in map(json.loads, open(sp / f"{n}.jsonl"))} if (sp / f"{n}.jsonl").exists() else None
        sims = {(arm, sfx): load(f"{n}{sfx}") for arm, n in (("B", nb), ("new", nn)) for sfx in ("", "_tts2")}
        if all(v is not None for v in sims.values()):
            ids = sorted(set.intersection(*[set(v) for v in sims.values()]))
            a = [0.5 * (sims[("B", "")][i] + sims[("B", "_tts2")][i]) for i in ids]
            b = [0.5 * (sims[("new", "")][i] + sims[("new", "_tts2")][i]) for i in ids]
            r["paired_sim"] = paired(a, b)
        tok = OUT / "tokens" / f"ood3_{ss}.jsonl"
        if tok.exists():
            r["fallback_share"] = float(np.mean([json.loads(l)["fallback"] for l in open(tok)]))
        r["gates"] = {"tcpASR_BLEU_not_lower": r["tcpASR_BLEU"]["diff_mean2"] >= 0,
                      "LAAL_not_larger": r["speech_LAAL_chunk"]["diff_mean2"] <= 0,
                      "identity_ge_-0.02": bool(r.get("paired_sim") and r["paired_sim"]["median"] >= -0.02)}
        S[ss] = r
    return S


if __name__ == "__main__":
    which = sys.argv[1]
    S = bench() if which == "bench" else ood3()
    p = out_path(which)
    p.write_text(json.dumps(S, indent=1))
    print(p)
    print(json.dumps(S, indent=1)[:4000])
