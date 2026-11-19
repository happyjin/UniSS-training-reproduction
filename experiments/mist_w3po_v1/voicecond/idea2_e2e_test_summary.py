"""Idea 2 e2e test summaries (prereg addendum 2026-11-19): e2e (main_v2, step 3000) vs baseline B AND vs the frozen
shrink+Vfb (M2), with exactly the statistics of idea2_test_summary.py (two-seed means; per-stream / per-item paired
differences seed-averaged first; median, bootstrap 95 % CI, Wilcoxon; Holm over the PESQ family per comparison).
Gates as M2: tcpASR-BLEU not lower, similarity median delta >= -0.02; LAAL per the user rule "BLEU 优先于 LAAL":
LAAL_not_larger is reported, a LAAL excess never hides and is listed separately (LAAL_excess_s), it does not override
the BLEU gate.
    python -m experiments.mist_w3po_v1.voicecond.idea2_e2e_test_summary bench|ood3     (conda_envs/uniss-eval python)
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/test_summary_e2e_<which>.json (a new _v<k> file if one exists).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from experiments.mist_w3po_v1.voicecond.idea2_test_summary import E, OUT, VB, holm, paired

CMP = {"vs_B": ("obsi_b_{c}_{d}__t1c", "idea2_ood3_b_{ss}__t1c"),
       "vs_shrinkvfb": ("obsi_i2_{c}_{d}__t1c", "idea2_ood3_sv_{ss}__t1c")}
NEW_BENCH, NEW_OOD3 = "obsi_e2e_{c}_{d}__t1c", "idea2_ood3_e2e_{ss}__t1c"


def out_path(which):
    p = OUT / f"test_summary_e2e_{which}.json"
    k = 2
    while p.exists():
        p, k = OUT / f"test_summary_e2e_{which}_v{k}.json", k + 1
    return p


def gates(r):
    lat = r["speech_LAAL_chunk"]["diff_mean2"]
    return {"tcpASR_BLEU_not_lower": r["tcpASR_BLEU"]["diff_mean2"] >= 0,
            "identity_ge_-0.02": bool(r.get("paired_sim") and r["paired_sim"]["median"] >= -0.02),
            "LAAL_not_larger": lat <= 0, "LAAL_excess_s": max(0.0, lat)}


def bench_one(ref_tpl):
    S, fam = {}, []
    for c in ("id2", "ood1", "clean"):
        for d in ("zh2en", "en2zh"):
            fb = VB / ref_tpl.format(c=c, d=d) / "metrics.json"
            fn = VB / NEW_BENCH.format(c=c, d=d) / "metrics.json"
            if not (fb.exists() and fn.exists()):
                S[f"{c}_{d}"] = {"status": "missing", "ref": fb.exists(), "new": fn.exists()}
                continue
            B, N = json.load(open(fb)), json.load(open(fn))
            r = {"status": "ok"}
            for k in ("tcpASR_BLEU", "blind_ASR_BLEU", "speech_LAAL_chunk", "EndOffset_chunk", "PESQ_median", "contrast_median",
                      "flatness_median", "SIM_mean"):
                b, n = B["summary"][k], N["summary"][k]
                r[k] = {"ref": b, "new": n, "diff_mean2": float(np.mean(n) - np.mean(b))}

            def stream_mean(ps, key):
                out = {}
                for run in ps:
                    for m, v in ps[run].items():
                        if v.get(key) is not None:
                            out.setdefault(m, []).append(v[key])
                return {m: float(np.mean(v)) for m, v in out.items()}
            for key in ("pesq", "contrast", "flatness", "sisdr", "sim"):
                a, bb = stream_mean(B["per_stream"], key), stream_mean(N["per_stream"], key)
                ms = sorted(set(a) & set(bb))
                r[f"paired_{key}"] = paired([a[m] for m in ms], [bb[m] for m in ms])
            tok = OUT / "tokens" / f"obsi_e2e_{c}_{d}.jsonl"
            if tok.exists():
                r["fallback_share"] = float(np.mean([json.loads(l)["fallback"] for l in open(tok)]))
            r["gates"] = gates(r)
            S[f"{c}_{d}"] = r
            if r["paired_pesq"]:
                fam.append((f"{c}_{d}", r["paired_pesq"]["p"]))
    if fam:
        for (k, _), h in zip(fam, holm([p for _, p in fam])):
            S[k]["paired_pesq"]["holm_p"] = h
    return S


def ood3_one(ref_tpl, cmp_tag):
    S = {}
    D = OUT / "ood3"
    for ss in ("ami", "en", "ali"):
        nb, nn = ref_tpl.format(ss=ss), NEW_OOD3.format(ss=ss)
        cf = D / f"compare_e2e_{cmp_tag}_{ss}.json"
        if not cf.exists():
            S[ss] = {"status": "missing", "compare": str(cf)}
            continue
        r = {"status": "ok"}
        bleu, lat = {}, {}
        for arm, n in (("ref", nb), ("new", nn)):
            bleu[arm] = [json.load(open(E / "decode" / f"{n}{sfx}" / "score_asr.json"))["tcpBLEU_all"] for sfx in ("", "_tts2")]
            lat[arm] = [json.load(open(D / f"lat_chunk_{n}{sfx}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
                        for sfx in ("", "_tts2")]
        r["tcpASR_BLEU"] = {**bleu, "diff_mean2": float(np.mean(bleu["new"]) - np.mean(bleu["ref"]))}
        r["speech_LAAL_chunk"] = {**lat, "diff_mean2": float(np.mean(lat["new"]) - np.mean(lat["ref"]))}
        cmp = json.load(open(cf))
        r["speech"] = {s: {k: (cmp[f"{ss}_{s}"][k] or {}).get("median_diff") for k in ("pesq", "contrast", "flatness", "sisdr")}
                       | {"pesq_ci": cmp[f"{ss}_{s}"]["pesq"]["median_ci"], "n": cmp[f"{ss}_{s}"]["pesq"]["n"]}
                       for s in ("s1", "s2")}
        sp = OUT / "spksim"

        def load(n):
            p = sp / f"{n}.jsonl"
            return {x["id"]: x["sim"] for x in map(json.loads, open(p))} if p.exists() else None
        sims = {(arm, sfx): load(f"{n}{sfx}") for arm, n in (("ref", nb), ("new", nn)) for sfx in ("", "_tts2")}
        if all(v is not None for v in sims.values()):
            ids = sorted(set.intersection(*[set(v) for v in sims.values()]))
            a = [0.5 * (sims[("ref", "")][i] + sims[("ref", "_tts2")][i]) for i in ids]
            b = [0.5 * (sims[("new", "")][i] + sims[("new", "_tts2")][i]) for i in ids]
            r["paired_sim"] = paired(a, b)
        tok = OUT / "tokens" / f"ood3_e2e_{ss}.jsonl"
        if tok.exists():
            r["fallback_share"] = float(np.mean([json.loads(l)["fallback"] for l in open(tok)]))
        r["gates"] = gates(r)
        S[ss] = r
    return S


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "bench":
        S = {k: bench_one(v[0]) for k, v in CMP.items()}
    else:
        S = {k: ood3_one(v[1], k.split("_", 1)[1].lower()) for k, v in CMP.items()}
    S["_meta"] = {"new": NEW_BENCH if which == "bench" else NEW_OOD3, "refs": CMP,
                  "rule": "BLEU 优先于 LAAL: LAAL excess reported (LAAL_excess_s), never overrides the BLEU gate"}
    p = out_path(which)
    p.write_text(json.dumps(S, indent=1))
    print(p)
    print(json.dumps(S, indent=1)[:6000])
