"""Idea 6 full-system test summary (prereg "完整系统测试预登记"): arms C_post (i6c) and E2E_post (i6e) vs B, Idea 2 frozen
shrink+Vfb (i2 / sv), Idea 1 C (i1c), and E2E vs C_post, with the statistics of idea2_test_summary.py (two-seed means;
per-stream / per-item paired differences seed-averaged first; median, bootstrap 95 % CI, Wilcoxon). Holm: bench, per
comparison and metric family (PESQ, SI-SDR, contrast, flatness) over the 6 sets; OOD-3 speech metrics: the Holm of
idea1_speech_metrics compare; OOD-3 per-window chrF (idea6_diag_harm.chrf_run, two-seed mean, paired Wilcoxon): Holm over
all (arm x reference x subset) tests. Gates (M2): tcpASR-BLEU not lower, similarity median delta >= -0.02, LAAL reported
with "BLEU 优先于 LAAL" (LAAL_excess_s listed, never overrides the BLEU gate).
    <uniss-eval python> -m experiments.mist_w3po_v1.voicecond.idea6_test_summary
Writes eval_outputs/mist_w3po_v1/idea6_test/test_summary.json (a new _v<k> file if one exists).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import wilcoxon

from experiments.mist_w3po_v1.voicecond.idea2_test_summary import E, VB, holm, paired
from experiments.mist_w3po_v1.voicecond.idea6_diag_harm import chrf_run

T = E / "idea6_test"
I2O, I1O, OUT = E / "voicecond_idea2/ood3", E / "voicecond_idea1/m3/ood3", T / "ood3"
SP = E / "voicecond_idea2/spksim"
SETS = [f"{c}_{d}" for c in ("id2", "ood1", "clean") for d in ("zh2en", "en2zh")]
SUBSETS = ("ami", "en", "ali")
BENCH = {"B": "obsi_b_{}__t1c", "I2": "obsi_i2_{}__t1c", "I1C": "obsi_i1c_{}__t1c", "i6c": "obsi_i6c_{}__t1c", "i6e": "obsi_i6e_{}__t1c"}
OOD3 = {"B": ("idea2_ood3_b_{}__t1c", I2O, "b"), "I2": ("idea2_ood3_sv_{}__t1c", I2O, "sv"),
        "I1C": ("idea1_ood3_c_{}__t1c", I1O, "i1c"), "i6c": ("idea2_ood3_i6c_{}__t1c", OUT, "i6c"),
        "i6e": ("idea2_ood3_i6e_{}__t1c", OUT, "i6e")}
COMPARISONS = [("i6c", "B"), ("i6c", "I2"), ("i6c", "I1C"), ("i6e", "B"), ("i6e", "I2"), ("i6e", "I1C"), ("i6e", "i6c")]
FAM = ("pesq", "sisdr", "contrast", "flatness")


def gates(r):
    lat = r["speech_LAAL_chunk"]["diff_mean2"]
    return {"tcpASR_BLEU_not_lower": bool(r["tcpASR_BLEU"]["diff_mean2"] >= 0),
            "identity_ge_-0.02": bool(r.get("paired_sim") and r["paired_sim"]["median"] >= -0.02),
            "LAAL_not_larger": bool(lat <= 0), "LAAL_excess_s": max(0.0, lat)}


def stream_mean(ps, key):
    out = {}
    for run in ps:
        for m, v in ps[run].items():
            if v.get(key) is not None:
                out.setdefault(m, []).append(v[key])
    return {m: float(np.mean(v)) for m, v in out.items()}


def bench():
    S = {}
    for new, ref in COMPARISONS:
        C = {}
        for s in SETS:
            fb, fn = VB / BENCH[ref].format(s) / "metrics.json", VB / BENCH[new].format(s) / "metrics.json"
            if not (fb.exists() and fn.exists()):
                C[s] = {"status": "missing", "ref": fb.exists(), "new": fn.exists()}
                continue
            B, N = json.load(open(fb)), json.load(open(fn))
            r = {"status": "ok"}
            for k in ("tcpASR_BLEU", "blind_ASR_BLEU", "speech_LAAL_chunk", "EndOffset_chunk", "PESQ_median", "contrast_median",
                      "flatness_median", "SIM_mean"):
                b, n = B["summary"][k], N["summary"][k]
                r[k] = {"ref": b, "new": n, "diff_mean2": float(np.mean(n) - np.mean(b))}
            for key in FAM + ("sim",):
                a, bb = stream_mean(B["per_stream"], key), stream_mean(N["per_stream"], key)
                ms = sorted(set(a) & set(bb))
                r[f"paired_{key}"] = paired([a[m] for m in ms], [bb[m] for m in ms])
            r["gates"] = gates(r)
            C[s] = r
        for key in FAM:
            fam = [(s, C[s][f"paired_{key}"]["p"]) for s in SETS if C[s].get("status") == "ok" and C[s][f"paired_{key}"]]
            for (s, _), h in zip(fam, holm([p for _, p in fam])):
                C[s][f"paired_{key}"]["holm_p"] = h
        S[f"{new}_vs_{ref}"] = C
    return S


def ood3():
    S, tests = {}, []
    jobs = [(OOD3[a][0].format(ss) + sd, ss) for a in OOD3 for ss in SUBSETS for sd in ("", "_tts2")]
    with ProcessPoolExecutor(min(16, len(jobs))) as ex:
        chrf = dict(ex.map(chrf_run, jobs))

    def cw(arm, ss):
        s1, s2 = chrf[OOD3[arm][0].format(ss)], chrf[OOD3[arm][0].format(ss) + "_tts2"]
        return {k: (s1[k] + s2[k]) / 2 for k in s1 if k in s2}

    def sims(n):
        p = SP / f"{n}.jsonl"
        return {x["id"]: x["sim"] for x in map(json.loads, open(p))} if p.exists() else None
    for new, ref in COMPARISONS:
        C = {}
        for ss in SUBSETS:
            nn, dn, _ = OOD3[new]
            nr, dr, tr = OOD3[ref]
            nn, nr = nn.format(ss), nr.format(ss)
            cf = OUT / f"compare_{new}_{tr}_{ss}.json"
            # 2026-11-19 17:50: the cross-arm compare (i6e vs i6c) is written by the runner only after all units; a
            # missing compare file no longer drops the cell, only its speech block (BLEU / LAAL / sim / chrF still filled)
            r = {"status": "ok" if cf.exists() else "ok_no_speech_compare"}
            bleu = {arm: [json.load(open(E / "decode" / f"{n}{sfx}" / "score_asr.json"))["tcpBLEU_all"] for sfx in ("", "_tts2")]
                    for arm, n in (("ref", nr), ("new", nn))}
            lat = {arm: [json.load(open(d / f"lat_chunk_{n}{sfx}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
                         for sfx in ("", "_tts2")] for arm, n, d in (("ref", nr, dr), ("new", nn, dn))}
            r["tcpASR_BLEU"] = {**bleu, "diff_mean2": float(np.mean(bleu["new"]) - np.mean(bleu["ref"]))}
            r["speech_LAAL_chunk"] = {**lat, "diff_mean2": float(np.mean(lat["new"]) - np.mean(lat["ref"]))}
            if cf.exists():
                cmp = json.load(open(cf))
                r["speech"] = {s: {k: {x: (cmp[f"{ss}_{s}"][k] or {}).get(x) for x in ("A_median", "B_median", "median_diff",
                                                                                       "median_ci", "holm_p", "n")}
                                   for k in FAM} for s in ("s1", "s2")}
            else:
                r["speech"] = None
            sm = {(arm, sfx): sims(f"{n}{sfx}") for arm, n in (("ref", nr), ("new", nn)) for sfx in ("", "_tts2")}
            if all(v is not None for v in sm.values()):
                ids = sorted(set.intersection(*[set(v) for v in sm.values()]))
                a = [0.5 * (sm[("ref", "")][i] + sm[("ref", "_tts2")][i]) for i in ids]
                b = [0.5 * (sm[("new", "")][i] + sm[("new", "_tts2")][i]) for i in ids]
                r["paired_sim"] = paired(a, b)
            ca, cb = cw(ref, ss), cw(new, ss)
            ks = sorted(set(ca) & set(cb))
            d = np.array([cb[k] - ca[k] for k in ks])
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0
            r["chrf_paired"] = {"n": len(d), "ref_mean": float(np.mean([ca[k] for k in ks])), "new_mean": float(np.mean([cb[k] for k in ks])),
                                "mean": float(d.mean()), "median": float(np.median(d)), "share_neg": float((d < 0).mean()),
                                "share_pos": float((d > 0).mean()), "p": p}
            tests.append((f"{new}_vs_{ref}", ss, p))
            r["gates"] = gates(r)
            C[ss] = r
        S[f"{new}_vs_{ref}"] = C
    for (k, ss, _), h in zip(tests, holm([t[2] for t in tests])):
        c = S[k][ss]["chrf_paired"]
        c["holm_p"] = h
        c["harm"] = bool(h < 0.05 and c["mean"] < 0)
    return S


if __name__ == "__main__":
    S = {"bench": bench(), "ood3": ood3(),
         "_meta": {"arms": {"i6c": "C_post", "i6e": "E2E_post (idea6_e2e_main_v1 step 3000)"},
                   "refs": {"B": "obsi_b / idea2_ood3_b", "I2": "obsi_i2 / idea2_ood3_sv", "I1C": "obsi_i1c / idea1_ood3_c"},
                   "rule": "BLEU 优先于 LAAL: LAAL excess reported (LAAL_excess_s), never overrides the BLEU gate"}}
    p, k = T / "test_summary.json", 2
    while p.exists():
        p, k = T / f"test_summary_v{k}.json", k + 1
    p.write_text(json.dumps(S, indent=1))
    print(p)
