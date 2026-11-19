"""Idea 1 M3 analysis, exactly per reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v4.3-v4.5.

bench  OpenBroadcast-SI v1 (bench_eval metrics.json of obsi_b_* = B, obsi_i1z_* = Z, obsi_i1c_* = C):
  per method x condition (id2, ood1, clean) x direction, both TTS seeds:
    hard gates (strict): tcpASR-BLEU two-seed mean >= B's; speech LAAL (chunk protocol) two-seed mean <= B's (the 0.05 s
    tolerance view reported too); identity: median over streams of (SIM two-seed mean, method - B) >= -0.02;
    primary quality: per-stream PESQ (two-seed mean of the stream means) paired with B: median diff, bootstrap 95 % CI
    (10000, seed 0), Wilcoxon p, Holm over the 6 condition x direction cells per method; blind ASR-BLEU, contrast,
    flatness, SI-SDR reported. clean no-harm: hard gates + the PESQ median-diff CI upper bound >= 0.
  H1 per direction: DiD_s = (Z - C)_ood1,s - (Z - C)_id2,s over twin streams s (bench_eval's stream key), PESQ two-seed
    means; statistic = mean DiD; one-sided bootstrap p = share of bootstrap means <= 0 (10000, seed 0); Holm over the
    2 directions; supported iff Holm-adjusted p < 0.05 (equivalently the one-sided 95 % lower bound > 0 for one test).
    Secondary DiDs (reported): contrast, SIM, and the code error of m3/codes_<cond>_<dir>.jsonl (err_z - err_c per item,
    stream mean over matched items).
  -> eval_outputs/mist_w3po_v1/voicecond_idea1/m3/test_summary.json (refuses an existing file)
    python -m experiments.mist_w3po_v1.voicecond.idea1_m3_analysis bench
ood3   AMI / AliMeeting subsets (v4.5): B = idea2_ood3_b_<ss>__t1c (Idea 2's arm, read only), Z / C =
  idea1_ood3_{z,c}_<ss>__t1c; tcpASR-BLEU from decode/<n>/score_asr.json; chunk LAAL from lat_chunk files
  (B: voicecond_idea2/ood3, Z / C: voicecond_idea1/m3/ood3); PESQ per stream from idea1_speech_metrics jsonl
  (voicecond_idea1/m3/ood3/<n>.jsonl); identity: TitaNet per stream (voicecond_idea1/m3/ood3/spksim_<ss>.json).
  Same strict gates; streams = talker items.  -> m3/ood3_summary.json
    python -m experiments.mist_w3po_v1.voicecond.idea1_m3_analysis ood3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EO = REPO / "eval_outputs/mist_w3po_v1"
VB = EO / "voicecond_bench"
M3 = EO / "voicecond_idea1/m3"
CONDS, DIRS = ("id2", "ood1", "clean"), ("zh2en", "en2zh")
METH = {"Z": "obsi_i1z", "C": "obsi_i1c"}


def boot_ci(d, fn=np.median, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = fn(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], v


def wilcoxon_p(d):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return 1.0


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for k, (key, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[key] = run
    return out


def load_run(name):
    p = VB / name / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def seeds(m):
    return [m["seeds"][k] for k in sorted(m["seeds"])]          # <name>, <name>_tts2


def stream_means(m, key):
    """{stream key: two-seed mean of the per-stream values}"""
    per = [m["per_stream"][k] for k in sorted(m["per_stream"])]
    out = {}
    for meet in per[0]:
        vals = [p[meet][key] for p in per if meet in p and p[meet].get(key) is not None]
        if len(vals) == len(per):
            out[per[0][meet]["stream"]] = float(np.mean(vals))
    return out


def paired(A: dict, B: dict):
    ks = sorted(set(A) & set(B))
    return np.array([B[k] - A[k] for k in ks]), ks


def cell(b, x):
    sb, sx = seeds(b), seeds(x)
    mean = lambda ss, f: float(np.mean([f(s) for s in ss]))  # noqa: E731
    tcp_b, tcp_x = mean(sb, lambda s: s["tcp_asr"]["tcpBLEU_all"]), mean(sx, lambda s: s["tcp_asr"]["tcpBLEU_all"])
    bl_b, bl_x = mean(sb, lambda s: s["blind_asr"]["corpus"]), mean(sx, lambda s: s["blind_asr"]["corpus"])
    la_b, la_x = mean(sb, lambda s: s["latency_chunk"]["speech_laal"]), mean(sx, lambda s: s["latency_chunk"]["speech_laal"])
    out = {"tcpASR_BLEU": {"B": tcp_b, "method": tcp_x, "diff": tcp_x - tcp_b},
           "blind_ASR_BLEU": {"B": bl_b, "method": bl_x, "diff": bl_x - bl_b},
           "speech_LAAL": {"B": la_b, "method": la_x, "diff": la_x - la_b}}
    for key in ("pesq", "contrast", "flatness", "sisdr", "sim"):
        d, ks = paired(stream_means(b, key), stream_means(x, key))
        if len(d) < 3:
            out[key] = None
            continue
        ci, _ = boot_ci(d)
        out[key] = {"n_streams": len(d), "median_diff": float(np.median(d)), "ci": ci, "mean_diff": float(d.mean()),
                    "wilcoxon_p": wilcoxon_p(d)}
    sim_md = out["sim"]["median_diff"] if out["sim"] else None
    out["gates"] = {"tcpASR_BLEU_ok": tcp_x >= tcp_b, "LAAL_ok_strict": la_x <= la_b, "LAAL_ok_tol0.05": la_x <= la_b + 0.05,
                    "identity_ok": sim_md is not None and sim_md >= -0.02}
    out["gates"]["hard_pass_strict"] = out["gates"]["tcpASR_BLEU_ok"] and out["gates"]["LAAL_ok_strict"] and out["gates"]["identity_ok"]
    return out


def did(z, c, key_fn):
    """per direction: {stream: (Z - C)} for one condition"""
    Z, C = key_fn(z), key_fn(c)
    return {k: Z[k] - C[k] for k in set(Z) & set(C)}


def code_stream_diffs(cond, d):
    p = M3 / f"codes_{cond}_{d}.jsonl"
    if not p.exists():
        return {}
    acc = {}
    for r in map(json.loads, open(p, encoding="utf-8")):
        if "err_z" in r:
            s = "s" + r["id"][3:6]
            acc.setdefault(s, []).append(r["err_z"] - r["err_c"])
    return {s: float(np.mean(v)) for s, v in acc.items()}


def bench(a) -> None:
    out_p = M3 / "test_summary.json"
    if out_p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out_p}")
    res = {"cells": {}, "missing": [], "holm_pesq": {}, "H1": {}, "clean_no_harm": {}}
    for meth, pre in METH.items():
        ps = {}
        for cond in CONDS:
            for d in DIRS:
                b, x = load_run(f"obsi_b_{cond}_{d}__t1c"), load_run(f"{pre}_{cond}_{d}__t1c")
                key = f"{meth}|{cond}|{d}"
                if b is None or x is None:
                    res["missing"].append(key)
                    continue
                c = cell(b, x)
                res["cells"][key] = c
                if c["pesq"]:
                    ps[key] = c["pesq"]["wilcoxon_p"]
        hp = holm(ps)
        res["holm_pesq"].update(hp)
        for key, p in hp.items():
            res["cells"][key]["pesq"]["holm_p"] = p
        for d in DIRS:
            key = f"{meth}|clean|{d}"
            if key in res["cells"]:
                c = res["cells"][key]
                res["clean_no_harm"][key] = {**c["gates"], "pesq_median_diff": c["pesq"]["median_diff"],
                                             "pesq_ci": c["pesq"]["ci"], "pesq_not_worse": c["pesq"]["ci"][1] >= 0,
                                             "no_harm_pass": c["gates"]["hard_pass_strict"] and c["pesq"]["ci"][1] >= 0}
    # H1
    p1 = {}
    for d in DIRS:
        runs = {(m, cond): load_run(f"{METH[m]}_{cond}_{d}__t1c") for m in METH for cond in ("id2", "ood1")}
        if any(v is None for v in runs.values()):
            res["H1"][d] = "missing runs"
            continue
        rec = {}
        for key in ("pesq", "contrast", "sim"):
            f = lambda m, k=key: stream_means(m, k)  # noqa: E731
            o = did(runs[("Z", "ood1")], runs[("C", "ood1")], f)
            i = did(runs[("Z", "id2")], runs[("C", "id2")], f)
            ks = sorted(set(o) & set(i))
            dd = np.array([o[k] - i[k] for k in ks])
            ci, bs = boot_ci(dd, np.mean)
            rec[key] = {"n_twin_streams": len(ks), "mean_DiD": float(dd.mean()), "median_DiD": float(np.median(dd)),
                        "ci95": ci, "one_sided_lower95": float(np.percentile(bs, 5)), "p_one_sided": float((bs <= 0).mean()),
                        "ZminusC_ood1_mean": float(np.mean([o[k] for k in ks])),
                        "ZminusC_id2_mean": float(np.mean([i[k] for k in ks]))}
        o, i = code_stream_diffs("ood1", d), code_stream_diffs("id2", d)
        ks = sorted(set(o) & set(i))
        if ks:
            dd = np.array([o[k] - i[k] for k in ks])
            ci, bs = boot_ci(dd, np.mean)
            rec["code_error"] = {"n_twin_streams": len(ks), "mean_DiD": float(dd.mean()), "ci95": ci,
                                 "note": "err_z - err_c; NEGATIVE DiD = Z gains more on OOD-1",
                                 "p_one_sided_negative": float((bs >= 0).mean())}
        res["H1"][d] = rec
        p1[d] = rec["pesq"]["p_one_sided"]
    if p1:
        hp = holm(p1)
        for d, p in hp.items():
            res["H1"][d]["pesq"]["holm_p"] = p
            res["H1"][d]["supported"] = p < 0.05
    out_p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for key, c in res["cells"].items():
        print(f"{key}: tcp {c['tcpASR_BLEU']['diff']:+.2f} LAAL {c['speech_LAAL']['diff']:+.3f} "
              f"PESQ {c['pesq']['median_diff']:+.3f} {np.round(c['pesq']['ci'], 3).tolist()} "
              f"SIM {c['sim']['median_diff'] if c['sim'] else None} gates {c['gates']}")
    for d, r in res["H1"].items():
        if isinstance(r, dict):
            print(f"H1 {d}: PESQ DiD {r['pesq']['mean_DiD']:+.3f} lower95 {r['pesq']['one_sided_lower95']:+.3f} "
                  f"holm p {r['pesq'].get('holm_p')} supported {r.get('supported')}")
    print("missing:", res["missing"])


def ood3(a) -> None:
    OD = M3 / "ood3"
    out_p = M3 / "ood3_summary.json"
    if out_p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out_p}")
    I2 = EO / "voicecond_idea2/ood3"
    res = {"cells": {}, "missing": []}
    for ss in ("ami", "en", "ali"):
        sims = {}
        sp = OD / f"spksim_{ss}.json"
        if sp.exists():
            sims = json.load(open(sp))["per_stream"]
        for meth in ("z", "c"):
            key = f"{meth.upper()}|{ss}"
            names = {arm: [f"{pre}_{ss}__t1c{s}" for s in ("", "_tts2")]
                     for arm, pre in (("B", "idea2_ood3_b"), ("X", f"idea1_ood3_{meth}"))}
            try:
                tcp = {arm: float(np.mean([json.load(open(EO / "decode" / n / "score_asr.json"))["tcpBLEU_all"] for n in ns]))
                       for arm, ns in names.items()}
                lat = {"B": float(np.mean([json.load(open(I2 / f"lat_chunk_{n}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
                                           for n in names["B"]])),
                       "X": float(np.mean([json.load(open(OD / f"lat_chunk_{n}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
                                           for n in names["X"]]))}
                met = {arm: [{r["id"]: r for r in map(json.loads, open((I2 if arm == "B" else OD) / f"{n}.jsonl")) if "pesq" in r}
                             for n in ns] for arm, ns in names.items()}
            except FileNotFoundError as e:
                res["missing"].append(f"{key}: {e.filename}")
                continue
            c = {"tcpASR_BLEU": {"B": tcp["B"], "method": tcp["X"], "diff": tcp["X"] - tcp["B"]},
                 "speech_LAAL": {"B": lat["B"], "method": lat["X"], "diff": lat["X"] - lat["B"]}}
            for k in ("pesq", "contrast", "flatness"):
                mB = {i: np.mean([m[i][k] for m in met["B"]]) for i in met["B"][0] if all(i in m and k in m[i] for m in met["B"])}
                mX = {i: np.mean([m[i][k] for m in met["X"]]) for i in met["X"][0] if all(i in m and k in m[i] for m in met["X"])}
                d, _ = paired(mB, mX)
                ci, _ = boot_ci(d)
                c[k] = {"n": len(d), "median_diff": float(np.median(d)), "ci": ci, "wilcoxon_p": wilcoxon_p(d)}
            if sims:
                sB = {i: np.mean([sims[n][i] for n in names["B"]]) for i in sims.get(names["B"][0], {})
                      if all(i in sims.get(n, {}) for n in names["B"])}
                sX = {i: np.mean([sims[n][i] for n in names["X"]]) for i in sims.get(names["X"][0], {})
                      if all(i in sims.get(n, {}) for n in names["X"])}
                d, _ = paired(sB, sX)
                c["sim"] = {"n": len(d), "median_diff": float(np.median(d)) if len(d) else None}
            sim_md = (c.get("sim") or {}).get("median_diff")
            c["gates"] = {"tcpASR_BLEU_ok": tcp["X"] >= tcp["B"], "LAAL_ok_strict": lat["X"] <= lat["B"],
                          "LAAL_ok_tol0.05": lat["X"] <= lat["B"] + 0.05,
                          "identity_ok": sim_md is not None and sim_md >= -0.02,
                          "pesq_not_worse": c["pesq"]["ci"][1] >= 0}
            c["gates"]["no_harm_pass"] = (c["gates"]["tcpASR_BLEU_ok"] and c["gates"]["LAAL_ok_strict"]
                                          and c["gates"]["identity_ok"] and c["gates"]["pesq_not_worse"])
            res["cells"][key] = c
            print(f"{key}: tcp {c['tcpASR_BLEU']['diff']:+.2f} LAAL {c['speech_LAAL']['diff']:+.3f} PESQ "
                  f"{c['pesq']['median_diff']:+.3f} SIM {sim_md} gates {c['gates']}")
    out_p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("missing:", res["missing"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bench", "ood3"])
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    bench(a) if a.cmd == "bench" else ood3(a)


if __name__ == "__main__":
    main()
