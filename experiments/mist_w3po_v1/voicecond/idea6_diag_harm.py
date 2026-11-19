"""Diagnosis (no training, no GPU): why does the trained denoising encoder C harm OOD-3 while Idea 2 (shrink+Vfb) does not?
Analyses fixed before reading any output of this script (LOG 2026-11-19 "C 伤害诊断"):

  D1 seed noise   : per subset and arm, tcpASR-BLEU and chunk speech LAAL of both TTS seeds; arm-minus-B difference vs the
                    seed-to-seed spread of the same arm; per-window chrF (window ASR text vs the reference utterances that
                    overlap the window in meeting time), two-seed mean, paired Wilcoxon arm vs B (Holm over arms x subsets).
  D2 token shift  : 32 FSQ ids -> 32x6 base-4 digits (192-d). Clean-voice Gaussian (Ledoit-Wolf) on g* of the open clean
                    reference speakers (idea2 refs, latest ustar table). Per item: Mahalanobis distance d_M of each arm's
                    tokens, Hamming distance to B's tokens.
  D3 behaviour    : per item two-seed mean Talker output duration (measure files, dur_s; the text is B's for every arm, so a
                    duration change is Talker behaviour). log(dur_arm / dur_B).
  D4 link         : Spearman of (d_M(arm) - d_M(B)) and Hamming-to-B with |log dur ratio| and with per-window chrF change,
                    pooled over arms C, Z, SV, E2E per subset.

    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/idea6_diag_harm.py --out eval_outputs/mist_w3po_v1/idea6_diag
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
I1 = E / "voicecond_idea1/m3/ood3"
I2 = E / "voicecond_idea2"
OD = REPO / "data/open_voicecond_v1/idea2/ood3"
SUBSETS = ("ami", "en", "ali")
ARMS = {"B": "idea2_ood3_b_{}__t1c", "C": "idea1_ood3_c_{}__t1c", "Z": "idea1_ood3_z_{}__t1c",
        "SV": "idea2_ood3_sv_{}__t1c", "E2E": "idea2_ood3_e2e_{}__t1c"}
SEEDS = ("", "_tts2")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def find(name, kind):
    for d in (I1, I2 / "ood3"):
        p = d / (f"lat_chunk_{name}.json" if kind == "lat" else f"{name}.jsonl")
        if p.exists():
            return p
    raise FileNotFoundError(f"{kind} {name}")


def tokens(ss):
    c = {r["key"]: r for r in jl(I1 / f"tokens_c_{ss}.jsonl")}
    out = {"B": {k: r["raw_ids"] for k, r in c.items()}, "C": {k: r["ids"] for k, r in c.items()},
           "Z": {r["key"]: r["ids"] for r in jl(I1 / f"tokens_z_{ss}.jsonl")},
           "SV": {r["key"]: r["ids"] for r in jl(I2 / "tokens" / f"ood3_{ss}.jsonl")},
           "E2E": {r["key"]: r["ids"] for r in jl(I2 / "tokens" / f"ood3_e2e_{ss}.jsonl")}}
    return out


def digits(ids):
    ids = np.asarray(ids, np.int64)
    return np.stack([(ids // 4 ** i) % 4 for i in range(6)], -1).reshape(*ids.shape[:-1], -1).astype(np.float64)


def clean_gauss():
    from sklearn.covariance import LedoitWolf
    R = REPO / "data/open_voicecond_v1/idea2/refs"
    k, p = 2, R / "ustar.npz"
    while (R / f"ustar_v{k}.npz").exists():
        p, k = R / f"ustar_v{k}.npz", k + 1
    g = np.load(p, allow_pickle=True)["gstar"].reshape(-1, 32)
    X = digits(g)
    lw = LedoitWolf().fit(X)
    return lw.location_, np.linalg.inv(lw.covariance_), str(p.relative_to(REPO)), len(X), X


def mahal(X, mu, P):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, P, d))


# ------------------------------------------------------------------------------------------------ chrF per window
def window_refs(ss):
    """online window id -> reference text of the reference utterances overlapping the window (meeting time)"""
    refs = defaultdict(list)
    for r in jl(OD / f"{ss}_ref.jsonl"):
        for u in r["utts"]:
            refs[r["meeting"]].append((r["start_s"] + u["begin_s"], r["start_s"] + u["end_s"], u["ref"]))
    out = {}
    for w in jl(OD / f"{ss}_win.jsonl"):
        a, b = w["start_s"], w["start_s"] + w["dur_ms"] / 1000
        txt = [t for s, e, t in sorted(refs[w["meeting"]]) if min(e, b) - max(s, a) > 0.5 * (e - s)]
        out[w["id"]] = (" ".join(txt), w.get("dir"))
    return out


def chrf_run(args):
    name, ss = args
    import sacrebleu
    ref = window_refs(ss)
    out = {}
    for r in jl(E / "decode" / name / "hyp_asr.jsonl"):
        if r["id"] not in ref or not ref[r["id"]][0]:
            continue
        hyp = " ".join(s["text"] for s in sorted(r["segments"], key=lambda s: s["start_s"]))
        out[r["id"]] = sacrebleu.sentence_chrf(hyp, [ref[r["id"]][0]]).score
    return name, out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od}/summary.json")
    od.mkdir(parents=True, exist_ok=True)
    jobs = [(ARMS[arm].format(ss) + sd, ss) for ss in SUBSETS for arm in ARMS for sd in SEEDS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        chrf = dict(ex.map(chrf_run, jobs))
    mu, P, gsrc, ng, Xclean = clean_gauss()
    S = {"clean_ref": gsrc, "n_clean": ng, "clean_dM_median": float(np.median(mahal(Xclean, mu, P))), "subsets": {}}
    pvals = []
    for ss in SUBSETS:
        R = {"D1": {}, "D2": {}, "D3": {}, "D4": {}}
        # D1 corpus metrics per seed
        bleu = {arm: [json.load(open(E / "decode" / (ARMS[arm].format(ss) + sd) / "score_asr.json"))["tcpBLEU_all"]
                      for sd in SEEDS] for arm in ARMS}
        laal = {arm: [json.load(open(find(ARMS[arm].format(ss) + sd, "lat")))["unit"]["speech_laal_online"]["all"]["mean"]
                      for sd in SEEDS] for arm in ARMS}
        cw = {arm: {} for arm in ARMS}
        for arm in ARMS:
            s1, s2 = (chrf[ARMS[arm].format(ss) + sd] for sd in SEEDS)
            cw[arm] = {k: (s1[k] + s2[k]) / 2 for k in s1 if k in s2}
        for arm in ARMS:
            r = {"bleu_seeds": bleu[arm], "bleu_mean": float(np.mean(bleu[arm])),
                 "bleu_seed_spread": float(abs(bleu[arm][0] - bleu[arm][1])),
                 "laal_seeds": laal[arm], "laal_mean": float(np.mean(laal[arm])),
                 "laal_seed_spread": float(abs(laal[arm][0] - laal[arm][1])),
                 "chrf_mean": float(np.mean(list(cw[arm].values())))}
            if arm != "B":
                r["bleu_diff_vs_B"] = r["bleu_mean"] - float(np.mean(bleu["B"]))
                r["laal_diff_vs_B"] = r["laal_mean"] - float(np.mean(laal["B"]))
                ks = sorted(set(cw[arm]) & set(cw["B"]))
                d = np.array([cw[arm][k] - cw["B"][k] for k in ks])
                nz = d[d != 0]
                p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0
                r["chrf_paired"] = {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
                                    "share_neg": float((d < 0).mean()), "share_pos": float((d > 0).mean()), "p": p}
                pvals.append((ss, arm, p))
            R["D1"][arm] = r
        # D2 tokens
        tk = tokens(ss)
        keys = sorted(set.intersection(*(set(v) for v in tk.values())))
        dig = {arm: digits(np.array([tk[arm][k] for k in keys])) for arm in ARMS}
        dM = {arm: mahal(dig[arm], mu, P) for arm in ARMS}
        ham = {arm: (np.array([tk[arm][k] for k in keys]) != np.array([tk["B"][k] for k in keys])).sum(1) for arm in ARMS}
        for arm in ARMS:
            R["D2"][arm] = {"n": len(keys), "dM_median": float(np.median(dM[arm])),
                            "dM_p90": float(np.percentile(dM[arm], 90)), "ham_vs_B_median": float(np.median(ham[arm]))}
        # D3 durations
        dur = {}
        for arm in ARMS:
            ms = [{r["id"]: r["dur_s"] for r in jl(find(ARMS[arm].format(ss) + sd, "meas")) if r.get("dur_s")} for sd in SEEDS]
            dur[arm] = {k: (ms[0][k] + ms[1][k]) / 2 for k in ms[0] if k in ms[1]}
        for arm in ARMS:
            if arm == "B":
                R["D3"][arm] = {"dur_total_s": float(sum(dur["B"].values()))}
                continue
            ks = sorted(set(dur[arm]) & set(dur["B"]))
            lr = np.log(np.array([dur[arm][k] for k in ks]) / np.array([dur["B"][k] for k in ks]))
            R["D3"][arm] = {"n": len(ks), "median_log_ratio": float(np.median(lr)), "mean_log_ratio": float(lr.mean()),
                            "total_ratio": float(sum(dur[arm][k] for k in ks) / sum(dur["B"][k] for k in ks)),
                            "share_abs_gt_10pct": float((np.abs(lr) > np.log(1.1)).mean()),
                            "p_wilcoxon": float(wilcoxon(lr[lr != 0]).pvalue)}
        # D4 links (item = window__Sk; chrF per window -> mean over the window's items)
        idx = {k: i for i, k in enumerate(keys)}
        rows = []
        for arm in ("C", "Z", "SV", "E2E"):
            for k in keys:
                if k in dur[arm] and k in dur["B"]:
                    w = k.split("__")[0]
                    dc = cw[arm][w] - cw["B"][w] if w in cw[arm] and w in cw["B"] else np.nan
                    rows.append((arm, dM[arm][idx[k]] - dM["B"][idx[k]], ham[arm][idx[k]],
                                 abs(np.log(dur[arm][k] / dur["B"][k])), dc))
        A = np.array([r[1:] for r in rows], np.float64)
        ok = ~np.isnan(A[:, 3])
        R["D4"] = {"n": len(rows),
                   "rho_ddM_absdur": float(spearmanr(A[:, 0], A[:, 2]).correlation),
                   "rho_ham_absdur": float(spearmanr(A[:, 1], A[:, 2]).correlation),
                   "rho_ddM_dchrf": float(spearmanr(A[ok, 0], A[ok, 3]).correlation),
                   "rho_ham_dchrf": float(spearmanr(A[ok, 1], A[ok, 3]).correlation),
                   "rho_absdur_dchrf": float(spearmanr(A[ok, 2], A[ok, 3]).correlation)}
        for arm in ("C", "Z", "SV", "E2E"):
            m = np.array([r[0] == arm for r in rows])
            R["D4"][f"{arm}_rho_ham_absdur"] = float(spearmanr(A[m, 1], A[m, 2]).correlation)
        S["subsets"][ss] = R
    # Holm over the chrF tests
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][2])
    m, run = len(pvals), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i][2]))
        ss, arm, _ = pvals[i]
        S["subsets"][ss]["D1"][arm]["chrf_paired"]["holm_p"] = run
    (od / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"clean ref {gsrc} n={ng} dM median {S['clean_dM_median']:.2f}")
    for ss in SUBSETS:
        R = S["subsets"][ss]
        print(f"== {ss}")
        for arm in ARMS:
            d1, d2, d3 = R["D1"][arm], R["D2"][arm], R["D3"][arm]
            cp = d1.get("chrf_paired", {})
            print(f"{arm:4s} BLEU {d1['bleu_seeds']} spread {d1['bleu_seed_spread']:.2f} dB {d1.get('bleu_diff_vs_B', 0):+.2f} | "
                  f"LAAL {d1['laal_mean']:.3f} spread {d1['laal_seed_spread']:.3f} d {d1.get('laal_diff_vs_B', 0):+.3f} | "
                  f"chrF d {cp.get('mean', 0):+.2f} holm {cp.get('holm_p', 1):.2g} | dM {d2['dM_median']:.2f} ham {d2['ham_vs_B_median']:.0f} | "
                  f"dur logr {d3.get('median_log_ratio', 0):+.3f} tot {d3.get('total_ratio', 1):.3f} >10% {d3.get('share_abs_gt_10pct', 0):.2f}")
        print("D4", json.dumps(R["D4"]))


if __name__ == "__main__":
    main()
