"""Idea 7 step 1 (prereg reports/mist_w3po_v1/voicecond_idea7_perception_prereg.zh-CN.md): perception-distortion mechanism
check on EXISTING test outputs only (no selection, no fitting beyond the clean-voice Gaussian).

Per set (6 OpenBroadcast-SI + 3 OOD-3) and arm (C, Z, I2, E2E2, i6c, i6e; OOD-3 also B): token digits (32x6 base-4 -> 192),
spread ratio tr(Cov)/tr(Cov_clean), Frechet distance to the clean Gaussian, median Mahalanobis d_M, median per-item PESQ
(two-seed mean). Items: non-fallback items of the Idea 6 i6c table, common to all arms. Gates M1 / M2 / M3 as preregistered.

    <uniss-eval python> -m experiments.mist_w3po_v1.voicecond.idea7_mechanism --out eval_outputs/mist_w3po_v1/idea7_mechanism
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
I1, I2T, I6T = E / "voicecond_idea1/m3", E / "voicecond_idea2/tokens", E / "idea6_test/tokens"
VB = E / "voicecond_bench"
BENCH = [f"{c}_{d}" for c in ("id2", "ood1", "clean") for d in ("zh2en", "en2zh")]
OOD3 = ["ami", "en", "ali"]
EST = ["C", "Z", "I2", "E2E2", "i6c", "i6e"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def digits(ids):
    ids = np.asarray(ids, np.int64)
    return np.stack([(ids // 4 ** i) % 4 for i in range(6)], -1).reshape(len(ids), -1).astype(np.float64)


def clean():
    from sklearn.covariance import LedoitWolf
    R = REPO / "data/open_voicecond_v1/idea2/refs"
    k, p = 2, R / "ustar.npz"
    while (R / f"ustar_v{k}.npz").exists():
        p, k = R / f"ustar_v{k}.npz", k + 1
    X = digits(np.load(p, allow_pickle=True)["gstar"].reshape(-1, 32))
    lw = LedoitWolf().fit(X)
    return lw.location_, lw.covariance_, np.linalg.inv(lw.covariance_), str(p.relative_to(REPO))


def _psd_sqrt(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fd(mu1, S1, mu2, S2):
    """Frechet distance; tr((S1^1/2 S2 S1^1/2)^1/2) via symmetric eigendecompositions (v1 used scipy sqrtm on S1 @ S2,
    which returned NaN for the near-singular token covariances)"""
    r = _psd_sqrt(S1)
    w = np.linalg.eigvalsh(r @ S2 @ r)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * np.sqrt(np.clip(w, 0, None)).sum())


def pesq_items(run_paths):
    seeds = []
    for p in run_paths:
        if not p.exists():
            return None
        seeds.append({r["id"]: r["pesq"] for r in jl(p) if r.get("pesq") is not None})
    return {k: (seeds[0][k] + seeds[1][k]) / 2 for k in seeds[0] if k in seeds[1]}


def load_set(s):
    if s in OOD3:
        c = {r["key"]: r for r in jl(I1 / "ood3" / f"tokens_c_{s}.jsonl")}
        tok = {"B": {k: r["raw_ids"] for k, r in c.items()}, "C": {k: r["ids"] for k, r in c.items()},
               "Z": {r["key"]: r["ids"] for r in jl(I1 / "ood3" / f"tokens_z_{s}.jsonl")},
               "I2": {r["key"]: r["ids"] for r in jl(I2T / f"ood3_{s}.jsonl")},
               "E2E2": {r["key"]: r["ids"] for r in jl(I2T / f"ood3_e2e_{s}.jsonl")},
               "i6c": {r["key"]: r["ids"] for r in jl(I6T / f"i6c_{s}.jsonl")},
               "i6e": {r["key"]: r["ids"] for r in jl(I6T / f"i6e_{s}.jsonl")}}
        O2, O1, O6 = E / "voicecond_idea2/ood3", I1 / "ood3", E / "idea6_test/ood3"
        runs = {"B": (O2, "idea2_ood3_b"), "C": (O1, "idea1_ood3_c"), "Z": (O1, "idea1_ood3_z"), "I2": (O2, "idea2_ood3_sv"),
                "E2E2": (O2, "idea2_ood3_e2e"), "i6c": (O6, "idea2_ood3_i6c"), "i6e": (O6, "idea2_ood3_i6e")}
        pe = {a: pesq_items([d / f"{n}_{s}__t1c{sd}.jsonl" for sd in ("", "_tts2")]) for a, (d, n) in runs.items()}
    else:
        tok = {"C": json.load(open(I1 / f"sidecar_c_{s}.json")), "Z": json.load(open(I1 / f"sidecar_z_{s}.json")),
               "I2": json.load(open(I2T / f"obsi_{s}.sidecar.json")), "E2E2": json.load(open(I2T / f"obsi_e2e_{s}.sidecar.json")),
               "i6c": json.load(open(I6T / f"i6c_{s}.sidecar.json")), "i6e": json.load(open(I6T / f"i6e_{s}.sidecar.json"))}
        runs = {"C": "obsi_i1c", "Z": "obsi_i1z", "I2": "obsi_i2", "E2E2": "obsi_e2e", "i6c": "obsi_i6c", "i6e": "obsi_i6e"}
        pe = {a: pesq_items([VB / f"{n}_{s}__t1c" / f"items_{n}_{s}__t1c{sd}.jsonl" for sd in ("", "_tts2")])
              for a, n in runs.items()}
    nofb = {r["key"] for r in jl(I6T / f"i6c_{s}.jsonl") if not r.get("fallback")}
    return tok, pe, nofb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    k = 1
    outp = od / "summary.json"
    while outp.exists():
        k += 1
        outp = od / f"summary_v{k}.json"
    mu_c, S_c, P_c, src = clean()
    trc = float(np.trace(S_c))
    S = {"clean_ref": src, "sets": {}}
    pooled = []
    for s in BENCH + OOD3:
        tok, pe, nofb = load_set(s)
        arms = [x for x in tok if pe.get(x) is not None]
        keys = sorted(nofb.intersection(*(set(tok[x]) for x in arms)).intersection(*(set(pe[x]) for x in arms)))
        R = {"n_items": len(keys), "arms": {}, "missing_pesq": [x for x in tok if pe.get(x) is None]}
        dm = {}
        for x in arms:
            X = digits([tok[x][kk] for kk in keys])
            mu, Sg = X.mean(0), np.cov(X, rowvar=False)
            d = X - mu_c
            dm[x] = np.sqrt(np.einsum("ij,jk,ik->i", d, P_c, d))
            R["arms"][x] = {"spread": float(np.trace(Sg)) / trc, "FD": fd(mu, Sg, mu_c, S_c),
                            "dM_median": float(np.median(dm[x])),
                            "pesq_median": float(np.median([pe[x][kk] for kk in keys]))}
        est = [x for x in EST if x in R["arms"]]
        if len(est) >= 4:
            P = [R["arms"][x]["pesq_median"] for x in est]
            R["rho_FD_pesq"] = float(spearmanr([R["arms"][x]["FD"] for x in est], P).correlation)
            R["rho_spread_pesq"] = float(spearmanr([R["arms"][x]["spread"] for x in est], P).correlation)
            R["M2"] = bool(R["rho_FD_pesq"] <= -0.5 and R["rho_spread_pesq"] >= 0.5)
        if "C" in R["arms"] and all(x in R["arms"] for x in ("I2", "i6c", "i6e")):
            R["M1"] = bool(all(R["arms"][x]["spread"] < R["arms"]["C"]["spread"] for x in ("I2", "i6c", "i6e")))
        for x, y in (("i6c", "C"), ("i6e", "C"), ("I2", "C"), ("i6e", "i6c")):
            if x in dm and y in dm:
                for i, kk in enumerate(keys):
                    pooled.append((dm[x][i] - dm[y][i], pe[x][kk] - pe[y][kk]))
        S["sets"][s] = R
    m1 = [R["M1"] for R in S["sets"].values() if "M1" in R]
    m2 = [R["M2"] for R in S["sets"].values() if "M2" in R]
    A = np.array(pooled)
    r3 = spearmanr(A[:, 0], A[:, 1])
    S["gates"] = {"M1_count": f"{sum(m1)}/{len(m1)}", "M1": bool(sum(m1) >= 7) if len(m1) == 9 else f"incomplete ({len(m1)}/9 sets)",
                  "M2_count": f"{sum(m2)}/{len(m2)}", "M2": bool(sum(m2) >= 6) if len(m2) == 9 else f"incomplete ({len(m2)}/9 sets)",
                  "M3_rho": float(r3.correlation), "M3_p": float(r3.pvalue), "M3_n": len(A),
                  "M3": bool(r3.correlation < 0 and r3.pvalue < 0.01)}
    outp.write_text(json.dumps(S, indent=1))
    print(outp)
    for s, R in S["sets"].items():
        print(f"== {s} n={R['n_items']} missing={R['missing_pesq']} M1={R.get('M1')} M2={R.get('M2')} "
              f"rhoFD={R.get('rho_FD_pesq', float('nan')):+.2f} rhoSpread={R.get('rho_spread_pesq', float('nan')):+.2f}")
        for x, v in R["arms"].items():
            print(f"   {x:5s} spread {v['spread']:.3f} FD {v['FD']:7.2f} dM {v['dM_median']:6.2f} PESQ {v['pesq_median']:.3f}")
    print(json.dumps(S["gates"]))


if __name__ == "__main__":
    main()
