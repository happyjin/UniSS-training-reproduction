"""Idea 5 pre-checks (CPU only): cross-speaker common-mode background as a stream-level random effect.
Prereg: reports/mist_w3po_v1/voicecond_idea5_commonmode_prereg.zh-CN.md (committed before any number).

  z_i = lam_i s_j(i) + (1 - lam_i)(c_lang + b) + eps_i,  b ~ N(0, Sigma_b),  eps_i ~ N(0, R_w / pi_i)

  fit  : Sigma_b, R_w from train oracle residuals (Idea 2 full params sha1 25aaf6cd kept fixed) + C1 on train
  c1   : sharing statistics (oracle) on a split
  c2   : causal joint posterior over (s_1..s_J, b) vs Idea 2 full on dev, code error at T_EVAL

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/idea5_prechecks.py \
        all --out eval_outputs/mist_w3po_v1/idea5_prechecks --procs 48

Refuses to overwrite <out>/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, T_EVAL, cluster_boot, load_refs,  # noqa: E402
                                                             load_stream, sids, slot_for_speaker, speech_clock)
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams, load_vt  # noqa: E402

FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
L, HOP, D, SEED = 2.0, 0.5, 192, 20261119
STEP = int(round(L / HOP))          # non-overlapping windows for covariance estimates (as in Idea 2 fit)

_G = {}


def _init():
    _G["p"] = ShrinkParams.load(FIT)
    _G["refs"] = load_refs()


# ------------------------------------------------------------------------------------------------ oracle residuals
def oracle_stream(sid):
    """matched windows (slot -> true speaker), every STEP-th window: residual e, lam, pi, speaker index"""
    p, refs = _G["p"], _G["refs"]
    d = load_stream(sid, L)
    meta = d["meta"]
    lang = meta["lang"]
    K = len(d["end"])
    vt = load_vt(sid, L, K)
    lam_all = p.lam(vt)
    c = p.cvec(lang)
    E, LM, PI, J = [], [], [], []
    for k in range(0, K, STEP):
        sl = int(d["slot"][k])
        if sl < 0 or sl >= len(d["slot2spk"]) or d["slot2spk"][sl] < 0 or d["pi"][k] <= 0:
            continue
        j = int(d["slot2spk"][sl])
        s = refs[meta["speakers"][j]][0] - p.bM
        lam = float(lam_all[k])
        E.append(d["z_mix"][k] - lam * s - (1 - lam) * c); LM.append(lam); PI.append(float(d["pi"][k])); J.append(j)
    E, LM, PI, J = map(np.asarray, (E, LM, PI, J))
    out = {"sid": sid, "lang": lang, "n": len(E)}
    if len(E):
        w = PI * (1 - LM)
        out["b"] = (w[:, None] * E).sum(0) / (PI * (1 - LM) ** 2).sum()
        out["E"], out["lam"], out["pi"], out["j"] = E, LM, PI, J
        out["Dj"] = {int(j): (PI[J == j][:, None] * E[J == j]).sum(0) / PI[J == j].sum() for j in np.unique(J)}
        out["nj"] = {int(j): int((J == j).sum()) for j in np.unique(J)}
    ns = d["cov"].sum(1) <= 0
    out["n_nonspeech"] = int(ns.sum())
    out["z_ns"] = d["z_mix"][ns].mean(0) if ns.any() else None
    return out


def c1_stats(recs, n_perm=5000):
    """sharing of per-(stream, speaker) mean residuals D across speakers of the same stream"""
    rng = np.random.default_rng(SEED)
    res = {}
    for lang in ("en", "zh"):
        U, S = [], []
        for r in recs:
            if r["lang"] != lang or r["n"] == 0:
                continue
            for j, Dj in r["Dj"].items():
                if r["nj"][j] >= 5:
                    U.append(Dj); S.append(r["sid"])
        U, S = np.asarray(U), np.asarray(S)
        U = U - U.mean(0)
        Un = U / np.linalg.norm(U, axis=1, keepdims=True)
        C = Un @ Un.T
        iu = np.triu_indices(len(U), 1)

        def same_mean(lab):
            m = lab[iu[0]] == lab[iu[1]]
            return float(C[iu][m].mean()), float(C[iu][~m].mean())
        s_obs, d_obs = same_mean(S)
        null = np.array([same_mean(rng.permutation(S))[0] for _ in range(n_perm)])
        p = float((1 + (null >= s_obs).sum()) / (n_perm + 1))
        # one-way random-effects ICC on vectors (sum over dims)
        streams = np.unique(S)
        N, Sn = len(U), len(streams)
        gm = U.mean(0)
        ns = np.array([(S == s).sum() for s in streams])
        means = np.stack([U[S == s].mean(0) for s in streams])
        ssb = float((ns[:, None] * (means - gm) ** 2).sum())
        ssw = float(sum(((U[S == s] - means[i]) ** 2).sum() for i, s in enumerate(streams)))
        msb, msw = ssb / (Sn - 1), ssw / (N - Sn)
        n0 = (N - (ns ** 2).sum() / N) / (Sn - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        # non-speech windows vs stream mean residual (information only)
        zs, bm = [], []
        for r in recs:
            if r["lang"] == lang and r["n"] and r["z_ns"] is not None and r["n_nonspeech"] >= 5:
                zs.append(r["z_ns"]); bm.append(np.mean([r["Dj"][j] for j in r["Dj"]], 0))
        ns_cos = None
        if len(zs) > 3:
            zs, bm = np.asarray(zs), np.asarray(bm)
            zs, bm = zs - zs.mean(0), bm - bm.mean(0)
            ns_cos = float(np.mean((zs * bm).sum(1) / (np.linalg.norm(zs, axis=1) * np.linalg.norm(bm, axis=1))))
        res[lang] = {"n_units": int(N), "n_streams": int(Sn), "cos_same_stream": s_obs, "cos_diff_stream": d_obs,
                     "perm_p": p, "perm_null_mean": float(null.mean()), "icc": float(icc),
                     "nonspeech_vs_stream_residual_cos": ns_cos}
    allu = sum(res[l]["n_units"] for l in res)
    res["icc_pooled"] = float(sum(res[l]["icc"] * res[l]["n_units"] for l in ("en", "zh")) / allu)
    res["pass"] = bool(all(res[l]["perm_p"] < 0.01 for l in ("en", "zh")) and res["icc_pooled"] >= 0.20)
    return res


def fit_params(recs):
    from sklearn.covariance import LedoitWolf
    B = np.stack([r["b"] for r in recs if r["n"] > 0])
    Sb = LedoitWolf(assume_centered=True).fit(B).covariance_
    num, n = np.zeros((D, D)), 0
    for r in recs:
        if r["n"] == 0:
            continue
        e = r["E"] - (1 - r["lam"])[:, None] * r["b"][None]
        num += (r["pi"][:, None] * e).T @ e
        n += len(e)
    return Sb, num / n


# ------------------------------------------------------------------------------------------------ C2 streaming
def _stats(d, lam, y):
    """per-slot cumulative sufficient statistics over all windows"""
    out = {}
    for s in sorted(set(int(x) for x in d["slot"] if x >= 0)):
        w = np.where((d["slot"] == s) & (d["pi"] > 0), d["pi"], 0.0)
        out[s] = (np.cumsum(w * lam * lam), np.cumsum(w * lam * (1 - lam)), np.cumsum(w * (1 - lam) ** 2),
                  np.cumsum((w * lam)[:, None] * y, 0), np.cumsum((w * (1 - lam))[:, None] * y, 0))
    return out


def joint_mean(st, kk, target, G, S0i, e0, mu0, Sbi):
    """posterior mean of s_target given all slots' statistics up to window kk (joint with b)"""
    if kk < 0 or target not in st or st[target][0][kk] <= 0:
        return mu0
    terms = []
    for s, (A, Bc, Cc, ys, yb) in st.items():
        if A[kk] <= 0 and Cc[kk] <= 0:
            continue
        Mi = np.linalg.inv(S0i + A[kk] * G)
        terms.append((s, A[kk], Bc[kk], Cc[kk], ys[kk], yb[kk], Mi))
    P = Sbi.copy()
    rhs = np.zeros(D)
    for s, A, Bc, Cc, ys, yb, Mi in terms:
        GMi = G @ Mi
        P += Cc * G - Bc * Bc * GMi @ G
        rhs += G @ yb - Bc * GMi @ (e0 + G @ ys)
    b = np.linalg.solve(P, rhs)
    for s, A, Bc, Cc, ys, yb, Mi in terms:
        if s == target:
            return Mi @ (e0 + G @ ys - Bc * (G @ b))
    return mu0


def indep_mean(st, kk, target, G, S0i, e0, mu0):
    if kk < 0 or target not in st or st[target][0][kk] <= 0:
        return mu0
    A, _, _, ys, _ = st[target]
    return np.linalg.solve(S0i + A[kk] * G, e0 + G @ ys[kk])


def c2_stream(args):
    sid, Sb, Rw, b_or = args
    p, refs = _G["p"], _G["refs"]
    d = load_stream(sid, L)
    meta = d["meta"]
    lang = meta["lang"]
    K = len(d["end"])
    vt = load_vt(sid, L, K)
    lam = p.lam(vt)
    c = p.cvec(lang)
    y = d["z_mix"] - (1 - lam)[:, None] * c[None]
    st = _stats(d, lam, y)
    scale = p.L / HOP
    G_I2 = np.linalg.inv(p.R * scale)
    G_w = np.linalg.inv(Rw * scale)
    S0i = np.linalg.inv(p.Sig0)
    e0 = S0i @ p.mu0
    Sbi = np.linalg.inv(Sb)
    st_or = _stats(d, lam, y - (1 - lam)[:, None] * b_or[None]) if b_or is not None else None
    avail = d["end"] + DIAR_DELAY
    J = len(meta["speakers"])
    out = []
    for j in range(J):
        s = slot_for_speaker(d, j)
        u_star = refs[meta["speakers"][j]][0]
        tau, tot = speech_clock(meta, j)
        rec = {"sid": sid, "lang": lang, "tot": tot, "J": J, "err": {}}
        meths = {"full": lambda kk: indep_mean(st, kk, s, G_I2, S0i, e0, p.mu0),
                 "hier": lambda kk: joint_mean(st, kk, s, G_w, S0i, e0, p.mu0, Sbi),
                 "hier_R": lambda kk: joint_mean(st, kk, s, G_I2, S0i, e0, p.mu0, Sbi)}
        if st_or is not None:
            meths["oracle_b"] = lambda kk: indep_mean(st_or, kk, s, G_w, S0i, e0, p.mu0)
        for m in meths:
            rec["err"][m] = np.full(len(T_EVAL), np.nan)
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            for m, f in meths.items():
                rec["err"][m][ti] = float(np.linalg.norm(f(kk) + p.bM - u_star))
        out.append(rec)
    return out


def c2_summary(units, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    meths = list(units[0]["err"])
    S = {"n_units": len(units), "n_streams": len(by), "median_err": {}, "paired": {}}
    for m in meths:
        S["median_err"][m] = {str(t): float(np.nanmedian([u["err"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
    for t in (8, 20, 60):
        i = T_EVAL.index(t)
        cl = [c for c in ([u for u in c if not np.isnan(u["err"]["full"][i])] for c in by.values()) if c]
        S["paired"][str(t)] = {"n_units": sum(len(c) for c in cl)}
        for m in meths:
            if m == "full":
                continue
            pm = lambda cs, m=m: float(np.nanmedian([u["err"][m][i] - u["err"]["full"][i] for c in cs for u in c]))
            est, lo, hi = cluster_boot(cl, pm, n_boot)
            base = float(np.nanmedian([u["err"]["full"][i] for c in cl for u in c]))
            S["paired"][str(t)][m] = {"median_diff": est, "ci": [lo, hi], "rel": est / base, "base_median": base}
    h60, h8 = S["paired"]["60"]["hier"], S["paired"]["8"]["hier"]
    S["gate"] = {"rel60_le_-0.10": bool(h60["rel"] <= -0.10), "ci60_upper_lt_0": bool(h60["ci"][1] < 0),
                 "t8_not_worse": bool(h8["median_diff"] <= 0)}
    S["gate"]["pass"] = bool(all(S["gate"].values()))
    return S


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od}/summary.json")
    od.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha1(FIT.read_bytes()).hexdigest() == FIT_SHA1, "Idea 2 fit changed"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    tr, dv = sids("all", "train"), sids("all", "dev")
    with Pool(a.procs, initializer=_init) as pool:
        rec_tr = pool.map(oracle_stream, tr, chunksize=4)
        log(f"train oracle residuals: {len(rec_tr)} streams")
        rec_dv = pool.map(oracle_stream, dv, chunksize=1)
        log(f"dev oracle residuals: {len(rec_dv)} streams")
        Sb, Rw = fit_params(rec_tr)
        np.savez(od / "params_train.npz", Sigma_b=Sb, R_w=Rw)
        _init()
        log(f"fit: tr(Sigma_b) {np.trace(Sb):.3f}, tr(R_w) {np.trace(Rw):.3f}, tr(R_I2) {np.trace(_G['p'].R):.3f}")
        c1 = {"train": c1_stats(rec_tr), "dev": c1_stats(rec_dv)}
        c1["pass"] = c1["dev"]["pass"]
        log("C1 " + json.dumps(c1))
        b_or = {r["sid"]: (r["b"] if r["n"] else None) for r in rec_dv}
        units = [u for us in pool.map(c2_stream, [(s, Sb, Rw, b_or[s]) for s in dv], chunksize=1) for u in us]
    log(f"C2 units {len(units)}")
    c2 = c2_summary(units)
    np.savez(od / "units_c2.npz", units=np.array(units, dtype=object))
    S = {"prereg": "reports/mist_w3po_v1/voicecond_idea5_commonmode_prereg.zh-CN.md", "fit_sha1": FIT_SHA1,
         "trace_Sigma_b": float(np.trace(Sb)), "trace_R_w": float(np.trace(Rw)),
         "C1": c1, "C2": c2, "pass": bool(c1["pass"] and c2["gate"]["pass"])}
    (od / "summary.json").write_text(json.dumps(S, indent=1))
    for m in c2["median_err"]:
        print(f"{m:9s}", " ".join(f"{c2['median_err'][m][str(t)]:.2f}" for t in T_EVAL))
    print(json.dumps({"C1_dev": c1["dev"], "C2_paired": c2["paired"], "C2_gate": c2["gate"], "pass": S["pass"]}, indent=1))


if __name__ == "__main__":
    main()
