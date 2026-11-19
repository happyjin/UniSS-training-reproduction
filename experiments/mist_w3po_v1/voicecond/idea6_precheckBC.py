"""Idea 6 pre-checks B (GMM speaker prior) and C (voice flicker), CPU only.
Prereg: reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md (committed before any number).

  B: Idea 2 full likelihood (lambda, c, R of the full fit sha1 25aaf6cd) with the single Gaussian prior replaced by a
     GMM fitted on train speakers' clean references (window space s = u* - b_M; full covariance, reg_covar = 1e-3 x mean
     variance, K = 1..8 by BIC, train only). Closed-form mixture posterior. Dev code error vs Idea 2 full.
  C: Idea 2 full posterior mean at every turn start of each speaker (available after the diarizer delay); 32 FSQ tokens and
     the decoder condition d (BiCodecPhi.d_vector); adjacent turns after >= 8 s of the speaker's speech.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/idea6_precheckBC.py \
        --out eval_outputs/mist_w3po_v1/idea6_precheckBC --procs 40
Refuses to overwrite <out>/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
_G = {}


def _init(gmm=None):
    import torch
    torch.set_num_threads(1)
    _G["p"] = ShrinkParams.load(FIT)
    _G["refs"] = load_refs()
    _G["gmm"] = gmm
    if gmm is not None:
        w, m, S = gmm
        _G["gS"] = [(np.log(w[k]), m[k], np.linalg.inv(S[k]), np.linalg.slogdet(S[k])[1]) for k in range(len(w))]
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    _G["P"] = BiCodecPhi("cpu")


def gmm_mean(A, a1, G):
    """closed-form mixture posterior mean; likelihood exp(-A/2 s'Gs + s'G a1)"""
    lz, means = [], []
    Ga1 = G @ a1
    for lw, m, Si, ld in _G["gS"]:
        P = Si + A * G
        h = Si @ m + Ga1
        mu = np.linalg.solve(P, h)
        lz.append(lw - 0.5 * ld - 0.5 * np.linalg.slogdet(P)[1] - 0.5 * m @ Si @ m + 0.5 * h @ mu)
        means.append(mu)
    lz = np.array(lz)
    r = np.exp(lz - lz.max())
    r /= r.sum()
    return (r[:, None] * np.stack(means)).sum(0), r


def dvec(P, ids):
    """decoder condition d for token sets (B, 32); BiCodec detokenize takes (B, 1, 32)"""
    import torch
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long).reshape(-1, 1, 32)
        return P.se.detokenize(t).cpu().numpy().reshape(t.shape[0], -1)


def stream(sid):
    p, refs, P = _G["p"], _G["refs"], _G["P"]
    d = load_stream(sid, L)
    meta = d["meta"]
    lang = meta["lang"]
    K = len(d["end"])
    lam = p.lam(load_vt(sid, L, K))
    y = d["z_mix"] - (1 - lam)[:, None] * p.cvec(lang)[None]
    G = np.linalg.inv(p.R * (p.L / HOP))
    S0i = np.linalg.inv(p.Sig0)
    e0 = S0i @ p.mu0
    avail = d["end"] + DIAR_DELAY
    out = []
    for j in range(len(meta["speakers"])):
        s = slot_for_speaker(d, j)
        spk = meta["speakers"][j]
        u_star, g_star = refs[spk]
        w = np.where((d["slot"] == s) & (d["pi"] > 0), d["pi"], 0.0)
        cA, c1 = np.cumsum(w * lam * lam), np.cumsum((w * lam)[:, None] * y, 0)

        def full(kk):
            if kk < 0 or cA[kk] <= 0:
                return p.mu0
            return np.linalg.solve(S0i + cA[kk] * G, e0 + G @ c1[kk])

        def gmm(kk):
            if kk < 0 or cA[kk] <= 0:
                return gmm_mean(0.0, np.zeros(D), G)[0]
            return gmm_mean(cA[kk], c1[kk], G)[0]
        tau, tot = speech_clock(meta, j)
        rec = {"sid": sid, "lang": lang, "tot": tot, "err": {m: np.full(len(T_EVAL), np.nan) for m in ("full", "gmm")}}
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            rec["err"]["full"][ti] = float(np.linalg.norm(full(kk) + p.bM - u_star))
            rec["err"]["gmm"][ti] = float(np.linalg.norm(gmm(kk) + p.bM - u_star))
        # C: turn starts after >= 8 s of this speaker's speech
        segs = sorted((g["start"], g["end"]) for g in meta["turns"] if g["spk"] == spk)
        U, starts = [], []
        for a, b in segs:
            spoken = sum(max(0.0, min(e, a) - s0) for s0, e in segs if s0 < a)
            if spoken < 8.0:
                continue
            kk = int(np.searchsorted(avail, a, side="right")) - 1
            U.append(full(kk) + p.bM); starts.append(a)
        rec["turns"] = len(U)
        if len(U) >= 2:
            ids = P.tokens(np.stack(U).reshape(-1, 32, 6).astype(np.float32))
            dv = dvec(P, ids)
            dg = dvec(P, np.asarray(g_star).reshape(1, 32))[0]
            cosd = lambda x, z: float(1 - x @ z / (np.linalg.norm(x) * np.linalg.norm(z)))
            rec["ham"] = [int((ids[i] != ids[i + 1]).sum()) for i in range(len(ids) - 1)]
            rec["d_turn"] = [cosd(dv[i], dv[i + 1]) for i in range(len(ids) - 1)]
            rec["d_id"] = [cosd(dv[i], dg) for i in range(len(ids))]
            rec["ham_changed_share"] = float(np.mean([h > 0 for h in rec["ham"]]))
        out.append(rec)
    return out


def fit_gmm(log):
    from sklearn.mixture import GaussianMixture
    p, refs = ShrinkParams.load(FIT), load_refs()
    spk = sorted({s for sid in sids("all", "train") for s in json.load(open(
        REPO / "data/open_voicecond_v1/idea2/streams" / sid / "meta.json"))["speakers"]})
    X = np.stack([refs[s][0] - p.bM for s in spk])
    reg = 1e-3 * float(X.var(0).mean())
    bics, models = [], []
    for k in range(1, 9):
        g = GaussianMixture(k, covariance_type="full", reg_covar=reg, n_init=3, max_iter=500, random_state=SEED).fit(X)
        bics.append(float(g.bic(X))); models.append(g)
        log(f"[B] K={k} BIC {bics[-1]:.1f}")
    kb = int(np.argmin(bics)) + 1
    g = models[kb - 1]
    return (g.weights_, g.means_, g.covariances_), {"n_train_speakers": len(spk), "bic": bics, "K": kb,
                                                     "weights": g.weights_.tolist(), "reg_covar": reg}


def summarize_B(units, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    S = {"median_err": {m: {str(t): float(np.nanmedian([u["err"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
                        for m in ("full", "gmm")}, "paired": {}}
    for t in (1, 2, 4, 8, 60):
        i = T_EVAL.index(t)
        cl = [c for c in ([u for u in c if not np.isnan(u["err"]["full"][i])] for c in by.values()) if c]
        pm = lambda cs: float(np.nanmedian([u["err"]["gmm"][i] - u["err"]["full"][i] for c in cs for u in c]))
        est, lo, hi = cluster_boot(cl, pm, n_boot)
        base = float(np.nanmedian([u["err"]["full"][i] for c in cl for u in c]))
        S["paired"][str(t)] = {"n": sum(len(c) for c in cl), "median_diff": est, "ci": [lo, hi], "rel": est / base}
    rel = np.mean([S["paired"][str(t)]["rel"] for t in (1, 2, 4)])
    S["gate"] = {"mean_rel_1_2_4": float(rel), "rel_le_-0.10": bool(rel <= -0.10),
                 "ci2_upper_lt_0": bool(S["paired"]["2"]["ci"][1] < 0), "t60_not_worse": bool(S["paired"]["60"]["median_diff"] <= 0)}
    S["gate"]["pass"] = bool(S["gate"]["rel_le_-0.10"] and S["gate"]["ci2_upper_lt_0"] and S["gate"]["t60_not_worse"])
    return S


def summarize_C(units):
    ham = [h for u in units for h in u.get("ham", [])]
    dt = [x for u in units for x in u.get("d_turn", [])]
    di = [x for u in units for x in u.get("d_id", [])]
    S = {"n_units_with_pairs": sum(1 for u in units if "ham" in u), "n_pairs": len(ham),
         "ham_median": float(np.median(ham)), "ham_mean": float(np.mean(ham)), "share_pairs_changed": float(np.mean(np.array(ham) > 0)),
         "D_turn_median": float(np.median(dt)), "D_id_median": float(np.median(di))}
    S["ratio"] = S["D_turn_median"] / S["D_id_median"]
    S["gate"] = {"ham_ge_8": bool(S["ham_median"] >= 8), "Dturn_ge_half_Did": bool(S["ratio"] >= 0.5)}
    S["gate"]["pass"] = bool(all(S["gate"].values()))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od}/summary.json")
    od.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha1(FIT.read_bytes()).hexdigest() == FIT_SHA1
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    gmm, ginfo = fit_gmm(log)
    log(f"[B] K by BIC = {ginfo['K']} ({ginfo['n_train_speakers']} train speakers)")
    with Pool(a.procs, initializer=_init, initargs=(gmm,)) as pool:
        units = [u for us in pool.map(stream, sids("all", "dev"), chunksize=1) for u in us]
    log(f"units {len(units)}")
    np.savez(od / "units.npz", units=np.array(units, dtype=object))
    S = {"prereg": "reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md", "fit_sha1": FIT_SHA1,
         "B": {"gmm": ginfo, **summarize_B(units)}, "C": summarize_C(units)}
    (od / "summary.json").write_text(json.dumps(S, indent=1))
    for m in ("full", "gmm"):
        print(f"{m:5s}", " ".join(f"{S['B']['median_err'][m][str(t)]:.2f}" for t in T_EVAL))
    print(json.dumps({"B_paired": S["B"]["paired"], "B_gate": S["B"]["gate"], "C": S["C"]}, indent=1))


if __name__ == "__main__":
    main()
