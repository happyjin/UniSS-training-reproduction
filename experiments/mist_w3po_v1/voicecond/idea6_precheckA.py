"""Idea 6 precheck A (prereg reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md, "预检 A"):
denoising encoder C (Idea 1 m2/ctrl) per 2 s window + causal temporal accumulation on the Idea 2 dev split.

  u_C,k = phi_C(mix[end_k - 2, end_k))  (prompt space; BiCodecPhi.phi_wave tiles to 6 s), hop 0.5 s, same windows as
  data/open_voicecond_v1/idea2/pre/win_L2.

Methods per (stream, true speaker) unit at t in T_EVAL (protocol of idea2_shrink.run_stream: slot_for_speaker,
speech_clock, a window is usable from end + DIAR_DELAY, own-slot windows with pi > 0 weighted by pi):
  C_first8   phi_C(first8_prompt(mix, solo_runs(post80, slot), tt - DIAR_DELAY))
  C_runmean  pi-weighted mean of own-slot u_C windows (fallback mu0 + bM)
  C_post     Gaussian conjugate posterior: prior N(mu0 + bM, Sig0) (Idea 2 full fit, sha1 25aaf6cd...),
             u_C,k = s + eps_k, eps_k ~ N(0, R_C (L/hop) / pi_k)  (ShrinkStream algebra with lambda = 1, c = 0)
  I2_full    Idea 2 full posterior (copy of run_stream's 'full' path) + bM
R_C: weighted residual covariance of u_C,k - u*_j on the first 64 train streams (sids("pilot", "train")), every 4th
window whose slot maps to a true speaker, R = (pi e)^T e / n (idea2_shrink.fit_one convention).

    python -I experiments/mist_w3po_v1/voicecond/idea6_precheckA.py encode --shard i --num-shards n --device cuda:0
    python -I experiments/mist_w3po_v1/voicecond/idea6_precheckA.py analyze
Outputs eval_outputs/mist_w3po_v1/idea6_precheckA/{enc/<sid>.npz, R_C.npz, units.npz, summary.json};
existing outputs are never rewritten (encode skips finished streams, analyze refuses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, I2, T_EVAL, cluster_boot,  # noqa: E402
                                                             load_refs, load_stream, sids, slot_for_speaker,
                                                             speech_clock)

OUT = REPO / "eval_outputs/mist_w3po_v1/idea6_precheckA"
ENC = OUT / "enc"
CKPT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
L, HOP, SR = 2.0, 0.5, 16000
D = 192
METHODS = ["C_first8", "C_runmean", "C_post", "I2_full"]
REPORT_T = [1, 2, 4, 8, 20, 60]
PAIRS = [("C_post", "C_first8"), ("C_post", "I2_full"), ("C_post", "C_runmean"), ("C_runmean", "C_first8"),
         ("C_first8", "I2_full")]


def all_sids():
    return sids("all", "dev") + sids("pilot", "train")


# ------------------------------------------------------------------------------------------------ encode (GPU)
def cmd_encode(a):
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea2_eval import first8_prompt, solo_runs
    torch.set_num_threads(4)
    P = load_encoder(BiCodecPhi(a.device), str(CKPT))
    ENC.mkdir(parents=True, exist_ok=True)
    ss = all_sids()[a.shard:: a.num_shards]
    print(f"[enc] shard {a.shard}/{a.num_shards}: {len(ss)} streams on {a.device}", flush=True)
    t0 = time.time()
    for i, sid in enumerate(ss, 1):
        dst = ENC / f"{sid}.npz"
        if dst.exists():
            print(f"[enc] {sid} exists, skipped", flush=True)
            continue
        mix = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0]
        end = np.load(I2 / "pre/win_L2" / f"{sid}.npz")["end_s"]
        W = int(L * SR)
        starts = (np.arange(len(end)) * HOP * SR).astype(int)
        assert np.allclose((starts + W) / SR, end, atol=1e-4), sid
        uC = []
        for b in range(0, len(starts), a.bs):
            uC.append(P.phi_wave([mix[s:s + W] for s in starts[b:b + a.bs]]).reshape(-1, D))
        res = {"end_s": end, "uC": np.concatenate(uC).astype(np.float32)}
        if sid.startswith("dev_"):
            d = load_stream(sid, L)
            meta = d["meta"]
            post = torch.load(I2 / "pre/diar" / f"{sid}.post.pt", weights_only=False)["post80"].float().numpy()
            J = len(meta["speakers"])
            f8 = np.full((J, len(T_EVAL), D), np.nan, np.float32)
            for j in range(J):
                s = slot_for_speaker(d, j)
                tau, _ = speech_clock(meta, j)
                runs = solo_runs(post, s)
                for ti, t in enumerate(T_EVAL):
                    tt = tau(t)
                    if tt is None:
                        continue
                    pr, _ = first8_prompt(mix, runs, tt - DIAR_DELAY)
                    f8[j, ti] = P.phi_wave(pr).reshape(-1)
            res["first8"] = f8
        tmp = ENC / f".{sid}.partial.npz"
        np.savez(tmp, **res)
        os.rename(tmp, dst)
        print(f"[enc] {i}/{len(ss)} {sid} K={len(end)} {time.time() - t0:.0f}s", flush=True)
    print("[enc] DONE", flush=True)


# ------------------------------------------------------------------------------------------------ analyze (CPU)
def load_uC(sid, K):
    return np.load(ENC / f"{sid}.npz")["uC"][:K].astype(np.float64)


def fit_RC(refs, log):
    step = int(round(L / HOP))
    E, Wt = [], []
    for sid in sids("pilot", "train"):
        d = load_stream(sid, L)
        K = len(d["end"])
        uC = load_uC(sid, K)
        spk = d["meta"]["speakers"]
        for k in range(0, K, step):
            sl = int(d["slot"][k])
            if sl < 0 or sl >= len(d["slot2spk"]) or d["slot2spk"][sl] < 0:
                continue
            j = int(d["slot2spk"][sl])
            E.append(uC[k] - refs[spk[j]][0]); Wt.append(d["pi"][k])
    E, Wt = np.asarray(E), np.asarray(Wt, np.float64)
    R = (Wt[:, None] * E).T @ E / len(E)
    log(f"[R_C] {len(E)} train windows (pilot 64 streams, every {step}th, slot -> true speaker); tr R_C {np.trace(R):.3f};"
        f" mean residual norm {np.linalg.norm((Wt[:, None] * E).sum(0) / Wt.sum()):.3f}")
    return R, len(E)


def run_stream(sid, pf, RC, refs):
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import load_vt
    d = load_stream(sid, L)
    meta = d["meta"]
    lang = meta["lang"]
    K = len(d["end"])
    z = np.load(ENC / f"{sid}.npz")
    uC = z["uC"][:K].astype(np.float64)
    f8 = z["first8"].astype(np.float64)
    vt = load_vt(sid, L, K)
    avail = d["end"] + DIAR_DELAY
    J = len(meta["speakers"])
    unit_slot = [slot_for_speaker(d, j) for j in range(J)]
    slots = sorted(set(unit_slot))
    bM = pf.bM
    # Idea 2 full path (copied from idea2_shrink.run_stream, mode 'full')
    lam = pf.lam(vt)
    y = d["z_mix"] - (1 - lam)[:, None] * pf.cvec(lang)[None]
    Rfi = np.linalg.inv(pf.R * (pf.L / 0.5))
    S0i = np.linalg.inv(pf.Sig0)
    e0 = S0i @ pf.mu0
    # C_post: lambda = 1, c = 0, prior N(mu0 + bM, Sig0), noise R_C (L / hop)
    RCfi = np.linalg.inv(RC * (L / HOP))
    e0C = S0i @ (pf.mu0 + bM)
    cum = {}
    for s in slots:
        w = np.where((d["slot"] == s) & (d["pi"] > 0), d["pi"], 0.0)
        cum[s] = {"full": (np.cumsum(w * lam * lam), np.cumsum((w * lam)[:, None] * y, 0)),
                  "C": (np.cumsum(w), np.cumsum(w[:, None] * uC, 0))}

    def post_full(s, kk):
        if kk < 0 or cum[s]["full"][0][kk] <= 0:
            return pf.mu0
        return np.linalg.solve(S0i + cum[s]["full"][0][kk] * Rfi, e0 + Rfi @ cum[s]["full"][1][kk])

    def post_C(s, kk):
        if kk < 0 or cum[s]["C"][0][kk] <= 0:
            return pf.mu0 + bM
        return np.linalg.solve(S0i + cum[s]["C"][0][kk] * RCfi, e0C + RCfi @ cum[s]["C"][1][kk])
    out = []
    for j in range(J):
        s = unit_slot[j]
        spk = meta["speakers"][j]
        u_star = refs[spk][0].astype(np.float64)
        tau, tot = speech_clock(meta, j)
        mine = np.nonzero(d["slot"] == s)[0]
        rec = {"sid": sid, "spk": spk, "lang": lang, "tot": tot, "slot": s,
               "err": {m: np.full(len(T_EVAL), np.nan) for m in METHODS}}
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            sel = mine[mine <= kk]
            w = d["pi"][sel]
            e = {"C_runmean": (w[:, None] * uC[sel]).sum(0) / w.sum() if len(sel) else pf.mu0 + bM,
                 "C_post": post_C(s, kk), "I2_full": post_full(s, kk) + bM, "C_first8": f8[j, ti]}
            assert np.isfinite(e["C_first8"]).all(), (sid, j, t)
            for m in METHODS:
                rec["err"][m][ti] = float(np.linalg.norm(e[m] - u_star))
        out.append(rec)
    # single-window errors (all windows whose slot maps to a true speaker): C vs frozen (z_mix + bM)
    sw = []
    for k in range(K):
        sl = int(d["slot"][k])
        if sl < 0 or sl >= len(d["slot2spk"]) or d["slot2spk"][sl] < 0:
            continue
        us = refs[meta["speakers"][int(d["slot2spk"][sl])]][0].astype(np.float64)
        sw.append((float(np.linalg.norm(uC[k] - us)), float(np.linalg.norm(d["z_mix"][k] + bM - us)), float(d["pi"][k])))
    return out, np.asarray(sw).reshape(-1, 3)


def summarize(units, sw_by, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    S = {"n_units": len(units), "n_streams": len(by), "median_err": {}, "n_units_t": {}, "paired": {}}
    for i, t in enumerate(T_EVAL):
        S["n_units_t"][str(t)] = int(sum(np.isfinite(u["err"]["C_post"][i]) for u in units))
    for m in METHODS:
        S["median_err"][m] = {str(t): float(np.nanmedian([u["err"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
    flat = lambda cl: [u for c in cl for u in c]
    for i, t in enumerate(T_EVAL):
        cl = [c for c in ([u for u in c if np.isfinite(u["err"]["C_post"][i])] for c in by.values()) if c]
        for m1, m2 in PAIRS:
            pm = lambda c, m1=m1, m2=m2: float(np.median([u["err"][m1][i] - u["err"][m2][i] for u in flat(c)]))
            est, lo, hi = cluster_boot(cl, pm, n_boot)
            base = float(np.median([u["err"][m2][i] for u in flat(cl)]))
            S["paired"].setdefault(f"{m1}-{m2}", {})[str(t)] = {
                "median_diff": est, "ci": [lo, hi], "base_median": base, "rel": est / base,
                "rel_ci": [lo / base, hi / base], "n_units": len(flat(cl)), "n_streams": len(cl)}
    p60, p8 = S["paired"], S["paired"]
    g = {"60s_vs_C_first8": bool(p60["C_post-C_first8"]["60"]["rel"] <= -0.10 and p60["C_post-C_first8"]["60"]["ci"][1] < 0),
         "60s_vs_I2_full": bool(p60["C_post-I2_full"]["60"]["rel"] <= -0.10 and p60["C_post-I2_full"]["60"]["ci"][1] < 0),
         "8s_not_worse_than_C_first8": bool(p8["C_post-C_first8"]["8"]["median_diff"] <= 0)}
    g["pass"] = bool(all(g.values()))
    S["gate"] = g
    # single-window C vs frozen
    cl = [v for v in sw_by if len(v)]
    allw = np.concatenate(cl)
    est, lo, hi = cluster_boot(cl, lambda c: float(np.median(np.concatenate(c)[:, 0] - np.concatenate(c)[:, 1])), n_boot)
    S["single_window"] = {"n_windows": int(len(allw)), "median_C": float(np.median(allw[:, 0])),
                          "median_frozen_plus_bM": float(np.median(allw[:, 1])),
                          "paired_median_C_minus_frozen": [est, lo, hi],
                          "pi_weighted_rms_C": float(np.sqrt(np.average(allw[:, 0] ** 2, weights=allw[:, 2] + 1e-12))),
                          "pi_weighted_rms_frozen": float(np.sqrt(np.average(allw[:, 1] ** 2, weights=allw[:, 2] + 1e-12)))}
    return S


def cmd_analyze(a):
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
    if (OUT / "summary.json").exists():
        sys.exit(f"refuse to overwrite {OUT / 'summary.json'}")
    missing = [s for s in all_sids() if not (ENC / f"{s}.npz").exists()]
    if missing:
        sys.exit(f"{len(missing)} streams not encoded yet, e.g. {missing[:3]}")
    sha = hashlib.sha1(FIT.read_bytes()).hexdigest()
    assert sha == FIT_SHA1, sha
    logf = open(OUT / "analyze.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    pf = ShrinkParams.load(FIT)
    refs = load_refs()
    t0 = time.time()
    RC, nR = fit_RC(refs, log)
    log(f"[R_C] tr R_I2 (window space, Idea 2 full fit) {np.trace(pf.R):.3f}")
    np.savez(OUT / "R_C.npz", R_C=RC, n=nR)
    units, sw_by = [], []
    for sid in sids("all", "dev"):
        u, sw = run_stream(sid, pf, RC, refs)
        units += u; sw_by.append(sw)
    np.savez(OUT / "units.npz", units=np.array(units, dtype=object))
    S = summarize(units, sw_by)
    S.update({"fit": str(FIT), "fit_sha1": sha, "ckpt": str(CKPT), "T_EVAL": T_EVAL, "tr_R_C": float(np.trace(RC)),
              "tr_R_I2": float(np.trace(pf.R)), "n_R_windows": nR, "analyze_s": time.time() - t0})
    with open(OUT / "summary.json", "x") as fh:
        fh.write(json.dumps(S, indent=1))
    log("method      " + " ".join(f"{t:>6}" for t in REPORT_T))
    for m in METHODS:
        log(f"{m:11s} " + " ".join(f"{S['median_err'][m][str(t)]:6.2f}" for t in REPORT_T))
    for k, v in S["paired"].items():
        for t in ("8", "60"):
            x = v[t]
            log(f"{k:20s} {t:>2}s diff {x['median_diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] rel {100 * x['rel']:+.1f}%"
                f" (n={x['n_units']}, streams={x['n_streams']})")
    log(json.dumps(S["single_window"]))
    log(json.dumps(S["gate"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "analyze"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    cmd_encode(a) if a.cmd == "encode" else cmd_analyze(a)
