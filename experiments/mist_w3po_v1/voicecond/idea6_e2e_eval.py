"""Idea 6 end-to-end dev check (prereg "端到端训练预登记", dev 检查): precheck A's exact dev protocol and units, method
E2E_post = closed-form posterior with the trained encoder (u_k) and precision head (alpha_k) on all own-slot windows.

    python -I experiments/mist_w3po_v1/voicecond/idea6_e2e_eval.py encode --ckpt <ckpt.pt> --enc <dir> \
        [--shard i --num-shards n] [--streams dev_en_0000,...] --device cuda:0
    python -I experiments/mist_w3po_v1/voicecond/idea6_e2e_eval.py analyze --enc <dir> --out <dir>
    python -I experiments/mist_w3po_v1/voicecond/idea6_e2e_eval.py check0 --enc <dir> --streams a,b,c
        (step-0 checkpoint: E2E_post errors and u_k must equal precheck A's C_post / u_C to 1e-4)
C_first8 / C_runmean / C_post / I2_full come from idea6_precheckA.run_stream (precheck A's stored encodes).
Outputs refuse to overwrite (encode skips finished streams).
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
from experiments.mist_w3po_v1.voicecond import idea6_precheckA as PA  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import (FIT, FIT_SHA1, HOP, L, RC_PATH, SR,  # noqa: E402
                                                               enc_forward, load_trained, post_numpy, prior_terms)

METHODS = ["C_first8", "C_runmean", "C_post", "I2_full", "E2E_post"]
REPORT_T = [1, 2, 4, 8, 20, 60]
PAIRS = [("E2E_post", "C_post"), ("E2E_post", "I2_full"), ("E2E_post", "C_first8"), ("C_post", "I2_full")]


def cmd_encode(a):
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    torch.set_num_threads(4)
    P = BiCodecPhi(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    head = load_trained(P, ck)
    P.model.eval()
    enc = Path(a.enc)
    enc.mkdir(parents=True, exist_ok=True)
    ss = a.streams.split(",") if a.streams else sids("all", "dev")
    assert all(s.startswith("dev_") for s in ss)
    ss = ss[a.shard:: a.num_shards]
    print(f"[enc] {a.ckpt} step {ck.get('step')}: {len(ss)} streams on {a.device}", flush=True)
    t0 = time.time()
    W = int(L * SR)
    for i, sid in enumerate(ss, 1):
        dst = enc / f"{sid}.npz"
        if dst.exists():
            continue
        mix = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0]
        end = np.load(I2 / "pre/win_L2" / f"{sid}.npz")["end_s"]
        starts = (np.arange(len(end)) * HOP * SR).astype(int)
        assert np.allclose((starts + W) / SR, end, atol=1e-4), sid
        us, als = [], []
        with torch.no_grad():
            for b in range(0, len(starts), a.bs):
                ref = torch.from_numpy(np.stack([P.ref_clip(mix[s:s + W]) for s in starts[b:b + a.bs]])).float().to(P.device)
                u, pooled = enc_forward(P, ref)
                us.append(u.cpu().numpy())
                als.append(head(pooled, u).cpu().numpy())
        tmp = enc / f".{sid}.partial.npz"
        np.savez(tmp, end_s=end, uE=np.concatenate(us).astype(np.float32), alpha=np.concatenate(als).astype(np.float32),
                 ckpt=str(a.ckpt), step=int(ck.get("step", -1)))
        os.rename(tmp, dst)
        print(f"[enc] {i}/{len(ss)} {sid} K={len(end)} {time.time() - t0:.0f}s", flush=True)
    print("[enc] DONE", flush=True)


def e2e_stream(sid, pf, RC, refs, enc):
    """E2E_post per unit (same unit order and protocol as idea6_precheckA.run_stream) + single-window rows + alpha rows"""
    d = load_stream(sid, L)
    meta = d["meta"]
    K = len(d["end"])
    z = np.load(Path(enc) / f"{sid}.npz")
    uE, al = z["uE"][:K].astype(np.float64), z["alpha"][:K].astype(np.float64)
    S0i, e0C, G, m0 = prior_terms(pf, RC)
    avail = d["end"] + DIAR_DELAY
    out = []
    for j in range(len(meta["speakers"])):
        s = slot_for_speaker(d, j)
        u_star = refs[meta["speakers"][j]][0].astype(np.float64)
        tau, tot = speech_clock(meta, j)
        mine = np.nonzero((d["slot"] == s) & (d["pi"] > 0))[0]
        err = np.full(len(T_EVAL), np.nan)
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            sel = mine[mine <= kk]
            m = post_numpy(uE[sel], al[sel], d["pi"][sel].astype(np.float64), S0i, e0C, G, m0)
            err[ti] = float(np.linalg.norm(m - u_star))
        out.append(err)
    sw, used = [], []
    for k in range(K):
        sl = int(d["slot"][k])
        if sl < 0 or sl >= len(d["slot2spk"]) or d["slot2spk"][sl] < 0:
            continue
        us = refs[meta["speakers"][int(d["slot2spk"][sl])]][0].astype(np.float64)
        sw.append(float(np.linalg.norm(uE[k] - us)))
        if d["pi"][k] > 0:
            used.append(k)
    return out, np.asarray(sw), al, al[used]


def summarize(units, sw_by, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    S = {"n_units": len(units), "n_streams": len(by), "median_err": {}, "n_units_t": {}, "paired": {}}
    for i, t in enumerate(T_EVAL):
        S["n_units_t"][str(t)] = int(sum(np.isfinite(u["err"]["E2E_post"][i]) for u in units))
    for m in METHODS:
        S["median_err"][m] = {str(t): float(np.nanmedian([u["err"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
    flat = lambda cl: [u for c in cl for u in c]
    for i, t in enumerate(T_EVAL):
        cl = [c for c in ([u for u in c if np.isfinite(u["err"]["E2E_post"][i])] for c in by.values()) if c]
        for m1, m2 in PAIRS:
            pm = lambda c, m1=m1, m2=m2: float(np.median([u["err"][m1][i] - u["err"][m2][i] for u in flat(c)]))
            est, lo, hi = cluster_boot(cl, pm, n_boot)
            base = float(np.median([u["err"][m2][i] for u in flat(cl)]))
            S["paired"].setdefault(f"{m1}-{m2}", {})[str(t)] = {
                "median_diff": est, "ci": [lo, hi], "base_median": base, "rel": est / base,
                "rel_ci": [lo / base, hi / base], "n_units": len(flat(cl)), "n_streams": len(cl)}
    p = S["paired"]["E2E_post-C_post"]
    g = {"60s_vs_C_post": bool(p["60"]["rel"] <= -0.10 and p["60"]["ci"][1] < 0),
         "8s_not_worse_than_C_post": bool(p["8"]["median_diff"] <= 0)}
    g["pass"] = bool(all(g.values()))
    S["gate"] = g
    cl = [v for v in sw_by if len(v)]
    allw = np.concatenate(cl)
    est, lo, hi = cluster_boot(cl, lambda c: float(np.median(np.concatenate(c)[:, 0] - np.concatenate(c)[:, 1])), n_boot)
    S["single_window"] = {"n_windows": int(len(allw)), "median_E2E": float(np.median(allw[:, 0])),
                          "median_C": float(np.median(allw[:, 1])), "median_frozen_plus_bM": float(np.median(allw[:, 2])),
                          "paired_median_E2E_minus_C": [est, lo, hi]}
    return S


def alpha_stats(A):
    return {"n_windows": int(len(A)), "mean": float(A.mean()), "std": float(A.std()),
            "pct": {str(q): float(np.percentile(A, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "winmean_mean": float(A.mean(1).mean()), "winmean_std": float(A.mean(1).std()),
            "winmean_pct": {str(q): float(np.percentile(A.mean(1), q)) for q in (5, 50, 95)},
            "dim_mean_std": float(A.mean(0).std()), "dim_mean_range": [float(A.mean(0).min()), float(A.mean(0).max())]}


def run_all(a, ss):
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
    assert hashlib.sha1(FIT.read_bytes()).hexdigest() == FIT_SHA1
    pf = ShrinkParams.load(FIT)
    RC = np.load(RC_PATH)["R_C"]
    refs = load_refs()
    units, sw_by, A_all, A_used = [], [], [], []
    for sid in ss:
        uc, sw = PA.run_stream(sid, pf, RC, refs)
        ue, swe, al, alu = e2e_stream(sid, pf, RC, refs, a.enc)
        assert len(uc) == len(ue) and len(sw) == len(swe), sid
        for u, e in zip(uc, ue):
            u["err"]["E2E_post"] = e
        units += uc
        sw_by.append(np.c_[swe, sw[:, 0], sw[:, 1]])
        A_all.append(al); A_used.append(alu)
    return units, sw_by, np.concatenate(A_all), np.concatenate(A_used)


def cmd_analyze(a):
    od = Path(a.out)
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od / 'summary.json'}")
    ss = sids("all", "dev")
    missing = [s for s in ss if not (Path(a.enc) / f"{s}.npz").exists()]
    if missing:
        sys.exit(f"{len(missing)} streams not encoded, e.g. {missing[:3]}")
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    units, sw_by, A_all, A_used = run_all(a, ss)
    S = summarize(units, sw_by)
    S["alpha_all_windows"] = alpha_stats(A_all)
    S["alpha_used_windows"] = alpha_stats(A_used)
    S.update({"enc": str(a.enc), "ckpt": str(np.load(Path(a.enc) / f"{ss[0]}.npz")["ckpt"]), "T_EVAL": T_EVAL,
              "analyze_s": time.time() - t0})
    np.savez(od / "units.npz", units=np.array(units, dtype=object))
    with open(od / "summary.json", "x") as fh:
        fh.write(json.dumps(S, indent=1))
    lines = ["method      " + " ".join(f"{t:>6}" for t in REPORT_T)]
    for m in METHODS:
        lines.append(f"{m:11s} " + " ".join(f"{S['median_err'][m][str(t)]:6.2f}" for t in REPORT_T))
    for k, v in S["paired"].items():
        for t in ("8", "60"):
            x = v[t]
            lines.append(f"{k:18s} {t:>2}s diff {x['median_diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] "
                         f"rel {100 * x['rel']:+.1f}% (n={x['n_units']}, streams={x['n_streams']})")
    lines += [json.dumps(S["single_window"]), json.dumps(S["alpha_used_windows"]), json.dumps(S["gate"])]
    (od / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_check0(a):
    ss = a.streams.split(",")
    units, sw_by, A_all, _ = run_all(a, ss)
    pre = {(u["sid"], u["spk"]): u for u in np.load(PA.OUT / "units.npz", allow_pickle=True)["units"] if u["sid"] in ss}
    de = max(float(np.nanmax(np.abs(u["err"]["E2E_post"] - pre[(u["sid"], u["spk"])]["err"]["C_post"]))) for u in units)
    du = 0.0
    for sid in ss:
        uE = np.load(Path(a.enc) / f"{sid}.npz")["uE"]
        uC = np.load(PA.ENC / f"{sid}.npz")["uC"]
        du = max(du, float(np.abs(uE - uC).max()))
    res = {"streams": ss, "n_units": len(units), "E2E_post_vs_precheckA_C_post_err_maxabs": de, "uE_vs_uC_maxabs": du,
           "alpha_maxabs": float(np.abs(A_all).max()), "pass_1e-4": bool(de <= 1e-4)}
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "analyze", "check0"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--enc", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--streams", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    {"encode": cmd_encode, "analyze": cmd_analyze, "check0": cmd_check0}[a.cmd](a)
