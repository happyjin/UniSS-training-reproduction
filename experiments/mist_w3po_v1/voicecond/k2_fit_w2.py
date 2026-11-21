"""K2 step 2 (reports/mist_w3po_v1/robust_k2_w2map_prereg.zh-CN.md): closed-form Gaussian W2 optimal-transport map from
the V1-code distribution to the clean-code distribution, per source language, fitted on the voicecond Idea 1 TRAIN pairs
only (data/open_voicecond_v1/idea1/train_{en,zh}, 20000 pairs each).
  V1 codes  = {phi_V1(y_k)} u {phi_V1(s_k)}  (the V1 encoder's input distribution: noisy and clean halves)
  clean     = {u*_k} = ctrl/u0_targets.npy (phi_0(s_k), rows train_en then train_zh, as idea1_e2e_train.py wrote them)
  T(x) = mu_c + A (x - mu_v),  A = Sv^{-1/2} (Sv^{1/2} Sc Sv^{1/2})^{1/2} Sv^{-1/2}   (symmetric eigendecompositions,
  ridge 1e-6 tr(S)/192 on both covariances)
Writes <out>/train_codes_<lang>.npz (vy, vs) and <out>/w2_map.npz (mu_v_<l>, mu_c_<l>, A_<l>) + w2_map.json (diagnostics).
Resumes from existing train_codes files; refuses an existing w2_map.npz.
    OMP_NUM_THREADS=48 qwen3omni/bin/python -I experiments/mist_w3po_v1/voicecond/k2_fit_w2.py --out <dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import CKPT_C  # noqa: E402

DATA = REPO / "data/open_voicecond_v1/idea1"
U0 = CKPT_C.parent / "u0_targets.npy"
BS = 64


def sqrtm_sym(S, inv=False):
    w, Q = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (Q * (w ** (-0.5 if inv else 0.5))) @ Q.T


def w2_map(Xv, Xc):
    mv, mc = Xv.mean(0), Xc.mean(0)
    Sv, Sc = np.cov(Xv, rowvar=False), np.cov(Xc, rowvar=False)
    d = Sv.shape[0]
    Sv = Sv + 1e-6 * np.trace(Sv) / d * np.eye(d)
    Sc = Sc + 1e-6 * np.trace(Sc) / d * np.eye(d)
    Rv, Rvi = sqrtm_sym(Sv), sqrtm_sym(Sv, inv=True)
    A = Rvi @ sqrtm_sym(Rv @ Sc @ Rv) @ Rvi
    A = 0.5 * (A + A.T)
    return mv, mc, A, Sv, Sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "w2_map.npz").exists():
        sys.exit(f"拒绝覆盖已有输出: {out / 'w2_map.npz'}")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "48")))
    U = np.load(U0).reshape(-1, 192)
    P = None
    codes, off = {}, 0
    for lg in ("en", "zh"):
        f = out / f"train_codes_{lg}.npz"
        Y = np.load(DATA / f"train_{lg}" / "y.npy", mmap_mode="r")
        n = Y.shape[0]
        if not f.exists():
            if P is None:
                from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
                from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
                P = load_encoder(BiCodecPhi(a.device), str(CKPT_C))
                P.model.eval()
            S = np.load(DATA / f"train_{lg}" / "s.npy", mmap_mode="r")
            vy, vs = np.zeros((n, 192), np.float32), np.zeros((n, 192), np.float32)
            t0 = time.time()
            for i in range(0, n, BS):
                vy[i:i + BS] = P.phi_wave([np.asarray(r, np.float64) for r in Y[i:i + BS]]).reshape(-1, 192)
                vs[i:i + BS] = P.phi_wave([np.asarray(r, np.float64) for r in S[i:i + BS]]).reshape(-1, 192)
                if (i // BS) % 20 == 0:
                    print(f"{lg} {i + BS}/{n} {time.time() - t0:.0f}s", flush=True)
            tmp = out / f"train_codes_{lg}.partial.npz"
            np.savez(tmp, vy=vy, vs=vs)
            os.replace(tmp, f)
        z = np.load(f)
        codes[lg] = (z["vy"], z["vs"], U[off: off + n])
        off += n
    assert off == len(U), (off, len(U))
    res, M = {}, {}
    for lg, (vy, vs, u) in codes.items():
        mv, mc, A, Sv, Sc = w2_map(np.concatenate([vy, vs]).astype(np.float64), u.astype(np.float64))
        M.update({f"mu_v_{lg}": mv, f"mu_c_{lg}": mc, f"A_{lg}": A})
        ev = np.linalg.eigvalsh(A)
        T = lambda x: mc + (x - mv) @ A  # noqa: E731  (A symmetric)
        Sy = np.cov(vy, rowvar=False)
        res[lg] = {"n_pairs": len(u), "trSv_over_trSc": float(np.trace(Sv) / np.trace(Sc)),
                   "trCov_vy_over_trSc": float(np.trace(Sy) / np.trace(Sc)),
                   "trCov_vs_over_trSc": float(np.trace(np.cov(vs, rowvar=False)) / np.trace(Sc)),
                   "A_eig_min_med_max": [float(ev.min()), float(np.median(ev)), float(ev.max())],
                   "mean_shift_norm": float(np.linalg.norm(mc - mv)),
                   "train_mse_vy_u": float(((vy - u) ** 2).mean()), "train_mse_Tvy_u": float(((T(vy) - u) ** 2).mean()),
                   "train_mse_vs_u": float(((vs - u) ** 2).mean()), "train_mse_Tvs_u": float(((T(vs) - u) ** 2).mean()),
                   "trCov_Tvy_over_trSc": float(np.trace(np.cov(T(vy), rowvar=False)) / np.trace(Sc)),
                   "trCov_Tvs_over_trSc": float(np.trace(np.cov(T(vs), rowvar=False)) / np.trace(Sc))}
    np.savez(out / "w2_map.npz", **M)
    with open(out / "w2_map.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
