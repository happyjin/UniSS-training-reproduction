"""Idea 2 E2.2: EM fit of the linear CA-Kalman on the synthetic TRAIN streams only (prereg §4.2).

    python -I experiments/mist_w3po_v1/voicecond/idea2_fit.py --which pilot|all --e20 <e20 dir> [--device cuda:0]

Non-overlapping windows (every (L/0.5)-th window of the 0.5 s grid); z = phi(mix window), v, diarizer slot / pi.
Init (fixed): A, B from the top-r SVD of the window cross-covariance Cov(z, v) (speaker windows), mu_nb = B^+ mean(v),
Sig_nb = I, rho = 0.9, S = Var(v), R = clean within-speaker window covariance (E2.0), mu0 / Sig0 from E2.0.
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/em_<which>/params.npz + fit.log; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_cakalman import Params, fit_em  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_common import OUTD, load_stream, sids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--e20", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    od = OUTD / f"em_{a.which}"
    if (od / "params.npz").exists():
        sys.exit(f"refuse to overwrite {od}/params.npz")
    od.mkdir(parents=True, exist_ok=True)
    logf = open(od / "fit.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    e20 = json.load(open(Path(a.e20) / "e20.json"))
    L, r = float(e20["L_chosen"]), int(e20["r"])
    pr = np.load(Path(a.e20) / f"prior_L{L:g}.npz")
    step = int(round(L / 0.5))
    streams, Zs, Vs = [], [], []
    for sid in sids(a.which, "train"):
        d = load_stream(sid, L)
        k = np.arange(0, len(d["end"]), step)
        z, v, sl, pi = d["z_mix"][k], d["v"][k].astype(np.float64), d["slot"][k], d["pi"][k].astype(np.float64)
        streams.append((z, v, sl, pi))
        Zs.append(z[sl >= 0]); Vs.append(v[sl >= 0])
    Z, V = np.concatenate(Zs), np.concatenate(Vs)
    Vall = np.concatenate([s[1] for s in streams])
    C = (Z - Z.mean(0)).T @ (V - V.mean(0)) / len(Z)
    U, sv, Wt = np.linalg.svd(C, full_matrices=False)
    A0 = U[:, :r] * np.sqrt(sv[:r] / 2)
    B0 = Wt[:r].T * np.sqrt(sv[:r] / 2)
    mu_nb = np.linalg.lstsq(B0, Vall.mean(0), rcond=None)[0]
    p = Params(A=A0, B=B0, S=Vall.var(0) + 1e-6, R=pr["R0"], rho=0.9, mu_nb=mu_nb, Sig_nb=np.eye(r), mu0=pr["mu0"],
               Sig0=pr["Sig0"], L=L)
    log(f"EM fit ({a.which}): {len(streams)} train streams, L={L:g}, r={r}, windows {sum(len(s[0]) for s in streams)},"
        f" speaker windows {len(Z)}; init |A| {np.linalg.norm(A0):.3f} |B| {np.linalg.norm(B0):.3f}")
    t0 = time.time()
    p = fit_em(streams, p, torch.device(a.device), iters=a.iters, log=log)
    p.save(od / "params.npz")
    hist = np.array(p.hist)
    mono = bool(np.all(np.diff(hist) >= -1e-6 * np.abs(hist[1:])))
    log(f"done in {time.time() - t0:.0f}s; loglik monotone: {mono}; rho {p.rho:.4f}")


if __name__ == "__main__":
    main()
