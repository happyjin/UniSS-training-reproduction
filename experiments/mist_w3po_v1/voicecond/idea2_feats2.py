"""Idea 2 M1b (prereg v2 §B): low-dimensional window nuisance summary v~_k for the contraction weight lambda_k.

    python -I experiments/mist_w3po_v1/voicecond/idea2_feats2.py --which pilot|all [--L 2] [--workers 32]

Window k = [k*0.5, k*0.5 + L] s (same grid as idea2_windows.py / idea2_feats.py). From the HDemucs estimates of the
window only (causal within the window): 64-band mel power P_v (vocals) and P_b (background), frame-averaged;
  v~_k = [ 10 log10(sum P_b / sum P_v),                                   window background-to-vocals ratio (dB)
           10 log10(mean_{band in G_g} P_b / mean_{all bands} P_v), g = 1..4 ]   G_g = mel bands 0-15, 16-31, 32-47, 48-63
clipped to [-60, 40] dB. Output data/open_voicecond_v1/idea2/pre/feats2_L<L>/<sid>.npz {end_s [K], vt [K, 5]}.
window_vt(vhat, bhat, ends, L) is reused for the esports demo. Never overwrites.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_diarize import stream_list  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_feats import melpow  # noqa: E402

I2 = Path(__import__("os").environ.get("IDEA2_ROOT", str(REPO / "data/open_voicecond_v1/idea2")))   # IDEA2_ROOT: e.g. the cvt root
SR, HOP = 16000, 0.5


def window_vt(vh: np.ndarray, bh: np.ndarray, ends: np.ndarray, L: float) -> np.ndarray:
    Pv, Pb = melpow(vh), melpow(bh)
    T = min(len(Pv), len(Pb))
    out = np.zeros((len(ends), 5), np.float32)
    for k, e in enumerate(ends):
        a, b = max(0, int(round((e - L) * 100))), min(T, int(round(e * 100)))
        if b <= a:
            continue
        pv, pb = Pv[a:b].mean(0), Pb[a:b].mean(0)
        ref = pv.mean() + 1e-10
        out[k, 0] = 10 * np.log10((pb.sum() + 1e-10) / (pv.sum() + 1e-10))
        for g in range(4):
            out[k, 1 + g] = 10 * np.log10(pb[16 * g:16 * (g + 1)].mean() / ref + 1e-10)
    return np.clip(out, -60, 40)


def one(job):
    sid, L = job
    od = I2 / f"pre/feats2_L{L:g}"
    if (od / f"{sid}.npz").exists():
        return sid, "skip"
    vp, bp = I2 / "pre/demucs" / f"{sid}.vhat.wav", I2 / "pre/demucs" / f"{sid}.bhat.wav"
    if not (vp.exists() and bp.exists()):
        return sid, "missing"
    vh = sf.read(str(vp), dtype="float32")[0]
    bh = sf.read(str(bp), dtype="float32")[0]
    W = int(L * SR)
    K = int((len(vh) - W) // (HOP * SR)) + 1
    ends = np.arange(K) * HOP + L
    vt = window_vt(vh, bh, ends, L)
    od.mkdir(parents=True, exist_ok=True)
    tmp = od / f".{sid}.partial.npz"
    np.savez(tmp, end_s=ends.astype(np.float32), vt=vt)
    os.rename(tmp, od / f"{sid}.npz")
    return sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--L", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(s, a.L) for s in stream_list(a.which)], chunksize=1))
    from collections import Counter
    print(Counter(st for _, st in res))


if __name__ == "__main__":
    main()
