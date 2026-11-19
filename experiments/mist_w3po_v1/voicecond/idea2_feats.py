"""Idea 2: per-window nuisance features v_k, diarizer slot j(k) / weight pi_k, and oracle labels (prereg §2, §3).

    python -I experiments/mist_w3po_v1/voicecond/idea2_feats.py --which pilot|all --L 2 [4] [--workers 32]

Window k = [k*0.5, k*0.5 + L] s (same grid as idea2_windows.py). Per stream writes
data/open_voicecond_v1/idea2/pre/feats_L<L>/<sid>.npz:
  end_s [K]; v [K, 64]   v_k = (P_mel(b_hat)_band / P_bar_k)^0.3, P_bar_k = mean v_hat mel power (band-averaged) over all
                          frames up to the window end where max_i p_i > 0.5 (none yet: the window's own mean)
  slot [K] (argmax_i sum_t p_i, -1 if pi < 0.3), pi [K] = mean_t p_j * prod_{i!=j} (1 - p_i), slot_raw [K], pi_raw [K]
  cov [K, J] oracle fraction of the window covered by each true speaker (meta speaker order); single [K] index of the
  only active true speaker if he covers >= 90% of the window and nobody else speaks, else -1
  slot2spk [n_slots] true-speaker index matched to each slot by maximal 80 ms overlap (-1 unmatched; evaluation only)
Never overwrites.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_diarize import stream_list  # noqa: E402

I2 = Path(__import__("os").environ.get("IDEA2_ROOT", str(REPO / "data/open_voicecond_v1/idea2")))   # IDEA2_ROOT: e.g. the cvt root
SR, HOP, NM = 16000, 0.5, 64
PI_MIN = 0.3
_FB = {}


def melpow(x: np.ndarray) -> np.ndarray:
    """[T10, 64] mel power (n_fft 512, win 400, hop 160, 0-8 kHz)"""
    import torch
    import torchaudio.functional as AF
    torch.set_num_threads(2)
    if "fb" not in _FB:
        _FB["fb"] = AF.melscale_fbanks(257, 0.0, 8000.0, NM, SR, norm="slaney", mel_scale="slaney")
        _FB["win"] = torch.hann_window(400)
    X = torch.stft(torch.from_numpy(x), 512, 160, 400, _FB["win"], center=True, return_complex=True)
    P = (X.abs() ** 2).T @ _FB["fb"]
    return P.numpy().astype(np.float64)


def one(job):
    sid, L = job
    od = I2 / f"pre/feats_L{L:g}"
    if (od / f"{sid}.npz").exists():
        return sid, "skip"
    import torch
    dp = I2 / "pre/diar" / f"{sid}.post.pt"
    vp, bp = I2 / "pre/demucs" / f"{sid}.vhat.wav", I2 / "pre/demucs" / f"{sid}.bhat.wav"
    if not (dp.exists() and vp.exists() and bp.exists()):
        return sid, "missing inputs"
    meta = json.load(open(I2 / "streams" / sid / "meta.json"))
    post = torch.load(dp, weights_only=False)["post80"].float().numpy()            # [T80, S]
    vh = sf.read(str(vp), dtype="float32")[0]
    bh = sf.read(str(bp), dtype="float32")[0]
    n = len(vh)
    W = int(L * SR)
    K = int((n - W) // (HOP * SR)) + 1
    st = np.arange(K) * HOP
    en = st + L
    Pv, Pb = melpow(vh), melpow(bh)                                                  # [T10, 64]
    T10 = min(len(Pv), len(Pb))
    ev = Pv[:T10].mean(1)
    act80 = post.max(1) > 0.5
    f10 = np.arange(T10)
    act10 = act80[np.minimum(f10 // 8, len(act80) - 1)] if len(act80) else np.zeros(T10, bool)
    cs_e = np.concatenate([[0.0], np.cumsum(ev * act10)])
    cs_n = np.concatenate([[0], np.cumsum(act10)])
    v = np.zeros((K, NM), np.float32)
    for k in range(K):
        a, b = int(st[k] * 100), min(T10, int(en[k] * 100))
        pb = Pb[a:b].mean(0)
        nb = cs_n[b]
        pbar = cs_e[b] / nb if nb > 0 else ev[a:b].mean()
        v[k] = (pb / max(pbar, 1e-10)) ** 0.3
    # diarizer slot and weight
    S = post.shape[1]
    slot_raw = np.zeros(K, np.int64)
    pi_raw = np.zeros(K, np.float32)
    for k in range(K):
        a, b = int(round(st[k] / 0.08)), min(len(post), int(round(en[k] / 0.08)))
        p = post[a:b]
        if len(p) == 0:
            slot_raw[k] = 0
            continue
        j = int(p.sum(0).argmax())
        others = np.prod(1 - np.delete(p, j, axis=1), axis=1)
        slot_raw[k] = j
        pi_raw[k] = float((p[:, j] * others).mean())
    slot = np.where(pi_raw >= PI_MIN, slot_raw, -1)
    # oracle
    J = len(meta["speakers"])
    sidx = {s: i for i, s in enumerate(meta["speakers"])}
    T100 = int(np.ceil(n / SR * 100)) + 1
    lab = np.zeros((J, T100), bool)
    for g in meta["segments"]:
        lab[sidx[g["spk"]], int(g["start"] * 100): int(g["end"] * 100)] = True
    cl = np.concatenate([np.zeros((J, 1)), np.cumsum(lab, 1)], 1)
    cov = np.zeros((K, J), np.float32)
    for k in range(K):
        a, b = int(st[k] * 100), int(en[k] * 100)
        cov[k] = (cl[:, b] - cl[:, a]) / (b - a)
    single = np.full(K, -1, np.int64)
    for k in range(K):
        nz = np.nonzero(cov[k] > 0.0)[0]
        if len(nz) == 1 and cov[k, nz[0]] >= 0.9:
            single[k] = nz[0]
    # slot -> true speaker by maximal overlap at 80 ms
    T80 = len(post)
    lab80 = lab[:, : T80 * 8].reshape(J, -1, 8).mean(2) > 0.5 if T80 * 8 <= T100 else lab[:, ::8][:, :T80]
    T80b = min(T80, lab80.shape[1])
    ov = (post[:T80b] > 0.5).T.astype(np.float64) @ lab80[:, :T80b].T.astype(np.float64)   # [S, J]
    slot2spk = np.where(ov.max(1) > 0, ov.argmax(1), -1)
    od.mkdir(parents=True, exist_ok=True)
    tmp = od / f".{sid}.partial.npz"
    np.savez(tmp, end_s=en.astype(np.float32), v=v, slot=slot, pi=np.where(slot >= 0, pi_raw, 0).astype(np.float32),
             slot_raw=slot_raw, pi_raw=pi_raw, cov=cov, single=single, slot2spk=slot2spk, ov=ov.astype(np.float32))
    os.rename(tmp, od / f"{sid}.npz")
    return sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--L", type=float, nargs="+", default=[2.0])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    jobs = [(s, L) for s in stream_list(a.which) for L in a.L]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=1))
    from collections import Counter
    print(Counter(st for _, st in res))
    bad = [s for s, st in res if st not in ("ok", "skip")]
    if bad:
        print("not done:", bad[:10])


if __name__ == "__main__":
    main()
