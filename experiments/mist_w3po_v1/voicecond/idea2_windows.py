"""Idea 2: window encodes z_k = phi(window) for the synthetic streams (prereg §2) and the clean reference codes u*.

    python -I experiments/mist_w3po_v1/voicecond/idea2_windows.py win --which pilot|all --L 2 4 --shard 0 --num-shards 4 --device cuda:0
    python -I experiments/mist_w3po_v1/voicecond/idea2_windows.py refs --device cuda:0

win: windows [k*0.5, k*0.5 + L] s (k = 0..), encoded by BiCodecPhi.phi_wave (volume normalisation, tiled to 6 s, exactly
as the tokenizer) on three signals: mix (noisy), speech (clean twin), vhat (HDemucs vocals). Output
data/open_voicecond_v1/idea2/pre/win_L<L>/<sid>.npz {end_s [K], z_mix, z_speech, z_vhat [K, 32, 6] float32}.
refs: refs/<spk>/p*.wav -> u* = mean_p phi(p) (pre-bound), g* = tokens(u*) -> refs/ustar.npz (spk, ustar, gstar,
n_prompts); refuses to overwrite. Existing outputs are skipped, never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_diarize import stream_list  # noqa: E402

I2 = REPO / "data/open_voicecond_v1/idea2"
SR, HOP = 16000, 0.5


def encode_all(P, wavs, bs=256):
    out = []
    for i in range(0, len(wavs), bs):
        out.append(P.phi_wave(wavs[i:i + bs]))
    return np.concatenate(out) if out else np.zeros((0, 32, 6), np.float32)


def cmd_win(a):
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    sids = stream_list(a.which)[a.shard:: a.num_shards]
    print(f"[win] {a.which} L={a.L} shard {a.shard}/{a.num_shards}: {len(sids)} streams on {a.device}", flush=True)
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        todo = [L for L in a.L if not (I2 / f"pre/win_L{L:g}" / f"{sid}.npz").exists()]
        if not todo:
            continue
        sig = {"mix": sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0],
               "speech": sf.read(str(I2 / "streams" / sid / "speech.wav"), dtype="float32")[0]}
        vp = I2 / "pre/demucs" / f"{sid}.vhat.wav"
        if not vp.exists():
            print(f"[win] {sid}: no vhat yet, skipped", flush=True)
            continue
        sig["vhat"] = sf.read(str(vp), dtype="float32")[0]
        n = len(sig["mix"])
        for L in todo:
            W = int(L * SR)
            K = int((n - W) // (HOP * SR)) + 1
            starts = (np.arange(K) * HOP * SR).astype(int)
            res = {"end_s": ((starts + W) / SR).astype(np.float32)}
            for name, x in sig.items():
                res[f"z_{name}"] = encode_all(P, [x[s:s + W] for s in starts])
            od = I2 / f"pre/win_L{L:g}"
            od.mkdir(parents=True, exist_ok=True)
            tmp = od / f".{sid}.partial.npz"
            np.savez(tmp, **res)
            os.rename(tmp, od / f"{sid}.npz")
        print(f"[win] {i}/{len(sids)} {sid} K={K} elapsed={time.time() - t0:.0f}s", flush=True)
    print("[win] DONE", flush=True)


def latest_refs():
    """the newest refs/ustar*.npz (ustar.npz, then ustar_v2.npz, ...), or None"""
    if not (I2 / "refs/ustar.npz").exists():
        return None
    k, p = 2, I2 / "refs/ustar.npz"
    while (I2 / f"refs/ustar_v{k}.npz").exists():
        p, k = I2 / f"refs/ustar_v{k}.npz", k + 1
    return p


def cmd_refs(a):
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    out = latest_refs()
    have = {}
    if out is not None:
        z = np.load(out, allow_pickle=True)
        have = {s: k for k, s in enumerate(z["spk"])}
    dirs = sorted(d for d in (I2 / "refs").iterdir() if d.is_dir() and not d.name.startswith("."))
    infos = [json.load(open(d / "info.json")) for d in dirs]
    new = [(d, inf) for d, inf in zip(dirs, infos) if inf["spk"] not in have]
    if not new:
        print("refs: nothing new")
        return
    P = BiCodecPhi(a.device)
    spk, us = [], []
    for d, inf in new:
        ps = [sf.read(str(d / f"p{k}.wav"), dtype="float32")[0] for k in range(inf["n_prompts"])]
        u = P.phi_wave(ps).mean(0)
        spk.append(inf["spk"]); us.append(u)
    us = np.stack(us)
    gs = P.tokens(us)
    if out is None:
        out = I2 / "refs/ustar.npz"
    else:   # a new version file next to the old one (never overwritten)
        k = 2
        while (I2 / f"refs/ustar_v{k}.npz").exists():
            k += 1
        z = np.load(out, allow_pickle=True)
        spk = list(z["spk"]) + spk
        us = np.concatenate([z["ustar"], us]); gs = np.concatenate([z["gstar"], gs])
        out = I2 / f"refs/ustar_v{k}.npz"
    np.savez(out, spk=np.array(spk), ustar=us.astype(np.float32), gstar=gs)
    print(f"refs: wrote {out} ({len(spk)} speakers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["win", "refs"])
    ap.add_argument("--which", choices=["pilot", "all"], default="pilot")
    ap.add_argument("--L", type=float, nargs="+", default=[2.0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cmd_win(a) if a.cmd == "win" else cmd_refs(a)
