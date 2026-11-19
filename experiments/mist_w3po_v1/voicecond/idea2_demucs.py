"""Idea 2: HDemucs background / vocals estimates for the synthetic streams (prereg §2).

Same model, normalisation and stems as experiments/mist_w3po_v1/evaluation/game_muffle_analysis.demucs():
hdemucs_high (MUSDB18-HQ+), input 44.1 kHz with the mono signal duplicated to stereo, (y - mean) / std; vocals = v_hat,
drums + bass + other = b_hat; both resampled back to 16 kHz. Whole streams are processed in 10 s segments with 1 s
linear crossfades (overlap-add). Output data/open_voicecond_v1/idea2/pre/demucs/<sid>.{vhat,bhat}.wav (PCM16); a stream
whose files exist is skipped, never rewritten.

    python -I experiments/mist_w3po_v1/voicecond/idea2_demucs.py --which pilot|all --shard 0 --num-shards 4 [--device cuda:0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_diarize import stream_list  # noqa: E402

I2 = Path(__import__("os").environ.get("IDEA2_ROOT", str(REPO / "data/open_voicecond_v1/idea2")))   # IDEA2_ROOT: e.g. the cvt root
CK = Path("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints")
SR, HS = 16000, 44100


@torch.no_grad()
def separate(model, y16: np.ndarray, dev, seg_s=10.0, ov_s=1.0, bs=8):
    y = AF.resample(torch.from_numpy(y16).to(dev), SR, HS)
    m, s = y.mean(), y.std() + 1e-8
    y = (y - m) / s
    N = y.shape[0]
    seg, ov = int(seg_s * HS), int(ov_s * HS)
    hop = seg - ov
    starts = list(range(0, max(1, N - ov), hop))
    out = torch.zeros(4, N, device=dev)
    wsum = torch.zeros(N, device=dev)
    ramp = torch.linspace(0, 1, ov, device=dev)
    for b0 in range(0, len(starts), bs):
        chunk = []
        for st in starts[b0:b0 + bs]:
            x = y[st:st + seg]
            if x.shape[0] < seg:
                x = torch.nn.functional.pad(x, (0, seg - x.shape[0]))
            chunk.append(torch.stack([x, x]))
        res = model(torch.stack(chunk))                      # [B, 4, 2, seg]
        for k, st in enumerate(starts[b0:b0 + bs]):
            L = min(seg, N - st)
            w = torch.ones(seg, device=dev)
            if st > 0:
                w[:ov] = ramp
            if st + seg < N:
                w[seg - ov:] = ramp.flip(0)
            w = w[:L]
            out[:, st:st + L] += res[k, :, :, :L].mean(1) * w
            wsum[st:st + L] += w
    out = out / wsum.clamp_min(1e-6) * s + m
    voc = AF.resample(out[3], HS, SR)
    bg = AF.resample(out[:3].sum(0), HS, SR)
    n = len(y16)
    fix = lambda z: torch.nn.functional.pad(z, (0, max(0, n - z.shape[0])))[:n].cpu().numpy()
    return fix(voc), fix(bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from torchaudio.models import hdemucs_high
    dev = torch.device(a.device)
    model = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load(CK / "hdemucs_high_trained.pt", map_location="cpu"))
    model = model.eval().to(dev)
    out = I2 / "pre/demucs"
    out.mkdir(parents=True, exist_ok=True)
    sids = stream_list(a.which)[a.shard:: a.num_shards]
    print(f"[demucs] {a.which} shard {a.shard}/{a.num_shards}: {len(sids)} streams on {dev}", flush=True)
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        if (out / f"{sid}.vhat.wav").exists() and (out / f"{sid}.bhat.wav").exists():
            continue
        y, sr = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")
        assert sr == SR
        voc, bg = separate(model, y, dev)
        for name, x in (("vhat", voc), ("bhat", bg)):
            tmp = out / f".{sid}.{name}.partial.wav"
            sf.write(str(tmp), np.clip(x, -1, 1), SR, subtype="PCM_16")
            os.rename(tmp, out / f"{sid}.{name}.wav")
        print(f"[demucs] {i}/{len(sids)} {sid} {len(y) / SR:.0f}s elapsed={time.time() - t0:.0f}s", flush=True)
    print("[demucs] DONE", flush=True)


if __name__ == "__main__":
    main()
