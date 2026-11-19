"""Idea 1 M2: HDemucs vocals for the training pairs on GPU (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v3.3),
batched but numerically the same recipe as evaluation/game_muffle_analysis.demucs(): 16 k -> 44.1 k (torchaudio
resample), mono duplicated to stereo, per-clip mean / std normalisation, hdemucs_high (MUSDB-HQ+), vocals = stem 3
averaged over channels, back to 16 k. Writes <dir>/vhat.npy (float16, N x 96000) and <dir>/sep_stats.npy (N x 4: c, kappa,
e_rel of bh = y - vhat = c b + kappa s + e with b = y - s, and the HDemucs BVR in dB). Refuses existing outputs.
    conda_envs/uniss-eval/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_hdemucs_gpu --dir <train dir> [--bs 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from torchaudio.models import hdemucs_high

CK = Path("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    D = Path(a.dir)
    for f in ("vhat.npy", "sep_stats.npy"):
        if (D / f).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {D / f}")
    dev = torch.device("cuda:0")
    h = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    h.load_state_dict(torch.load(CK / "hdemucs_high_trained.pt", map_location="cpu"))
    h = h.to(dev).eval()
    Y = np.load(D / "y.npy", mmap_mode="r")
    S = np.load(D / "s.npy", mmap_mode="r")
    N, T = Y.shape
    V = np.lib.format.open_memmap(str(D / "vhat.npy.partial"), mode="w+", dtype=np.float16, shape=(N, T))
    st = np.zeros((N, 4), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, a.bs):
            x = torch.from_numpy(np.asarray(Y[i:i + a.bs], dtype=np.float32)).to(dev)
            y = AF.resample(x, 16000, 44100)
            y = torch.stack([y, y], 1)                                   # (B, 2, T44)
            ref = y.mean(1)
            m = ref.mean(-1)[:, None, None]
            sd = ref.std(-1)[:, None, None] + 1e-8
            out = h((y - m) / sd) * sd[:, None] + m[:, None]             # (B, 4, 2, T44)
            voc = out[:, 3].mean(1)
            bg = out[:, :3].sum(1).mean(1)
            bvr = 10 * torch.log10((bg ** 2).sum(-1) + 1e-12) - 10 * torch.log10((voc ** 2).sum(-1) + 1e-12)
            v16 = AF.resample(voc, 44100, 16000)[:, :T]
            if v16.shape[1] < T:
                v16 = torch.nn.functional.pad(v16, (0, T - v16.shape[1]))
            V[i:i + len(x)] = v16.cpu().numpy().astype(np.float16)
            s = torch.from_numpy(np.asarray(S[i:i + a.bs], dtype=np.float32)).to(dev).double()
            xd = x.double()
            bh = xd - v16.double()
            b = xd - s
            X = torch.stack([b, s], -1)                                  # (B, T, 2)
            coef = torch.linalg.lstsq(X, bh[..., None]).solution[..., 0]  # (B, 2)
            e = bh - (X @ coef[..., None])[..., 0]
            erel = e.norm(dim=-1) / (bh.norm(dim=-1) + 1e-12)
            st[i:i + len(x)] = torch.stack([coef[:, 0], coef[:, 1], erel, bvr.double()], 1).cpu().numpy()
            if i % (a.bs * 250) == 0:
                print(f"{i}/{N}", flush=True)
    V.flush()
    del V
    (D / "vhat.npy.partial").rename(D / "vhat.npy")
    np.save(D / "sep_stats.npy", st)
    print(json.dumps({"dir": str(D), "n": N, "c_median": float(np.median(st[:, 0])),
                      "kappa_median": float(np.median(st[:, 1])), "e_rel_median": float(np.median(st[:, 2]))}))


if __name__ == "__main__":
    main()
