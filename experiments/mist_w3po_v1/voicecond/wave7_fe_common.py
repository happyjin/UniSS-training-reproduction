"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): shared pieces of the task-driven
pseudo-clean causal front-end.

  logmel_torch   the Thinker's Whisper log-mel (WhisperFeatureExtractor._torch_extract_fbank_features of the
                 Qwen3-Omni preprocessor: n_fft 400, hop 160, 128 mel, log10 clamp 1e-10, per-item max - 8 floor,
                 (x + 4) / 4), written without detach so that a gradient reaches the waveform; same padding as the
                 processor (padding=True: zero-pad to the longest waveform of the batch, last STFT frame dropped)
  FrontEnd       causal STFT magnitude-mask network: frame 32 ms (n_fft 512) / hop 16 ms (256), periodic Hann,
                 no look-ahead beyond the 32 ms frame (algorithmic latency 32 ms); log-magnitude -> 2 causal conv1d
                 (k = 3, left padded, 384 ch, GELU) -> 2-layer GRU (384) -> linear (zero-initialised) -> mask
                 m = 2 sigmoid(.) in [0, 2] (exactly 1 = identity at initialisation); noisy phase kept; iSTFT by
                 overlap-add (Hann^2 normalised), output trimmed to the input length
  mrstft_l1      multi-resolution STFT magnitude L1 over (n_fft, hop) = (512, 128), (1024, 256), (2048, 512) (Hann)
                 + waveform L1 (the prereg's loss, equal weights)
  mix_single     wave 5's single-layer mixing (voicecond/wave5_m_train.Sampler.sample) with wave-7 seeds: one file of
                 the env's pool (Idea 2 background index split "train", held-out files asserted absent by
                 wave1_data.bg_pools), a segment covering the clip (looped if short), unit RMS, 0.02 s fades, BVR at a
                 U(0.05, 0.95) quantile of the OpenBroadcast-SI BVR values, bench limiter
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SR = 16000
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave7/FE"
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
TGT = {"en2zh": "cmn", "zh2en": "eng"}
IM_START, IM_END, IGNORE = 151644, 151645, -100


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


# ------------------------------------------------------------------------------------------------ differentiable mel
_MEL: dict = {}


def mel_filters(device) -> torch.Tensor:
    key = str(device)
    if key not in _MEL:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(str(QWEN3))
        assert (fe.n_fft, fe.hop_length, fe.feature_size, fe.dither) == (400, 160, 128, 0.0)
        _MEL[key] = torch.from_numpy(fe.mel_filters).to(device, torch.float32)
    return _MEL[key]


def logmel_torch(wavs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """wavs: list of 1-D float32 tensors (may require grad). Returns (features [B, 128, T], attention mask [B, T]) exactly
    as Qwen3OmniMoeProcessor(audio=..., padding=True) builds input_features / feature_attention_mask."""
    dev = wavs[0].device
    L = max(int(w.shape[0]) for w in wavs)
    x = torch.stack([F.pad(w.float(), (0, L - w.shape[0])) for w in wavs])           # [B, L]
    win = torch.hann_window(400, device=dev)
    st = torch.stft(x, 400, 160, window=win, return_complex=True)
    mag = st[..., :-1].abs() ** 2
    mel = mel_filters(dev).T @ mag
    ls = torch.clamp(mel, min=1e-10).log10()
    mx = ls.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    ls = torch.maximum(ls, mx - 8.0)
    ls = (ls + 4.0) / 4.0
    att = torch.zeros(len(wavs), L, dtype=torch.long, device=dev)
    for i, w in enumerate(wavs):
        att[i, : w.shape[0]] = 1
    att = att[:, ::160]
    if L % 160 != 0:
        att = att[:, :-1]
    return ls, att


# --------------------------------------------------------------------------------------------------------- front-end
class FrontEnd(nn.Module):
    NFFT, HOP, CH = 512, 256, 384

    def __init__(self):
        super().__init__()
        nb = self.NFFT // 2 + 1
        self.c1 = nn.Conv1d(nb, self.CH, 3)
        self.c2 = nn.Conv1d(self.CH, self.CH, 3)
        self.gru = nn.GRU(self.CH, self.CH, num_layers=2, batch_first=True)
        self.out = nn.Linear(self.CH, nb)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("win", torch.hann_window(self.NFFT, periodic=True), persistent=False)

    def stft(self, x: torch.Tensor):
        n = x.shape[-1]
        nfr = max(1, math.ceil((n - self.NFFT) / self.HOP) + 1) if n > self.NFFT else 1
        tot = (nfr - 1) * self.HOP + self.NFFT
        x = F.pad(x, (0, tot - n))
        return torch.stft(x, self.NFFT, self.HOP, window=self.win, center=False, return_complex=True), tot

    def mask(self, spec: torch.Tensor) -> torch.Tensor:
        lm = torch.log(spec.abs() + 1e-5)                                   # [B, F, T]
        h = F.gelu(self.c1(F.pad(lm, (2, 0))))                              # causal: left pad k - 1
        h = F.gelu(self.c2(F.pad(h, (2, 0))))
        h, _ = self.gru(h.transpose(1, 2))                                  # [B, T, C]
        return 2.0 * torch.sigmoid(self.out(h)).transpose(1, 2)             # [B, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N] -> [B, N]; output sample t depends on input samples < t + NFFT only within its own frames"""
        n = x.shape[-1]
        spec, tot = self.stft(x)
        y = spec * self.mask(spec)
        out = self.ola(y, tot)
        return out[..., :n]

    def ola(self, spec: torch.Tensor, tot: int) -> torch.Tensor:
        fr = torch.fft.irfft(spec.transpose(1, 2), n=self.NFFT) * self.win  # [B, T, NFFT]
        B, T, _ = fr.shape
        out = F.fold(fr.transpose(1, 2), output_size=(1, tot), kernel_size=(1, self.NFFT), stride=(1, self.HOP))
        wsq = F.fold((self.win ** 2).expand(1, T, -1).transpose(1, 2), output_size=(1, tot),
                     kernel_size=(1, self.NFFT), stride=(1, self.HOP))
        return (out / wsq.clamp_min(1e-8)).reshape(B, tot)


def mrstft_l1(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    loss = (y - t).abs().mean()
    for nfft, hop in ((512, 128), (1024, 256), (2048, 512)):
        w = torch.hann_window(nfft, device=y.device)
        Y = torch.stft(y, nfft, hop, window=w, return_complex=True).abs()
        T = torch.stft(t, nfft, hop, window=w, return_complex=True).abs()
        loss = loss + (Y - T).abs().mean()
    return loss


# ------------------------------------------------------------------------------------------------------------ mixing
class Mixer:
    def __init__(self):
        from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
        self.pools = W1D.bg_pools("train")
        self.bv = W1D.bvr_values()

    def mix(self, x: np.ndarray, env: str, tag: str) -> tuple[np.ndarray, float]:
        from experiments.mist_w3po_v1.voicecond import bench_build as BB
        rng = random.Random(h(f"20261119|w7fe|mix|{tag}"))
        bvr = float(np.quantile(self.bv, rng.uniform(0.05, 0.95)))
        P = self.pools[POOL[env]]
        f = P[int(rng.random() * len(P)) % len(P)]
        b = BB.load(f["file"])
        n = len(x)
        if len(b) < n:
            b = np.tile(b, 1 + n // max(1, len(b)))
        off = int(rng.random() * max(0, len(b) - n))
        seg = BB.fade(BB.unit_rms(b[off: off + n]), 0.02, 0.02)
        es, eb = float((x.astype(np.float64) ** 2).sum()), float((seg.astype(np.float64) ** 2).sum())
        alpha = math.sqrt(10 ** (bvr / 10) * es / max(eb, 1e-12))
        y = x + alpha * seg
        lim = min(1.0, 0.99 / float(np.abs(y).max()))
        return (y * lim).astype(np.float32), lim
