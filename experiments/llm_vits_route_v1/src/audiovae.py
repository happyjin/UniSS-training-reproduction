"""Frozen VoxCPM2 AudioVAE (S3): 16 kHz causal encoder with mu/logvar, 25 Hz x 64 latents, 48 kHz causal decoder.
Needs the tts_schemes environment (voxcpm package)."""
import json
from pathlib import Path

import torch
import torchaudio

VOXCPM2 = Path("/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots")
LAT_HZ, OUT_SR = 25.0, 48000


def load_audiovae(dev):
  from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2
  snap = next(VOXCPM2.iterdir())
  cfg = AudioVAEConfigV2(**json.load(open(snap / "config.json"))["audio_vae_config"])
  vae = AudioVAEV2(config=cfg)
  ck = torch.load(snap / "audiovae.pth", map_location="cpu", weights_only=True)
  missing, unexpected = vae.load_state_dict(ck.get("state_dict", ck), strict=True), None
  vae = vae.to(dev).eval()
  for p in vae.parameters():
    p.requires_grad_(False)
  return vae


@torch.no_grad()
def encode_stats(vae, wav_22k):
  """wav [B, S] at 22.05 kHz -> mu, logvar [B, 64, T_A]."""
  x = torchaudio.functional.resample(wav_22k, 22050, 16000)[:, None]
  x = vae.preprocess(x, 16000)
  out = vae.encoder(x)
  return out["mu"], out["logvar"]


@torch.no_grad()
def decode_22k(vae, u):
  """u [B, 64, T_A] -> waveform [B, S] at 22.05 kHz (decoded at 48 kHz, resampled for the shared evaluation)."""
  y = vae.decode(u)[:, 0]
  return torchaudio.functional.resample(y.float(), OUT_SR, 22050)
