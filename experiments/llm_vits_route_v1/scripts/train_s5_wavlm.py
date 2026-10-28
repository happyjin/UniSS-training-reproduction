"""S5: training-only WavLM perceptual discriminator (handbook S5.1-S5.3) on the E0 backbone, DDP.

Only the HiFi-GAN decoder is trained (text encoder, posterior, flow and SDP stay frozen so the decoder drops into
every streaming system unchanged). Discriminators: the original MPD plus a new head on frozen WavLM-Large layers
6/12/18/24 (LayerNorm, 1x1 to 128, mean pool, 2-layer MLP), least squares (S5.2). WavLM parameters are frozen but
the generator graph runs through WavLM and a differentiable 22.05->16 kHz resampler (S5.3 chain rule).
Schedule: head-only warm-up, then alternating D/G with lambda_SLM ramped from 0 to its target.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
import commons  # noqa: E402
from acoustic_common import AlignedDataset, backbone_frames, collate_aligned, load_rows  # noqa: E402
from data import BucketSampler  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch  # noqa: E402
from models import MultiPeriodDiscriminator  # noqa: E402
from stream_common import load_backbone  # noqa: E402
from train_backbone import bf16_forward  # noqa: E402

WAVLM = "/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"


class SLMHead(nn.Module):
  def __init__(self, width, layers, proj=128):
    super().__init__()
    self.layers = layers
    self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in layers])
    self.projs = nn.ModuleList([nn.Linear(width, proj) for _ in layers])
    self.mlp = nn.Sequential(nn.Linear(proj * len(layers), 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

  def forward(self, hidden_states):
    feats = [p(n(hidden_states[l])).mean(1) for l, n, p in zip(self.layers, self.norms, self.projs)]
    return self.mlp(torch.cat(feats, -1)).squeeze(-1)


class WavLMFeatures(nn.Module):
  """Frozen WavLM with a differentiable resampler; returns all hidden states."""

  def __init__(self, path):
    super().__init__()
    from transformers import WavLMModel
    self.m = WavLMModel.from_pretrained(path).eval()
    for p in self.m.parameters():
      p.requires_grad_(False)

  def forward(self, y):
    y = torchaudio.functional.resample(y.squeeze(1), 22050, 16000)
    y = (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-5)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      hs = self.m(y, output_hidden_states=True).hidden_states
    return [h.float() for h in hs]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", default=str(ROOT / "configs/ljs_char_base.json"))
  ap.add_argument("--backbone-dir", required=True, help="dir with G_<step>.pth and D_<step>.pth")
  ap.add_argument("--backbone-step", type=int, default=60000)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--steps", type=int, default=20000)
  ap.add_argument("--head-warmup", type=int, default=1000)
  ap.add_argument("--ramp", type=int, default=5000)
  ap.add_argument("--lambda-slm", type=float, default=0.1)
  ap.add_argument("--layers", default="6,12,18,24")
  ap.add_argument("--seg-frames", type=int, default=172, help="about 2 s crops (S5.3 proposes 2-4 s)")
  ap.add_argument("--batch", type=int, default=16)
  ap.add_argument("--lr-g", type=float, default=2e-5)
  ap.add_argument("--lr-head", type=float, default=1e-4)
  ap.add_argument("--save-every", type=int, default=5000)
  args = ap.parse_args()
  hps = json.load(open(args.config))
  tr, dd = hps["train"], hps["data"]

  dist.init_process_group("nccl")
  rank = dist.get_rank()
  local = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local)
  dev = torch.device("cuda", local)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(1234 + rank)
  run = Path(args.run_dir)
  if rank == 0:
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run / "args.json", "w"), indent=1)

  durs = dict(np.load(Path(args.align_dir) / "durations.npz"))
  rows = [r for r in load_rows(ROOT / "data/manifest_v2.jsonl", {"train"})
          if r["utt_id"] in durs and int(durs[r["utt_id"]].sum()) > args.seg_frames]
  ds = AlignedDataset(rows, durs)
  sampler = BucketSampler(ds.lengths, args.batch, seed=1234)
  loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_aligned, num_workers=8,
                                       pin_memory=True, persistent_workers=True)

  ck = Path(args.backbone_dir) / f"G_{args.backbone_step}.pth"
  net = load_backbone(args.config, str(ck)).to(dev).eval()
  for p in net.parameters():
    p.requires_grad_(False)
  dec = net.dec
  dec.train()
  for p in dec.parameters():
    p.requires_grad_(True)
  mpd = MultiPeriodDiscriminator(False).to(dev)
  mpd.load_state_dict(torch.load(Path(args.backbone_dir) / f"D_{args.backbone_step}.pth", map_location="cpu")["model"])
  bf16_forward(dec)
  bf16_forward(mpd)
  layers = [int(l) for l in args.layers.split(",")]
  wavlm = WavLMFeatures(WAVLM).to(dev)
  head = SLMHead(wavlm.m.config.hidden_size, layers).to(dev)
  dec_ddp = DDP(dec, device_ids=[local])
  mpd_ddp = DDP(mpd, device_ids=[local])
  head_ddp = DDP(head, device_ids=[local])
  opt_g = torch.optim.AdamW(dec.parameters(), args.lr_g, betas=(0.8, 0.99), eps=1e-9)
  opt_d = torch.optim.AdamW(mpd.parameters(), args.lr_g, betas=(0.8, 0.99), eps=1e-9)
  opt_h = torch.optim.AdamW(head.parameters(), args.lr_head, betas=(0.8, 0.99), eps=1e-9)
  seg = args.seg_frames

  step, epoch, t0 = 0, 0, time.time()
  while step < args.steps:
    sampler.set_epoch(epoch)
    for idx, x, xl, wav, wl, d, _ in loader:
      x, xl, wav, wl, d = x.to(dev), xl.to(dev), wav.to(dev), wl.to(dev), d.to(dev)
      fr = backbone_frames(net, x, xl, wav, wl, d, dd)
      z_slice, ids = commons.rand_slice_segments(fr["z"], fr["t"], seg)
      y = commons.slice_segments(wav, ids * dd["hop_length"], seg * dd["hop_length"])
      warm = step < args.head_warmup
      lam = 0.0 if warm else args.lambda_slm * min(1.0, (step - args.head_warmup) / max(1, args.ramp))
      if warm:
        with torch.no_grad():
          y_hat = dec(z_slice)
      else:
        y_hat = dec_ddp(z_slice)

      # discriminators: MPD (after warm-up) and the SLM head, generated audio detached
      hs_r = wavlm(y)
      with torch.no_grad():
        hs_g = wavlm(y_hat.detach())
      loss_h = ((head_ddp(hs_r) - 1) ** 2).mean() + (head_ddp(hs_g) ** 2).mean()
      opt_h.zero_grad(set_to_none=True)
      loss_h.backward()
      opt_h.step()
      if not warm:
        r_, g_, _, _ = mpd_ddp(y, y_hat.detach())
        loss_d, _, _ = discriminator_loss(r_, g_)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # generator: mel + MPD adversarial + feature matching + SLM adversarial through frozen WavLM; the head runs
        # unwrapped (its stray gradients are cleared before its next update)
        r_, g_, fm_r, fm_g = mpd_ddp(y, y_hat)
        mel_kw = dict(n_fft=dd["filter_length"], num_mels=dd["n_mel_channels"], sampling_rate=dd["sampling_rate"],
                      hop_size=dd["hop_length"], win_size=dd["win_length"], fmin=dd["mel_fmin"], fmax=dd["mel_fmax"])
        loss_mel = F.l1_loss(mel_spectrogram_torch(y.squeeze(1), **mel_kw), mel_spectrogram_torch(y_hat.squeeze(1), **mel_kw)) * tr["c_mel"]
        loss_gen, _ = generator_loss(g_)
        loss_fm = feature_loss(fm_r, fm_g)
        loss_slm = ((head(wavlm(y_hat)) - 1) ** 2).mean()
        loss_g = loss_mel + loss_gen + loss_fm + lam * loss_slm
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
      step += 1
      if rank == 0 and step % 100 == 0:
        rec = {"step": step, "head": loss_h.item(), "lambda": lam, "h_per_it": (time.time() - t0) / step / 3600}
        if not warm:
          rec.update(mel=loss_mel.item(), gen=loss_gen.item(), fm=loss_fm.item(), slm=loss_slm.item(), disc=loss_d.item())
        print(json.dumps(rec), flush=True)
      if rank == 0 and (step % args.save_every == 0 or step == args.steps):
        torch.save({"dec": dec.state_dict(), "head": head.state_dict(), "step": step}, run / f"S5_{step}.pth")
      if step >= args.steps:
        break
    epoch += 1
  if rank == 0:
    (run / "DONE").write_text(str(step))
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
