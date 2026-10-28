"""T4: prefix-consistent causal acoustic flow and causal waveform decoder (handbook T4.1-T4.5), DDP.

  --part flow  causal mean-only coupling flow (T4.3 with a = 0, causal conditioners) trained with the VITS KL
               against the frozen posterior; prior statistics are the frozen text encoder's, expanded with d*.
  --part dec   causal HiFi-GAN: every convolution left-padded, every transposed convolution cropped on the right,
               initialised from the E0 decoder and fine-tuned with mel + MPD + feature matching on posterior z.
Both parts are prefix invariant: appending frames never changes earlier outputs (T4.2), and their receptive
field bounds the memory a streaming renderer needs (T4.4 with exact replay inside the field).
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
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
import commons  # noqa: E402
from acoustic_common import AlignedDataset, CausalFlow, backbone_frames, collate_aligned, load_rows  # noqa: E402
from data import BucketSampler  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch  # noqa: E402
from models import MultiPeriodDiscriminator  # noqa: E402
from stream_common import load_backbone  # noqa: E402
from train_backbone import bf16_forward  # noqa: E402


class LeftPad(nn.Module):
  def __init__(self, conv):
    super().__init__()
    self.pad = conv.dilation[0] * (conv.kernel_size[0] - 1)
    conv.padding = (0,)
    self.conv = conv

  def forward(self, x):
    return self.conv(F.pad(x, (self.pad, 0)))


class RightCrop(nn.Module):
  def __init__(self, convt):
    super().__init__()
    convt.padding = (0,)
    self.convt, self.u = convt, convt.stride[0]

  def forward(self, x):
    return self.convt(x)[..., :x.size(-1) * self.u]


def make_causal(dec):
  """In place: the official Generator becomes causal at the sample level (weights kept)."""
  def swap(parent):
    for name, child in parent.named_children():
      if isinstance(child, nn.ConvTranspose1d):
        setattr(parent, name, RightCrop(child))
      elif isinstance(child, nn.Conv1d):
        setattr(parent, name, LeftPad(child))
      else:
        swap(child)
  swap(dec)
  return dec


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", required=True, choices=["flow", "dec"])
  ap.add_argument("--config", default=str(ROOT / "configs/ljs_char_base.json"))
  ap.add_argument("--backbone-dir", required=True)
  ap.add_argument("--backbone-step", type=int, default=60000)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--steps", type=int, default=30000)
  ap.add_argument("--batch", type=int, default=32)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--seg-frames", type=int, default=32)
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
  net = load_backbone(args.config, str(Path(args.backbone_dir) / f"G_{args.backbone_step}.pth")).to(dev).eval()
  for p in net.parameters():
    p.requires_grad_(False)

  if args.part == "flow":
    flow = CausalFlow().to(dev)
    model = DDP(flow, device_ids=[local])
    opt = torch.optim.AdamW(flow.parameters(), args.lr, betas=(0.8, 0.99), eps=1e-9)
  else:
    dec = make_causal(net.dec).train()
    for p in dec.parameters():
      p.requires_grad_(True)
    mpd = MultiPeriodDiscriminator(False).to(dev)
    mpd.load_state_dict(torch.load(Path(args.backbone_dir) / f"D_{args.backbone_step}.pth", map_location="cpu")["model"])
    bf16_forward(dec)
    bf16_forward(mpd)
    model, mpd_ddp = DDP(dec, device_ids=[local]), DDP(mpd, device_ids=[local])
    opt = torch.optim.AdamW(dec.parameters(), args.lr, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(mpd.parameters(), args.lr, betas=(0.8, 0.99), eps=1e-9)
    mel_kw = dict(n_fft=dd["filter_length"], num_mels=dd["n_mel_channels"], sampling_rate=dd["sampling_rate"],
                  hop_size=dd["hop_length"], win_size=dd["win_length"], fmin=dd["mel_fmin"], fmax=dd["mel_fmax"])

  step, epoch, t0 = 0, 0, time.time()
  while step < args.steps:
    sampler.set_epoch(epoch)
    for idx, x, xl, wav, wl, d, _ in loader:
      x, xl, wav, wl, d = x.to(dev), xl.to(dev), wav.to(dev), wl.to(dev), d.to(dev)
      fr = backbone_frames(net, x, xl, wav, wl, d, dd)
      if args.part == "flow":
        z_p = model(fr["z"], fr["y_mask"])
        loss = kl_loss(z_p, fr["logs_q"], fr["m_f"], fr["logs_f"], fr["y_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {"kl": loss.item()}
      else:
        z_slice, ids = commons.rand_slice_segments(fr["z"], fr["t"], args.seg_frames)
        y = commons.slice_segments(wav, ids * dd["hop_length"], args.seg_frames * dd["hop_length"])
        y_hat = model(z_slice)
        r_, g_, _, _ = mpd_ddp(y, y_hat.detach())
        loss_d, _, _ = discriminator_loss(r_, g_)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        r_, g_, fm_r, fm_g = mpd_ddp(y, y_hat)
        loss_mel = F.l1_loss(mel_spectrogram_torch(y.squeeze(1), **mel_kw),
                             mel_spectrogram_torch(y_hat.squeeze(1), **mel_kw)) * tr["c_mel"]
        loss_gen, _ = generator_loss(g_)
        loss_fm = feature_loss(fm_r, fm_g)
        loss = loss_mel + loss_gen + loss_fm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {"mel": loss_mel.item(), "gen": loss_gen.item(), "fm": loss_fm.item(), "disc": loss_d.item()}
      step += 1
      if rank == 0 and step % 100 == 0:
        print(json.dumps({"step": step, **rec, "h_per_1k": (time.time() - t0) / step * 1000 / 3600}), flush=True)
      if rank == 0 and (step % args.save_every == 0 or step == args.steps):
        mod = model.module
        torch.save({"part": args.part, "model": mod.state_dict(), "step": step}, run / f"T4{args.part}_{step}.pth")
      if step >= args.steps:
        break
    epoch += 1
  if rank == 0:
    (run / "DONE").write_text(str(step))
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
