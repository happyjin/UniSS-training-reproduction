"""E0 backbone: official VITS objective (MAS + SDP + flow + HiFi-GAN + MPD) on LJSpeech characters, DDP.

Run with torchrun. The HiFi-GAN decoder and the discriminators run under bf16 autocast; the posterior,
flow, KL and MAS stay in fp32 (handbook 3.2: density math in fp32).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
import commons  # noqa: E402
from data import BucketSampler, TextAudioDataset, collate, load_manifest  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch  # noqa: E402
from models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from text import intersperse, symbols, text_to_ids  # noqa: E402


def to_float(o):
  if torch.is_tensor(o):
    return o.float()
  if isinstance(o, (list, tuple)):
    return type(o)(to_float(t) for t in o)
  return o


def bf16_forward(module):
  orig = module.forward

  def forward(*a, **k):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return to_float(orig(*a, **k))
  module.forward = forward


def build_generator(hps):
  d = hps["data"]
  return SynthesizerTrn(len(symbols), d["filter_length"] // 2 + 1, hps["train"]["segment_size"] // d["hop_length"],
                        **hps["model"])


def latest_ckpt(run_dir, prefix):
  cks = sorted(run_dir.glob(f"{prefix}_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
  return cks[-1] if cks else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", required=True)
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--max-steps", type=int, default=None)
  args = ap.parse_args()
  hps = json.load(open(args.config))
  tr, dd = hps["train"], hps["data"]
  max_steps = args.max_steps or tr["max_steps"]

  dist.init_process_group("nccl")
  rank, world = dist.get_rank(), dist.get_world_size()
  local = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local)
  dev = torch.device("cuda", local)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(tr["seed"] + rank)
  run_dir = Path(args.run_dir)
  if rank == 0:
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(hps, open(run_dir / "config.json", "w"), indent=2)
  dist.barrier()

  rows = load_manifest(ROOT / dd["manifest"], {"train"})
  ds = TextAudioDataset(rows, hop=dd["hop_length"])
  sampler = BucketSampler(ds.lengths, tr["batch_size_per_gpu"], seed=tr["seed"])
  loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=8,
                                       pin_memory=True, persistent_workers=True)

  net_g = build_generator(hps).to(dev)
  net_d = MultiPeriodDiscriminator(hps["model"]["use_spectral_norm"]).to(dev)
  if tr.get("amp_bf16_dec_disc"):
    bf16_forward(net_g.dec)
    bf16_forward(net_d)
  opt_g = torch.optim.AdamW(net_g.parameters(), tr["learning_rate"], betas=tr["betas"], eps=tr["eps"])
  opt_d = torch.optim.AdamW(net_d.parameters(), tr["learning_rate"], betas=tr["betas"], eps=tr["eps"])
  step, epoch = 0, 0
  ck = latest_ckpt(run_dir, "G")
  if ck is not None:
    sg = torch.load(ck, map_location="cpu")
    sd = torch.load(run_dir / ck.name.replace("G_", "D_"), map_location="cpu")
    net_g.load_state_dict(sg["model"]); net_d.load_state_dict(sd["model"])
    opt_g.load_state_dict(sg["optim"]); opt_d.load_state_dict(sd["optim"])
    step, epoch = sg["step"], sg["epoch"]
    if rank == 0:
      print(f"resumed {ck} step={step} epoch={epoch}", flush=True)
  sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, tr["lr_decay"], last_epoch=epoch - 2 if epoch > 1 else -1)
  sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, tr["lr_decay"], last_epoch=epoch - 2 if epoch > 1 else -1)
  net_g = DDP(net_g, device_ids=[local], find_unused_parameters=False)
  net_d = DDP(net_d, device_ids=[local])

  val_rows = load_manifest(ROOT / dd["manifest"], {"val"})[:2]
  t0, log_t = time.time(), time.time()
  while step < max_steps:
    sampler.set_epoch(epoch)
    for idx, x, xl, wav, wl in loader:
      x, xl, wav, wl = x.to(dev, non_blocking=True), xl.to(dev), wav.to(dev, non_blocking=True), wl.to(dev)
      with torch.no_grad():
        spec = spectrogram_torch(wav.squeeze(1), dd["filter_length"], dd["sampling_rate"], dd["hop_length"],
                                 dd["win_length"], center=False)
        sl = wl // dd["hop_length"]
        spec = spec[:, :, :int(sl.max())]
      y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, xl, spec, sl)
      mel = spec_to_mel_torch(spec, dd["filter_length"], dd["n_mel_channels"], dd["sampling_rate"], dd["mel_fmin"],
                              dd["mel_fmax"])
      y_mel = commons.slice_segments(mel, ids_slice, tr["segment_size"] // dd["hop_length"])
      y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), dd["filter_length"], dd["n_mel_channels"],
                                        dd["sampling_rate"], dd["hop_length"], dd["win_length"], dd["mel_fmin"],
                                        dd["mel_fmax"])
      y = commons.slice_segments(wav, ids_slice * dd["hop_length"], tr["segment_size"])

      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
      opt_d.zero_grad(set_to_none=True)
      loss_disc.backward()
      log_now = rank == 0 and (step + 1) % tr["log_every"] == 0
      gn_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), float("inf")) if log_now else 0.0
      opt_d.step()

      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      loss_dur = torch.sum(l_length.float())
      loss_mel = F.l1_loss(y_mel, y_hat_mel) * tr["c_mel"]
      loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * tr["c_kl"]
      loss_fm = feature_loss(fmap_r, fmap_g)
      loss_gen, _ = generator_loss(y_d_hat_g)
      loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
      opt_g.zero_grad(set_to_none=True)
      loss_g.backward()
      gn_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), float("inf")) if log_now else 0.0
      opt_g.step()
      step += 1

      if rank == 0 and step % tr["log_every"] == 0:
        dt = time.time() - log_t
        log_t = time.time()
        print(json.dumps({"step": step, "epoch": epoch, "lr": opt_g.param_groups[0]["lr"], "disc": loss_disc.item(),
                          "gen": loss_gen.item(), "fm": loss_fm.item(), "mel": loss_mel.item(), "dur": loss_dur.item(),
                          "kl": loss_kl.item(), "gn_g": float(gn_g), "gn_d": float(gn_d),
                          "it_s": tr["log_every"] / dt, "elapsed_h": (time.time() - t0) / 3600}), flush=True)
      if rank == 0 and step % tr["save_every"] == 0:
        g_mod, d_mod = net_g.module, net_d.module
        torch.save({"model": g_mod.state_dict(), "optim": opt_g.state_dict(), "step": step, "epoch": epoch},
                   run_dir / f"G_{step}.pth")
        torch.save({"model": d_mod.state_dict(), "optim": opt_d.state_dict(), "step": step, "epoch": epoch},
                   run_dir / f"D_{step}.pth")
        g_mod.eval()
        with torch.no_grad():
          for r in val_rows:
            ids = torch.LongTensor(intersperse(text_to_ids(r["clean"])))[None].to(dev)
            o = g_mod.infer(ids, torch.LongTensor([ids.size(1)]).to(dev), noise_scale=0.667, noise_scale_w=0.8)[0]
            sf.write(run_dir / f"sample_{step}_{r['utt_id']}.wav", o[0, 0].float().cpu().numpy(), dd["sampling_rate"])
        g_mod.train()
      if step >= max_steps:
        break
    epoch += 1
    sch_g.step(); sch_d.step()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
