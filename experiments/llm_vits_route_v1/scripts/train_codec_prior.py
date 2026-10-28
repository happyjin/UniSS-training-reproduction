"""Stages B-C of the pretrained-codec schemes on cached targets (tts_schemes environment), DDP.

  --scheme s3   S3Prior on VoxCPM2 AudioVAE latents: u ~ q_A (mu, logvar kept: strict VAE prior, S3.3-S3.4);
                loss = -log p(u | overlap-mapped token conditions), reported with the posterior entropy as KL.
  --scheme s4   Qwen3-TTS 12.5 Hz tokens: token-predictor cross-entropy (S4.3 stage 2) + adapter KL against the
                frozen E0 posterior through the frozen E0 flow (stage 1); from --sample-from the adapter is fed
                predicted tokens with a probability ramped to --sample-max (stage 3, no gradient through samples).
Text conditions come from the frozen E0 encoder on the full text and d*; the codec stays frozen throughout.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from acoustic_common import (AlignedDataset, FRAME_S, backbone_frames, clock_bounds, collate_aligned,  # noqa: E402
                             interval_overlap, load_rows, overlap_matrix, token_bounds)
from codec_priors import S3Prior, TokenAdapter, TokenPredictor  # noqa: E402
from data import BucketSampler  # noqa: E402
from losses import kl_loss  # noqa: E402
from stream_common import load_backbone  # noqa: E402

TOK_HZ, LAT_HZ = 12.5, 25.0


def load_cache(pattern):
  out = {}
  for f in sorted(glob.glob(pattern)):
    out.update(dict(np.load(f)))
  return out


class Extra:
  def __init__(self, scheme, cache, durs):
    self.scheme, self.cache, self.durs = scheme, cache, durs

  def __call__(self, uid):
    d = self.durs[uid]
    c = self.cache[uid]
    if self.scheme == "s3":
      return {"u": torch.from_numpy(c.astype(np.float32)), "o": torch.from_numpy(overlap_matrix(d, LAT_HZ, c.shape[-1]))}
    tb = token_bounds(d)
    o_tok = interval_overlap(clock_bounds(len(c), TOK_HZ), tb)                       # [L, N]
    o_frm = interval_overlap(clock_bounds(int(d.sum()), 1 / FRAME_S), clock_bounds(len(c), TOK_HZ))   # [T, L]
    return {"a": torch.from_numpy(c.astype(np.int64)), "o": torch.from_numpy(o_tok), "o_frm": torch.from_numpy(o_frm)}


def pad_stack(ts, dims):
  shape = [len(ts)] + [max(t.shape[i] for t in ts) for i in range(ts[0].dim())]
  out = ts[0].new_zeros(shape)
  for k, t in enumerate(ts):
    out[(k,) + tuple(slice(0, n) for n in t.shape)] = t
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scheme", required=True, choices=["s3", "s4"])
  ap.add_argument("--config", default=str(ROOT / "configs/ljs_char_base.json"))
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--cache", required=True, help="glob of shard npz files")
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--steps", type=int, default=40000)
  ap.add_argument("--batch", type=int, default=32)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--sample-from", type=int, default=20000)
  ap.add_argument("--sample-max", type=float, default=0.5)
  ap.add_argument("--save-every", type=int, default=10000)
  args = ap.parse_args()
  dd = json.load(open(args.config))["data"]

  dist.init_process_group("nccl")
  rank = dist.get_rank()
  local = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local)
  dev = torch.device("cuda", local)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.manual_seed(1234 + rank)
  run = Path(args.run_dir)
  if rank == 0:
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run / "args.json", "w"), indent=1)

  durs = dict(np.load(Path(args.align_dir) / "durations.npz"))
  cache = load_cache(args.cache)
  rows = [r for r in load_rows(ROOT / "data/manifest_v2.jsonl", {"train"}) if r["utt_id"] in cache]
  ds = AlignedDataset(rows, durs, Extra(args.scheme, cache, durs))
  sampler = BucketSampler(ds.lengths, args.batch, seed=1234)
  loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_aligned, num_workers=8,
                                       pin_memory=True, persistent_workers=True)
  net = load_backbone(args.config, args.backbone_ckpt).to(dev).eval()
  for p in net.parameters():
    p.requires_grad_(False)

  if args.scheme == "s3":
    model = S3Prior().to(dev)
    mods = {"prior": model}
  else:
    pred, adapt = TokenPredictor().to(dev), TokenAdapter().to(dev)
    mods = {"pred": pred, "adapt": adapt}
  ddp = {k: DDP(m, device_ids=[local]) for k, m in mods.items()}
  params = [p for m in mods.values() for p in m.parameters()]
  opt = torch.optim.AdamW(params, args.lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

  step, epoch, t0 = 0, 0, time.time()
  while step < args.steps:
    sampler.set_epoch(epoch)
    for idx, x, xl, wav, wl, d, ex in loader:
      x, xl, wav, wl, d = x.to(dev), xl.to(dev), wav.to(dev), wl.to(dev), d.to(dev)
      fr = backbone_frames(net, x, xl, wav, wl, d, dd, posterior=args.scheme == "s4")
      tok = torch.cat([fr["h"], fr["m_p"]], 1).transpose(1, 2)              # [B, N, 384]
      o = pad_stack([e["o"] for e in ex], 2).to(dev)                        # [B, T_low, N]
      cond = torch.bmm(o, tok)                                              # [B, T_low, 384]
      lmask = (o.sum(-1) > 0).float()[:, None]                              # [B, 1, T_low]
      if args.scheme == "s3":
        st = pad_stack([e["u"] for e in ex], 3).to(dev)                     # [B, 2, 64, T_A]
        mu, lv = st[:, 0], st[:, 1]
        u = (mu + torch.exp(0.5 * lv) * torch.randn_like(mu)) * lmask
        loss = ddp["prior"](u, cond.transpose(1, 2), lmask)
        ent = (0.5 * lv + 0.5 * math.log(2 * math.pi * math.e)) * lmask
        rec = {"nll": loss.item(), "kl": loss.item() - (ent.sum() / lmask.sum()).item()}
      else:
        a = pad_stack([e["a"] for e in ex], 2).to(dev)                       # [B, L, 16]
        logits = ddp["pred"](cond, TokenPredictor.shift(a, pred.vocab))
        ce = F.cross_entropy(logits.reshape(-1, pred.vocab), a.reshape(-1), reduction="none").view(a.shape)
        ce = (ce * lmask[:, 0, :, None]).sum() / (lmask.sum() * a.size(-1))
        a_in = a
        p_s = 0.0 if step < args.sample_from else args.sample_max * min(1.0, (step - args.sample_from) / max(1, args.steps - args.sample_from))
        if p_s > 0:
          with torch.no_grad():
            sampled = torch.distributions.Categorical(logits=logits.float()).sample()
          keep = (torch.rand(a.shape[:2], device=dev) < p_s)[..., None]
          a_in = torch.where(keep, sampled, a)
        o_frm = pad_stack([e["o_frm"] for e in ex], 2).to(dev)
        m2, l2 = ddp["adapt"](a_in, o_frm, fr["h_f"], fr["m_f"], fr["logs_f"])
        with torch.no_grad():
          z_p = net.flow(fr["z"], fr["y_mask"])
        kl = kl_loss(z_p, fr["logs_q"], m2, l2, fr["y_mask"])
        with torch.no_grad():
          kl0 = kl_loss(z_p, fr["logs_q"], fr["m_f"], fr["logs_f"], fr["y_mask"])
        loss = ce + kl
        rec = {"ce": ce.item(), "kl": kl.item(), "kl_e0_prior": kl0.item(), "p_sample": p_s}
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(params, 5.0)
      opt.step()
      sched.step()
      step += 1
      if rank == 0 and step % 100 == 0:
        print(json.dumps({"step": step, **rec, "lr": sched.get_last_lr()[0], "h_per_1k": (time.time() - t0) / step * 1000 / 3600}), flush=True)
      if rank == 0 and (step % args.save_every == 0 or step == args.steps):
        torch.save({k: m.state_dict() for k, m in mods.items()} | {"step": step, "scheme": args.scheme},
                   run / f"{args.scheme}_{step}.pth")
      if step >= args.steps:
        break
    epoch += 1
  if rank == 0:
    (run / "DONE").write_text(str(step))
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
