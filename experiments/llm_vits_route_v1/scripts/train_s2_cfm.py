"""S2-E: train the CFM vector field on frozen E0 posterior latents (handbook S2.2-S2.3), DDP.

Per utterance one window: a target block of 8-48 frames (the size of committed units) after up to 96 clean context
frames; loss = masked MSE between the predicted and the straight-path velocity z1 - z0 on block frames (S2.2).
z1 is a posterior sample (cached statistics keep stochasticity); conditions come from the frozen text encoder + d*.
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
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from acoustic_common import AlignedDataset, backbone_frames, collate_aligned, load_rows  # noqa: E402
from cfm import CFMField  # noqa: E402
from data import BucketSampler  # noqa: E402
from stream_common import load_backbone  # noqa: E402


def windows(z, cond, t, rng, lmin=8, lmax=48, cmax=96):
  """Cut one (context, block) window per utterance; returns padded z, cond, is_ctx, target mask, pad mask."""
  segs = []
  for k in range(z.size(0)):
    tk = int(t[k])
    l = int(rng.integers(lmin, min(lmax, tk) + 1))
    s = int(rng.integers(0, tk - l + 1))
    c = min(s, int(rng.integers(0, cmax + 1)))
    segs.append((k, s - c, s, s + l))
  w = max(e - a for _, a, _, e in segs)
  b = len(segs)
  zz = z.new_zeros(b, w, z.size(1))
  cc = z.new_zeros(b, w, cond.size(1))
  is_ctx = torch.zeros(b, w, device=z.device)
  tgt = torch.zeros(b, w, device=z.device)
  pad = torch.ones(b, w, dtype=torch.bool, device=z.device)
  for i, (k, a, s, e) in enumerate(segs):
    zz[i, :e - a] = z[k, :, a:e].T
    cc[i, :e - a] = cond[k, :, a:e].T
    is_ctx[i, :s - a] = 1
    tgt[i, s - a:e - a] = 1
    pad[i, :e - a] = False
  return zz, cc, is_ctx, tgt, pad


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", default=str(ROOT / "configs/ljs_char_base.json"))
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--steps", type=int, default=60000)
  ap.add_argument("--batch", type=int, default=64)
  ap.add_argument("--lr", type=float, default=2e-4)
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
  rng = np.random.default_rng(1234 + rank)
  run = Path(args.run_dir)
  if rank == 0:
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run / "args.json", "w"), indent=1)
  durs = dict(np.load(Path(args.align_dir) / "durations.npz"))
  rows = [r for r in load_rows(ROOT / "data/manifest_v2.jsonl", {"train"}) if r["utt_id"] in durs]
  ds = AlignedDataset(rows, durs)
  sampler = BucketSampler(ds.lengths, args.batch, seed=1234)
  loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_aligned, num_workers=8,
                                       pin_memory=True, persistent_workers=True)
  net = load_backbone(args.config, args.backbone_ckpt).to(dev).eval()
  for p in net.parameters():
    p.requires_grad_(False)
  field = CFMField().to(dev)
  model = DDP(field, device_ids=[local])
  opt = torch.optim.AdamW(field.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=0.01)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 2000))
  step, epoch, t0 = 0, 0, time.time()
  while step < args.steps:
    sampler.set_epoch(epoch)
    for idx, x, xl, wav, wl, d, _ in loader:
      x, xl, wav, wl, d = x.to(dev), xl.to(dev), wav.to(dev), wl.to(dev), d.to(dev)
      fr = backbone_frames(net, x, xl, wav, wl, d, dd)
      z1, cond, is_ctx, tgt, pad = windows(fr["z"], torch.cat([fr["h_f"], fr["m_f"]], 1), fr["t"], rng)
      z0 = torch.randn_like(z1)
      tau = torch.rand(z1.size(0), device=dev)
      zt = (1 - tau[:, None, None]) * z0 + tau[:, None, None] * z1
      zin = torch.where(is_ctx[..., None] > 0, z1, zt)
      with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(zin, cond, is_ctx, tau, pad)
      err = ((v.float() - (z1 - z0)) ** 2).mean(-1)
      loss = (err * tgt).sum() / tgt.sum()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
      opt.step()
      sched.step()
      step += 1
      if rank == 0 and step % 100 == 0:
        print(json.dumps({"step": step, "cfm": loss.item(), "h_per_1k": (time.time() - t0) / step * 1000 / 3600}), flush=True)
      if rank == 0 and (step % args.save_every == 0 or step == args.steps):
        torch.save({"model": field.state_dict(), "step": step}, run / f"S2_{step}.pth")
      if step >= args.steps:
        break
    epoch += 1
  if rank == 0:
    (run / "DONE").write_text(str(step))
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
