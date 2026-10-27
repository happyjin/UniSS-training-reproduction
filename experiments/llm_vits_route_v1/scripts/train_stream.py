"""Fine-tune a streaming variant (E1 / E2 / S6 family) on the frozen E0 backbone.

Loss per commit step: acoustic KL on the block's frames (posterior from the full recording, prior from the visible
prefix) + SDP duration NLL bound on the block's tokens + beta * KL(q(r) || p(r)) for S6. The raw negative bound
(beta = 1) is logged next to the optimised surrogate (handbook S6.3).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from mel_processing import spectrogram_torch  # noqa: E402
from stream_common import load_backbone  # noqa: E402
from stream_data import StreamDataset, assemble, collate, load_cands  # noqa: E402
from stream_model import StreamVITS  # noqa: E402

VARIANTS = {
  "e1": {},
  "e2": {"use_qwen": True},
  "s6_future": {"use_qwen": True, "use_future": True, "use_r": True},
  "s6_rho": {"use_qwen": True, "use_rho": True},
  "s6": {"use_qwen": True, "use_future": True, "use_r": True, "use_rho": True},
}


def build(variant, backbone_config, backbone_ckpt):
  return StreamVITS(load_backbone(backbone_config, backbone_ckpt), **VARIANTS[variant])


def step_losses(model, batch, dev, dd, r_mode):
  b, inst = assemble(batch, dev)
  u = b["inst_utt"]
  n = u.numel()
  with torch.no_grad():
    spec = spectrogram_torch(b["wav"].squeeze(1), dd["filter_length"], dd["sampling_rate"], dd["hop_length"],
                             dd["win_length"])
    sl = b["wav_len"] // dd["hop_length"]
    spec = spec[:, :, :int(sl.max())]
    z, m_q, logs_q, y_mask = model.frozen["enc_q"](spec, sl)
    z_p = model.frozen["flow"](z, y_mask)
  fut = model.future_features(b["cand"], b["cand_len"], n, inst["blk"]) if model.use_future else None
  x, m_p, logs_p, x_mask, dur_cond, kl_r = model.forward_step(
      inst["ids"], inst["lens"], inst["blk"], qfeat=inst.get("qfeat"), q_last_idx=inst["last"], fut=fut,
      hist=b["hist"], hist_lens=b["hist_len"], w=inst["w"], r_mode=r_mode)
  nll = model.dp(x, inst["blk"], w=inst["w"], cond=dur_cond)
  n_tok = inst["blk"].sum()
  # acoustic KL on block frames
  nf = b["f1"] - b["f0"]
  f_max = int(nf.max())
  ar = torch.arange(f_max, device=dev)[None]
  fmask = (ar < nf[:, None]).float()[:, None]
  fi = (b["f0"][:, None] + ar).clamp_max(z_p.size(-1) - 1)
  cum = torch.cumsum(b["dur"][u], -1)
  tok = torch.searchsorted(cum, fi, right=True).clamp_max(m_p.size(-1) - 1)
  gi = fi[:, None].expand(-1, z_p.size(1), -1)
  gt = tok[:, None].expand(-1, m_p.size(1), -1)
  zp_b = torch.gather(z_p[u], 2, gi)
  lq_b = torch.gather(logs_q[u], 2, gi)
  mp_b = torch.gather(m_p, 2, gt)
  lp_b = torch.gather(logs_p, 2, gt)
  kl = lp_b - lq_b - 0.5 + 0.5 * (zp_b - mp_b) ** 2 * torch.exp(-2.0 * lp_b)
  kl_frames = (kl * fmask).sum()
  return {"kl_ac": kl_frames / fmask.sum(), "dur": nll.sum() / n_tok, "kl_r": kl_r.sum() / n_tok,
          "neg_bound_per_tok": (kl_frames + nll.sum() + kl_r.sum()) / n_tok}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", required=True, choices=list(VARIANTS))
  ap.add_argument("--backbone-config", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--qwen-dir", default=str(ROOT / "data/qwen_prefix"))
  ap.add_argument("--cands", default=str(ROOT / "data/future_cands/shard*.jsonl"))
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--steps", type=int, default=20000)
  ap.add_argument("--batch-utts", type=int, default=32)
  ap.add_argument("--per-utt", type=int, default=4)
  ap.add_argument("--lr", type=float, default=1e-4)
  ap.add_argument("--lr-pretrained", type=float, default=5e-5)
  ap.add_argument("--kl-r-warmup", type=int, default=2000)
  ap.add_argument("--val-every", type=int, default=2000)
  ap.add_argument("--seed", type=int, default=17)
  args = ap.parse_args()
  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  dev = torch.device("cuda")
  run = Path(args.run_dir)
  run.mkdir(parents=True, exist_ok=True)
  if list(run.glob("stream_*.pth")):
    raise FileExistsError(f"{run} already has checkpoints")
  json.dump(vars(args), open(run / "args.json", "w"), indent=2)
  dd = json.load(open(args.backbone_config))["data"]
  flags = VARIANTS[args.variant]
  model = build(args.variant, args.backbone_config, args.backbone_ckpt).to(dev)
  rows = [json.loads(l) for l in open(args.manifest)]
  durations = dict(np.load(Path(args.align_dir) / "durations.npz"))
  prosody = {d["utt_id"]: [u for u in d["units"]] for d in map(json.loads, open(Path(args.align_dir) / "unit_prosody.jsonl"))}
  cands = load_cands(sorted(glob.glob(args.cands))) if flags.get("use_future") else None
  qdir = args.qwen_dir if flags.get("use_qwen") else None
  mk = lambda split, fixed, seed: StreamDataset([r for r in rows if r["split"] == split], durations, prosody, qdir,
                                                cands, args.per_utt, seed=seed, fixed=fixed)
  train_ds, val_ds = mk("train", False, args.seed), mk("val", True, 1)
  loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_utts, shuffle=True, collate_fn=collate,
                                       num_workers=12, pin_memory=True, drop_last=True, persistent_workers=True)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_utts, collate_fn=collate, num_workers=4)
  pre = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("enc_p.") or n.startswith("dp."))]
  new = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("enc_p.") or n.startswith("dp."))]
  opt = torch.optim.AdamW([{"params": pre, "lr": args.lr_pretrained}, {"params": new, "lr": args.lr, "weight_decay": 0.01}],
                          betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0)
  print(json.dumps({"variant": args.variant, "flags": flags, "pretrained_params": sum(p.numel() for p in pre),
                    "new_params": sum(p.numel() for p in new)}), flush=True)
  step, t0 = 0, time.time()
  best = None
  while step < args.steps:
    for batch in loader:
      model.train()
      beta = min(1.0, step / max(1, args.kl_r_warmup))
      L = step_losses(model, batch, dev, dd, "posterior")
      loss = L["kl_ac"] + L["dur"] + beta * L["kl_r"]
      opt.zero_grad(set_to_none=True)
      loss.backward()
      gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
      opt.step()
      step += 1
      if step % 100 == 0:
        print(json.dumps({"step": step, **{k: round(v.item(), 4) for k, v in L.items()}, "beta": beta,
                          "gn": round(gn.item(), 3), "s_per_it": round((time.time() - t0) / step, 3)}), flush=True)
      if step % args.val_every == 0 or step == args.steps:
        model.eval()
        acc = {}
        with torch.no_grad():
          torch.manual_seed(0)
          for mode in ["posterior", "prior"]:
            tot = {}
            for vb in val_loader:
              for k, v in step_losses(model, vb, dev, dd, mode).items():
                tot[k] = tot.get(k, 0.0) + v.item()
            acc[mode] = {k: v / len(val_loader) for k, v in tot.items()}
        sel = acc["prior"]["kl_ac"] + acc["prior"]["dur"]
        state = {n: p for n, p in model.state_dict().items() if not n.startswith("frozen.")}
        torch.save({"model": state, "variant": args.variant, "flags": flags, "step": step, "val": acc},
                   run / f"stream_{step}.pth")
        if best is None or sel < best[0]:
          best = (sel, step)
        print(json.dumps({"val_step": step, "val": acc, "select": sel, "best": best}), flush=True)
      if step >= args.steps:
        break
  json.dump({"best_step": best[1], "best_select": best[0]}, open(run / "best.json", "w"))


if __name__ == "__main__":
  main()
