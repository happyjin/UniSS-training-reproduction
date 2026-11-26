"""sst_hsmm_v1 DTC-RL (rl_plan_20261126 §3, prereg change 40 / 41): one policy-gradient iteration of the drafter LoRA.

Policy = wave17 (merged_enc_wise05) + ONE bidirectional LoRA (q/k/v/o of every text layer, r16 a32 = model.build_thinker;
the direction is only an input). Reference = the same module with the adapters off. Each sample is one sampled raw-view
draft at a fork read t (hsc12 / hsc11 rollouts): prompt = model.processor_and_heads()[dir] + the committed prefix (the
exact string vLLM was given), audio = the raw source up to t; action = the sampled continuation token ids (as vLLM
returned them) with their rollout log-probs (old policy); adv = the group-relative advantage from rl_rewards.py.
Loss per sample (GSPO sequence ratio, DAPO clip): s = exp(mean_i(log pi(y_i) - log pi_old(y_i))),
pg = -min(s A, clip(s, 1 - 0.2, 1 + 0.28) A); kl = mean_i k3(pi_ref || pi) (k3 = e^(r) - r - 1, r = log pi_ref - log pi);
step loss = sum over the step's samples of (pg + beta kl) / (number of samples in the step, all ranks).
The iteration's samples are shuffled (seed) and split into --minibatches optimiser steps; gradients all-reduced (sum).
--init: the previous iteration's adapters (+ opt_state.pt beside them, AdamW moments carried over). Saves
<out>/adapters_final.pt (+ .json: rank / alpha for wave16_make_lora) and <out>/opt_state.pt; metrics in <out>/metrics.jsonl.
Refuses to overwrite.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_dtc --samples <jsonl> --out <dir> [--init <adapters.pt>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402
from experiments.uniss_streaming_p2st_traj_v1.training import lora as L  # noqa: E402

CLIP_LO, CLIP_HI = 0.2, 0.28


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_wav(path, cache):
    if path not in cache:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == 16000, path
        cache.clear()
        cache[path] = x.mean(axis=1)
    return cache[path]


def action_logp(thinker, proc, head, s, wav, dev):
    """log pi(y_i | audio <= t, prompt, y_<i) for the sampled continuation tokens (graph kept if grads are on)"""
    inp = proc(text=[head[s["dir"]] + s["prefix"]], audio=[wav[: int(s["t"]) * 16]], return_tensors="pt", padding=True)
    ids = inp["input_ids"]
    P = ids.shape[1]
    cont = torch.tensor([s["tok"]], dtype=ids.dtype)
    inp["input_ids"] = torch.cat([ids, cont], 1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(cont)], 1)
    inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
    lmh = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        hid = thinker(**inp, output_router_logits=False, use_cache=False).logits
    finally:
        thinker.lm_head = lmh
    n = cont.shape[1]
    lsm = torch.log_softmax(lmh(hid[0, P - 1: P - 1 + n]).float(), dim=-1)
    return lsm.gather(-1, cont[0].to(dev)[:, None])[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--max-samples", type=int, default=0, help="smoke: use only the first N samples")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    torch.set_num_threads(4)
    dev = f"cuda:{lr_}"
    out = Path(a.out)
    if rank == 0:
        if (out / "adapters_final.pt").exists():
            sys.exit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    S = [json.loads(x) for x in open(a.samples, encoding="utf-8") if x.strip()]
    if a.max_samples:
        S = S[: a.max_samples]
    random.Random(a.seed).shuffle(S)
    proc, head = MD.processor_and_heads()
    torch.manual_seed(a.seed)
    thinker, names = MD.build_thinker(dev, lora=True)
    layers = [m for m in thinker.modules() if isinstance(m, L.LoRALinear)]
    lora_p = list(names.values())
    if a.init:
        st = torch.load(a.init, map_location="cpu")
        assert set(st) == set(names), "init adapter keys"
        with torch.no_grad():
            for k, p in names.items():
                p.copy_(st[k].to(p.dtype))
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}], betas=(0.9, 0.999), weight_decay=0.0)
    if a.init and (Path(a.init).parent / "opt_state.pt").exists():
        opt.load_state_dict(torch.load(Path(a.init).parent / "opt_state.pt", map_location="cpu"))
        for g_ in opt.param_groups:
            g_["lr"] = a.lr
    mb = [S[i::a.minibatches] for i in range(a.minibatches)]
    mf = open(out / f"metrics_rank{rank}.jsonl", "a") if rank == 0 else None
    cache, t0 = {}, time.time()
    for m, batch in enumerate(mb):
        mine = batch[rank::world]
        nglob = len(batch)
        st_ = torch.zeros(8, device=dev)    # n, pg, kl, ratio mean, clipped, |A|, ratio-1 abs, lp_new - lp_old
        for s in mine:
            wav = load_wav(s["audio"], cache)
            with torch.no_grad(), L.adapters_disabled(layers):
                lref = action_logp(thinker, proc, head, s, wav, dev)
            lnew = action_logp(thinker, proc, head, s, wav, dev)
            lold = torch.tensor(s["lp"], device=dev, dtype=torch.float32)
            assert lnew.shape == lold.shape, (s["id"], lnew.shape, lold.shape)
            A = float(s["adv"])
            ratio = torch.exp((lnew - lold).mean())
            pg = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - CLIP_LO, 1 + CLIP_HI) * A)
            r_ = lref - lnew
            kl = (torch.exp(r_) - r_ - 1).mean()
            ((pg + a.beta * kl) / nglob).backward()
            with torch.no_grad():
                rv = float(ratio)
                st_ += torch.tensor([1.0, float(pg), float(kl), rv, float(rv < 1 - CLIP_LO or rv > 1 + CLIP_HI), abs(A),
                                     abs(rv - 1), float((lnew - lold).mean())], device=dev)
        for p_ in lora_p:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
        flat = torch.cat([p_.grad.reshape(-1) for p_ in lora_p])
        dist.all_reduce(flat)
        k = 0
        for p_ in lora_p:
            n_ = p_.numel()
            p_.grad.copy_(flat[k: k + n_].view_as(p_))
            k += n_
        gn = float(flat.float().norm())
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(st_)
        if rank == 0:
            n = max(float(st_[0]), 1.0)
            rec = {"minibatch": m, "samples": int(st_[0]), "pg": float(st_[1]) / n, "kl": float(st_[2]) / n,
                   "ratio": float(st_[3]) / n, "clipped": float(st_[4]) / n, "abs_adv": float(st_[5]) / n,
                   "abs_ratio_dev": float(st_[6]) / n, "dlogp": float(st_[7]) / n, "grad_norm": gn, "sec": round(time.time() - t0, 1)}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            log(json.dumps(rec))
    if rank == 0:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        torch.save(opt.state_dict(), out / "opt_state.pt")
        json.dump({"method": "DTC-RL iteration (change 41)", "base": str(MD.THINKER), "rank": MD.RANK_LORA, "alpha": MD.ALPHA,
                   "world": world, "lr": a.lr, "beta": a.beta, "minibatches": a.minibatches, "samples": len(S),
                   "init": a.init, "seed": a.seed, "samples_file": a.samples}, open(out / "adapters_final.json", "w"))
        log("saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
