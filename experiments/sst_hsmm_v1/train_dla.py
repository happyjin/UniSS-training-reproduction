"""sst_hsmm_v1 prereg change 21: Differentiable Local Agreement (DLA) training of a bidirectional LoRA on the N translator.
Thinker = wave17 merged_enc_wise05 + a FRESH LoRA (model.build_thinker: q/k/v/o r16 a32, b = 0 -> starts at wave17).
Per sample (the p4 noisy UniST mixtures, step s slot j, reference row; g = 1 everywhere: no alignment constraint):
  teacher-forced prefix forward at every read c (audio <= c s) over the REFERENCE + <|im_end|>; WAIT masked (-1e4);
  p_c = softmax; agreement a(c, i) = sum_v p_c(v) p_{c-1}(v) (read 0: no previous draft -> no commit);
  wrong-agreement e(c, i) = a - p_c(y_i) p_{c-1}(y_i); per-write cost e / a;
  dla.lattice_risk_dla(lp_wait = log(1 - a), lcost = log(e / a), ...) -> E[q], E[L] on the speech timeline (OB 2 bins);
  loss = E[q] / n + lam_L * (E[L] / n - X (n - 1) / (2 n)) + kappa * KL(p_theta || p_wave17) (mean over reads x positions,
  adapters disabled for p_wave17), kappa = 1.
lam_L fixed by a rule at step 0: mean(q/n) / mean(L) over the step-0 samples of all ranks (equal weight at the start).
1000 steps, 16 slots, AdamW lr 1e-4 (LoRA), 50 warm-up steps, cosine. Saves adapters_final.pt (+ json).
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_dla --out eval_outputs/sst_hsmm_v1/q/train
"""
from __future__ import annotations

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
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402
from experiments.sst_hsmm_v1 import proxy as PX  # noqa: E402
from experiments.sst_hsmm_v1.dla import lattice_risk_dla  # noqa: E402
from experiments.uniss_streaming_p2st_traj_v1.training import lora as L  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
LR, KAPPA = 1e-4, 1.0
DIRS = ("en2zh", "zh2en")
SEED = 20261125


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def hidden(thinker, inp):
    head = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        out = thinker(**inp, output_router_logits=False, use_cache=False)
    finally:
        thinker.lm_head = head
    return out.logits


def lsm_at(thinker, hid, off, pos, dev):
    C, n = pos.shape
    b = (off + torch.arange(C, device=dev))[:, None].expand(C, n)
    z = thinker.lm_head(hid[b, pos.to(dev)]).float()
    z = z.masked_fill(torch.arange(z.shape[-1], device=dev) == MD.WAIT, -1e4)
    return torch.log_softmax(z, dim=-1), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps-run", type=int, default=0, help="smoke: stop after this many steps (no final save)")
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
    refs = {d: {r["id"]: r for r in map(json.loads, open(DATA / f"train_{d}.jsonl", encoding="utf-8"))} for d in DIRS}
    samples = {(x["s"], x["j"]): x for x in map(json.loads, open(P4 / "samples.jsonl", encoding="utf-8"))}
    proc, head = MD.processor_and_heads()
    torch.manual_seed(SEED)                               # identical LoRA init on every rank
    thinker, names = MD.build_thinker(dev, lora=True)
    layers = [m for m in thinker.modules() if isinstance(m, L.LoRALinear)]
    assert len(layers) * 2 == len(names), (len(layers), len(names))
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": LR}], betas=(0.9, 0.999), weight_decay=0.0)
    lam_L = None

    def lr_at(s):
        return s / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, STEPS - WARM)))

    my_slots = [j for j in range(SLOTS) if j % world == rank]
    mf = open(out / f"metrics_rank{rank}.jsonl", "a")
    t0 = time.time()
    for s in range(STEPS):
        if a.max_steps_run and s >= a.max_steps_run:
            break
        for g_ in opt.param_groups:
            g_["lr"] = LR * lr_at(s)
        stats = torch.zeros(4, device=dev)       # samples, skipped, tokens, -
        dstat = torch.zeros(2, 7, device=dev)    # per dir: n, q, L, KL, mean agreement (reads >= 1), greedy-wrong rate, P
        rows, wavs = [], []
        for j in my_slots:
            x = samples[(s, j)]
            rr = dict(refs[x["dir"]][x["row"]])
            rr["g"] = [1] * len(rr["tok"])
            rows.append(rr)
            wavs.append(sf.read(x["wav"], dtype="float32")[0])
        terms = []
        inp, meta = MD.prefix_batch_multi(proc, head, rows, wavs)
        stats[1] += len(rows) - len(meta)
        if meta:
            inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.no_grad(), L.adapters_disabled(layers):
                hid0 = hidden(thinker, inp)
                base = {jj: lsm_at(thinker, hid0, off, pos, dev)[0] for jj, off, pos in meta}
            del hid0
            hid = hidden(thinker, inp)
            for jj, off, pos in meta:
                r = rows[jj]
                d = r["dir"]
                C, n1 = pos.shape
                n_ = n1 - 1
                lsm, b = lsm_at(thinker, hid, off, pos, dev)
                ids = inp["input_ids"][b, pos.to(dev) + 1]
                p = lsm.exp()
                kl = (p * (lsm - base[jj])).sum(-1).mean()
                py = p.gather(-1, ids[..., None])[..., 0]
                agree = (p[1:] * p[:-1]).sum(-1).clamp(1e-9, 1 - 1e-6)            # (C-1, n1)
                wrong = (agree - py[1:] * py[:-1]).clamp_min(1e-12)
                lp_wait = torch.cat([torch.zeros(1, n1, device=dev), torch.log1p(-agree)], 0)
                lcost = torch.cat([torch.zeros(1, n1, device=dev), torch.log(wrong) - torch.log(agree)], 0)
                Eq, EL, EP = lattice_risk_dla(lp_wait.cpu(), lcost.cpu(), r["g"] + [r["C"]], r["dbin"] + [0], r["punct"] + [1],
                                              r["X"], PX.OB, eos=True)
                X = r["X"] / 1000.0
                Lm = EL / n_ - X * (n_ - 1) / (2 * n_)
                q = Eq / n_
                terms.append((q, Lm, kl))
                with torch.no_grad():
                    gw = (lsm.argmax(-1) != ids).float().mean()
                k_ = DIRS.index(d)
                dstat[k_] += torch.tensor([1.0, float(q), float(Lm), float(kl), float(agree[:, :n_].mean()) if C > 1 else 0.0,
                                           float(gw), float(EP)], device=dev)
                stats[0] += 1; stats[2] += n_
        if lam_L is None:                                  # rule (change 21): equal weight of q/n and L at step 0
            tq = torch.tensor([sum(float(t[0]) for t in terms), sum(float(t[1]) for t in terms), len(terms)], device=dev)
            dist.all_reduce(tq)
            lam_L = float(tq[0] / tq[1])
            if rank == 0:
                log(f"lam_L = {lam_L:.5f} (mean q/n {float(tq[0] / tq[2]):.4f}, mean L {float(tq[1] / tq[2]):.4f})")
        total = sum(((q + lam_L * Lm).to(dev) + KAPPA * kl) / SLOTS for q, Lm, kl in terms) if terms else None
        if total is not None:
            total.backward()
        for p_ in lora_p:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
        flat = torch.cat([p_.grad.reshape(-1) for p_ in lora_p])
        dist.all_reduce(flat)
        k = 0
        for p_ in lora_p:
            m_ = p_.numel()
            p_.grad.copy_(flat[k: k + m_].view_as(p_))
            k += m_
        gn = float(flat.float().norm())
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        dist.all_reduce(dstat)
        if rank == 0:
            rec = {"step": s, "lr": LR * lr_at(s), "lam_L": lam_L, "samples": int(stats[0]), "skipped": int(stats[1]),
                   "grad_norm": gn, "sec": round(time.time() - t0, 1),
                   **{f"{d}_{k}": float(dstat[i_, 1 + j_] / max(float(dstat[i_, 0]), 1)) for i_, d in enumerate(DIRS)
                      for j_, k in enumerate(("q", "L", "KL", "agree", "greedy_wrong", "P"))}}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / f"adapters_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"method": "DLA (change 21)", "base": str(MD.THINKER), "steps": STEPS, "rank": MD.RANK_LORA, "alpha": MD.ALPHA,
                   "world": world, "lam_L": lam_L, "kappa": KAPPA, "lr": LR, "ob": PX.OB, "seed": SEED},
                  open(out / "adapters_final.json", "w"))
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
