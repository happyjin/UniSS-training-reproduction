"""sst_hsmm_v1 prereg change 22: CR-LA = counterfactual-regret local-agreement training of a bidirectional LoRA on wave17.
Per sample (a crla regret row: wave17 LA2 rollout on an open UniST p4 mixture, its non-final commits k with marginal regret r_k
scored by prefix-forced full-audio continuation): teacher-forced prefix forward at every read c (audio <= c s) over the
ROLLOUT's final translation + <|im_end|>; WAIT masked; tokens of commit k (read c_k = t_k / 1000) get the LA2 commit
probability a_j = p_{c_k}(y_j) p_{c_k - 1}(y_j); loss = - sum_k A_k mean_j a_j / K + kappa KL(p_theta || p_wave17),
A_k = clip(r_k / 100, -0.5, 0.5), kappa = 0.1. 1000 steps, 16 slots, AdamW lr 1e-4, 50 warm-up, cosine. Saves adapters_final.pt.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_crla --regret <regret.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
from experiments.uniss_streaming_p2st_traj_v1.training import lora as L  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
LR, KAPPA, ACLIP = 1e-4, 0.1, 0.5
DIRS = ("en2zh", "zh2en")
TOKMS = {"en2zh": 345.6, "zh2en": 332.5}
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")     # = e2e_si_thinker.PUNCT
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
    ap.add_argument("--regret", required=True)
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
    pool = [x for x in map(json.loads, open(a.regret, encoding="utf-8")) if x["events"]]
    import random
    random.Random(SEED).shuffle(pool)
    proc, head = MD.processor_and_heads()
    tk = proc.tokenizer
    torch.manual_seed(SEED)                               # identical LoRA init on every rank
    thinker, names = MD.build_thinker(dev, lora=True)
    layers = [m for m in thinker.modules() if isinstance(m, L.LoRALinear)]
    assert len(layers) * 2 == len(names), (len(layers), len(names))
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": LR}], betas=(0.9, 0.999), weight_decay=0.0)

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
        stats = torch.zeros(4, device=dev)       # samples, skipped, events, -
        dstat = torch.zeros(2, 7, device=dev)    # per dir: n, pg term, KL, a on harmful commits, a on helpful, n harmful, n helpful
        rows, wavs = [], []
        for j in my_slots:
            x = pool[(s * SLOTS + j) % len(pool)]
            ids_ = tk(x["translation"], add_special_tokens=False)["input_ids"]
            ends = [len(tk.decode(ids_[: q_ + 1])) for q_ in range(len(ids_))]
            X = int(x["dur_ms"])
            spans, lo = [], 0
            for e in x["events"]:
                hi = len(e["pre"])
                js = [q_ for q_, en in enumerate(ends) if lo < en <= hi]
                c = e["t"] // 1000                                   # read c (1-based); lsm rows c-1 (this read), c-2 (previous)
                if js and 2 <= c <= -(-X // 1000) - 1:
                    spans.append((c - 1, js, max(-ACLIP, min(ACLIP, e["r"] / 100.0))))
                lo = max(lo, hi)
            rows.append({"dir": x["dir"], "ref": x["translation"], "tok": ids_, "C": -(-X // 1000), "X": X, "spans": spans})
            w = sf.read(x["wav"], dtype="float32", always_2d=True)[0].mean(axis=1)
            wavs.append(w)
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
                lsm, b = lsm_at(thinker, hid, off, pos, dev)
                ids = inp["input_ids"][b, pos.to(dev) + 1]
                p = lsm.exp()
                kl = (p * (lsm - base[jj])).sum(-1).mean()
                py = p.gather(-1, ids[..., None])[..., 0]                      # (C, n1)
                pg = torch.zeros((), device=dev)
                h_a, h_n, g_a, g_n = 0.0, 0, 0.0, 0
                for c0, js, A in r["spans"]:
                    jt = torch.tensor(js, device=dev)
                    am = (py[c0, jt] * py[c0 - 1, jt]).mean()
                    pg = pg - A * am / len(r["spans"])
                    if A < 0:
                        h_a += float(am); h_n += 1
                    elif A > 0:
                        g_a += float(am); g_n += 1
                terms.append((pg, kl))
                k_ = DIRS.index(d)
                dstat[k_] += torch.tensor([1.0, float(pg), float(kl), h_a, g_a, h_n, g_n], device=dev)
                stats[0] += 1; stats[2] += len(r["spans"])
        total = sum((pg + KAPPA * kl) / SLOTS for pg, kl in terms) if terms else None
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
            rec = {"step": s, "lr": LR * lr_at(s), "samples": int(stats[0]), "skipped": int(stats[1]), "events": int(stats[2]),
                   "grad_norm": gn, "sec": round(time.time() - t0, 1),
                   **{f"{d}_{k}": float(dstat[i_, 1 + j_] / max(float(dstat[i_, 0]), 1)) for i_, d in enumerate(DIRS)
                      for j_, k in enumerate(("pg", "KL"))},
                   **{f"{d}_a_harm": float(dstat[i_, 3] / max(float(dstat[i_, 5]), 1)) for i_, d in enumerate(DIRS)},
                   **{f"{d}_a_help": float(dstat[i_, 4] / max(float(dstat[i_, 6]), 1)) for i_, d in enumerate(DIRS)}}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / f"adapters_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"method": "CR-LA (change 22)", "base": str(MD.THINKER), "steps": STEPS, "rank": MD.RANK_LORA, "alpha": MD.ALPHA,
                   "world": world, "kappa": KAPPA, "aclip": ACLIP, "lr": LR, "seed": SEED, "regret": a.regret, "pool": len(pool)},
                  open(out / "adapters_final.json", "w"))
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
