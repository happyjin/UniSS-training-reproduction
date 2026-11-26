"""sst_hsmm_v1 prereg change 37-4: SAP2 = train_sap.py (not modified) with data2 rows (80000 / direction) and p5sap2 samples (risk_prep2: e1 rows
excluded), 4000 steps, checkpoints every 1000, output adapters as before. Was: sst_hsmm_v1 prereg change 24: SAP = supported-prefix training of a bidirectional LoRA on wave17 (pure LA2 / combo inference).
Per sample (the p4 noisy UniST mixtures, step s slot j, reference row with the speech-timeline support chunk g_i of every token):
teacher-forced prefix forward at every read c = 1..C (audio <= c s) over the REFERENCE + <|im_end|>; WAIT masked;
targets at read c: the k_c = #{i : g_i <= c} supported reference tokens, then <|im_end|> at position k_c (read C: the whole
reference + <|im_end|>); loss = mean token cross-entropy over all reads x target positions. So a draft from a partial read
is the supported reference prefix and stops (no anticipated tail), and LA2's agreement of consecutive drafts approximates the
supported prefix. 1000 steps, 16 slots, AdamW lr 1e-4 (LoRA), 50 warm-up steps, cosine, seed 20261125. Saves adapters_final.pt.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_sap --out eval_outputs/sst_hsmm_v1/t/train
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

DATA = REPO / "eval_outputs/sst_hsmm_v1/data2"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p5sap2"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 1000, 4000
LR = 1e-4
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
        stats = torch.zeros(4, device=dev)       # samples, skipped, target positions, -
        dstat = torch.zeros(2, 5, device=dev)    # per dir: n, CE partial reads, CE final read, P(stop) at the support boundary, KL
        rows, wavs = [], []
        for j in my_slots:
            x = samples[(s, j)]
            rr = dict(refs[x["dir"]][x["row"]])
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
                ids = inp["input_ids"][b, pos.to(dev) + 1].clone()            # (C, n1): y_1..y_n, <|im_end|>
                gt = torch.tensor(r["g"], device=dev)
                k = (gt[None, :] <= torch.arange(1, C + 1, device=dev)[:, None]).sum(1)   # supported tokens per read
                k[-1] = n_                                                     # final read: the whole reference
                col = torch.arange(n1, device=dev)[None, :]
                mask = col <= k[:, None]
                ids[col.expand(C, n1) == k[:, None]] = MD.IM_END               # stop right after the supported prefix
                nll = -lsm.gather(-1, ids[..., None])[..., 0]
                ce = (nll * mask).sum() / mask.sum()
                with torch.no_grad():
                    kl = ((lsm.exp() * (lsm - base[jj])).sum(-1) * mask).sum() / mask.sum()
                    cep = float((nll[:-1] * mask[:-1]).sum() / mask[:-1].sum().clamp_min(1)) if C > 1 else 0.0
                    cef = float(nll[-1].mean())
                    pst = float(torch.exp(-nll[:-1].gather(1, k[:-1, None].clamp_max(n_)))[:, 0].mean()) if C > 1 else 0.0
                terms.append(ce)
                k_ = DIRS.index(d)
                dstat[k_] += torch.tensor([1.0, cep, cef, pst, float(kl)], device=dev)
                stats[0] += 1; stats[2] += float(mask.sum())
        total = sum(ce / SLOTS for ce in terms) if terms else None
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
            rec = {"step": s, "lr": LR * lr_at(s), "samples": int(stats[0]), "skipped": int(stats[1]),
                   "grad_norm": gn, "sec": round(time.time() - t0, 1),
                   **{f"{d}_{k}": float(dstat[i_, 1 + j_] / max(float(dstat[i_, 0]), 1)) for i_, d in enumerate(DIRS)
                      for j_, k in enumerate(("ce_partial", "ce_final", "p_stop", "KL"))}}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / f"adapters_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"method": "SAP2 (change 37-4)", "base": str(MD.THINKER), "steps": STEPS, "rank": MD.RANK_LORA, "alpha": MD.ALPHA,
                   "world": world, "lr": LR, "seed": SEED},
                  open(out / "adapters_final.json", "w"))
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
