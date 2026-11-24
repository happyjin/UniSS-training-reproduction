"""sst_hsmm_v1 P11 (prereg change 19): = train_head.py (P10) with WriteHead2 on 10 inputs: head_feats (5) + prev_feats
(agreement of the greedy token with the previous chunk's greedy token at the same prefix, its log-prob, known flag) + (z_L, z_P)
of the sampled multipliers. Same targets (p10/own_targets.jsonl), samples, lambda draws (same keys), steps, optimiser.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_head2 --targets eval_outputs/sst_hsmm_v1/p10/own_targets.jsonl --out eval_outputs/sst_hsmm_v1/p11/train
Was: sst_hsmm_v1 P10 (prereg change 18): FROZEN translator (wave17 merged_enc_wise05, no LoRA) + a trained write head.
= train_risk6 (P7': multiplier-conditioned Gate2, lam_L, lam_P ~ log-uniform [0.003, 10], loss q/n + lam_L * L + lam_P * P on
hsmm.lattice_risk_block with OB = 2 bins, beta 1 -> 4, 1000 steps, 16 slots, same p4 samples / mixtures, AdamW lr 1e-3, 50
warm-up steps, cosine) with two changes:
  1. the lattice's l(c, i) = WriteHead(headfeat.head_feats(frozen log-softmax at chunk c, position i)) instead of the LoRA
     model's log(1 - P_wait) - log P_wait (lp_wait := logsigmoid(-head)); trained: WriteHead + Gate2 (bias0 = 0);
  2. target sequences = the frozen model's own full-audio translations of the same mixtures (gen_own_targets.py, p10/own_targets.jsonl)
     instead of the P1a model's streaming rollouts; dis (q cost) = the frozen model's greedy token at chunk c != y_i.
The Thinker runs under no_grad (nothing in it is trained). Saves head_final.pt, gate_final.pt, gate_final.json.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_head --targets eval_outputs/sst_hsmm_v1/p10/own_targets.jsonl --out eval_outputs/sst_hsmm_v1/p10/train
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.sst_hsmm_v1 import hsmm as H  # noqa: E402
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402
from experiments.sst_hsmm_v1 import proxy as PX  # noqa: E402
from experiments.sst_hsmm_v1.headfeat import WriteHead2, head_feats, prev_feats  # noqa: E402

H.LAM_HI = 10.0                                   # = train_risk6 (P7')
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
BETA0, BETA1 = 1.0, 4.0
DIRS = ("en2zh", "zh2en")
TOKMS = {"en2zh": 345.6, "zh2en": 332.5}
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")     # = e2e_si_thinker.PUNCT
SEED = 20261124


def lam_of(key):
    import hashlib
    u = (int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 ** 12) / 10 ** 12
    return math.exp(math.log(H.LAM_LO) + u * (math.log(H.LAM_HI) - math.log(H.LAM_LO)))


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def own_row(ref_row, text, tok, punct_ids):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pu = [int(i in punct_ids) for i in ids]
    d = int(round(TOKMS[ref_row["dir"]] / 100))
    return {"dir": ref_row["dir"], "ref": text, "tok": ids, "C": ref_row["C"], "X": ref_row["X"],
            "g": [1] * len(ids), "dbin": [0 if p else d for p in pu], "punct": pu}


@torch.no_grad()
def feats_multi(thinker, inp, meta, dev, punct_idx):
    """frozen forward for prefix_batch_multi's batch -> [(row index, feats (C, n, 5) cpu, dis (C, n) cpu, n_shape)]"""
    inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
    head = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        out = thinker(**inp, output_router_logits=False, use_cache=False)
    finally:
        thinker.lm_head = head
    hid = out.logits
    res = []
    for j, off, pos in meta:
        C, n = pos.shape
        b = (off + torch.arange(C, device=dev))[:, None].expand(C, n)
        p = pos.to(dev)
        lsm = torch.log_softmax(head(hid[b, p]).float(), dim=-1)
        ids = inp["input_ids"][b, p + 1]
        mx = lsm.clone()
        mx[..., MD.WAIT] = -float("inf")
        dis = (mx.argmax(-1) != ids).float()
        res.append((j, torch.cat([head_feats(lsm, punct_idx), prev_feats(lsm)], -1).cpu(), dis.cpu()))
        del lsm, mx
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
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
        if (out / "gate_final.pt").exists() or (out / "head_final.pt").exists():
            sys.exit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.manual_seed(SEED)
    refs = {d: {r["id"]: r for r in map(json.loads, open(DATA / f"train_{d}.jsonl", encoding="utf-8"))} for d in DIRS}
    samples = {(x["s"], x["j"]): x for x in map(json.loads, open(P4 / "samples.jsonl", encoding="utf-8"))}
    own = {x["key"]: x["translation"] for x in map(json.loads, open(a.targets, encoding="utf-8"))}
    proc, head = MD.processor_and_heads()
    tk = proc.tokenizer
    punct_ids = {i for i in range(len(tk)) if PUNCT.match(tk.decode([i]) or "x")}
    punct_idx = torch.tensor(sorted(punct_ids))
    thinker, _ = MD.build_thinker(dev, lora=False)
    whead = WriteHead2()
    gate = H.Gate2(0.0)
    params = list(whead.parameters()) + list(gate.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-3}], betas=(0.9, 0.999), weight_decay=0.0)
    base_lr = [1e-3]

    def lr_at(s):
        return s / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, STEPS - WARM)))

    my_slots = [j for j in range(SLOTS) if j % world == rank]
    mf = open(out / f"metrics_rank{rank}.jsonl", "a")
    t0 = time.time()
    for s in range(STEPS):
        if a.max_steps_run and s >= a.max_steps_run:
            break
        for g_, b in zip(opt.param_groups, base_lr):
            g_["lr"] = b * lr_at(s)
        beta = BETA0 + (BETA1 - BETA0) * s / max(1, STEPS - 1)
        stats = torch.zeros(6, device=dev)       # risk sum, -, samples, skipped, own tokens, -
        dstat = torch.zeros(2, 7, device=dev)    # per dir: n, log lamL, log lamP, q, L, P, mean write prob of the head alone
        orows, wavs = [], []
        for j in my_slots:
            x = samples[(s, j)]
            text = own.get(f"s{s:04d}_j{j:02d}", "")
            if not text.strip():
                stats[3] += 1
                continue
            rr = refs[x["dir"]][x["row"]]
            orows.append(own_row(rr, text, tk, punct_ids))
            orows[-1]["lam"] = (lam_of(f"{SEED}|p10|lamL|{s}|{j}"), lam_of(f"{SEED}|p10|lamP|{s}|{j}"))
            wavs.append(sf.read(x["wav"], dtype="float32")[0])
        total = 0.0
        if orows:
            inp, meta = MD.prefix_batch_multi(proc, head, orows, wavs)
            stats[3] += len(orows) - len(meta)
            for jj, ft, dis in (feats_multi(thinker, inp, meta, dev, punct_idx) if meta else []):
                r = orows[jj]
                d = r["dir"]
                n_ = len(r["tok"])
                lL, lP = r["lam"]
                gate.z = (H.lam_z(lL), H.lam_z(lP))
                zz = torch.tensor([H.lam_z(lL), H.lam_z(lP)], dtype=ft.dtype).expand(*ft.shape[:-1], 2)
                hv = whead(torch.cat([ft, zz], -1))                  # (C, n)
                lw = F.logsigmoid(-hv)                               # l(c, i) = log(1 - P_w) - log P_w = hv
                Eq, EL, EP = H.lattice_risk_block(dis, lw, dis, r["g"] + [r["C"]], r["dbin"] + [0],
                                                  r["punct"] + [1], r["X"], gate, PX.OB, beta=beta, eos=True)
                X = r["X"] / 1000.0
                Lm = EL / n_ - X * (n_ - 1) / (2 * n_)            # LAAL form (same as proxy.item_proxy)
                q = Eq / n_
                loss = q + lL * Lm + lP * EP
                total = total + loss / SLOTS
                k_ = DIRS.index(d)
                dstat[k_] += torch.tensor([1.0, math.log(lL), math.log(lP), float(q), float(Lm), float(EP),
                                           float(torch.sigmoid(hv.detach()).mean())], device=dev)
                stats[0] += float(loss); stats[2] += 1; stats[4] += n_
        if torch.is_tensor(total) and total.requires_grad:
            total.backward()
        gg = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch.cat([x_.reshape(-1) for x_ in gg]).to(dev)
        dist.all_reduce(flat)
        k = 0
        for p in params:
            n_ = p.numel()
            p.grad = flat[k: k + n_].view_as(p).cpu()
            k += n_
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        dist.all_reduce(dstat)
        if rank == 0:
            rec = {"step": s, "lr": base_lr[0] * lr_at(s), "beta": round(beta, 3),
                   "risk": float(stats[0] / max(stats[2], 1)), "samples": int(stats[2]), "skipped": int(stats[3]),
                   "grad_norm": gn, "gate_bias": float(gate.net[2].bias), "head_bias": float(whead.net[2].bias),
                   "head_skip": [round(float(x), 4) for x in whead.skip.weight.view(-1)],
                   "gate_skip": [round(float(x), 4) for x in gate.skip.weight.view(-1)], "sec": round(time.time() - t0, 1),
                   **{f"{d}_{k}": float(dstat[i_, 1 + j_] / max(float(dstat[i_, 0]), 1)) for i_, d in enumerate(DIRS)
                      for j_, k in enumerate(("loglamL", "loglamP", "q", "L", "P", "pw_head"))}}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({"step": s + 1, "gate": gate.state_dict(), "head": whead.state_dict()}, out / f"ckpt_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        json.dump({"translator": str(MD.THINKER), "frozen": True, "targets": a.targets, "steps": STEPS, "world": world,
                   "lam_range": [H.LAM_LO, H.LAM_HI], "gate": "Gate2", "head": "WriteHead2(headfeat + prev_feats + z)", "ob": PX.OB,
                   "beta": [BETA0, BETA1]}, open(out / "gate_final.json", "w"))
        torch.save(gate.state_dict(), out / "gate_final.pt")
        torch.save(whead.state_dict(), out / "head_final.pt")
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
