"""sst_hsmm_v1 P4 (reports/sst_hsmm_v1/prereg.zh-CN.md change 7): exact expected-risk training of the write policy on the
policy's OWN outputs, warm-started from the change-5 P1a (adapters_final + gate_final), 8 ranks (torchrun).

Step s, slot j (rank r takes j % world == r): the P4 sample (risk_prep: row, mixed wav) and its rollout output Y^ (P1a,
threshold rule). Policy loss: hsmm.lattice_risk on Y^ + <|im_end|> (g_i = 1, end token g = C; durations = the direction's
mean token duration, punctuation 0; lam 0.5 / 0.5; beta = 1 -> 4 linearly over the steps), emissions detached inside
(gradients reach the wait logit and the gate only), / n(Y^). Translation loss: token NLL of the reference + <|im_end|>
with the whole audio, / n(ref). loss = sum over the step's samples of (policy + translation) / 16; one backward; LoRA
gradients all-reduced as one flat buffer, gate gradients likewise; AdamW (LoRA 1e-4, gate 1e-3), 50 warmup steps, cosine
to 0, no clipping. Saves ckpt every 250 steps; adapters_final.pt (+ .json) and gate_final.pt in P1a's format (merge.py).
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_risk --out eval_outputs/sst_hsmm_v1/p4/train
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.sst_hsmm_v1 import hsmm as H  # noqa: E402
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P1 = REPO / "eval_outputs/sst_hsmm_v1/p1/train"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
LAM_L, LAM_P, BETA0, BETA1 = 0.5, 0.5, 1.0, 4.0
TOKMS = {"en2zh": 345.6, "zh2en": 332.5}
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")     # = e2e_si_thinker.PUNCT
SEED = 20261123


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def full_batch(proc, head, rows, wavs):
    """reference + <|im_end|> with the WHOLE audio, one sequence per sample -> (inp, [(row index, offset, pos (1, n))])"""
    inp = proc(text=[head[r["dir"]] + r["ref"] + "<|im_end|>" for r in rows],
               audio=[w[: r["X"] * 16] for r, w in zip(rows, wavs)], return_tensors="pt", padding=True)
    ids = inp["input_ids"]
    meta = []
    for b, r in enumerate(rows):
        t = ids[b].tolist()
        st = max(i for i, x in enumerate(t) if x == MD.IM_START) + 3
        tg = r["tok"] + [MD.IM_END]
        if t[st: st + len(tg)] == tg:
            meta.append((b, b, torch.tensor([list(range(st - 1, st - 1 + len(tg)))])))
    return inp, meta


def own_row(ref_row, text, tok, punct_ids):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pu = [int(i in punct_ids) for i in ids]
    d = int(round(TOKMS[ref_row["dir"]] / 100))
    return {"dir": ref_row["dir"], "ref": text, "tok": ids, "C": ref_row["C"], "X": ref_row["X"],
            "g": [1] * len(ids), "dbin": [0 if p else d for p in pu], "punct": pu}


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
    torch.manual_seed(SEED)
    refs = {d: {r["id"]: r for r in map(json.loads, open(DATA / f"train_{d}.jsonl", encoding="utf-8"))} for d in ("en2zh", "zh2en")}
    samples = {(x["s"], x["j"]): x for x in map(json.loads, open(P4 / "samples.jsonl", encoding="utf-8"))}
    roll = {}
    for d in ("en2zh", "zh2en"):
        for k in range(4):
            for x in map(json.loads, open(P4 / f"rollout_{d}_{k}" / f"blocks_{d}_w1.jsonl", encoding="utf-8")):
                roll[x["id"]] = x["translation"]
    proc, head = MD.processor_and_heads()
    tk = proc.tokenizer
    punct_ids = {i for i in range(len(tk)) if PUNCT.match(tk.decode([i]) or "x")}
    thinker, names = MD.build_thinker(dev, lora=True)
    ad0 = torch.load(P1 / "adapters_final.pt", map_location="cpu")
    assert set(ad0) == set(names), (len(ad0), len(names))
    with torch.no_grad():
        for k2, p in names.items():
            p.copy_(ad0[k2].to(p.dtype))
    gate = H.Gate()
    gate.load_state_dict(torch.load(P1 / "gate_final.pt", map_location="cpu"))
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": 1e-4}, {"params": list(gate.parameters()), "lr": 1e-3}],
                            betas=(0.9, 0.999), weight_decay=0.0)
    base_lr = [1e-4, 1e-3]

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
        stats = torch.zeros(6, device=dev)       # risk sum, nll sum, samples, skipped, own tokens, ref-nll samples
        orows, rrows, wavs = [], [], []
        for j in my_slots:
            x = samples[(s, j)]
            text = roll.get(f"s{s:04d}_j{j:02d}", "")
            if not text.strip():
                stats[3] += 1
                continue
            rr = refs[x["dir"]][x["row"]]
            orows.append(own_row(rr, text, tk, punct_ids)); rrows.append(rr)
            wavs.append(sf.read(x["wav"], dtype="float32")[0])
        total = 0.0
        if orows:
            inp, meta = MD.prefix_batch_multi(proc, head, orows, wavs)
            stats[3] += len(orows) - len(meta)
            for jj, lp, lw in (MD.emissions_multi(thinker, inp, meta, dev) if meta else []):
                r = orows[jj]
                risk = H.lattice_risk(lp.cpu(), lw.cpu(), r["g"] + [r["C"]], r["dbin"] + [0], r["punct"] + [1], r["X"],
                                      LAM_L, LAM_P, gate, beta=beta, eos=True)
                loss = risk / len(r["tok"])
                total = total + loss / SLOTS
                stats[0] += float(loss); stats[2] += 1; stats[4] += len(r["tok"])
            finp, fmeta = full_batch(proc, head, rrows, wavs)
            for jj, lp, _ in (MD.emissions_multi(thinker, finp, fmeta, dev) if fmeta else []):
                nll = -lp.mean()
                total = total + nll.cpu() / SLOTS
                stats[1] += float(nll); stats[5] += 1
        if torch.is_tensor(total):
            total.backward()
        for p in lora_p:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat_l = torch.cat([p.grad.reshape(-1) for p in lora_p])
        dist.all_reduce(flat_l)
        k = 0
        for p in lora_p:
            n_ = p.numel()
            p.grad.copy_(flat_l[k: k + n_].view_as(p))
            k += n_
        gg = [p.grad if p.grad is not None else torch.zeros_like(p) for p in gate.parameters()]
        flat = torch.cat([x_.reshape(-1) for x_ in gg]).to(dev)
        dist.all_reduce(flat)
        k = 0
        for p in gate.parameters():
            n_ = p.numel()
            p.grad = flat[k: k + n_].view_as(p).cpu()
            k += n_
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in lora_p)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        if rank == 0:
            rec = {"step": s, "lr": base_lr[0] * lr_at(s), "beta": round(beta, 3),
                   "risk": float(stats[0] / max(stats[2], 1)), "nll_ref": float(stats[1] / max(stats[5], 1)),
                   "samples": int(stats[2]), "skipped": int(stats[3]), "grad_norm_lora": gn,
                   "gate_bias": float(gate.net[2].bias), "sec": round(time.time() - t0, 1)}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({"step": s + 1, "adapters": {k2: v.detach().float().cpu() for k2, v in names.items()},
                        "gate": gate.state_dict()}, out / f"ckpt_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"steps": STEPS, "rank": MD.RANK_LORA, "alpha": MD.ALPHA, "world": world, "lam_L": LAM_L, "lam_P": LAM_P,
                   "beta": [BETA0, BETA1], "warm_start": str(P1)}, open(out / "adapters_final.json", "w"))
        torch.save(gate.state_dict(), out / "gate_final.pt")
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
