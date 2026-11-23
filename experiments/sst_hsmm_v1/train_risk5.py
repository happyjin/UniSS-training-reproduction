"""sst_hsmm_v1 P6b (prereg change 13) = train_risk4 (P6) with the text LoRA FROZEN at P5' (p5b adapters_final) and only Gate2
trained (same lam sampling, loss, data, rollouts, schedule, beta; translation NLL dropped since nothing it touches is trained);
Gate2 bias0 = P5' gate at (0, 0, 1, 0.5). Inference uses p5b/merged + this gate.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_risk5 --out eval_outputs/sst_hsmm_v1/p6b/train
Was: sst_hsmm_v1 P6 (prereg change 11) = train_risk2 (change 8) with a MULTIPLIER-CONDITIONED policy instead of constraints:
per sample (s, j): lam_L, lam_P ~ log-uniform [hsmm.LAM_LO, hsmm.LAM_HI] (sha1 keys "20261123|p6|lamL|s|j" / "...lamP..."),
gate = hsmm.Gate2(s, f, started, z_L, z_P) (no c/C), loss = q/n + lam_L * L + lam_P * P (+ translation NLL); no budgets,
no dual ascent. Warm start: P1a adapters_final; Gate2 last-layer bias = P1a gate at (s 0, f 0, started 1, c/C 0.5),
everything else zero -> the initial policy = P1a's (whose gate is ~constant). Same data, rollouts, optimiser, schedule, beta.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_risk4 --out eval_outputs/sst_hsmm_v1/p6/train
Was: sst_hsmm_v1 P5 (reports/sst_hsmm_v1/prereg.zh-CN.md change 8): constrained exact expected-risk training of the write
policy on its own outputs; same data, rollouts (P1a), warm start, optimiser and schedule as P4 (train_risk.py). Changes:
  - lattice: hsmm.lattice_risk_block (block bit, Talker per-block overhead OB = 2 bins, three expected costs q, L, P);
  - quality cost q = expected number of tokens written while the prefix model's greedy non-wait token != y_i (per token);
  - latency L (LAAL form, s) and pauses P are constraints, not fixed weights: per sample budget from the same item's P1a
    rollout under proxy.item_proxy: b_L = L_P1a + dL[dir], b_P = rho[dir] * P_P1a (p5/budget.json, rule in change 8);
    Lagrangian q/n + lam_L[dir] * L + lam_P[dir] * P; multipliers by multiplicative dual ascent after every step:
    lam <- clip(lam * exp(ETA * v / scale), 1e-3, 20), v = mean over the step's samples of the direction (all ranks) of
    (L - b_L) resp. (P - b_P), scale 0.25 s resp. 0.25 pauses, ETA 0.02, lam_L = lam_P = 0.5 at step 0.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_risk2 --out eval_outputs/sst_hsmm_v1/p5/train
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
from experiments.sst_hsmm_v1 import proxy as PX  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P1 = REPO / "eval_outputs/sst_hsmm_v1/p5b/train"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
P5 = REPO / "eval_outputs/sst_hsmm_v1/p5"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
LAM0, BETA0, BETA1, ETA, SC_L, SC_P = 0.5, 1.0, 4.0, 0.02, 0.25, 0.25
DIRS = ("en2zh", "zh2en")
TOKMS = {"en2zh": 345.6, "zh2en": 332.5}
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")     # = e2e_si_thinker.PUNCT
SEED = 20261123


def lam_of(key):
    import hashlib
    u = (int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 ** 12) / 10 ** 12
    return math.exp(math.log(H.LAM_LO) + u * (math.log(H.LAM_HI) - math.log(H.LAM_LO)))


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
        if (out / "gate_final.pt").exists():
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
                roll[x["id"]] = (x["translation"], x["blocks"])
    proc, head = MD.processor_and_heads()
    tk = proc.tokenizer
    punct_ids = {i for i in range(len(tk)) if PUNCT.match(tk.decode([i]) or "x")}
    thinker, names = MD.build_thinker(dev, lora=True)
    ad0 = torch.load(P1 / "adapters_final.pt", map_location="cpu")
    assert set(ad0) == set(names), (len(ad0), len(names))
    with torch.no_grad():
        for k2, p in names.items():
            p.copy_(ad0[k2].to(p.dtype))
    for p in names.values():
        p.requires_grad_(False)
    g1 = H.Gate()
    g1.load_state_dict(torch.load(P1 / "gate_final.pt", map_location="cpu"))
    with torch.no_grad():
        bias0 = float(g1.net(torch.tensor([[0.0, 0.0, 1.0, 0.5]])))
    gate = H.Gate2(bias0)
    if rank == 0:
        log(f"Gate2 bias0 = {bias0:.4f}")
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": list(gate.parameters()), "lr": 1e-3}], betas=(0.9, 0.999), weight_decay=0.0)
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
        stats = torch.zeros(6, device=dev)       # risk sum, nll sum, samples, skipped, own tokens, ref-nll samples
        dstat = torch.zeros(2, 6, device=dev)    # per dir: n, sum (L - b_L), sum (P - b_P), sum q/n, sum L, sum P
        orows, rrows, wavs = [], [], []
        for j in my_slots:
            x = samples[(s, j)]
            text = roll.get(f"s{s:04d}_j{j:02d}", ("", []))[0]
            if not text.strip():
                stats[3] += 1
                continue
            rr = refs[x["dir"]][x["row"]]
            orows.append(own_row(rr, text, tk, punct_ids)); rrows.append(rr)
            orows[-1]["lam"] = (lam_of(f"{SEED}|p6|lamL|{s}|{j}"), lam_of(f"{SEED}|p6|lamP|{s}|{j}"))
            wavs.append(sf.read(x["wav"], dtype="float32")[0])
        total = 0.0
        if orows:
            inp, meta = MD.prefix_batch_multi(proc, head, orows, wavs)
            stats[3] += len(orows) - len(meta)
            for jj, lp, lw, dis in (MD.emissions_multi_dis(thinker, inp, meta, dev) if meta else []):
                r = orows[jj]
                d = r["dir"]
                n_ = len(r["tok"])
                lL, lP = r["lam"]
                gate.z = (H.lam_z(lL), H.lam_z(lP))
                Eq, EL, EP = H.lattice_risk_block(lp.detach().cpu(), lw.cpu(), dis.cpu(), r["g"] + [r["C"]], r["dbin"] + [0],
                                                  r["punct"] + [1], r["X"], gate, PX.OB, beta=beta, eos=True)
                X = r["X"] / 1000.0
                Lm = EL / n_ - X * (n_ - 1) / (2 * n_)            # LAAL form (same as proxy.item_proxy)
                q = Eq / n_
                loss = q + lL * Lm + lP * EP
                total = total + loss / SLOTS
                k_ = DIRS.index(d)
                dstat[k_] += torch.tensor([1.0, math.log(lL), math.log(lP), float(q), float(Lm), float(EP)], device=dev)
                stats[0] += float(loss); stats[2] += 1; stats[4] += n_
        if torch.is_tensor(total) and total.requires_grad:
            total.backward()
        gg = [p.grad if p.grad is not None else torch.zeros_like(p) for p in gate.parameters()]
        flat = torch.cat([x_.reshape(-1) for x_ in gg]).to(dev)
        dist.all_reduce(flat)
        k = 0
        for p in gate.parameters():
            n_ = p.numel()
            p.grad = flat[k: k + n_].view_as(p).cpu()
            k += n_
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in gate.parameters())))
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        dist.all_reduce(dstat)
        if rank == 0:
            rec = {"step": s, "lr": base_lr[0] * lr_at(s), "beta": round(beta, 3),
                   "risk": float(stats[0] / max(stats[2], 1)), "nll_ref": float(stats[1] / max(stats[5], 1)),
                   "samples": int(stats[2]), "skipped": int(stats[3]), "grad_norm_gate": gn,
                   "gate_bias": float(gate.net[2].bias), "sec": round(time.time() - t0, 1),
                   "gate_skip": [round(float(x), 4) for x in gate.skip.weight.view(-1)],
                   **{f"{d}_{k}": float(dstat[i_, 1 + j_] / max(float(dstat[i_, 0]), 1)) for i_, d in enumerate(DIRS)
                      for j_, k in enumerate(("loglamL", "loglamP", "q", "L", "P"))}}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({"step": s + 1, "gate": gate.state_dict()}, out / f"ckpt_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        json.dump({"frozen_lora": str(P1 / "adapters_final.pt"), "steps": STEPS, "rank": MD.RANK_LORA, "alpha": MD.ALPHA, "world": world, "lam_range": [H.LAM_LO, H.LAM_HI], "gate": "Gate2", "ob": PX.OB, "beta": [BETA0, BETA1], "warm_start": str(P1)}, open(out / "gate_final.json", "w"))
        torch.save(gate.state_dict(), out / "gate_final.pt")
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
