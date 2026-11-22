"""sst_hsmm_v1 P1a (reports/sst_hsmm_v1/prereg.zh-CN.md 3.2 + change 1): end-to-end training of the UE Thinker LoRA and the
HSMM gate g_psi by the exact tilted marginal likelihood of the speech-timeline lattice (hsmm.lattice), 8 ranks (torchrun).

Step s, slot j = 0..15 (16 samples / step; rank r takes the slots with j % world == r): direction ("en2zh", "zh2en")[j % 2],
row = sha1("20261122|hsmm|row|s|j") mod n_rows(dir), audio mixed on the fly (mix.Mixer, key "20261122|hsmm|mix|s|j",
clean with p 0.5). Per sample: prefix forward (model.emissions, graph kept) -> lattice on CPU (gate on CPU) ->
loss = -(1/n) log Z, backward scaled by 1/16; gradients all-reduced (sum) over ranks; AdamW (LoRA lr 1e-4, gate 1e-3,
betas 0.9/0.999, wd 0), 100 warmup steps, cosine to 0; no clipping. lam_L, lam_P, gate bias init and steps from
eval_outputs/sst_hsmm_v1/p0/summary.json. Saves adapters + gate every 500 steps and at the end.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train --out eval_outputs/sst_hsmm_v1/p1/train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from experiments.sst_hsmm_v1.mix import Mixer  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P0 = REPO / "eval_outputs/sst_hsmm_v1/p0/summary.json"
SLOTS, WARM, SAVE_EVERY = 16, 100, 500
SEED = 20261122


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps-run", type=int, default=0, help="smoke: stop after this many steps (no final save)")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    torch.set_num_threads(4)              # CPU lattice: avoid 192-thread oversubscription x 8 ranks
    dev = f"cuda:{lr_}"
    out = Path(a.out)
    p0 = json.load(open(P0))
    lam_L, lam_P = p0["P0c"]["chosen"]["lam_L"], p0["P0c"]["chosen"]["lam_P"]
    steps = int(p0["P0b"]["steps"])
    if rank == 0:
        if (out / "adapters_final.pt").exists():
            sys.exit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.manual_seed(SEED)
    rows = {d: [json.loads(l) for l in open(DATA / f"train_{d}.jsonl", encoding="utf-8")] for d in ("en2zh", "zh2en")}
    proc, head = MD.processor_and_heads()
    thinker, names = MD.build_thinker(dev, lora=True)
    gate = H.Gate()
    with torch.no_grad():
        gate.net[2].bias.fill_(float(p0["P0b"]["gate_bias_init"]))
    mixer = Mixer()
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": 1e-4}, {"params": list(gate.parameters()), "lr": 1e-3}],
                            betas=(0.9, 0.999), weight_decay=0.0)
    base_lr = [1e-4, 1e-3]

    def lr_at(s):
        return s / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, steps - WARM)))

    my_slots = [j for j in range(SLOTS) if j % world == rank]
    mf = open(out / f"metrics_rank{rank}.jsonl", "a")
    t0 = time.time()
    for s in range(steps):
        if a.max_steps_run and s >= a.max_steps_run:
            break
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * lr_at(s)
        stats = torch.zeros(4, device=dev)          # loss sum, samples, mismatches, tokens
        for j in my_slots:
            d = ("en2zh", "zh2en")[j % 2]
            r = rows[d][h(f"{SEED}|hsmm|row|{s}|{j}") % len(rows[d])]
            wav = sf.read(r["audio"], dtype="float32")[0]
            wav, _, _ = mixer.mix(wav, f"{SEED}|hsmm|mix|{s}|{j}", p_clean=0.5)
            inp, pos = MD.prefix_batch(proc, head, r, wav)
            if inp is None:
                stats[2] += 1
                continue
            lp, lw = MD.emissions(thinker, inp, pos, dev)
            tot, _ = H.lattice(lp.cpu(), lw.cpu(), r["g"], [int(round(x / 100)) for x in r["d_ms"]], r["punct"], r["X"],
                               lam_L, lam_P, gate)
            loss = -tot / len(r["tok"])
            (loss / SLOTS).backward()
            stats[0] += float(loss); stats[1] += 1; stats[3] += len(r["tok"])
        for p in lora_p:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
        gg = [p.grad if p.grad is not None else torch.zeros_like(p) for p in gate.parameters()]
        flat = torch.cat([x.reshape(-1) for x in gg]).to(dev)
        dist.all_reduce(flat)
        k = 0
        for p in gate.parameters():
            n = p.numel()
            p.grad = flat[k: k + n].view_as(p).cpu()
            k += n
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in lora_p)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        if rank == 0:
            rec = {"step": s, "lr": base_lr[0] * lr_at(s), "loss": float(stats[0] / max(stats[1], 1)),
                   "samples": int(stats[1]), "mismatch": int(stats[2]), "grad_norm_lora": gn,
                   "gate_bias": float(gate.net[2].bias), "sec": round(time.time() - t0, 1)}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({"step": s + 1, "adapters": {k2: v.detach().float().cpu() for k2, v in names.items()},
                        "gate": gate.state_dict()}, out / f"ckpt_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"steps": steps, "rank": MD.RANK_LORA, "alpha": MD.ALPHA, "world": world, "lam_L": lam_L, "lam_P": lam_P,
                   "gate_bias_init": p0["P0b"]["gate_bias_init"]}, open(out / "adapters_final.json", "w"))
        torch.save(gate.state_dict(), out / "gate_final.pt")
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
