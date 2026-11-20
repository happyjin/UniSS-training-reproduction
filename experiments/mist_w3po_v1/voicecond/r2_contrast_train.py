"""R2 (reports/mist_w3po_v1/robust_r2_contrast_prereg.zh-CN.md): target-free invariance adaptation of the audio encoder.

LoRA placement / optimiser / steps / batch / seed = wave 17 (wave17_enc_train.py): q/k/v/out_proj of the 32 self-attention
layers of the Thinker's audio encoder, rank 16, alpha 32, b = 0 at init, fp32 adapters; AdamW 1e-4 (0.9, 0.999), wd 0,
cosine to 0 over 2300 steps, no clipping, global batch 64, seed 20261119. Data = wave 14's UniST rows, wave 14/17 slot
keys (row = sha1("20261119|w5m|row|s|j"); noisy-env slots use wave 17's exact mix key; clean-env slots draw a noise type
by sha1("20261119|r2|type|s|j") mod 3 and mix with sha1("20261119|r2|mix|s|j")). Every slot yields a pair (clean =
original audio, noisy = mixed audio).
Objective (the only change): symmetric InfoNCE between frame-mean-pooled, L2-normalised encoder outputs (after proj2,
i.e. what the LLM consumes) of the clean and noisy views; positives on the diagonal, the other sentences of the global
batch (all_gather with autograd) as negatives; temperature 0.1. The LLM is never run and no reference translation is
used (not distillation: no teacher, no pseudo-label, symmetric contrastive learning with one encoder).
The global batch is handled with gradient caching (embeddings without a graph -> global loss -> per-microbatch re-encode
and backward of the cached embedding gradients), so the loss is the exact global-batch InfoNCE while peak memory is that
of one microbatch (coordinator: training must stay <= ~70 GB per card because R4's vLLM shares cards 4-5).
Only the audio encoder is loaded (weights thinker.audio_tower.* of the release). Collapse monitor (logged only): mean
positive / negative cosine, mean per-dimension std of z. Saves adapters every 500 steps (resume) and at the end, with
adapters_final.json module = "thinker.audio_tower" (wave17_merge.py format).

    torchrun --nproc-per-node 2 experiments/mist_w3po_v1/voicecond/r2_contrast_train.py --steps 2300 --out DIR --mb 32
"""
from __future__ import annotations

import argparse
import hashlib
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave14/data"
ENVS = [(d, n) for d in ("zh2en", "en2zh") for n in ("clean", "music", "crowd", "sfx")]
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
NOISY = ("music", "crowd", "sfx")
PER_ENV = 8
RANK_LORA, ALPHA, TAU = 16, 32.0, 0.1


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


class PairSampler:
    def __init__(self):
        from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
        self.rows = {d: [json.loads(l) for l in open(DATA / f"rows_{d}.jsonl", encoding="utf-8")] for d in ("zh2en", "en2zh")}
        self.pools = W1D.bg_pools("train")
        self.bv = W1D.bvr_values()

    def mix(self, x, env, key):
        from experiments.mist_w3po_v1.voicecond import bench_build as BB
        rng = random.Random(h(key))
        bvr = float(np.quantile(self.bv, rng.uniform(0.05, 0.95)))
        P = self.pools[POOL[env]]
        f = P[int(rng.random() * len(P)) % len(P)]
        b = BB.load(f["file"])
        n = len(x)
        if len(b) < n:
            b = np.tile(b, 1 + n // max(1, len(b)))
        off = int(rng.random() * max(0, len(b) - n))
        seg = BB.fade(BB.unit_rms(b[off: off + n]), 0.02, 0.02)
        es, eb = float((x.astype(np.float64) ** 2).sum()), float((seg.astype(np.float64) ** 2).sum())
        alpha = math.sqrt(10 ** (bvr / 10) * es / max(eb, 1e-12))
        y = x + alpha * seg
        return (y * min(1.0, 0.99 / float(np.abs(y).max()))).astype(np.float32), bvr

    def sample(self, s: int, j: int) -> dict:
        d, env = ENVS[j // PER_ENV]
        rows = self.rows[d]
        r = rows[h(f"20261119|w5m|row|{s}|{j}") % len(rows)]
        x = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
        if env == "clean":
            env_n, key = NOISY[h(f"20261119|r2|type|{s}|{j}") % 3], f"20261119|r2|mix|{s}|{j}"
        else:
            env_n, key = env, f"20261119|w5m|mix|{s}|{j}"
        xn, bvr = self.mix(x, env_n, key)
        return {"clean": x, "noisy": xn, "env": j // PER_ENV, "id": r["id"], "bvr": bvr}


class StepData(torch.utils.data.IterableDataset):
    def __init__(self, start, steps, rank, world, mb):
        self.start, self.steps, self.rank, self.world, self.mb = start, steps, rank, world, mb

    def __iter__(self):
        from transformers import WhisperFeatureExtractor
        wi = torch.utils.data.get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        fe = WhisperFeatureExtractor.from_pretrained(str(QWEN3))
        S = PairSampler()
        slots = [j for j in range(len(ENVS) * PER_ENV) if (j % PER_ENV) % self.world == self.rank]
        for s in range(self.start, self.steps):
            if (s - self.start) % nw != wid:
                continue
            items = [S.sample(s, j) for j in slots]
            mbs = []
            for k in range(0, len(items), self.mb):
                part = items[k: k + self.mb]
                wavs = [it["clean"] for it in part] + [it["noisy"] for it in part]
                f = fe(wavs, sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True,
                       return_tensors="pt")
                mbs.append({"input_features": f["input_features"], "mask": f["attention_mask"],
                            "slot": torch.tensor([slots[k + i] for i in range(len(part))]),
                            "env": torch.tensor([it["env"] for it in part])})
            yield s, mbs


def build_encoder(dev):
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
    from experiments.uniss_streaming_p2st_traj_v1.training import lora
    cfg = AutoConfig.from_pretrained(str(QWEN3)).thinker_config.audio_config
    cfg._attn_implementation = "sdpa"
    enc = Qwen3OmniMoeAudioEncoder(cfg)
    idx = json.loads((QWEN3 / "model.safetensors.index.json").read_text())["weight_map"]
    pre = "thinker.audio_tower."
    sd = {}
    for shard in sorted({v for k, v in idx.items() if k.startswith(pre)}):
        sd.update({k[len(pre):]: v for k, v in load_file(str(QWEN3 / shard)).items() if k.startswith(pre)})
    miss, unexp = enc.load_state_dict(sd, strict=False)
    assert not unexp and all("positional_embedding" in m for m in miss), (miss[:5], unexp[:5])
    enc = enc.to(device=dev, dtype=torch.bfloat16)
    enc.requires_grad_(False)
    ad = lora.inject(enc, rank=RANK_LORA, alpha=ALPHA, targets=("q_proj", "k_proj", "v_proj", "out_proj"))
    assert len(enc.layers) == 32 and len(ad) == 128, (len(enc.layers), len(ad))
    names = {nm: p for nm, p in enc.named_parameters() if p.requires_grad}
    assert len(names) == 256 and all(nm.endswith((".a", ".b")) and ".self_attn." in nm for nm in names), list(names)[:3]
    enc.eval()                                                       # all dropouts are 0 (as wave 17)
    return enc, names


def pooled(enc, feats, mask):
    """frame-mean-pooled encoder outputs (after proj2), one row per audio, fp32"""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
    lens = mask.sum(1)
    x = feats.permute(0, 2, 1)[mask.bool()].permute(1, 0)
    hs = enc(x, feature_lens=lens).last_hidden_state                  # [sum frames, D]
    out_lens = _get_feat_extract_output_lengths(lens).tolist()
    assert sum(out_lens) == hs.shape[0], (sum(out_lens), hs.shape)
    return torch.stack([c.float().mean(0) for c in torch.split(hs, out_lens, dim=0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=int, default=32)
    ap.add_argument("--max-steps-run", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.manual_seed(20261119)
    enc, names = build_encoder(dev)
    params = list(names.values())
    log(f"rank {rank}/{world}: {len(params)} adapter tensors, {sum(p.numel() for p in params) / 1e6:.2f} M; "
        f"mem {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB")
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    start = 0
    ck = sorted(out.glob("resume_*.pt"))
    if ck:
        st = torch.load(ck[-1], map_location="cpu", weights_only=False)
        for k, p in names.items():
            p.data.copy_(st["adapters"][k].to(p.device))
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"resumed from {ck[-1].name} at step {start}")

    def lr_at(s):
        return 1e-4 * 0.5 * (1 + math.cos(math.pi * s / a.steps))

    from torch.distributed.nn.functional import all_gather as ag
    dl = torch.utils.data.DataLoader(StepData(start, a.steps, rank, world, a.mb), batch_size=None,
                                     num_workers=a.workers, prefetch_factor=2, persistent_workers=False)
    metrics = open(out / f"metrics_rank{rank}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for s, mbs in dl:
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        # gradient caching (Gao et al. 2021): exact global-batch InfoNCE with memory of one microbatch.
        # pass 1: embeddings of every microbatch without a graph
        zc, zn, slot, sizes = [], [], [], []
        with torch.no_grad():
            for mb in mbs:
                feats = mb["input_features"].to(dev, torch.bfloat16)
                z = F.normalize(pooled(enc, feats, mb["mask"].to(dev)), dim=-1)
                n = z.shape[0] // 2
                zc.append(z[:n]); zn.append(z[n:]); slot.append(mb["slot"].to(dev)); sizes.append(n)
        zc = torch.cat(zc).requires_grad_(True)
        zn = torch.cat(zn).requires_grad_(True)
        slot = torch.cat(slot)
        # pass 2: the global loss on the cached embeddings -> gradients w.r.t. this rank's embeddings
        Zc, Zn = torch.cat(ag(zc)), torch.cat(ag(zn))
        sl = [torch.zeros_like(slot) for _ in range(world)]
        dist.all_gather(sl, slot)
        SL = torch.cat(sl)
        order = torch.argsort(SL)                                    # global slot order (same on every rank)
        Zc, Zn = Zc[order], Zn[order]
        logits = Zc @ Zn.T / TAU
        tgt = torch.arange(logits.shape[0], device=dev)
        loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
        loss.backward()
        gc, gn = zc.grad.split(sizes), zn.grad.split(sizes)
        # pass 3: re-encode each microbatch with a graph and back-propagate its cached embedding gradients
        for k, mb in enumerate(mbs):
            feats = mb["input_features"].to(dev, torch.bfloat16)
            z = F.normalize(pooled(enc, feats, mb["mask"].to(dev)), dim=-1)
            torch.autograd.backward(z, torch.cat([gc[k], gn[k]]))
        Zc, Zn = Zc.detach(), Zn.detach()
        bad = torch.zeros((), device=dev)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= world
            bad += (~torch.isfinite(p.grad)).sum()
        dist.all_reduce(bad)
        if s == start:
            gb = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for nm, p in names.items() if nm.endswith(".b")))
            log(f"rank {rank} step {s}: encoder LoRA-b gradient norm {gb:.4e}")
            assert gb > 0, "no gradient reached the adapters"
        if bad.item() > 0:
            log(f"step {s}: non-finite gradient ({int(bad.item())} entries), step skipped")
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0:
            with torch.no_grad():
                sim = (Zc @ Zn.T).float()
                N = sim.shape[0]
                pos = float(sim.diag().mean())
                neg = float((sim.sum() - sim.diag().sum()) / (N * N - N))
                zstd = float(torch.cat([Zc, Zn]).float().std(0).mean())
                acc = float((logits.argmax(1) == tgt).float().mean())
            rec = {"step": s, "lr": lr_at(s), "loss": float(loss), "pos_cos": pos, "neg_cos": neg, "z_std": zstd,
                   "top1": acc, "n": N, "skipped": bool(bad.item() > 0), "sec": round(time.time() - t0, 1),
                   "mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            if s % 10 == 0 or s < 5:
                log(json.dumps(rec))
        done = s + 1
        if a.max_steps_run and done - start >= a.max_steps_run:
            log(f"smoke stop after {done - start} steps")
            break
        if done % 500 == 0 and done < a.steps and rank == 0:
            torch.save({"step": done, "adapters": {k: v.detach().float().cpu() for k, v in names.items()},
                        "opt": opt.state_dict()}, out / f"resume_{done:05d}.pt")
        if done == a.steps and rank == 0:
            torch.save({k: v.detach().float().cpu() for k, v in names.items()}, out / "adapters_final.pt")
            json.dump({"arm": "r2", "module": "thinker.audio_tower", "targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
                       "steps": a.steps, "rank": RANK_LORA, "alpha": ALPHA, "world": world, "mb": a.mb,
                       "per_env": PER_ENV, "objective": "symmetric InfoNCE clean vs noisy, frame-mean pool, tau 0.1"},
                      open(out / "adapters_final.json", "w"))
            log("final adapters saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
