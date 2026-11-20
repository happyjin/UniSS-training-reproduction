"""Robust wave 9 (reports/mist_w3po_v1/robust_wave9_anchor_prereg.zh-CN.md): voice-anchored Thinker (arm A) and its
crosstalk-augmented control (arm C). A copy of wave5_m_train.py's ERM path (lam 0; the Fishr code removed, nothing
else changed): same base Thinker, LoRA (q/k/v/o of all 48 text layers, rank 16, alpha 32, b = 0 at init, fp32), the
8 environments x 8 per step, steps 2300, row / noise random numbers ("20261119|w5m|row|s|j", "20261119|w5m|mix|s|j"),
AdamW lr 1e-4 cosine, no clipping, gradient checkpointing of layers 0..43 / eager experts + math SDPA in the last 4,
adapters-only saves.
Added for wave 9 (both arms identical samples): the crosstalk augmentation of wave9_common.crosstalk BEFORE the wave-5
background noise (which is then added to the mixture by wave 5's rule: BVR from the mixture's energy).
Arm A only: u = C(first 1.0 s of the final (mixed) audio) -> 8 anchor ids (wave9_common.Anchor, fixed PCA_FILE),
inserted after <|audio_end|> (labels IGNORE); the 256 anchor embedding rows are a trainable fp32 parameter
(initialised to the mean of the embedding rows the tokenizer uses, 0..151675), put in by a forward hook on the
Thinker's embed_tokens; same AdamW group as the LoRA. Arm C: no anchor tokens.
    torchrun --nproc-per-node 2 experiments/mist_w3po_v1/voicecond/wave9_train.py --arm A|C --steps 2300 --out DIR
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
from experiments.mist_w3po_v1.voicecond import wave9_common as W9C  # noqa: E402

QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave5/M/data"
SR = 16000
ENVS = [(d, n) for d in ("zh2en", "en2zh") for n in ("clean", "music", "crowd", "sfx")]
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
TGT = {"en2zh": "cmn", "zh2en": "eng"}
PER_ENV, N_LAST = 8, 4
RANK_LORA, ALPHA = 16, 32.0
IM_START, IM_END, IGNORE = 151644, 151645, -100


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------- data
class Sampler:
    def __init__(self):
        from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
        self.rows = {d: [json.loads(l) for l in open(DATA / f"rows_{d}.jsonl", encoding="utf-8")] for d in ("zh2en", "en2zh")}
        self.held = W1D.held_speakers()
        for d in self.rows:
            assert not {r["spk"] for r in self.rows[d]} & self.held, d
        self.pools = W1D.bg_pools("train")
        self.bv = W1D.bvr_values()

    def sample(self, s: int, j: int) -> dict:
        from experiments.mist_w3po_v1.voicecond import bench_build as BB
        d, env = ENVS[j // PER_ENV]
        rows = self.rows[d]
        r = rows[h(f"20261119|w5m|row|{s}|{j}") % len(rows)]
        x = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
        x, xt = W9C.crosstalk(x, r, rows, s, j, self.held)          # wave 9: crosstalk first (own random numbers)
        bvr = None
        if env != "clean":                                          # then wave 5's background noise, unchanged
            rng = random.Random(h(f"20261119|w5m|mix|{s}|{j}"))
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
            x = (y * min(1.0, 0.99 / float(np.abs(y).max()))).astype(np.float32)
        return {"wav": x, "dir": d, "env": j // PER_ENV, "ref": r["ref"], "id": r["id"], "bvr": bvr, "xt": xt}


class StepData(torch.utils.data.IterableDataset):
    """yields, for each step from start, this rank's microbatches as processor tensors + labels + env ids"""

    def __init__(self, start, steps, rank, world, mb):
        self.start, self.steps, self.rank, self.world, self.mb = start, steps, rank, world, mb

    def __iter__(self):
        from transformers import Qwen3OmniMoeProcessor
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
        wi = torch.utils.data.get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        proc = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3))
        proc.tokenizer.padding_side = "right"
        ins = inference_instruction()
        head = {}
        for d in ("zh2en", "en2zh"):
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]]}]}]
            head[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        S = Sampler()
        slots = [j for j in range(len(ENVS) * PER_ENV) if (j % PER_ENV) % self.world == self.rank]
        for s in range(self.start, self.steps):
            if (s - self.start) % nw != wid:
                continue
            items = [S.sample(s, j) for j in slots]
            mbs = []
            for k in range(0, len(items), self.mb):
                part = items[k: k + self.mb]
                texts = [head[it["dir"]] + it["ref"] + "<|im_end|>" for it in part]
                inp = proc(text=texts, audio=[it["wav"] for it in part], return_tensors="pt", padding=True)
                ids = inp["input_ids"]
                lab = torch.full_like(ids, IGNORE)
                for b in range(ids.shape[0]):
                    row = ids[b].tolist()
                    st = max(i for i, t in enumerate(row) if t == IM_START) + 3      # <|im_start|> assistant \n
                    en = max(i for i, t in enumerate(row) if t == IM_END)
                    assert en > st, "empty target"
                    lab[b, st: en + 1] = ids[b, st: en + 1]
                mbs.append({**{k2: v for k2, v in inp.items()}, "labels": lab,
                            "env": torch.tensor([it["env"] for it in part]), "ids": [it["id"] for it in part],
                            "head": [it["wav"][: W9C.HEAD].copy() for it in part],
                            "xt": [it["xt"] is not None for it in part]})
            yield s, mbs


def insert_anchor(mb: dict, anc: np.ndarray) -> dict:
    """8 anchor ids right after <|audio_end|> in every row (same count per row, so right padding stays at the end)"""
    ids, am, lab = mb["input_ids"], mb["attention_mask"], mb["labels"]
    B, L = ids.shape
    n = anc.shape[1]
    out = {k: torch.empty((B, L + n), dtype=t.dtype) for k, t in (("input_ids", ids), ("attention_mask", am), ("labels", lab))}
    for b in range(B):
        p = (ids[b] == W9C.AUDIO_END).nonzero()
        assert len(p) == 1
        p = int(p[0]) + 1
        a = torch.as_tensor(anc[b], dtype=ids.dtype)
        out["input_ids"][b] = torch.cat([ids[b, :p], a, ids[b, p:]])
        out["attention_mask"][b] = torch.cat([am[b, :p], torch.ones(n, dtype=am.dtype), am[b, p:]])
        out["labels"][b] = torch.cat([lab[b, :p], torch.full((n,), IGNORE, dtype=lab.dtype), lab[b, p:]])
    return {**mb, **out}


# ------------------------------------------------------------------------------------------------------------ model
def build_model(dev, anchor: bool):
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M
    from experiments.uniss_streaming_p2st_traj_v1.training import lora
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(QWEN3), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    thinker = omni.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni, attr):
            delattr(omni, attr)
    del omni
    thinker.requires_grad_(False)
    text = thinker.model
    ad = lora.inject(text, rank=RANK_LORA, alpha=ALPHA, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    n_layers = len(text.layers)
    assert len(ad) == 4 * n_layers, (len(ad), n_layers)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eager_experts = M.Qwen3OmniMoeThinkerTextExperts.forward.__wrapped__
    import types
    for li in range(n_layers - N_LAST, n_layers):
        text.layers[li].gradient_checkpointing = False
        ex = text.layers[li].mlp.experts
        ex.forward = types.MethodType(eager_experts, ex)
    names = {}
    for nm, p in text.named_parameters():
        if p.requires_grad:
            names[nm] = p
    E0 = None
    if anchor:
        emb = text.embed_tokens
        assert thinker.get_input_embeddings() is emb and emb.weight.shape[0] == W9C.VOCAB
        assert emb.weight.data_ptr() != thinker.lm_head.weight.data_ptr()          # not tied
        mu = emb.weight[: W9C.TOK_N].float().mean(0)                              # rows the tokenizer can produce
        E = torch.nn.Parameter(mu[None, :].repeat(W9C.ANCHOR_N, 1).clone())
        E0 = E.detach().clone()

        def hook(mod, inp, out):
            ids = inp[0]
            m = ids >= W9C.ANCHOR_LO
            if not bool(m.any()):
                return out
            rep = E[(ids - W9C.ANCHOR_LO).clamp(min=0)].to(out.dtype)
            return torch.where(m[..., None], rep, out)
        emb.register_forward_hook(hook)
        names["anchor.E"] = E
    thinker.train()
    thinker.audio_tower.eval()
    return thinker, names, E0


def adapter_state(names, E0):
    st = {k: v.detach().float().cpu() for k, v in names.items()}
    if E0 is not None:
        st["anchor.E0"] = E0.float().cpu()
    return st


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("A", "C"), required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--max-steps-run", type=int, default=0, help="smoke: stop after this many steps (no final save)")
    ap.add_argument("--workers", type=int, default=6)
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
    anchor = a.arm == "A"
    thinker, names, E0 = build_model(dev, anchor)
    A = W9C.Anchor(str(dev)) if anchor else None
    params = list(names.values())
    log(f"rank {rank}/{world} arm {a.arm}: {len(params)} trainable tensors, {sum(p.numel() for p in params) / 1e6:.2f} M; "
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

    dl = torch.utils.data.DataLoader(StepData(start, a.steps, rank, world, a.mb), batch_size=None,
                                     num_workers=a.workers, prefetch_factor=2, persistent_workers=False)
    metrics = open(out / f"metrics_rank{rank}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    # math SDPA everywhere, as wave 5 (both of its arms)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    n_env = len(ENVS)
    shown = False
    for s, mbs in dl:
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        n_glob = len(ENVS) * PER_ENV
        ce_env = torch.zeros(n_env, device=dev)
        cnt_env = torch.zeros(n_env, device=dev)
        n_xt = torch.zeros((), device=dev)
        for mb in mbs:
            n_xt += sum(mb["xt"])
            if anchor:
                anc = A.ids(mb["head"])
                mb = insert_anchor(mb, anc)
                if not shown and rank == 0:
                    r0 = mb["input_ids"][0].tolist()
                    p = r0.index(W9C.AUDIO_END)
                    log(f"anchor check step {s}: ids around <|audio_end|> {r0[p - 1: p + 11]}; anchors of the mb "
                        f"{anc.tolist()}; xt {mb['xt']}")
                    shown = True
            elif not shown and rank == 0:
                log(f"arm C step {s}: no anchor; xt {mb['xt']}")
                shown = True
            env = mb["env"].to(dev)
            kw = {k: v.to(dev) for k, v in mb.items() if k in ("input_ids", "attention_mask", "feature_attention_mask")}
            kw["input_features"] = mb["input_features"].to(dev, torch.bfloat16)
            lab = mb["labels"].to(dev)
            out_ = thinker(**kw, return_dict=True)
            tl = lab[:, 1:]
            sel = tl != IGNORE
            tok = F.cross_entropy(out_.logits[:, :-1][sel].float(), tl[sel], reduction="none")
            bidx = sel.nonzero()[:, 0]
            nb = lab.shape[0]
            ce_i = torch.zeros(nb, device=dev).index_add(0, bidx, tok) / sel.sum(1).float()   # per-sample mean CE
            ce_env.index_add_(0, env, ce_i.detach())
            cnt_env.index_add_(0, env, torch.ones_like(ce_i))
            loss_ce = ce_i.sum() / n_glob * world                          # mean over the global batch after grad averaging
            loss_ce.backward()
        bad = torch.zeros((), device=dev)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= world
            bad += (~torch.isfinite(p.grad)).sum()
        dist.all_reduce(bad)
        if bad.item() > 0:
            log(f"step {s}: non-finite gradient ({int(bad.item())} entries), step skipped")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
        dist.all_reduce(ce_env)
        dist.all_reduce(cnt_env)
        dist.all_reduce(n_xt)
        ce_e = (ce_env / cnt_env.clamp(min=1)).tolist()
        if rank == 0:
            rec = {"step": s, "lr": lr_at(s), "ce": float(sum(ce_e) / n_env), "ce_env": ce_e,
                   "xt_frac": float(n_xt) / n_glob, "skipped": bool(bad.item() > 0), "sec": round(time.time() - t0, 1),
                   "mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)}
            if anchor:
                rec["anchor_drift"] = round(float((names["anchor.E"].detach() - E0.to(dev)).norm(dim=1).mean()), 5)
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            if s % 10 == 0 or s < 5:
                log(json.dumps(rec))
        done = s + 1
        if a.max_steps_run and done - start >= a.max_steps_run:
            log(f"smoke stop after {done - start} steps")
            break
        if done % 500 == 0 and done < a.steps and rank == 0:
            torch.save({"step": done, "adapters": adapter_state(names, E0), "opt": opt.state_dict()},
                       out / f"resume_{done:05d}.pt")
        if done == a.steps and rank == 0:
            torch.save(adapter_state(names, E0), out / "adapters_final.pt")
            json.dump({"arm": a.arm, "steps": a.steps, "rank": RANK_LORA, "alpha": ALPHA, "world": world,
                       "per_env": PER_ENV, "lam": 0.0, "anchor": anchor,
                       **({"anchor_lo": W9C.ANCHOR_LO, "anchor_n": W9C.ANCHOR_N} if anchor else {})},
                      open(out / "adapters_final.json", "w"))
            log("final adapters saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
