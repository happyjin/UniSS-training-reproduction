"""Robust wave 17 (reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md; method: robust_mainline_plan_20261120 4.3):
perception-layer LoRA. Copy of wave14_train.py with ONE change, the LoRA placement: LoRA (lora.inject, b = 0 at init,
fp32, rank 16, alpha 32) on q_proj / k_proj / v_proj / out_proj of the self-attention of every one of the 32 layers of
the Thinker's audio encoder (thinker.audio_tower.layers.<i>.self_attn.*; the encoder names the output projection
out_proj), 128 adapters, 5.24 M parameters; NO LoRA on the 48 text layers (all frozen). Data, environments, sampler
sha1 keys, steps, batch, optimiser, schedule and seed are wave 14's. The encoder stays in eval mode as in wave 14 (all
its dropouts are 0, so train/eval give the same forward); nothing in the Thinker wraps it in no_grad, so the gradient
reaches it through the frozen text layers (layers 0..43 gradient-checkpointed, non-reentrant, as in wave 14); the
encoder layers are not checkpointed (GradientCheckpointingLayer only checkpoints in training mode; its activations are
small). Adapter keys are audio_tower-relative ("layers.<i>.self_attn.q_proj.a"); adapters_final.json records
module = "thinker.audio_tower" for wave17_merge.py. Run with --arm erm, world 2, 32 slots per rank.

--- wave14_train.py docstring ---
Robust wave 14 (reports/mist_w3po_v1/robust_wave14_unist_prereg.zh-CN.md): wave 5's ERM recipe on UniST pilot data.

Copy of wave5_m_train.py with ONE change: DATA = eval_outputs/mist_w3po_v1/wave14/data (rows_<dir>.jsonl built by
wave14_data.py from the UniST pilot). Steps, batch, LoRA, optimiser, seeds and sha1 keys ("20261119|w5m|row|s|j",
"20261119|w5m|mix|s|j") are untouched. Run with --arm erm, world 8 and --mb 8 (one microbatch per rank).

--- original wave5_m_train.py docstring ---
Robust wave 5, main line M (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md): one bidirectional LoRA on the base
Qwen3-Omni Thinker, ERM (control) or Fishr (Rame et al., ICML 2022), identical data, steps and random numbers.

Model: Qwen3-Omni-30B-A3B-Instruct Thinker, bf16, audio tower frozen; LoRA (lora.inject, b = 0 at init, fp32) on q/k/v/o
of all 48 text layers, rank 16, alpha 32. Prompt: si_thinker_stream's base prompt (E1 instruction for the target
language, chat template with the audio); target: the reference translation + <|im_end|>; loss: per-sample mean
cross-entropy over the target tokens (no model output is a target: not distillation).
Environments (8): {zh2en, en2zh} x {clean, music (bed), crowd (amb), sfx/MUSAN noise}; global batch 64 = 8 per env;
step s, slot j (env j // 8): row = sha1("20261119|w5m|row|s|j") mod n_rows(dir), noise from
random.Random(sha1("20261119|w5m|mix|s|j")): one file of the env's pool (Idea 2 background index, split "train", held-out
files asserted absent), a segment covering the sentence (looped if short), unit RMS, 0.02 s fades, scaled to a BVR drawn
at a U(0.05, 0.95) quantile of the OpenBroadcast-SI BVR values, bench limiter. Both arms see the same samples.
Ranks (torchrun, world W): rank r takes slots with (j % 8) % W == r.
ERM:   L = mean_i CE_i.
Fishr: L = mean_i CE_i + lam * P, P = mean_e mean((v_e - mean_e v_e)^2), v_e = EMA (0.95, 1/(1-ema) corrected, previous
       value detached) of the per-env variance of per-sample gradients of CE_i over the LoRA parameters of the last 4
       text layers (moments all-reduced across ranks with autograd); lam = 0 before step 1500 (statistics still
       tracked), 1000 from step 1500, when Adam is re-initialised (DomainBed Fishr defaults).
Optimiser: AdamW lr 1e-4, betas (0.9, 0.999), wd 0, cosine to 0 over the run; no clipping; a non-finite step is
skipped and logged. Layers 0..43 use gradient checkpointing; the last 4 are not checkpointed, run their experts with
the eager loop and all attention uses the math SDPA kernel (double backward), identically in both arms.
Saves adapters only (fp32 a/b), every 500 steps (resume) and at the end; no checkpoint selection.

    torchrun --nproc-per-node 4 experiments/mist_w3po_v1/voicecond/wave5_m_train.py --arm erm|fishr --steps N --out DIR
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
DATA = REPO / "eval_outputs/mist_w3po_v1/wave14/data"            # wave 14: UniST pilot rows (only change)
SR = 16000
ENVS = [(d, n) for d in ("zh2en", "en2zh") for n in ("clean", "music", "crowd", "sfx")]
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
TGT = {"en2zh": "cmn", "zh2en": "eng"}
PER_ENV, N_LAST = 8, 4
RANK_LORA, ALPHA = 16, 32.0
LAM, ANNEAL, EMA = 1000.0, 1500, 0.95
TRACK_FROM = 100
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
        self.pools = W1D.bg_pools("train")
        self.bv = W1D.bvr_values()

    def sample(self, s: int, j: int) -> dict:
        from experiments.mist_w3po_v1.voicecond import bench_build as BB
        d, env = ENVS[j // PER_ENV]
        rows = self.rows[d]
        r = rows[h(f"20261119|w5m|row|{s}|{j}") % len(rows)]
        x = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
        bvr = None
        if env != "clean":
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
        return {"wav": x, "dir": d, "env": j // PER_ENV, "ref": r["ref"], "id": r["id"], "bvr": bvr}


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
                            "env": torch.tensor([it["env"] for it in part]), "ids": [it["id"] for it in part]})
            yield s, mbs


# ------------------------------------------------------------------------------------------------------------ model
def build_model(dev):
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
    enc = thinker.audio_tower                                       # wave 17: the only change -- LoRA in the encoder
    ad = lora.inject(enc, rank=RANK_LORA, alpha=ALPHA, targets=("q_proj", "k_proj", "v_proj", "out_proj"))
    n_enc = len(enc.layers)
    assert n_enc == 32 and len(ad) == 4 * n_enc, (len(ad), n_enc)
    assert all(".self_attn." in l.lora_name for l in ad), [l.lora_name for l in ad][:4]
    assert not any(p.requires_grad for p in text.parameters()), "text layers must stay frozen"
    n_layers = len(text.layers)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eager_experts = M.Qwen3OmniMoeThinkerTextExperts.forward.__wrapped__
    import types
    for li in range(n_layers - N_LAST, n_layers):
        text.layers[li].gradient_checkpointing = False
        ex = text.layers[li].mlp.experts
        ex.forward = types.MethodType(eager_experts, ex)
    names = {}
    for nm, p in enc.named_parameters():
        if p.requires_grad:
            names[nm] = p
    assert len(names) == 2 * 4 * n_enc and all(nm.endswith((".a", ".b")) for nm in names), len(names)
    tr = [nm for nm, p in thinker.named_parameters() if p.requires_grad]
    assert len(tr) == len(names) and all(nm.startswith("audio_tower.") for nm in tr), tr[:3]
    last = []                                                       # Fishr only; wave 17 runs ERM
    thinker.train()
    thinker.audio_tower.eval()
    return thinker, names, last


def adapter_state(names):
    return {k: v.detach().float().cpu() for k, v in names.items()}


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("erm", "fishr"), required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--max-steps-run", type=int, default=0, help="smoke: stop after this many steps (no final save)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke-anneal", type=int, default=0, help="smoke only: switch the penalty on at this step")
    a = ap.parse_args()
    assert a.arm == "erm", "wave 17 trains ERM only (no Fishr statistics over encoder adapters)"
    global ANNEAL
    if a.smoke_anneal:
        assert a.max_steps_run, "--smoke-anneal is for smoke runs only"
        ANNEAL = a.smoke_anneal
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
    thinker, names, last = build_model(dev)
    params = list(names.values())
    log(f"rank {rank}/{world}: {len(params)} adapter tensors, {sum(p.numel() for p in params) / 1e6:.2f} M; "
        f"last-{N_LAST} {sum(p.numel() for p in last) / 1e6:.3f} M; mem {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB")

    def make_opt():
        return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    opt = make_opt()
    start, ema, ema_n = 0, None, 0
    ck = sorted(out.glob("resume_*.pt"))
    if ck:
        st = torch.load(ck[-1], map_location="cpu", weights_only=False)
        for k, p in names.items():
            p.data.copy_(st["adapters"][k].to(p.device))
        opt.load_state_dict(st["opt"])
        start, ema_n = st["step"], st["ema_n"]
        ema = None if st["ema"] is None else st["ema"].to(dev)
        log(f"resumed from {ck[-1].name} at step {start}")

    def lr_at(s):
        return 1e-4 * 0.5 * (1 + math.cos(math.pi * s / a.steps))

    dl = torch.utils.data.DataLoader(StepData(start, a.steps, rank, world, a.mb), batch_size=None,
                                     num_workers=a.workers, prefetch_factor=2, persistent_workers=False)
    metrics = open(out / f"metrics_rank{rank}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    # math SDPA everywhere (forward, checkpoint recomputation, double backward): set globally, not by a context
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    from torch.distributed.nn.functional import all_reduce as ar_grad
    n_env = len(ENVS)
    for s, mbs in dl:
        if a.arm == "fishr" and s == ANNEAL and ema_n > 0:
            opt = make_opt()
            log(f"step {s}: penalty on (lambda {LAM:g}), Adam re-initialised")
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        lam = LAM if (a.arm == "fishr" and s >= ANNEAL) else 0.0
        want_graph = lam > 0
        # Fishr statistics: tracked from TRACK_FROM steps before the penalty switches on (EMA 0.95 -> 0.95^100 = 0.6 %
        # residual weight of anything earlier), every step after it
        track = a.arm == "fishr" and s >= ANNEAL - TRACK_FROM
        n_glob = len(ENVS) * PER_ENV
        ce_env = torch.zeros(n_env, device=dev)
        cnt_env = torch.zeros(n_env, device=dev)
        S1 = S2 = None
        cnt = torch.zeros(n_env, device=dev)
        keep_graphs = []
        for mb in mbs:
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
            if track:
                G = []
                for i in range(len(ce_i)):
                    gi = torch.autograd.grad(ce_i[i], last, retain_graph=True, create_graph=want_graph)
                    G.append(torch.cat([g.reshape(-1) for g in gi]))
                G = torch.stack(G)                                         # [mb, P]
                oh = F.one_hot(env, n_env).to(G.dtype)                     # [mb, E]
                s1, s2 = oh.T @ G, oh.T @ (G * G)
                S1 = s1 if S1 is None else S1 + s1
                S2 = s2 if S2 is None else S2 + s2
                cnt += oh.sum(0)
            if want_graph:
                keep_graphs.append(loss_ce)
            else:
                loss_ce.backward()
        pen = torch.zeros((), device=dev)
        if track:
            if want_graph:
                S1, S2 = ar_grad(S1), ar_grad(S2)
            else:
                dist.all_reduce(S1)
                dist.all_reduce(S2)
            dist.all_reduce(cnt)
            mean = S1 / cnt[:, None]
            var = S2 / cnt[:, None] - mean * mean                          # population variance per env
            prev = torch.zeros_like(var) if ema is None else ema
            cur = EMA * prev + (1 - EMA) * var
            ema = cur.detach()
            ema_n += 1
            v = cur / (1 - EMA)
            pen = ((v - v.mean(0, keepdim=True)) ** 2).mean(1).mean()
            if want_graph:
                (sum(keep_graphs) + lam * pen).backward()
        bad = torch.zeros((), device=dev)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= world
            bad += (~torch.isfinite(p.grad)).sum()
        dist.all_reduce(bad)
        if s == start:                                              # wave 17: the gradient must reach the encoder
            gb = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for nm, p in names.items() if nm.endswith(".b")))
            log(f"rank {rank} step {s}: encoder LoRA-b gradient norm {gb:.4e}")
            assert gb > 0, "no gradient reached the audio encoder"
        if bad.item() > 0:
            log(f"step {s}: non-finite gradient ({int(bad.item())} entries), step skipped")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
        dist.all_reduce(ce_env)
        dist.all_reduce(cnt_env)
        ce_e = (ce_env / cnt_env.clamp(min=1)).tolist()
        if rank == 0:
            rec = {"step": s, "lr": lr_at(s), "lam": lam, "ce": float(sum(ce_e) / n_env), "ce_env": ce_e,
                   "pen": float(pen), "skipped": bool(bad.item() > 0), "sec": round(time.time() - t0, 1),
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
            torch.save({"step": done, "adapters": adapter_state(names), "opt": opt.state_dict(), "ema": ema,
                        "ema_n": ema_n}, out / f"resume_{done:05d}.pt")
        if done == a.steps and rank == 0:
            torch.save(adapter_state(names), out / "adapters_final.pt")
            json.dump({"arm": "enc", "module": "thinker.audio_tower", "targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
                       "steps": a.steps, "rank": RANK_LORA, "alpha": ALPHA, "world": world, "mb": a.mb,
                       "per_env": PER_ENV, "lam": LAM if a.arm == "fishr" else 0.0, "anneal": ANNEAL, "ema": EMA},
                      open(out / "adapters_final.json", "w"))
            log("final adapters saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
