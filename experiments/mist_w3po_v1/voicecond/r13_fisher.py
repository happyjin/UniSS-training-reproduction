"""Robust R13 (reports/mist_w3po_v1/robust_r13_fishermask_prereg.zh-CN.md, section 3): empirical diagonal Fisher of the
BASE Thinker's text-layer q/k/v/o weights on paired clean / noisy UniST training rows.

512 rows (256 per direction, sha1("20261121|r13|row|<dir>|<k>")), each clean and noisy (r3_subspace.mix, its own sha1 mix
keys); one forward/backward per (row, condition) with batch 1; loss = mean CE over the reference tokens (prompt as
si_thinker_stream / wave 14); F_cond[W] += grad(W)^2. torchrun world W: rank r takes k with k % W == r; sums are
all-reduced; rank 0 writes <out>/fisher.pt {weight name: {"clean": F, "noisy": F}} (fp32) + stats.json. Refuses to overwrite.

    torchrun --nproc-per-node 4 experiments/mist_w3po_v1/voicecond/r13_fisher.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave14/data"
TGT = {"en2zh": "cmn", "zh2en": "eng"}
IM_START, IM_END, IGNORE = 151644, 151645, -100
N_PER_DIR = 256


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    if rank == 0 and out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dist.barrier()
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
    from experiments.mist_w3po_v1.voicecond.r3_subspace import mix
    pools, bv = W1D.bg_pools("train"), W1D.bvr_values()
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(QWEN3), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    thinker = omni.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni, attr):
            delattr(omni, attr)
    del omni
    thinker.requires_grad_(False)
    text = thinker.model
    W = {}
    for i, L in enumerate(text.layers):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = getattr(L.self_attn, p).weight
            w.requires_grad_(True)
            W[f"thinker.model.layers.{i}.self_attn.{p}.weight"] = w
    assert len(W) == 4 * len(text.layers)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    thinker.train()
    thinker.audio_tower.eval()
    FS = {c: {k: torch.zeros_like(w, dtype=torch.float32) for k, w in W.items()} for c in ("clean", "noisy")}
    log(f"rank {rank}: {len(W)} weights, {sum(w.numel() for w in W.values()) / 1e6:.1f} M; mem {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB")
    proc = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3))
    proc.tokenizer.padding_side = "right"
    ins = inference_instruction()
    head = {}
    for d in ("zh2en", "en2zh"):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]]}]}]
        head[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    n, ce_sum, t0 = 0, {"clean": 0.0, "noisy": 0.0}, time.time()
    for d in ("zh2en", "en2zh"):
        rows = [json.loads(l) for l in open(DATA / f"rows_{d}.jsonl", encoding="utf-8")]
        for k in range(N_PER_DIR):
            if k % world != rank:
                continue
            r = rows[h(f"20261121|r13|row|{d}|{k}") % len(rows)]
            x = sf.read(str(REPO / r["audio"]), dtype="float32")[0]
            xn = mix(x, d, k, pools, bv)[0]
            for cond, wav in (("clean", x), ("noisy", xn)):
                inp = proc(text=[head[d] + r["ref"] + "<|im_end|>"], audio=[wav], return_tensors="pt", padding=True)
                ids = inp["input_ids"]
                row = ids[0].tolist()
                st = max(i for i, t in enumerate(row) if t == IM_START) + 3
                en = max(i for i, t in enumerate(row) if t == IM_END)
                lab = torch.full_like(ids, IGNORE)
                lab[0, st: en + 1] = ids[0, st: en + 1]
                kw = {k2: v.to(dev) for k2, v in inp.items() if k2 in ("input_ids", "attention_mask", "feature_attention_mask")}
                kw["input_features"] = inp["input_features"].to(dev, torch.bfloat16)
                o = thinker(**kw, return_dict=True)
                tl = lab[:, 1:].to(dev)
                sel = tl != IGNORE
                loss = F.cross_entropy(o.logits[:, :-1][sel].float(), tl[sel])
                loss.backward()
                ce_sum[cond] += float(loss)
                for kk, w in W.items():
                    FS[cond][kk] += w.grad.float() ** 2
                    w.grad = None
                del o, loss
            n += 1
            if n % 16 == 0:
                log(f"rank {rank}: {n} rows, {time.time() - t0:.0f}s, mean CE clean {ce_sum['clean'] / n:.3f} noisy {ce_sum['noisy'] / n:.3f}")
    nt = torch.tensor([float(n), ce_sum["clean"], ce_sum["noisy"]], device=dev)
    dist.all_reduce(nt)
    for c in FS:
        for kk in FS[c]:
            dist.all_reduce(FS[c][kk])
    if rank == 0:
        out.mkdir(parents=True)
        torch.save({kk: {"clean": FS["clean"][kk].cpu(), "noisy": FS["noisy"][kk].cpu()} for kk in W}, out / "fisher.pt")
        tot = {c: sum(float(v.sum()) for v in FS[c].values()) for c in FS}
        st = {"n_rows": int(nt[0]), "ce_clean": float(nt[1] / nt[0]), "ce_noisy": float(nt[2] / nt[0]),
              "fisher_total": tot, "seconds": round(time.time() - t0, 1)}
        json.dump(st, open(out / "stats.json", "x"), indent=1)
        log(json.dumps(st))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
