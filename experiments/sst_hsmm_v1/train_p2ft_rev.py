"""change 65 (reversed-pair en2zh adapter): a copy of train_p2ft.py (not modified) where the en2zh slots come from
eval_outputs/sst_hsmm_v1/rev (rev_prep.py): input = UniST's English target speech (BiCodec-decoded, p4 noise mixing), target = the
original spoken Chinese transcription (fillers 呃 / 嗯 removed). zh2en slots, tags, steps, seed, LR and losses are unchanged.
Original header follows.
portfolio 2026-11-27 method E1 (P2F-T, observation-horizon tags): a copy of train_p2f.py (not modified) where (a) full-audio
examples use the instruction + p2ft_tags.TAG_C ("complete") and (b) truncated-audio examples use + TAG_P ("stops early"), so the
adapter is told whether the input is finished (fix for continuation leakage). Seed 20261128. Original header follows.
sst_hsmm_v1 change 49 (S3): P2F = a bidirectional LoRA on wave17 trained for ANTICIPATORY translation (copy of train_d3.py).
Per sample (the p4 noisy UniST mixtures, step s slot j, as D3), two teacher-forced sequences with wave5's translation prompt:
  (a) full audio -> reference + <|im_end|>                                       (token-mean CE, = D3's CE_a)
  (b) prefix-to-full: a random read c in 1..C-1 (seeded per sample), audio <= c s -> the FULL reference + <|im_end|>
L = CE_a + CE_b (weights 1). Targets are references only: no teacher outputs. Used only on non-final reads (hsc17 --draft-lora,
faithful finalisation, change 48). Monitors: KL(a) to the frozen base. 1000 steps, 16 slots, AdamW lr 1e-4, 50 warm-up, cosine,
seed 20261127. Saves adapters_final.pt.
    torchrun --nproc_per_node 8 -m experiments.sst_hsmm_v1.train_p2f --out eval_outputs/sst_hsmm_v1/p2f/train
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
REV = REPO / "eval_outputs/sst_hsmm_v1/rev"
SRCW = REPO / "eval_outputs/sst_hsmm_v1/d3/src_words.jsonl"
SLOTS, WARM, SAVE_EVERY, STEPS = 16, 50, 250, 1000
LR = 1e-4
DIRS = ("en2zh", "zh2en")
SEED = 20261128
ASR_INS = "Transcribe this speech, then continue with what the speaker is most likely to say next."


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def word_spans(text, words):
    low, p, out = text.lower(), 0, []
    for w in words:
        k = low.find(w["text"].lower(), p)
        if k < 0:
            return None
        out.append((k, k + len(w["text"]), int(w["start_ms"])))
        p = k + len(w["text"])
    return out


def prep():
    import pyarrow.parquet as pq
    if SRCW.exists():
        sys.exit(f"exists: {SRCW}")
    ids = {json.loads(l)["id"] for d in DIRS for l in open(DATA / f"train_{d}.jsonl", encoding="utf-8")}
    SRCW.parent.mkdir(parents=True, exist_ok=True)
    n, bad = 0, 0
    with open(SRCW, "x", encoding="utf-8") as fh:
        for f in sorted((REPO / "data/unist_simul_v1/rows").glob("part-*.parquet")):
            for r in pq.read_table(f, columns=["id", "transcription", "source_words"]).to_pylist():
                if r["id"] not in ids:
                    continue
                ids.discard(r["id"])
                text = r["transcription"].strip()
                ws = word_spans(text, json.loads(r["source_words"]))
                if not ws:
                    bad += 1
                    continue
                fh.write(json.dumps({"id": r["id"], "text": text, "w": ws}, ensure_ascii=False) + "\n")
                n += 1
    log(f"src_words: {n} written, {bad} unaligned, {len(ids)} train ids without a row")


def hidden(thinker, inp):
    head = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        out = thinker(**inp, output_router_logits=False, use_cache=False)
    finally:
        thinker.lm_head = head
    return out.logits


def lsm_rows(thinker, hid, rows_pos, dev):
    """[(batch row, positions (n,))] -> list of (n, V) log-softmax with WAIT masked"""
    out = []
    for b, pos in rows_pos:
        z = thinker.lm_head(hid[b, pos.to(dev)]).float()
        z[:, MD.WAIT] = -1e4
        out.append(torch.log_softmax(z, dim=-1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--max-steps-run", type=int, default=0, help="smoke: stop after this many steps (no final save)")
    ap.add_argument("--smoke-slots", type=int, default=0, help="smoke only: use this many slots per step")
    a = ap.parse_args()
    if a.prep:
        return prep()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    torch.set_num_threads(4)
    dev = f"cuda:{lr_}"
    out = Path(a.out)
    if rank == 0:
        if (out / "adapters_final.pt").exists() or (out / "metrics_rank0.jsonl").exists():
            sys.exit(f"拒绝覆盖已有输出: {out}")
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    refs = {d: {r["id"]: r for r in map(json.loads, open((REV if d == "en2zh" else DATA) / f"train_{d}.jsonl", encoding="utf-8"))}
            for d in DIRS}
    samples = {(x["s"], x["j"]): x for x in map(json.loads, open(REV / "samples.jsonl", encoding="utf-8"))}
    proc, head = MD.processor_and_heads()
    from experiments.sst_hsmm_v1.p2ft_tags import tagged_heads
    head_p, head_c = tagged_heads(proc)          # E1: horizon-tagged heads
    tk = proc.tokenizer
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ASR_INS}]}]
    head_b = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    torch.manual_seed(SEED)                               # identical LoRA init on every rank
    thinker, names = MD.build_thinker(dev, lora=True)
    layers = [m for m in thinker.modules() if isinstance(m, L.LoRALinear)]
    assert len(layers) * 2 == len(names), (len(layers), len(names))
    lora_p = list(names.values())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": LR}], betas=(0.9, 0.999), weight_decay=0.0)

    def lr_at(s):
        return s / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, STEPS - WARM)))

    my_slots = [j for j in range(a.smoke_slots or SLOTS) if j % world == rank]
    assert not a.smoke_slots or a.max_steps_run, "--smoke-slots needs --max-steps-run"
    mf = open(out / f"metrics_rank{rank}.jsonl", "a")
    t0 = time.time()
    for s in range(STEPS):
        if a.max_steps_run and s >= a.max_steps_run:
            break
        for g_ in opt.param_groups:
            g_["lr"] = LR * lr_at(s)
        stats = torch.zeros(4, device=dev)       # samples (a), samples (b), skipped (a), skipped (b)
        dstat = torch.zeros(2, 6, device=dev)    # per dir: n, CE_a, KL_a, CE_b, CE_b heard, CE_b future (token sums / counts below)
        dcnt = torch.zeros(2, 3, device=dev)     # per dir: n_b, heard tokens, future tokens
        texts, auds, tgts, kinds = [], [], [], []
        for j in my_slots:
            x = samples.get((s, j))
            if x is None:                                 # change 65: rev slot without decoded audio (counted as skipped)
                stats[2] += 1
                continue
            r = refs[x["dir"]][x["row"]]
            C, X = int(r["C"]), int(r["X"])
            w = sf.read(x["wav"], dtype="float32")[0]
            texts.append(head_c[r["dir"]] + r["ref"] + "<|im_end|>"); auds.append(w[: X * 16])
            tgts.append(json.loads(r["tok"]) if isinstance(r["tok"], str) else list(r["tok"])); kinds.append((r["dir"], "a", None))
            if C < 2:
                stats[3] += 1
                continue
            c = random.Random(f"{SEED}-{s}-{j}").randint(1, C - 1)
            tg_ = json.loads(r["tok"]) if isinstance(r["tok"], str) else list(r["tok"])
            texts.append(head_p[r["dir"]] + r["ref"] + "<|im_end|>"); auds.append(w[: min(c * 16000, X * 16)])
            tgts.append(tg_); kinds.append((r["dir"], "b", [0] * (len(tg_) + 1)))
        inp = proc(text=texts, audio=auds, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        keep = []
        for b, (tg, kd) in enumerate(zip(tgts, kinds)):
            tg = tg + [MD.IM_END]
            t = ids[b].tolist()
            st = max(i for i, v in enumerate(t) if v == MD.IM_START) + 3
            if t[st: st + len(tg)] != tg:
                stats[2 if kd[1] == "a" else 3] += 1
                continue
            keep.append((b, torch.arange(st - 1, st - 1 + len(tg)), torch.tensor(tg), kd))
        terms = []
        if keep:
            inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
            ka = [(b, pos) for b, pos, tg, kd in keep if kd[1] == "a"]
            if ka:
                with torch.no_grad(), L.adapters_disabled(layers):
                    hid0 = hidden(thinker, inp)
                    base = lsm_rows(thinker, hid0, ka, dev)
                del hid0
            hid = hidden(thinker, inp)
            lsms = lsm_rows(thinker, hid, [(b, pos) for b, pos, tg, kd in keep], dev)
            ia = 0
            for (b, pos, tg, kd), lsm in zip(keep, lsms):
                d, kind, fut = kd
                k_ = DIRS.index(d)
                nll = -lsm.gather(-1, tg.to(dev)[:, None])[:, 0]
                ce = nll.mean()
                terms.append(ce)
                with torch.no_grad():
                    if kind == "a":
                        kl = float((lsm.exp() * (lsm - base[ia])).sum(-1).mean())
                        ia += 1
                        stats[0] += 1
                        dstat[k_, 0] += 1; dstat[k_, 1] += float(ce); dstat[k_, 2] += kl
                    else:
                        f = torch.tensor(fut, device=dev, dtype=torch.bool)
                        stats[1] += 1
                        dstat[k_, 3] += float(ce)
                        dstat[k_, 4] += float(nll[~f].sum()); dstat[k_, 5] += float(nll[f].sum())
                        dcnt[k_, 0] += 1; dcnt[k_, 1] += float((~f).sum()); dcnt[k_, 2] += float(f.sum())
        total = sum(terms) / SLOTS if terms else None
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
        dist.all_reduce(dcnt)
        if rank == 0:
            rec = {"step": s, "lr": LR * lr_at(s), "n_a": int(stats[0]), "n_b": int(stats[1]), "skip_a": int(stats[2]),
                   "skip_b": int(stats[3]), "grad_norm": gn, "sec": round(time.time() - t0, 1)}
            for i_, d in enumerate(DIRS):
                na, nb = max(float(dstat[i_, 0]), 1), max(float(dcnt[i_, 0]), 1)
                rec.update({f"{d}_ce_a": float(dstat[i_, 1]) / na, f"{d}_kl_a": float(dstat[i_, 2]) / na,
                            f"{d}_ce_b": float(dstat[i_, 3]) / nb,
                            f"{d}_ce_b_heard": float(dstat[i_, 4]) / max(float(dcnt[i_, 1]), 1),
                            f"{d}_ce_b_future": float(dstat[i_, 5]) / max(float(dcnt[i_, 2]), 1)})
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if s % 10 == 0:
                log(json.dumps(rec))
        if rank == 0 and (s + 1) % SAVE_EVERY == 0:
            torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / f"adapters_{s + 1:05d}.pt")
    if rank == 0 and not a.max_steps_run:
        torch.save({k2: v.detach().float().cpu() for k2, v in names.items()}, out / "adapters_final.pt")
        json.dump({"method": "P2F-T with reversed-pair en2zh data (change 65)", "base": str(MD.THINKER), "steps": STEPS,
                   "rank": MD.RANK_LORA, "alpha": MD.ALPHA, "world": world, "lr": LR, "seed": SEED, "asr_instruction": None},
                  open(out / "adapters_final.json", "w"))
        log("final saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
