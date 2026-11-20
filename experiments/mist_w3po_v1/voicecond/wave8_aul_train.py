"""Robust wave 8, Agreement-Unlikelihood (AUL; reports/mist_w3po_v1/robust_wave8_aul_prereg.zh-CN.md): the wave-5 ERM
recipe (wave5_m_train.py --arm erm: same model, LoRA, prompt, data, sampler random numbers, optimiser, schedule, 2300
steps, global batch 64 = 8 envs x 8) plus an unlikelihood term on LocalAgreement-2's agreed-but-wrong unit.

AUL samples: on every step, each rank takes its first slot of every environment (global slot j = 8 e + rank; world 4 ->
32 per step). Read window w = 1 s; k = 1 + sha1("20261120|w8aul|cut|s|j") mod (floor(dur) - 1) (skipped when
floor(dur) < 2). In eval mode under no_grad the Thinker (LoRA active) greedily continues the base prompt (empty assistant
prefix) on x[:k w] and on x[:(k+1) w], capped at max(#reference units + 8, 16) new tokens; a draft that hits its cap
without <|im_end|> loses its last (possibly cut) unit. Units are si_thinker_stream.units (English words / Chinese
characters); compared after norm(): English words without punctuation, lower-cased; Chinese characters as they are.
A = the longest common prefix of the two drafts (raw units of the later draft); j = the first position where A differs
from the reference R. If j < |A|, A_j is agreed but wrong: the UL context is the prompt on x[:(k+1) w] with the
assistant prefix join(R[:j]) and y- = the first token of the draft's continuation from A_j (leading space in English
when j > 0); l_UL = -log(1 - p(y- | context)). If y- equals the first token of the reference continuation (a shared BPE
prefix, e.g. " un"), the sample has no negative (the UL term must never push down the reference token).
Loss: L = mean_i CE_i + LAM_UL * mean over the global negatives of l_UL, LAM_UL = 1.0; a step without negatives is
plain ERM. Saves adapters only (fp32 a/b), every 500 steps (resume) and at the end.

    torchrun --nproc-per-node 4 experiments/mist_w3po_v1/voicecond/wave8_aul_train.py --steps 2300 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from experiments.mist_w3po_v1.voicecond import wave5_m_train as W5  # noqa: E402  (unchanged; only imported)
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units, join  # noqa: E402

SR, WIN = 16000, 1.0
LAM_UL = 1.0
IM_START, IM_END, IGNORE = W5.IM_START, W5.IM_END, W5.IGNORE
ENVS, PER_ENV = W5.ENVS, W5.PER_ENV


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


log = W5.log


# ------------------------------------------------------------------------------------------------ unit logic (pure)
def norm(u: str, zh: bool) -> str:
    return u if zh else "".join(c for c in u if not unicodedata.category(c).startswith("P")).lower()


def agreed_wrong(d1: list[str], d2: list[str], ref: list[str], zh: bool):
    """(j, A) when LA2 on the drafts d1 (earlier read), d2 (later read) agrees on a prefix A whose unit j is the first
    one differing from the reference ref; None when the agreed prefix is all correct (or empty)."""
    n1, n2, nr = [norm(u, zh) for u in d1], [norm(u, zh) for u in d2], [norm(u, zh) for u in ref]
    L = 0
    while L < min(len(n1), len(n2)) and n1[L] == n2[L]:
        L += 1
    j = 0
    while j < L and j < len(nr) and n2[j] == nr[j]:
        j += 1
    return (j, d2[:L]) if j < L else None


def cut_k(s: int, j: int, n_samples: int):
    """the read index k for global slot j at step s (None: clip shorter than 2 windows)"""
    m = int(n_samples / SR // WIN) - 1
    return None if m < 1 else 1 + h(f"20261120|w8aul|cut|{s}|{j}") % m


def cont_text(us: list[str], j: int, zh: bool) -> str:
    """the reply's text from unit j on, as it follows join(us[:j])"""
    return join(us[j:], zh) if zh else ("" if j == 0 else " ") + join(us[j:], zh)


# ------------------------------------------------------------------------------------------------------------- data
class StepData(W5.StepData):
    """wave 5's microbatches (identical samples) + this rank's AUL generation inputs (left-padded prompts on the cuts)"""

    def __iter__(self):
        from transformers import Qwen3OmniMoeProcessor
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
        wi = torch.utils.data.get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        proc = Qwen3OmniMoeProcessor.from_pretrained(str(W5.QWEN3))
        ins = inference_instruction()
        head = {}
        for d in ("zh2en", "en2zh"):
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[W5.TGT[d]]}]}]
            head[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        S = W5.Sampler()
        slots = [j for j in range(len(ENVS) * PER_ENV) if (j % PER_ENV) % self.world == self.rank]
        for s in range(self.start, self.steps):
            if (s - self.start) % nw != wid:
                continue
            items = [S.sample(s, j) for j in slots]
            mbs = []
            proc.tokenizer.padding_side = "right"
            for k in range(0, len(items), self.mb):
                part = items[k: k + self.mb]
                texts = [head[it["dir"]] + it["ref"] + "<|im_end|>" for it in part]
                inp = proc(text=texts, audio=[it["wav"] for it in part], return_tensors="pt", padding=True)
                ids = inp["input_ids"]
                lab = torch.full_like(ids, IGNORE)
                for b in range(ids.shape[0]):
                    row = ids[b].tolist()
                    st = max(i for i, t in enumerate(row) if t == IM_START) + 3
                    en = max(i for i, t in enumerate(row) if t == IM_END)
                    assert en > st, "empty target"
                    lab[b, st: en + 1] = ids[b, st: en + 1]
                mbs.append({**{k2: v for k2, v in inp.items()}, "labels": lab,
                            "env": torch.tensor([it["env"] for it in part]), "ids": [it["id"] for it in part]})
            # AUL: first slot of every env on this rank
            aul = []
            for j, it in zip(slots, items):
                if j % PER_ENV != self.rank:
                    continue
                k = cut_k(s, j, len(it["wav"]))
                if k is None:
                    continue
                zh = it["dir"] == "en2zh"
                ru = units(it["ref"], zh)
                aul.append({"slot": j, "k": k, "dir": it["dir"], "zh": zh, "ref": it["ref"], "ref_units": ru,
                            "cap": max(len(ru) + 8, 16), "head": head[it["dir"]], "id": it["id"],
                            "wav_long": it["wav"][: int((k + 1) * WIN * SR)]})
            gen = None
            if aul:
                proc.tokenizer.padding_side = "left"
                cuts = [a_["wav_long"][: int(a_["k"] * WIN * SR)] for a_ in aul] + [a_["wav_long"] for a_ in aul]
                gen = dict(proc(text=[a_["head"] for a_ in aul] * 2, audio=cuts, return_tensors="pt", padding=True))
                proc.tokenizer.padding_side = "right"
            yield s, mbs, aul, gen


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
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
    thinker, names, _ = W5.build_model(dev)
    params = list(names.values())
    from transformers import Qwen3OmniMoeProcessor
    proc = Qwen3OmniMoeProcessor.from_pretrained(str(W5.QWEN3))
    tok = proc.tokenizer
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

    dl = torch.utils.data.DataLoader(StepData(start, a.steps, rank, world, a.mb), batch_size=None,
                                     num_workers=a.workers, prefetch_factor=2, persistent_workers=False)
    metrics = open(out / f"metrics_rank{rank}.jsonl", "a", encoding="utf-8")
    examples = open(out / f"aul_examples_rank{rank}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    n_env, n_glob = len(ENVS), len(ENVS) * PER_ENV
    for s, mbs, aul, gen in dl:
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        # ---- AUL: two greedy drafts per sample, eval mode, no grad
        negs = []
        tg = time.time()
        for x in aul:   # the DataLoader's default_convert turned the numpy cuts into tensors
            x["wav_long"] = np.asarray(x["wav_long"], dtype=np.float32)
        if aul:
            thinker.eval()
            with torch.no_grad():
                kw = {k: v.to(dev) for k, v in gen.items() if k in ("input_ids", "attention_mask", "feature_attention_mask")}
                kw["input_features"] = gen["input_features"].to(dev, torch.bfloat16)
                seq = thinker.generate(**kw, max_new_tokens=max(x["cap"] for x in aul), do_sample=False,
                                       eos_token_id=IM_END, pad_token_id=IM_END, use_cache=True)
            thinker.train()
            thinker.audio_tower.eval()
            new = seq[:, kw["input_ids"].shape[1]:].tolist()
            na = len(aul)
            drafts = []
            for b in range(2 * na):
                x = aul[b % na]
                t_ = new[b][: x["cap"]]
                ended = IM_END in t_
                if ended:
                    t_ = t_[: t_.index(IM_END)]
                u = units(tok.decode(t_, skip_special_tokens=True), x["zh"])
                drafts.append(u if ended else u[:-1])
            for i, x in enumerate(aul):
                r = agreed_wrong(drafts[i], drafts[na + i], x["ref_units"], x["zh"])
                if r is None:
                    continue
                j, A = r
                y_neg = tok(cont_text(A, j, x["zh"]), add_special_tokens=False)["input_ids"][0]
                y_ref = (tok(cont_text(x["ref_units"], j, x["zh"]), add_special_tokens=False)["input_ids"][0]
                         if j < len(x["ref_units"]) else IM_END)
                if y_neg == y_ref:
                    continue
                negs.append({**x, "j": j, "A": A, "y_neg": y_neg, "d1": drafts[i], "d2": drafts[na + i]})
        t_gen = time.time() - tg
        cnt = torch.tensor([float(len(negs)), float(len(aul))], device=dev)
        dist.all_reduce(cnt)
        n_neg_g, n_aul_g = int(cnt[0].item()), int(cnt[1].item())
        ul_sum = torch.zeros((), device=dev)
        if negs:
            texts = [x["head"] + join(x["ref_units"][: x["j"]], x["zh"]) for x in negs]
            inp = proc(text=texts, audio=[x["wav_long"] for x in negs], return_tensors="pt", padding=True)
            kw = {k: v.to(dev) for k, v in inp.items() if k in ("input_ids", "attention_mask", "feature_attention_mask")}
            kw["input_features"] = inp["input_features"].to(dev, torch.bfloat16)
            o = thinker(**kw, return_dict=True)
            am = kw["attention_mask"]   # the Qwen3-Omni processor pads on the left whatever tokenizer.padding_side says
            last = (am * torch.arange(am.shape[1], device=dev)).argmax(1)
            lg = o.logits[torch.arange(len(negs), device=dev), last].float()
            p = lg.softmax(-1)[torch.arange(len(negs), device=dev), torch.tensor([x["y_neg"] for x in negs], device=dev)]
            ul = -torch.log1p(-p.clamp(max=1 - 1e-6))
            ul_sum = ul.detach().sum()
            (LAM_UL * ul.sum() / n_neg_g * world).backward()
            del o, lg
            if rank == 0 and (s % 10 == 0 or s < 5):
                for x, pv in zip(negs, p.tolist()):
                    examples.write(json.dumps({"step": s, "slot": x["slot"], "id": x["id"], "k": x["k"], "dir": x["dir"],
                                               "ref": x["ref"], "d1": join(x["d1"], x["zh"]), "d2": join(x["d2"], x["zh"]),
                                               "j": x["j"], "A_j": x["A"][x["j"]], "y_neg": tok.decode([x["y_neg"]]),
                                               "p": round(pv, 4)}, ensure_ascii=False) + "\n")
                examples.flush()
        dist.all_reduce(ul_sum)
        # ---- CE, identical to wave 5 ERM
        ce_env = torch.zeros(n_env, device=dev)
        cnt_env = torch.zeros(n_env, device=dev)
        for mb in mbs:
            env = mb["env"].to(dev)
            kw = {k: v.to(dev) for k, v in mb.items() if k in ("input_ids", "attention_mask", "feature_attention_mask")}
            kw["input_features"] = mb["input_features"].to(dev, torch.bfloat16)
            lab = mb["labels"].to(dev)
            out_ = thinker(**kw, return_dict=True)
            tl = lab[:, 1:]
            sel = tl != IGNORE
            tk = F.cross_entropy(out_.logits[:, :-1][sel].float(), tl[sel], reduction="none")
            bidx = sel.nonzero()[:, 0]
            ce_i = torch.zeros(lab.shape[0], device=dev).index_add(0, bidx, tk) / sel.sum(1).float()
            ce_env.index_add_(0, env, ce_i.detach())
            cnt_env.index_add_(0, env, torch.ones_like(ce_i))
            (ce_i.sum() / n_glob * world).backward()
            del out_
        bad = torch.zeros((), device=dev)
        for p_ in params:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
            dist.all_reduce(p_.grad)
            p_.grad /= world
            bad += (~torch.isfinite(p_.grad)).sum()
        dist.all_reduce(bad)
        if bad.item() > 0:
            log(f"step {s}: non-finite gradient ({int(bad.item())} entries), step skipped")
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(ce_env)
        dist.all_reduce(cnt_env)
        ce_e = (ce_env / cnt_env.clamp(min=1)).tolist()
        if rank == 0:
            rec = {"step": s, "lr": lr_at(s), "ce": float(sum(ce_e) / n_env), "ce_env": ce_e,
                   "ul": float(ul_sum.item() / n_neg_g) if n_neg_g else None, "n_neg": n_neg_g, "n_aul": n_aul_g,
                   "skipped": bool(bad.item() > 0), "gen_sec": round(t_gen, 1), "step_sec": round(time.time() - ts, 1),
                   "sec": round(time.time() - t0, 1), "mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            if s % 10 == 0 or s < 5:
                log(json.dumps(rec))
        done = s + 1
        if a.max_steps_run and done - start >= a.max_steps_run:
            log(f"smoke stop after {done - start} steps")
            break
        if done % 500 == 0 and done < a.steps and rank == 0:
            torch.save({"step": done, "adapters": W5.adapter_state(names), "opt": opt.state_dict()}, out / f"resume_{done:05d}.pt")
        if done == a.steps and rank == 0:
            torch.save(W5.adapter_state(names), out / "adapters_final.pt")
            json.dump({"arm": "aul", "steps": a.steps, "rank": W5.RANK_LORA, "alpha": W5.ALPHA, "world": world,
                       "per_env": PER_ENV, "lam": 0.0, "lam_ul": LAM_UL, "win": WIN}, open(out / "adapters_final.json", "w"))
            log("final adapters saved")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
