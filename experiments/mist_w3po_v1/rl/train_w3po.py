"""W3PO trainer (docs/mist_grpo_plan.zh-CN.md §3, §5) and the RL baselines / ablations of §6 (--mode, see advantage.py).

One process per GPU (torchrun), each holding the whole stage-0 merged Thinker (bf16) with a new LoRA (r 32, alpha 64,
q/k/v/o of every text layer; the reference policy is the same module with the adapters off). Per step:
  1. windows: --windows-per-step x --oversample from the training pool (AMI train diarizer windows, 50 % "hard":
     overlap, an utterance <= 1 s or >= 3 speaker changes), each process its share; gold cross-window context (S0-4)
  2. rollouts: G sampled continuations (temperature --temp, --temp-when at when positions), every read of every
     member batched; anchors: greedy fixed-lag Lag1/2/3 with the adapters off (FLAD, C3), cached per window
  3. rewards: STAMP credit per member (stamp.credit), advantages per token (advantage.group_advantages); groups whose
     members all committed the same text are dropped (DAPO dynamic sampling); ACR per token type logged
  4. update: --minibatches optimiser steps over the step's read blocks, teacher-forced in the same HF engine;
     ratios: when -> projected binary (p_wait, 1 - p_wait), who -> softmax over the 8 speaker tokens, what -> GSPO
     geometric mean over the segment's tokens within a read block (baselines: over the read block); clip
     [1 - 0.2, 1 + 0.28]; anchors in the group (--anchor-in-group) use LUFFY's pi / (pi + 0.1) without clipping;
     loss = L_when / N_when + w_who L_who / N_who + w_what L_what / N_what (global token counts) + beta KL to the
     reference (projected KL for when / who, k3 for what)
  5. every --save-every steps: LoRA + optimiser + sampler state under <out>/ckpt/step_<n> (resumable: --resume)
Metrics: <out>/metrics.jsonl (one line per step, rank 0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from experiments.mist_w3po_v1.core.engine import (
    AUDIO_PAD, NEW_END, SPK0, WAIT_ID, Engine, Prompter, ReadItem, ctx_string)
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.advantage import W3_MODES, categories, group_advantages
from experiments.mist_w3po_v1.rl.rollout import anchors as run_anchors
from experiments.mist_w3po_v1.rl.rollout import load_window, rollout
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import ATTN, N_SLOT, SDProjector
from experiments.uniss_streaming_p2st_traj_v1.training import lora

CLIP_LO, CLIP_HI, LUFFY_G = 0.2, 0.28, 0.1


def log0(*a):
    if dist.get_rank() == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_policy(path: str, rank_lora: int, alpha: float, dev: torch.device, direction: str):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"thinker": dev.index, "talker": "cpu", "code2wav": "cpu"})
    thinker = omni.thinker
    del omni
    keep = lambda parent: re.search(r"(?:^|\.)model\.layers\.(\d+)\.", parent + ".") is not None  # noqa: E731
    adapters = lora.inject(thinker, rank=rank_lora, alpha=alpha, targets=ATTN, layer_filter=keep)
    sd = SDProjector(N_SLOT, thinker.config.text_config.hidden_size)
    sd.load_state_dict(torch.load(Path(path) / "sd_proj.pt"))
    sd = sd.to(dev, torch.bfloat16).eval()
    sd.requires_grad_(False)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eng = Engine(thinker, sd, Prompter(direction), dev)
    return eng, adapters


def hard(w: dict) -> bool:
    us = sorted(w["utts"], key=lambda u: u["begin_s"])
    if any(u["end_s"] - u["begin_s"] <= 1.0 for u in us):
        return True
    for i, u in enumerate(us):
        if any(v["speaker_id"] != u["speaker_id"] and min(u["end_s"], v["end_s"]) - max(u["begin_s"], v["begin_s"]) > 0
               for v in us[i + 1:]):
            return True
    return sum(us[i]["speaker_id"] != us[i - 1]["speaker_id"] for i in range(1, len(us))) >= 3


class Sampler:
    """Deterministic stratified stream of windows: alternating hard / normal, each pool reshuffled per pass."""

    def __init__(self, windows: list[dict], seed: int):
        self.pools = [[w for w in windows if hard(w)], [w for w in windows if not hard(w)]]
        self.seed = seed

    def take(self, step: int, n: int) -> list[dict]:
        out = []
        for k in range(n):
            pool = self.pools[k % 2]
            idx = step * ((n + 1) // 2) + k // 2
            p, i = divmod(idx, len(pool))
            perm = np.random.default_rng(self.seed + 1000 * p + 7 * (k % 2)).permutation(len(pool))
            out.append(pool[perm[i]])
        return out


def gold_ctx(w: dict, meet_utts: dict, sess: dict, seed: int, ctx_max: int = 2) -> str:
    rng = random.Random(f"{seed}|{w['id']}")
    x = rng.random()
    k = min(ctx_max, 0 if x < 0.1 else (1 if x < 0.55 else 2))
    prev = [u for u in meet_utts.get(w["meeting"], []) if u[0] <= w["start_s"]][-k:] if k else []
    return ctx_string([{"spk": sess[w["meeting"]][s], "text": t} for _, s, t in prev], k)


def segments_seq(tr) -> list:
    """(read, [positions]) per segment piece within a read, for GSPO ratios."""
    out = {}
    for r, ts in enumerate(tr.tok_seg):
        for p, j in enumerate(ts):
            if j >= 0:
                out.setdefault((r, j), []).append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="stage-0 merged Thinker")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="w3po", choices=list(W3_MODES) + ["noC2", "seqpo", "hibiki", "hpo", "dia"])
    ap.add_argument("--dir", default="zh2en")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--windows-per-step", type=int, default=64)
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lags", default="1000,2000,3000")
    ap.add_argument("--anchor-in-group", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-when", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--w-who", type=float, default=1.0)
    ap.add_argument("--w-what", type=float, default=1.0)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--tok-budget", type=int, default=12000, help="prompt+block tokens per micro-batch")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--anchor-cache", default="", help="comma list of dirs with anchor records (precompute_anchors.py part_*.jsonl or an earlier run's anchors/rank*.jsonl, same base and ctx), read each step")
    ap.add_argument("--ctx-max", type=int, default=2, help="cross-window context segments (0 = v8 prompts, no context)")
    ap.add_argument("--max-dur-ms", type=int, default=20500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ref-src", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--utts", default="data/ami_multispk_si_v1/zh_mix_v8/utts")
    ap.add_argument("--en-norm", default="data/ami_multispk_si_v1/en_norm_v1.jsonl")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    out = Path(args.out)
    if rank == 0:
        if out.exists() and not args.resume:
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        (out / "anchors").mkdir(exist_ok=True)
        (out / "samples").mkdir(exist_ok=True)
        if not args.resume:
            (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dist.barrier()

    # data
    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    W = [w for w in W if w.get("utts") and w["dur_ms"] <= args.max_dur_ms]
    sampler = Sampler(W, args.seed)
    sess = json.loads((Path(args.ref_src) / "slot_map.json").read_text())["lowlat"]
    en = {r["record_id"]: r["en"] for r in map(json.loads, open(args.en_norm, encoding="utf-8"))}
    meet_utts = {}
    for m in {w["meeting"] for w in W}:
        us = []
        for u in map(json.loads, open(Path(args.utts) / f"{m}.jsonl", encoding="utf-8")):
            if u["hes"] or not u.get("placed", True) or any(f.startswith("excluded:") for f in u.get("flags", [])):
                continue
            ref = en.get(u["record_id"], "").strip()
            if ref and u["speaker_id"] in sess.get(m, {}):
                us.append((u["zh_end"], u["speaker_id"], ref))
        meet_utts[m] = sorted(us)
    log0(f"pool {len(W)} windows ({len(sampler.pools[0])} hard / {len(sampler.pools[1])} normal), world {world}")

    eng, adapters = load_policy(args.base, args.lora_rank, args.lora_alpha, dev, args.dir)
    params = lora.trainable_parameters(adapters)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    start = 0
    if args.resume:
        cks = sorted((out / "ckpt").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        cks = [c for c in cks if (c / "DONE").exists()]
        if cks:
            ck = cks[-1]
            lora.load_state_dict(adapters, torch.load(ck / "lora.pt", map_location="cpu"))
            opt.load_state_dict(torch.load(ck / "optim.pt", map_location="cpu"))
            start = int(ck.name.split("_")[1])
            log0(f"resumed from {ck}")
    lags = [int(x) for x in args.lags.split(",")] if args.lags else []
    use_anchors = args.mode not in ("noC3", "seqpo", "hibiki", "hpo", "dia") and lags
    anchor_cache = {}
    acf = out / "anchors" / f"rank{rank}.jsonl"
    if acf.exists():
        for l in open(acf, encoding="utf-8"):
            try:
                r = json.loads(l)
                anchor_cache[r["id"]] = r
            except json.JSONDecodeError:
                pass
    pre_off = {}

    def refresh_precomputed():
        """new complete lines of the precomputed anchor files (precompute_anchors.py) -> anchor_cache."""
        if not args.anchor_cache:
            return 0
        n = 0
        for f in sorted(f for d in args.anchor_cache.split(",") for f in Path(d).glob("*.jsonl")):
            with open(f, "rb") as fh:
                fh.seek(pre_off.get(f, 0))
                data = fh.read()
            end = data.rfind(b"\n") + 1
            pre_off[f] = pre_off.get(f, 0) + end
            for l in data[:end].splitlines():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r["id"] not in anchor_cache:
                    anchor_cache[r["id"]] = r
                    n += 1
        return n

    gen = torch.Generator(device=dev)
    w3 = args.mode in W3_MODES
    n_local = math.ceil(args.windows_per_step * args.oversample / world)

    for step in range(start, args.steps):
        t0 = time.time()
        gen.manual_seed(args.seed * 1000 + step * 97 + rank)
        allw = sampler.take(step, n_local * world)
        mine = allw[rank::world]
        ctxs = {w["id"]: gold_ctx(w, meet_utts, sess, args.seed, args.ctx_max) for w in mine}
        cache = {}
        eng.thinker.eval()
        trajs = rollout(eng, mine, ctxs, args.G, temp=args.temp, temp_when=args.temp_when, gen=gen,
                        max_new=args.max_new, cache=cache)
        t_roll = time.time() - t0
        # anchors (adapters off), cached
        anc = {}
        if use_anchors:
            refresh_precomputed()
            need = [w for w in mine if w["id"] not in anchor_cache]
            if need:
                with lora.adapters_disabled(adapters):
                    got = run_anchors(eng, need, ctxs, lags, max_new=args.max_new, cache=cache)
                with open(acf, "a", encoding="utf-8") as fh:
                    for w in need:
                        rec = {"id": w["id"], "ctx": ctxs[w["id"]], "trajs": [
                            {"kind": tr.kind, "reads": [{"t": rd.t_ms, "final": rd.final, "before": rd.before,
                                                         "ids": rd.ids, "akey": rd.akey} for rd in tr.reads]}
                            for tr in got[w["id"]]]}
                        anchor_cache[w["id"]] = rec
                        fh.write(json.dumps(rec) + "\n")
            from experiments.mist_w3po_v1.rl.rollout import Read, Traj, parse
            for w in mine:
                rec = anchor_cache[w["id"]]
                lst = []
                for t in rec["trajs"]:
                    tr = Traj(w["id"], t["kind"], rec["ctx"])
                    for rd in t["reads"]:
                        n = len(rd["ids"])
                        tr.reads.append(Read(rd["t"], rd["final"], rd["before"], rd["ids"], [False] * n, [False] * n,
                                             [0.0] * n, [0.0] * n, [0.0] * n, rd["akey"]))
                    parse(tr, eng.tok, w["dur_ms"] / 1000)
                    lst.append(tr)
                anc[w["id"]] = lst
        t_anc = time.time() - t0 - t_roll
        # rewards and advantages
        seqs = []     # training sequences: (traj, read, adv tokens, is_anchor, seg pieces)
        gstats = []
        for w in mine:
            units = S.units_of(w["utts"])
            dur = w["dur_ms"] / 1000
            _, post = load_window(w)
            members = list(trajs[w["id"]])
            anc_tr = anc.get(w["id"], [])
            in_group = members + (anc_tr if (args.anchor_in_group and w3 and args.mode != "noC3") else [])
            creds = []
            for tr in in_group:
                creds.append(S.credit(tr.segs, units, dur, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads], post=post))
            aq = [(c.R, c.lat) for c in (S.credit(tr.segs, units, dur, tr.wait_reads,
                                                   [rd.t_ms / 1000 for rd in tr.reads], post=post) for tr in anc_tr)]
            texts = {"".join(s["text"] for s in tr.segs) for tr in members}
            adv = group_advantages(args.mode, in_group, creds, units, dur, aq)
            st = adv["stats"]
            st["dropped"] = len(texts) == 1
            st["n_units"] = len(units)
            st["anchors"] = aq
            st["segs"] = [len(tr.segs) for tr in members]
            st["spurious"] = [c.spurious for c in creds[: len(members)]]
            st["dup"] = [sum(1 for d in c.delta[:] if d < 0) for c in creds[: len(members)]]
            gstats.append(st)
            if st["dropped"]:
                continue
            for m, tr in enumerate(in_group):
                for r, rd in enumerate(tr.reads):
                    if rd.ids:
                        seqs.append((tr, r, adv["tok"][m][r], m >= len(members), w))
        t_rew = time.time() - t0 - t_roll - t_anc

        # global token counts per category
        cnt = torch.zeros(3, device=dev)
        for tr, r, toks, is_a, w in seqs:
            for (wh, who, aw, ah, at) in toks:
                if w3:
                    cnt[0] += wh
                    cnt[1] += who
                    cnt[2] += not who
                else:
                    cnt[2] += 1
        dist.all_reduce(cnt)
        cnt = cnt.clamp(min=1)

        # update
        eng.thinker.train()
        order = list(range(len(seqs)))
        random.Random(step * 31 + rank).shuffle(order)
        mbs = [order[k::args.minibatches] for k in range(args.minibatches)]
        logs = {"kl": 0.0, "clip": 0.0, "ntok": 0.0, "ratio_dev": 0.0, "loss": 0.0, "gnorm": 0.0,
                "p_wait_when": 0.0, "n_when": 0.0}
        for mb in mbs:
            opt.zero_grad(set_to_none=True)
            mb = sorted(mb, key=lambda i: len(seqs[i][0].reads[seqs[i][1]].before) + len(seqs[i][0].reads[seqs[i][1]].ids))
            micro, cur, cur_tok = [], [], 0
            for i in mb:
                tr, r = seqs[i][0], seqs[i][1]
                n = len(tr.reads[r].before) + len(tr.reads[r].ids) + 400
                if cur and cur_tok + n > args.tok_budget:
                    micro.append(cur)
                    cur, cur_tok = [], 0
                cur.append(i)
                cur_tok += n
            if cur:
                micro.append(cur)
            for mi in micro:
                items, blocks = [], []
                for i in mi:
                    tr, r, toks, is_a, w = seqs[i]
                    rd = tr.reads[r]
                    wav, post = load_window(w) if rd.akey not in cache else (None, None)
                    items.append(ReadItem(wav=wav, post=post, t_ms=rd.t_ms, final=rd.final, committed=rd.before,
                                          ctx=tr.ctx, akey=rd.akey))
                    blocks.append(rd.ids)
                loss, lg = micro_loss(eng, adapters, items, blocks, [seqs[i] for i in mi], cache, cnt, args, w3)
                loss.backward()
                for k in lg:
                    logs[k] += lg[k]
            for p in params:   # every rank all-reduces the same tensors, even with no tokens of its own
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            buf = torch._utils._flatten_dense_tensors([p.grad for p in params])
            dist.all_reduce(buf)
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(buf, [p.grad for p in params])):
                p.grad.copy_(g)
            del buf
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            logs["gnorm"] += float(gn) / args.minibatches
            opt.step()
        eng.thinker.eval()
        t_upd = time.time() - t0 - t_roll - t_anc - t_rew

        # metrics
        def agg(vals):
            x = torch.tensor(vals, dtype=torch.float64, device=dev)
            dist.all_reduce(x)
            return x.tolist()
        Rs = [v for st in gstats for v in st["R"][: args.G]]
        acr = [st.get("acr_when"), st.get("acr_who"), st.get("acr_what")]
        loc = [len(gstats), sum(st["dropped"] for st in gstats), float(np.sum(Rs)), float(len(Rs)),
               float(np.sum([np.std(st["R"][: args.G]) for st in gstats])),
               float(np.sum([np.mean(st["lat"][: args.G]) for st in gstats])),
               float(sum(1 for st in gstats if st.get("acr_when"))), float(sum(1 for st in gstats if st.get("acr_who"))),
               float(sum(1 for st in gstats if st.get("acr_what"))),
               float(sum(sum(st["segs"]) for st in gstats)), float(sum(sum(st["spurious"]) for st in gstats)),
               float(np.sum([np.mean(st.get("r_lat", [0])[: args.G]) for st in gstats])),
               float(np.sum([np.mean([a[0] for a in st["anchors"]]) if st["anchors"] else 0 for st in gstats])),
               logs["kl"], logs["clip"], logs["ntok"], logs["ratio_dev"], logs["p_wait_when"], logs["n_when"],
               t_roll, t_anc, t_rew, t_upd]
        g = agg(loc)
        if rank == 0:
            ng = max(g[0], 1)
            rec = {"step": step + 1, "groups": int(g[0]), "dropped": int(g[1]), "R": g[2] / max(g[3], 1),
                   "R_std_in_group": g[4] / ng, "lat": g[5] / ng, "ACR_when": g[6] / ng, "ACR_who": g[7] / ng,
                   "ACR_what": g[8] / ng, "segs_per_traj": g[9] / max(g[3], 1), "spurious_per_traj": g[10] / max(g[3], 1),
                   "r_lat": g[11] / ng, "anchor_R": g[12] / ng, "kl": g[13] / max(g[15], 1),
                   "clip_frac": g[14] / max(g[15], 1), "ratio_dev": g[16] / max(g[15], 1),
                   "p_wait_at_when": g[17] / max(g[18], 1), "gnorm": logs["gnorm"],
                   "t_rollout": g[19] / world, "t_anchor": g[20] / world, "t_reward": g[21] / world, "t_update": g[22] / world,
                   "t_step": time.time() - t0, "gpu_h": (time.time() - t0) * world / 3600, "tokens": cnt.tolist()}
            with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            log0(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
        if step < 3 or (step + 1) % 10 == 0:   # a few sampled trajectories for inspection
            with open(out / "samples" / f"rank{rank}.jsonl", "a", encoding="utf-8") as fh:
                w = mine[0]
                fh.write(json.dumps({"step": step + 1, "id": w["id"], "ref": [u["ref"] for u in S.units_of(w["utts"])],
                                     "R": gstats[0]["R"][: args.G],
                                     "trajs": [[(s["spk"], s["a"], s["b"], s["text"], s["r"]) for s in tr.segs]
                                               for tr in trajs[w["id"]]],
                                     "anchors": [[(s["spk"], s["a"], s["b"], s["text"], s["r"]) for s in tr.segs]
                                                 for tr in anc.get(w["id"], [])]}, ensure_ascii=False) + "\n")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ck = out / "ckpt" / f"step_{step + 1:05d}"
            if rank == 0:
                ck.mkdir(parents=True, exist_ok=True)
                torch.save(lora.state_dict(adapters), ck / "lora.pt")
                torch.save(opt.state_dict(), ck / "optim.pt")
                (ck / "DONE").write_text(json.dumps({"step": step + 1, "time": time.time()}))
            dist.barrier()
        del cache
        torch.cuda.empty_cache()
    log0("W3PO_TRAIN_DONE")
    dist.destroy_process_group()


def micro_loss(eng, adapters, items, blocks, seqrows, cache, cnt, args, w3):
    """Teacher-forced log-probabilities of the blocks under the policy (grad) and the reference (no grad), and the
    W3 / baseline loss of the micro-batch."""
    th = eng.thinker
    emb_layer = eng.emb_layer
    ae = eng.audio_embeds(items, emb_layer.weight.dtype, cache=cache)
    seqs, tgt_pos = [], []
    for it, (n, _), blk in zip(items, ae, blocks):
        p = eng.pr.ids(n, it.ctx, it.committed)
        seqs.append(p + blk[:-1])
        tgt_pos.append((len(p) - 1, blk))
    B, L = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(s)
        mask[i, : len(s)] = 1
    ids, mask = ids.to(eng.dev), mask.to(eng.dev)
    rows, cols, tg = [], [], []
    for i, (p0, blk) in enumerate(tgt_pos):
        for k, t in enumerate(blk):
            rows.append(i)
            cols.append(p0 + k)
            tg.append(t)
    rows_t, cols_t = torch.as_tensor(rows, device=eng.dev), torch.as_tensor(cols, device=eng.dev)
    tg_t = torch.as_tensor(tg, device=eng.dev)

    def fwd():
        emb = emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        h = th.model(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        return th.lm_head(h[rows_t, cols_t]).float()

    with torch.no_grad(), lora.adapters_disabled(adapters):
        ref_lg = fwd()
        ref_ls = torch.log_softmax(ref_lg, -1)
        ref_lp = ref_ls.gather(1, tg_t[:, None])[:, 0]
        ref_pw = ref_ls[:, WAIT_ID].exp()
        ref_spk = torch.log_softmax(ref_lg[:, SPK0:NEW_END], -1)
        del ref_lg, ref_ls
    lg = fwd()
    ls = torch.log_softmax(lg, -1)
    lp = ls.gather(1, tg_t[:, None])[:, 0]
    pw = ls[:, WAIT_ID].exp().clamp(1e-6, 1 - 1e-6)
    lspk = torch.log_softmax(lg[:, SPK0:NEW_END], -1)
    del lg, ls

    # flatten per-token metadata in the same order
    old_lp, old_lw, old_lh, aw, ah, at, is_w, is_h, is_anchor, chose_wait, spk_k, seg_key = ([] for _ in range(12))
    for (tr, r, toks, is_a, w), blk in zip(seqrows, blocks):
        rd = tr.reads[r]
        for k, t in enumerate(blk):
            wh, who, a_w, a_h, a_t = toks[k]
            is_w.append(wh)
            is_h.append(who)
            aw.append(a_w)
            ah.append(a_h)
            at.append(a_t)
            is_anchor.append(is_a)
            chose_wait.append(t == WAIT_ID)
            spk_k.append(max(0, min(N_SLOT - 1, t - SPK0)))
            old_lp.append(rd.logp[k] if not is_a else 0.0)
            old_lw.append(rd.logp_wait[k] if not is_a else 0.0)
            old_lh.append(rd.logp_who[k] if not is_a else 0.0)
            seg_key.append((id(tr), r, tr.tok_seg[r][k]) if (w3 and not who) else (id(tr), r, -2))
    T = lambda x, dt=torch.float32: torch.as_tensor(x, device=eng.dev, dtype=dt)  # noqa: E731
    is_w, is_h, is_anchor, chose_wait = T(is_w, torch.bool), T(is_h, torch.bool), T(is_anchor, torch.bool), T(chose_wait, torch.bool)
    aw, ah, at = T(aw), T(ah), T(at)
    old_lp, old_lw, old_lh = T(old_lp), T(old_lw), T(old_lh)
    spk_k = T(spk_k, torch.long)

    def ppo(ratio, adv, mask_):
        if not bool(mask_.any()):
            return torch.zeros((), device=eng.dev), 0.0
        r_ = ratio[mask_]
        a_ = adv[mask_]
        anc = is_anchor[mask_]
        s1 = r_ * a_
        s2 = r_.clamp(1 - CLIP_LO, 1 + CLIP_HI) * a_
        l_pol = -torch.minimum(s1, s2)
        luffy = -(r_ / (r_ + LUFFY_G)) * a_       # anchors: pi / (pi + gamma) with pi = ratio's numerator
        l = torch.where(anc, luffy, l_pol)
        clipped = ((r_ < 1 - CLIP_LO) | (r_ > 1 + CLIP_HI)) & ~anc
        return l.sum(), float(clipped.float().sum())

    logs = {}
    total = torch.zeros((), device=eng.dev)
    kl_sum = torch.zeros((), device=eng.dev)
    clip = 0.0
    # what (and every token in single-advantage modes): GSPO over segment pieces (or read blocks)
    what = ~is_h if w3 else torch.ones_like(is_h)
    keys = {}
    for k, key in enumerate(seg_key):
        if bool(what[k]):
            keys.setdefault(key, []).append(k)
    seq_ratio = torch.ones_like(lp)
    for key, idx in keys.items():
        ii = torch.as_tensor(idx, device=eng.dev)
        if bool(is_anchor[ii[0]]):
            seq_ratio[ii] = lp[ii].exp()      # LUFFY on the token probability
        else:
            s = torch.exp((lp[ii] - old_lp[ii]).mean())
            seq_ratio[ii] = s
    l_what, c = ppo(seq_ratio, at, what)
    clip += c
    k3 = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1
    kl_what = (k3 * (what & ~is_anchor)).sum()
    if w3:
        # when: projected binary policy
        p_choice = torch.where(chose_wait, pw, 1 - pw)
        old_choice = torch.where(chose_wait, old_lw.exp(), 1 - old_lw.exp()).clamp(1e-6, 1)
        r_when = torch.where(is_anchor, p_choice, p_choice / old_choice)
        l_when, c = ppo(r_when, aw, is_w)
        clip += c
        rp = ref_pw.clamp(1e-6, 1 - 1e-6)
        kl_b = pw * (pw / rp).log() + (1 - pw) * ((1 - pw) / (1 - rp)).log()
        kl_when = (kl_b * (is_w & ~is_anchor)).sum()
        # who: projected 8-way policy
        lh = lspk.gather(1, spk_k[:, None])[:, 0]
        r_who = torch.where(is_anchor, lh.exp(), torch.exp(lh - old_lh))
        l_who, c = ppo(r_who, ah, is_h)
        clip += c
        kl_8 = (lspk.exp() * (lspk - ref_spk)).sum(-1)
        kl_who = (kl_8 * (is_h & ~is_anchor)).sum()
        total = l_when / cnt[0] + args.w_who * l_who / cnt[1] + args.w_what * l_what / cnt[2]
        kl_sum = kl_when / cnt[0] + kl_who / cnt[1] + kl_what / cnt[2]
        logs["p_wait_when"] = float(pw[is_w & ~is_anchor].sum())
        logs["n_when"] = float((is_w & ~is_anchor).sum())
    else:
        total = l_what / cnt[2]
        kl_sum = kl_what / cnt[2]
    loss = total + args.beta * kl_sum
    pol = ~is_anchor
    logs["kl"] = float(k3[pol].sum())
    logs["clip"] = clip
    logs["ntok"] = float(pol.sum())
    logs["ratio_dev"] = float((lp - old_lp)[pol].abs().sum())
    logs["loss"] = float(loss)
    return loss, logs


if __name__ == "__main__":
    main()
