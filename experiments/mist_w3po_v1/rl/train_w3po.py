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
from experiments.mist_w3po_v1.rl.advantage import W3_MODES, admit_anchors, categories, group_advantages, rep_score
from experiments.mist_w3po_v1.rl.rollout import anchors as run_anchors
from experiments.mist_w3po_v1.rl.rollout import load_window, rollout
from experiments.mist_w3po_v1.rl.advantage import ecf_apply, ecf_specs
from experiments.mist_w3po_v1.rl.advantage import hica_weights
from experiments.mist_w3po_v1.rl.rollout import branch_rollout
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

    def __init__(self, windows: list[dict], seed: int, hard_only: bool = False):
        self.pools = [[w for w in windows if hard(w)], [w for w in windows if not hard(w)]]
        self.seed = seed
        self.hard_only = hard_only   # --hard-only (2026-11-08): event-centred sampling, every window "hard"

    def take(self, step: int, n: int) -> list[dict]:
        if self.hard_only:
            pool, out = self.pools[0], []
            for k in range(n):
                p, i = divmod(step * n + k, len(pool))
                perm = np.random.default_rng(self.seed + 1000 * p + 13).permutation(len(pool))
                out.append(pool[perm[i]])
            return out
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
    ap.add_argument("--mode", default="w3po", choices=list(W3_MODES) + ["noC2", "seqpo", "hibiki", "hpo", "dia", "eagrpo"])
    ap.add_argument("--dir", default="zh2en")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--windows-per-step", type=int, default=64)
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lags", default="1000,2000,3000")
    ap.add_argument("--anchor-in-group", type=int, default=1)
    ap.add_argument("--anchor-gate", default="minpol", choices=["none", "minpol"],
                    help="which anchors join the group (advantage.admit_anchors); none = pilot steps 1-13")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-when", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--w-when", type=float, default=1.0,
                    help="weight of the when-channel policy loss (1.0 = the pilot's loss, bit-identical; 2026-11-07 drift fix)")
    ap.add_argument("--w-who", type=float, default=1.0)
    ap.add_argument("--kl-cap", type=float, default=0.0,
                    help="cap of the per-token k3 KL term in the loss (logged KL stays uncapped); 0 = off (default)")
    ap.add_argument("--hard-only", action="store_true",
                    help="sample only 'hard' windows (overlap, an utterance <= 1 s, or >= 3 speaker changes); default: "
                         "the 50/50 hard/normal stream")
    ap.add_argument("--rep-penalty", type=float, default=0.0,
                    help="dia / w3dia / noC1: quality minus this x advantage.rep_score (repeated word 4-gram share); 0 = off")
    ap.add_argument("--who-sc", type=float, default=0.5,
                    help="--mode w3dia2 only: weight of the group-normalised speaker correctness on who tokens")
    ap.add_argument("--who-rbc", type=float, default=0.0,
                    help="--mode w3dia2 only: weight of the group-normalised RBC on who tokens (0 = off)")
    ap.add_argument("--lam-ev", type=float, default=1.0,
                    help="--mode eagrpo only: weight of the event-anchored term (0 = dia exactly)")
    ap.add_argument("--ev-norm", default="center", choices=["center", "std"],
                    help="--mode eagrpo only: per-event x - mean (center, default) or the first version's std division")
    ap.add_argument("--ev-clip", type=float, default=1.0,
                    help="--mode eagrpo only (center): clip of the event term E(t) per token")
    ap.add_argument("--ev-clip-scaled", action="store_true",
                    help="--mode eagrpo only: also clip lam_ev x E(t) to +-sqrt(G - 1) (the backbone's range); default off")
    ap.add_argument("--ecf-branches", type=int, default=0,
                    help="--mode dia only (ECF-GRPO, 2026-11-09): per window at most K counterfactual flush branches at "
                         "missed short reference units; 0 (default) = off, dia unchanged")
    ap.add_argument("--lam-ecf", type=float, default=1.0, help="--ecf-branches only: weight of the branch credit")
    ap.add_argument("--hica-ms", type=int, default=0,
                    help="--mode dia only (HiCA-GRPO, 2026-11-09): hindsight credit allocation with the frozen reference hearing D ms more source audio as the privileged teacher; 0 (default) = off, dia unchanged")
    ap.add_argument("--hica-clip", type=float, default=1.0, help="--hica-ms only: clip of sign(A) x delta (log weight)")
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

    import datetime as _dt
    dist.init_process_group("nccl", timeout=_dt.timedelta(hours=2))   # ranks may lag a lot on shared cards (2026-11-07)
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
    sampler = Sampler(W, args.seed, hard_only=args.hard_only)
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
    use_anchors = args.mode not in ("noC3", "seqpo", "hibiki", "hpo", "dia", "eagrpo") and lags
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
    who_kw = {"who_sc": args.who_sc, "who_rbc": args.who_rbc} if args.mode == "w3dia2" else {}   # --mode w3dia2 only
    if args.mode == "eagrpo":
        who_kw = {"lam_ev": args.lam_ev, "ev_norm": args.ev_norm, "ev_clip": args.ev_clip, "ev_clip_scaled": args.ev_clip_scaled}
    if args.ecf_branches > 0 and args.mode != "dia":
        raise SystemExit("--ecf-branches needs --mode dia")
    if args.hica_ms > 0 and (args.mode != "dia" or args.ecf_branches > 0):
        raise SystemExit("--hica-ms needs --mode dia (and no --ecf-branches)")
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
        ecf = {}   # --ecf-branches: wid -> (specs, branch trajs); empty (default): nothing below changes
        if args.ecf_branches > 0:
            sp_all = []
            for w in mine:
                units_w = S.units_of(w["utts"])
                _, post_w = load_window(w)
                cr_w = [S.credit(tr.segs, units_w, w["dur_ms"] / 1000, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads],
                                 post=post_w) for tr in trajs[w["id"]]]
                sp = ecf_specs(trajs[w["id"]], cr_w, units_w, random.Random(f"ecf|{args.seed}|{step}|{w['id']}"),
                               args.ecf_branches)
                ecf[w["id"]] = (sp, [])
                sp_all += [(w["id"], trajs[w["id"]][m], r) for m, r, _ in sp]
            if sp_all:
                br = branch_rollout(eng, {w["id"]: w for w in mine}, sp_all, temp=args.temp, temp_when=args.temp_when,
                                    gen=gen, max_new=args.max_new, cache=cache)
                for (wid, _, _), b in zip(sp_all, br):
                    ecf[wid][1].append(b)
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
            cred = lambda tr: S.credit(tr.segs, units, dur, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads], post=post)  # noqa: E731
            creds = [cred(tr) for tr in members]
            anc_creds = [cred(tr) for tr in anc_tr]
            aq = [(c.R, c.lat) for c in anc_creds]
            in_group = list(members)
            n_admit = 0
            if args.anchor_in_group and w3 and args.mode != "noC3":
                ok = admit_anchors([c.R for c in creds], [c.R for c in anc_creds], args.anchor_gate)
                in_group += [tr for tr, k in zip(anc_tr, ok) if k]
                creds += [c for c, k in zip(anc_creds, ok) if k]
                n_admit = sum(ok)
            texts = {"".join(s["text"] for s in tr.segs) for tr in members}
            if args.rep_penalty > 0:
                qp = [args.rep_penalty * rep_score(tr.segs) for tr in in_group]
                adv = group_advantages(args.mode, in_group, creds, units, dur, aq, q_penalty=qp, **who_kw)
            else:
                qp = None
                adv = group_advantages(args.mode, in_group, creds, units, dur, aq, **who_kw)
            st = adv["stats"]
            st["dropped"] = len(texts) == 1
            if ecf.get(w["id"], ([], []))[0] and not st["dropped"]:
                sp, br = ecf[w["id"]]
                pen = (lambda segs: args.rep_penalty * rep_score(segs)) if args.rep_penalty > 0 else (lambda segs: 0.0)
                q_m = [S.seq_cpbleu(tr.segs, units) - pen(tr.segs) for tr in members]
                q_b = [S.seq_cpbleu(b.segs, units) - pen(b.segs) for b in br]
                st["ecf"] = ecf_apply(adv["tok"], q_m, sp, q_b, args.lam_ecf)
            st["n_units"] = len(units)
            st["anchors"] = aq
            st["n_admit"] = n_admit
            st["rep"] = [rep_score(tr.segs) for tr in members]     # logged in every mode (loops: rep_score > 0.5)
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
        adv_mag = adv_magnitudes(seqs) if w3 else [0.0] * 6

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
                    logs[k] = logs.get(k, 0.0) + lg[k]
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
               t_roll, t_anc, t_rew, t_upd, float(sum(st["n_admit"] for st in gstats)),
               float(sum(len(st["anchors"]) for st in gstats))] + adv_mag + [
               float(sum(sum(st["rep"]) for st in gstats)), float(sum(sum(r > 0.5 for r in st["rep"]) for st in gstats))]
        EVK = ("units", "units_nz", "abs_when", "n_when", "abs_who", "n_who", "abs_what", "n_what", "short", "short_cov",
               "abs_E", "abs_z", "n_tok", "n_scaled_clip")
        ECFK = ("n", "better", "dq", "abs_extra")
        if args.ecf_branches > 0:   # branch diagnostics (default: loc unchanged)
            loc += [float(sum(st["ecf"][k] for st in gstats if "ecf" in st)) for k in ECFK]
        HK = ("hica_n", "hica_absd", "hica_w", "hica_wclip")
        if args.hica_ms > 0:   # HiCA diagnostics (default: loc unchanged)
            loc += [float(logs.get(k, 0.0)) for k in HK]
        if args.mode == "eagrpo":   # event-term diagnostics (other modes: loc unchanged)
            loc += [float(sum(st["ev"][k] for st in gstats if "ev" in st)) for k in EVK]
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
                   "t_step": time.time() - t0, "gpu_h": (time.time() - t0) * world / 3600, "tokens": cnt.tolist(),
                   "anchor_admit": g[23] / max(g[24], 1),
                   "adv_abs_when": g[25] / max(g[26], 1), "adv_abs_who": g[27] / max(g[28], 1),
                   "adv_abs_what": g[29] / max(g[30], 1),
                   "rep_mean": g[31] / max(g[3], 1), "loop_frac": g[32] / max(g[3], 1)}
            if args.ecf_branches > 0:
                e = dict(zip(ECFK, g[33:]))
                rec.update(ecf_n=e["n"] / ng, ecf_better=e["better"] / max(e["n"], 1), ecf_dq=e["dq"] / max(e["n"], 1),
                           ecf_abs_extra=e["abs_extra"] / max(e["n"], 1))
            if args.hica_ms > 0:
                e = dict(zip(HK, g[33:]))
                rec.update(hica_n=e["hica_n"], hica_absd=e["hica_absd"] / max(e["hica_n"], 1),
                           hica_w=e["hica_w"] / max(e["hica_n"], 1), hica_wclip=e["hica_wclip"] / max(e["hica_n"], 1))
            if args.mode == "eagrpo":
                e = dict(zip(EVK, g[33:]))
                rec.update(ev_units_nz_frac=e["units_nz"] / max(e["units"], 1), ev_abs_when=e["abs_when"] / max(e["n_when"], 1),
                           ev_abs_who=e["abs_who"] / max(e["n_who"], 1), ev_abs_what=e["abs_what"] / max(e["n_what"], 1),
                           ev_short_cov=e["short_cov"] / max(e["short"], 1),
                           ev_E_over_z=e["abs_E"] / max(e["abs_z"], 1e-9),
                           ev_scaled_clip_frac=e["n_scaled_clip"] / max(e["n_tok"], 1))
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


def w3_total(l_when, l_who, l_what, cnt, w_when: float = 1.0, w_who: float = 1.0, w_what: float = 1.0):
    """W3 policy loss: per-channel sums over their global token counts, weighted per channel. w_when = 1.0 gives the
    pilot's l_when / cnt[0] exactly (1.0 * x is exact in IEEE arithmetic; test_w_when.py)."""
    return w_when * l_when / cnt[0] + w_who * l_who / cnt[1] + w_what * l_what / cnt[2]


def adv_magnitudes(seqs) -> list[float]:
    """sum |a| and token counts per channel over the policy (non-anchor) tokens of a step's training sequences:
    [sum|a_when|, n_when, sum|a_who|, n_who, sum|a_what|, n_what] (what = every non-who token, as in the loss)."""
    out = [0.0] * 6
    for tr, r, toks, is_a, w in seqs:
        if is_a:
            continue
        for (wh, who, aw, ah, at) in toks:
            if wh:
                out[0] += abs(aw)
                out[1] += 1
            if who:
                out[2] += abs(ah)
                out[3] += 1
            else:
                out[4] += abs(at)
                out[5] += 1
    return out


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
    hica_logs = {}
    if getattr(args, "hica_ms", 0) > 0:   # HiCA-GRPO (default 0: this block never runs, dia unchanged)
        at, hica_logs = hica_reweight(eng, adapters, items, blocks, seqrows, cache, args, at, tg_t, ref_lp, ref_spk,
                                      spk_k, is_w, is_h, is_anchor)

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
    # --kl-cap (2026-11-08): one sampled tail token with ref - lp ~ 12 nat gave k3 ~ 2.6e5 and a 15x step; cap per token
    k3_loss = k3.clamp(max=args.kl_cap) if getattr(args, "kl_cap", 0.0) > 0 else k3
    kl_what = (k3_loss * (what & ~is_anchor)).sum()
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
        total = w3_total(l_when, l_who, l_what, cnt, args.w_when, args.w_who, args.w_what)
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
    logs.update(hica_logs)
    return loss, logs


def hica_items(items, seqrows, d_ms):
    """Lookahead copies of the read items: audio up to min(t + d_ms, window end), same committed prefix and ctx; the
    akey is the rollout's key of that read time (read times are 1 s multiples capped at the window end), so the
    encoding comes from the rollout cache when it is there."""
    out = []
    for it, (tr, r, toks, is_a, w) in zip(items, seqrows):
        t2 = min(it.t_ms + d_ms, int(w["dur_ms"]))
        out.append(ReadItem(wav=it.wav, post=it.post, t_ms=t2, final=it.final, committed=it.committed, ctx=it.ctx,
                            akey=f"{w['id']}|{t2}"))
    return out


@torch.no_grad()
def hica_reweight(eng, adapters, items, blocks, seqrows, cache, args, at, tg_t, ref_lp, ref_spk, spk_k, is_w, is_h,
                  is_anchor):
    """HiCA-GRPO: teacher = the frozen reference hearing args.hica_ms more audio, teacher-forced on the same blocks.
    delta = teacher - reference log-probability of the sampled token (who tokens: of the 8-way speaker projection,
    so a speaker token at a when position compares only which speaker); pure when tokens and anchors keep weight 1."""
    th, emb_layer = eng.thinker, eng.emb_layer
    items2 = hica_items(items, seqrows, args.hica_ms)
    for j, (it, row) in enumerate(zip(items2, seqrows)):
        if it.wav is None and (cache is None or it.akey not in cache):
            wav, post = load_window(row[4])
            items2[j] = ReadItem(wav=wav, post=post, t_ms=it.t_ms, final=it.final, committed=it.committed, ctx=it.ctx,
                                 akey=it.akey)
    ae = eng.audio_embeds(items2, emb_layer.weight.dtype, cache=cache)
    seqs, tgt_pos = [], []
    for it, (n, _), blk in zip(items2, ae, blocks):
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
    rows, cols = [], []
    for i, (p0, blk) in enumerate(tgt_pos):
        for k in range(len(blk)):
            rows.append(i)
            cols.append(p0 + k)
    rows_t, cols_t = torch.as_tensor(rows, device=eng.dev), torch.as_tensor(cols, device=eng.dev)
    with lora.adapters_disabled(adapters):
        emb = emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        h = th.model(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        lg = th.lm_head(h[rows_t, cols_t]).float()
    del h, emb
    t_lp = torch.log_softmax(lg, -1).gather(1, tg_t[:, None])[:, 0]
    t_spk = torch.log_softmax(lg[:, SPK0:NEW_END], -1)
    del lg
    d_what = t_lp - ref_lp
    d_who = t_spk.gather(1, spk_k[:, None])[:, 0] - ref_spk.gather(1, spk_k[:, None])[:, 0]
    delta = torch.where(is_h, d_who, d_what)
    apply = ~is_anchor & ~(is_w & ~is_h)
    w, clipped = hica_weights(at, delta, apply, args.hica_clip)
    logs = {"hica_n": float(apply.sum()), "hica_absd": float(delta.abs()[apply].sum()),
            "hica_w": float(w[apply].sum()), "hica_wclip": float(clipped.sum())}
    return at * w, logs


if __name__ == "__main__":
    main()
