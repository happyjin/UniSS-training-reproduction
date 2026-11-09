"""Scheme 1' trainer and its shared control C0 (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §10.2-§10.5, §11;
design note: reports/mist_w3po_v1/sched_crf_cpu_stage.zh-CN.md §8): continue training v8 on the sched_v1 windows.

Model (= v8's SFT model, thinker_gstar_model.ThinkerGStarModel with --gstar-init-merged, re-hosted in HF like
rl/train_w3po.py): the v8 merged Thinker (bf16, frozen) + a new LoRA r 32 / alpha 64 on q/k/v/o of all 48 text layers
(b = 0 at start, so step 0 is v8 exactly) + the speaker projector (fp32, from v8's sd_proj.pt) + the 259 new token rows
(new_emb / new_head, fp32, from v8's merged rows). ThinkerGStarModel.embed / .logits_at are called unbound on this
module, so the forward is v8's SFT code.

Data (identical for both objectives): one window of eval_outputs/mist_w3po_v1/crf/sched_v1 (AMI train, v8 recipe: S,
reads t_r, lo/hi, lag, heuristic schedule beta) gives R rows; row r = v8 prompt (audio <= t_r, posteriors zeroed from
max(0, t_r - 1040) // 80, v8's --gstar-perturb 0.2 augmentation, ctx "") + S. One forward per row gives at the
position after S^{<b}:  tok_lp[r, b] = log p(S_b | .)  and  end_lp[r, b] = log p(e_r | .),  e_r = <wait>, <eos> at the
last read. Rows are cut after their last needed position and end with e_r as input, exactly v8's row layout
(causal attention: neither changes a read-out position).

  --objective heuristic  (C0)  one-hot weights from beta: the committed block S[beta_{r-1}:beta_r) and e_r at beta_r.
        The row is then literally v8's SFT row (prefix S[:beta_{r-1}], target block, end) and the loss is v8's
        (test_train_crf_sft (a): equal to ThinkerGStarModel.forward on rows built by v8's build_batch).
  --objective sched      (scheme 1')
        1. pass 1 (no grad): tok_lp / end_lp of every row of every window of the step (rows cut at
           b_max(r) = first token with lo > r; later states are unreachable); at the last read <wait> is barred
           before the softmax at every position, as engine.read does when decoding (engine.py:246) and as
           precheck_sched.py scores, so q below is the exact posterior of v8's own decoding (--final-bar-wait 1)
        2. lam* by bisection (schedule_lattice.solve_lambda) on the step's whole global batch so that the mean over
           its windows of E_q[Lat] = --lat-target (L_ref = 0.826 s, sched_v1/summary.json); q_lam ~ p(S,z|x) e^{-lam Lat}
        3. w_tok[r, j] = P_q(beta_{r-1} <= j < beta_r), w_end[r, b] = P_q(beta_r = b) (schedule_lattice.marginals)
        4. pass 2 (grad): loss = -(sum h_j w_tok tok_lp + sum w_end end_lp) / D  +  state_w * L_state
           L_state = mean over A of -tok_lp[r, j], A = the (r, j) that some feasible schedule commits (lo <= r <= hi
           and reachable; 5.9 % of the lo/hi pairs are unreachable: their prefix would hold unheard text), pooled
           over the global batch; state_w = 1 (pre-registered)
        Fisher identity: with h = 1 and --final-bar-wait 0 the first term's gradient is exactly -grad log Z_lam
        (test (b)). The loss keeps v8's plain cross-entropy at the last read in BOTH objectives (so the two arms
        differ only in their weights); with the bar (default) the update therefore adds, to grad J of the decode
        model, v8-SFT's push-down of p(<wait>) at last-read positions, weighted by q.
  Normalisation (both objectives): D = sum over the global batch of (sum_j h_j + R) = the total weight, the same for
  both because every token is committed once and every read ends once (sum_r w_tok[r, j] = 1, sum_b w_end[r, b] = 1;
  test (d)). h_j = v8's hierarchical weights (time x1.5, speaker x2; --hier-weights 0: h = 1). With h != 1 the update
  is an EM step whose E-step is the exact model posterior and whose M-step is v8's weighted log-likelihood.
  The loss at the final <eos> is v8's plain cross-entropy in both objectives (C0 = v8 SFT exactly).

Recipe match with v8 (thinker_gstar_s2dl_zh2en_3k_20261030: run_thinker_gstar.sh via run_v5_train.sh): AdamW
betas (0.9, 0.95), eps 1e-8, weight decay 0, global grad-norm clip 1.0, lr 1e-4 with 50 linear warm-up iterations
(lr_k = 1e-4 k / 50) then cosine to 1e-5 at the last step (Megatron's schedule), 3000 steps, base bf16 + fp32
trainable parameters, posterior perturbation p 0.2. Global batch: v8 = 64 rows per step (GBS 64, 8 ranks x MBS 8);
here every step takes whole windows from a seeded per-pass permutation until the step holds ~64 read rows (a window
is added while rows + R/2 <= 64), so a step is ~64 rows, 3000 steps ~1.0 pass over the 188,853 read rows (v8: 1.19
passes over its 160,957 rows, 81 % read rows). Loss = the global token-weighted mean over the step (Megatron: the
mean of the per-micro-batch means). Rows are micro-batched by --tok-budget (padded tokens); gradients are summed over
micro-batches and ranks (the loss is already divided by the global D).

Checkpoints: <out>/ckpt/step_NNNNN/ with lora.pt (lora.state_dict, the rl/train_w3po.py format, loadable by
evaluation/online_decode.py --lora), optim.pt, extra.pt (sd_proj, new_emb, new_head, extra_params, last lam) and DONE.
The LoRA alone is NOT the model: the projector and the token rows also move. Use crf/export_merged.py (a merged
Thinker dir for eval_pipeline.sh / launch_decode.sh / train_w3po --base) or crf/decode_crf_ckpt.py (online_decode
with the checkpoint applied in memory). --resume continues from the newest DONE checkpoint (the data stream is a pure
function of the step).

HOOK API (stable; scheme 2 / TurnCRF plugs in here; default None everywhere = v8 bit-identical)

    from experiments.mist_w3po_v1.crf import train_crf_sft as T
    T.main(argv, post_transform=fn, extra_loss=fn2, extra_params=[module, ...])

  Where P enters the model: v8's loader turns the diarizer posteriors P [F, 8] (80 ms frames) into the audio-token
  grid (thinker_gstar_model.post_to_tokens, linear interpolation), ThinkerGStarModel.embed feeds that to the speaker
  projector and interleaves the result K:1 with the audio-encoder states (engine.Engine.audio_embeds: the same at
  inference). post_transform sits right before post_to_tokens, on the 80 ms grid.

  post_transform(P_prefix, enc, meta) -> Tensor [T, 8]
    P_prefix  float32 [T, 8] on the device: the row's posteriors (after the perturbation) at 80 ms frames 0..T-1,
              T = max(0, t_r - 1040) // 80 = exactly the frames v8 sees un-zeroed at read time t_r; frames >= T are 0
              for v8 and stay 0. T may be 0 (shape [0, 8]).
    enc       float32 [T, D] (D = 2048 for v8), detached: the frozen audio encoder's states of this row (audio <= t_r),
              computed in the same forward just before the projector, linearly resampled (np.interp semantics) from
              their own grid (13 per s, centres (i + 0.5) dur / n) to the frame centres (k + 0.5) 0.08 s; cost one
              [T x n] interpolation. None when the callable has an attribute wants_enc = False.
    meta      dict: id, widx, read, t_ms, final, dur_ms, T, F (frames of the window's npy), n_audio, frame_s (0.08),
              lookahead_ms (1040), phase ("score" = the no-grad pass 1 of --objective sched, "train" = the gradient
              pass), step, rec (the sched_v1 record: units with a_ms / b_ms / slot / ov_s, chunks, ...; AMI train)
    returns   [T, 8], differentiable in the hook's parameters; it replaces P_prefix, frames >= T stay 0, then the v8
              path continues (post_to_tokens re-implemented in torch so gradients flow, projector, interleave).
  extra_loss(info) -> scalar tensor, added to every gradient micro-batch's loss before backward. Gradients are summed
    over micro-batches and ranks, so normalise with the global counts in info:
    info = {"metas": [meta per row of the micro-batch], "P_in": [P_prefix], "P_out": [post_transform output],
            "enc": [enc or None] (empty lists when post_transform is None), "rows_global", "windows_global", "step",
            "device"}
  extra_params: list of nn.Module; moved to the device (dtype kept); their parameters join the optimiser (same AdamW,
    lr schedule), the gradient all-reduce and the global grad-norm clip; saved in extra.pt["extra_params"] (state
    dicts, in order) and restored by --resume.

    torchrun --nproc_per_node 4 -m experiments.mist_w3po_v1.crf.train_crf_sft --objective sched --out <new dir>
    (single process without torchrun also works; launchers: scripts/run_crf_c0.sh, scripts/run_crf_sched.sh)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from experiments.mist_w3po_v1.crf import schedule_lattice as sl
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    ATTN, AUDIO_PAD, EOS_ID, NEW0, NEW_END, N_SLOT, SPK0, SDProjector, ThinkerGStarModel, interleave, post_to_tokens)
from experiments.uniss_streaming_p2st_traj_v1.training import lora

WAIT_ID = 151649
LOOKAHEAD_MS = 1040
FRAME_S = 0.08
SCHED = "eval_outputs/mist_w3po_v1/crf/sched_v1"
V8 = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"
LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")


# ------------------------------------------------------------------------------------------------ distributed helpers
def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def world() -> int:
    return torch.distributed.get_world_size() if is_dist() else 1


def log0(*a):
    if rank() == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ windows and weights
def hier_weights(S: list[int], on: bool = True) -> np.ndarray:
    """v8's hierarchical cross-entropy weights (thinker_gstar_model.forward): time tokens 1.5, speaker tokens 2."""
    s = np.asarray(S)
    w = np.ones(len(s), np.float64)
    if on:
        w += 0.5 * ((s >= NEW0) & (s < SPK0)) + 1.0 * ((s >= SPK0) & (s < NEW_END))
    return w


@dataclass
class Window:
    idx: int
    rec: dict
    S: list
    N: int
    R: int
    t_ms: list
    lo: np.ndarray
    hi: np.ndarray
    beta: np.ndarray          # heuristic states (committed count after each read)
    lat_heur: float
    h: np.ndarray             # [N] token weights
    bmax: np.ndarray          # [R] the largest state reachable after read r (first token with lo > r, else N)
    reach: np.ndarray | None = None   # [R, N] bool: (r, j) committed by some feasible schedule
    feasible: bool = True

    @property
    def lag(self) -> np.ndarray:
        return (np.asarray(self.t_ms, np.float64)[:, None] - np.asarray(self.rec["tau_ms"], np.float64)[None, :]) / 1000.0

    @property
    def end_ids(self) -> list[int]:
        return [WAIT_ID] * (self.R - 1) + [EOS_ID]


def make_window(idx: int, rec: dict, hier: bool = True) -> Window:
    lo, hi = np.asarray(rec["lo"], np.int64), np.asarray(rec["hi"], np.int64)
    N, R = rec["n_tok"], rec["n_read"]
    bmax = np.full(R, N, np.int64)
    for r in range(R):
        bad = np.nonzero(lo > r)[0]
        if len(bad):
            bmax[r] = int(bad[0])
    return Window(idx, rec, list(rec["S_ids"]), N, R, list(rec["t_ms"]), lo, hi, np.asarray(rec["beta_heur"], np.int64),
                  float(rec["lat_heur"]), hier_weights(rec["S_ids"], hier), bmax)


def load_windows(sched: str, hier: bool = True) -> list[Window]:
    recs = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    return [make_window(i, r, hier) for i, r in enumerate(recs)]


def set_reach(wins: list[Window], W: int) -> None:
    """reach and feasibility from the lattice with all scores 0 (every feasible schedule has the same weight, so
    w_tok > 0 exactly on the pairs some feasible schedule uses); CPU float64, deterministic on every rank."""
    todo = [w for w in wins if w.reach is None]
    if not todo:
        return
    lat = sl.pack([{"tok_lp": np.zeros((w.R, w.N)), "end_lp": np.zeros((w.R, w.N + 1)), "lo": w.lo, "hi": w.hi,
                    "lag": np.zeros((w.R, w.N))} for w in todo], W=W, device="cpu")
    mg = sl.marginals(lat, 0.0)
    for i, w in enumerate(todo):
        w.feasible = bool(mg["feasible"][i])
        w.reach = (mg["w_tok"][i, : w.R, : w.N] > 0).numpy()


def heur_weights(w: Window) -> tuple[np.ndarray, np.ndarray]:
    """One-hot weights of the heuristic schedule: w_tok [R, N], w_end [R, N+1]."""
    wt = np.zeros((w.R, w.N))
    we = np.zeros((w.R, w.N + 1))
    prev = 0
    for r, b in enumerate(w.beta.tolist()):
        wt[r, prev:b] = 1.0
        we[r, b] = 1.0
        prev = b
    return wt, we


def heur_states(w: Window, R_pad: int) -> np.ndarray:
    s = np.full(R_pad, w.N, np.int64)
    s[: w.R] = w.beta
    return s


# ------------------------------------------------------------------------------------------------ step plan
def plan_steps(Rs: list[int], seed: int, steps: int, rows_per_step: int) -> list[list[int]]:
    """Deterministic window stream (a seeded permutation per pass), cut into steps of ~rows_per_step read rows: a
    window joins the step while rows + R / 2 <= rows_per_step (at least one window per step)."""
    out, cur, rows, p, perm, k = [], [], 0, 0, None, 0
    n = len(Rs)
    while len(out) < steps:
        if perm is None or k == n:
            perm = np.random.default_rng(seed + 1000 * p).permutation(n)
            p, k = p + 1, 0
        i = int(perm[k])
        if cur and rows + Rs[i] / 2 > rows_per_step:
            out.append(cur)
            cur, rows = [], 0
            continue
        cur.append(i)
        rows += Rs[i]
        k += 1
    return out


def est_cost(w: Window) -> float:
    """Tokens of the window's rows (prompt ~ 120 + 15.6 per s of audio, plus S)."""
    return float(sum(120 + 15.6 * t / 1000.0 + w.N for t in w.t_ms))


def assign(wins: list[Window], n_ranks: int) -> list[list[Window]]:
    """Greedy longest-first balance of the step's windows over the ranks (deterministic)."""
    load = [0.0] * n_ranks
    out = [[] for _ in range(n_ranks)]
    for w in sorted(wins, key=lambda w: (-est_cost(w), w.idx)):
        k = min(range(n_ranks), key=lambda i: (load[i], i))
        out[k].append(w)
        load[k] += est_cost(w)
    return out


def lr_at(k: int, args) -> float:
    """Megatron's schedule at iteration k (1-based): linear warm-up from 0, then cosine to min_lr at --steps."""
    if args.warmup > 0 and k <= args.warmup:
        return args.lr * k / args.warmup
    ratio = (k - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (math.cos(math.pi * min(1.0, ratio)) + 1.0)


# ------------------------------------------------------------------------------------------------ rows and batches
@dataclass
class Row:
    win: Window
    r: int
    K: int                    # tokens of S fed after the prompt
    pos: np.ndarray           # states b (0..K) read out; position Lp - 1 + b
    c_tok: np.ndarray | None = None    # pass 2: coefficient of -tok_lp at pos (main part, h * w / D)
    c_state: np.ndarray | None = None  # pass 2: coefficient of -tok_lp at pos (state part)
    c_end: np.ndarray | None = None    # pass 2: coefficient of -end_lp at pos


@dataclass
class Batch:
    rows: list
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_features: torch.Tensor
    feature_attention_mask: torch.Tensor
    n_audio: torch.Tensor
    has_sd: torch.Tensor
    sd_post: torch.Tensor | None                  # [B, nmax, 8] (v8 path)
    P_prefix: list = field(default_factory=list)  # hook path: [T, 8] float32
    F_frames: list = field(default_factory=list)
    dur_s: list = field(default_factory=list)
    metas: list = field(default_factory=list)
    sel_row: torch.Tensor | None = None
    sel_col: torch.Tensor | None = None
    tok_tgt: torch.Tensor | None = None
    end_tgt: torch.Tensor | None = None
    final_pos: torch.Tensor | None = None


class AudioCache:
    """Per step: each window's audio (mono, 16 kHz) and posteriors, read once."""

    def __init__(self):
        self.d = {}

    def get(self, w: Window):
        if w.idx not in self.d:
            import soundfile as sf

            x, sr = sf.read(w.rec["audio"], dtype="float32", always_2d=True)
            assert sr == 16000, w.rec["audio"]
            self.d[w.idx] = (x.mean(axis=1), np.load(w.rec["post"]).astype(np.float32))
        return self.d[w.idx]


def row_inputs(w: Window, r: int, audio: AudioCache, perturb_p: float, seed: int, step: int):
    """v8's GStarRows.__getitem__ for read r: wav[:t*16] (padded to 0.1 s), perturbed posteriors zeroed from
    max(0, t - 1040) // 80. Returns (wav, post, T)."""
    from experiments.uniss_qwen3_omni_v1.training.pretrain_thinker_gstar import perturb

    wav, post = audio.get(w)
    t = int(w.t_ms[r])
    x = wav[: t * 16]
    if len(x) < 1600:
        x = np.pad(x, (0, 1600 - len(x)))
    p = perturb(post, np.random.default_rng([seed, step, w.idx, r]), perturb_p).copy()
    T = max(0, t - LOOKAHEAD_MS) // 80
    p[T:] = 0.0
    return x, p, min(T, len(p))


def collate(rows: list[Row], audio: AudioCache, proc, prompter, hook: bool, perturb_p: float, seed: int, step: int,
            phase: str) -> Batch:
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    tok = proc.tokenizer
    ins = [row_inputs(rw.win, rw.r, audio, perturb_p, seed, step) for rw in rows]
    f = proc.feature_extractor([x for x, _, _ in ins], sampling_rate=16000, padding=True, truncation=False,
                               return_attention_mask=True, return_tensors="pt")
    n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1))
    seqs, sel_r, sel_c, tt, et, fin = [], [], [], [], [], []
    for i, (rw, n) in enumerate(zip(rows, n_audio.tolist())):
        S = rw.win.S
        ids = prompter.ids(n, "", S[: rw.K]) + [rw.win.end_ids[rw.r]]   # v8's row layout: ... + block + end token
        Lp = len(ids) - rw.K - 1
        seqs.append(ids)
        for b in rw.pos.tolist():
            sel_r.append(i)
            sel_c.append(Lp - 1 + b)
            tt.append(S[b] if b < rw.win.N else 0)
            et.append(rw.win.end_ids[rw.r])
            fin.append(rw.r == rw.win.R - 1)
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(s)
        mask[i, : len(s)] = 1
    b = Batch(rows, ids, mask, f["input_features"], f["attention_mask"], n_audio,
              torch.ones(len(rows), dtype=torch.bool), None)
    if hook:
        for rw, (x, p, T), n in zip(rows, ins, n_audio.tolist()):
            b.P_prefix.append(torch.as_tensor(p[:T]))
            b.F_frames.append(len(p))
            b.dur_s.append(len(x) / 16000)
            b.metas.append({"id": rw.win.rec["id"], "widx": rw.win.idx, "read": rw.r, "t_ms": int(rw.win.t_ms[rw.r]),
                            "final": rw.r == rw.win.R - 1, "dur_ms": rw.win.rec["dur_ms"], "T": T, "F": len(p),
                            "n_audio": n, "frame_s": FRAME_S, "lookahead_ms": LOOKAHEAD_MS, "phase": phase,
                            "step": step, "rec": rw.win.rec})
    else:   # v8's build_batch: posteriors on the audio-token grid, numpy
        nmax = max(int(n_audio.max()), 1)
        sd = torch.zeros((len(rows), nmax, N_SLOT), dtype=torch.float32)
        for i, ((x, p, T), n) in enumerate(zip(ins, n_audio.tolist())):
            q = post_to_tokens(p, n, len(x) / 16000)
            sd[i, : len(q)] = torch.as_tensor(q)
        b.sd_post = sd
    b.sel_row, b.sel_col = torch.as_tensor(sel_r), torch.as_tensor(sel_c)
    b.tok_tgt, b.end_tgt, b.final_pos = torch.as_tensor(tt), torch.as_tensor(et), torch.as_tensor(fin)
    return b


def micro_batches(rows: list[Row], budget: int, n_prompt0: int) -> list[list[Row]]:
    """Rows sorted by length, packed while (rows x longest) <= budget padded tokens."""
    def ln(rw):
        t = rw.win.t_ms[rw.r] / 1000.0
        n = math.ceil(13 * t) + 1
        return n_prompt0 + n + math.ceil(n / 5) + rw.K
    out, cur, mx = [], [], 0
    for rw in sorted(rows, key=ln):
        L = ln(rw)
        if cur and (len(cur) + 1) * max(mx, L) > budget:
            out.append(cur)
            cur, mx = [], 0
        cur.append(rw)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------------------------------------------ torch interpolation
def interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp(x, xp, fp[:, c]) for every column c, in float64 (as numpy), differentiable in fp."""
    fp64 = fp.to(torch.float64)
    if xp.numel() == 1:
        return fp64[:1].expand(x.numel(), fp.shape[1])
    j = (torch.searchsorted(xp, x, right=True) - 1).clamp(0, xp.numel() - 2)
    slope = (fp64[j + 1] - fp64[j]) / (xp[j + 1] - xp[j])[:, None]
    y = slope * (x - xp[j])[:, None] + fp64[j]
    y = torch.where((x < xp[0])[:, None], fp64[:1].expand_as(y), y)
    return torch.where((x >= xp[-1])[:, None], fp64[-1:].expand_as(y), y)


def post_to_tokens_torch(post: torch.Tensor, n: int, dur_s: float) -> torch.Tensor:
    """thinker_gstar_model.post_to_tokens in torch: [F, 8] at 80 ms frame centres -> [n, 8] at audio-token centres."""
    if n <= 0:
        return post.new_zeros((0, post.shape[1]), dtype=torch.float32)
    dev = post.device
    fc = (torch.arange(post.shape[0], device=dev, dtype=torch.float64) + 0.5) * FRAME_S
    tc = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) * dur_s / n
    return interp_rows(tc, fc, post).to(torch.float32)


def enc_to_frames(a: torch.Tensor, T: int, dur_s: float) -> torch.Tensor:
    """Audio-encoder states [n, D] (centres (i + 0.5) dur / n) -> [T, D] at the 80 ms frame centres."""
    n = a.shape[0]
    if T <= 0 or n == 0:
        return a.new_zeros((0, a.shape[1]), dtype=torch.float32)
    dev = a.device
    tc = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) * dur_s / n
    fc = (torch.arange(T, device=dev, dtype=torch.float64) + 0.5) * FRAME_S
    return interp_rows(fc, tc, a.detach()).to(torch.float32)


# ------------------------------------------------------------------------------------------------ model
class CrfThinker(nn.Module):
    """v8's SFT model in HF: thinker + LoRA + sd_proj + new token rows (ThinkerGStarModel with init_merged)."""

    def __init__(self, thinker, sd_state: dict, rank_lora: int = 32, alpha: float = 64.0, post_transform=None):
        super().__init__()
        self.thinker = thinker
        keep = lambda parent: LAYER_RE.search(parent + ".") is not None   # noqa: E731
        self.adapters = lora.inject(thinker, rank=rank_lora, alpha=alpha, targets=ATTN, layer_filter=keep)
        emb = thinker.get_input_embeddings().weight
        head = thinker.lm_head.weight
        d = emb.shape[1]
        # created after inject (which freezes everything else), so these train (ThinkerGStarModel, init_merged)
        self.sd_proj = SDProjector(N_SLOT, d).float().to(emb.device)
        self.sd_proj.load_state_dict(sd_state)
        self.new_emb = nn.Parameter(emb[NEW0:NEW_END].detach().float().clone())
        self.new_head = nn.Parameter(head[NEW0:NEW_END].detach().float().clone())
        self.use_sd = True
        self.init_merged = True
        self._post_transform = post_transform
        self.hook_record = None

    def trainable(self) -> list[nn.Parameter]:
        return lora.trainable_parameters(self.adapters) + list(self.sd_proj.parameters()) + [self.new_emb, self.new_head]

    def audio_dtype(self):
        return next(self.thinker.audio_tower.parameters()).dtype if hasattr(self.thinker, "audio_tower") else \
            self.thinker.get_input_embeddings().weight.dtype

    def embed(self, input_ids, input_features, feature_attention_mask, sd_post, n_audio, has_sd):
        """v8's ThinkerGStarModel.embed (same signature, so ThinkerGStarModel.forward also runs on this module)."""
        return ThinkerGStarModel.embed(self, input_ids, input_features, feature_attention_mask, sd_post, n_audio, has_sd)

    def logits_at(self, h: torch.Tensor) -> torch.Tensor:
        return ThinkerGStarModel.logits_at(self, h)

    def embed_batch(self, b: Batch) -> torch.Tensor:
        if self._post_transform is None:   # v8 verbatim
            return self.embed(b.input_ids, b.input_features, b.feature_attention_mask, b.sd_post, b.n_audio, b.has_sd)
        return self.embed_hook(b)

    def embed_hook(self, b: Batch) -> torch.Tensor:
        """ThinkerGStarModel.embed with the posteriors routed through post_transform on the 80 ms grid."""
        input_ids = b.input_ids
        is_new = (input_ids >= NEW0) & (input_ids < NEW_END)
        emb = self.thinker.get_input_embeddings()(input_ids.masked_fill(is_new, 0))
        if is_new.any():
            emb = emb.clone()
            emb[is_new] = self.new_emb[input_ids[is_new] - NEW0].to(emb.dtype)
        a = self.thinker.get_audio_features(b.input_features, b.feature_attention_mask).last_hidden_state.to(emb.dtype)
        assert a.shape[0] == int(b.n_audio.sum()), (a.shape, b.n_audio)
        wants_enc = getattr(self._post_transform, "wants_enc", True)
        rec = {"P_in": [], "P_out": [], "enc": []}
        parts, pos = [], 0
        for i in range(input_ids.shape[0]):
            n = int(b.n_audio[i])
            ai = a[pos: pos + n]
            pos += n
            P = b.P_prefix[i].to(emb.device, torch.float32)
            T = P.shape[0]
            enc = enc_to_frames(ai, T, b.dur_s[i]) if wants_enc else None
            Pn = self._post_transform(P, enc, b.metas[i])
            if tuple(Pn.shape) != (T, N_SLOT):
                raise ValueError(f"post_transform must return [{T}, {N_SLOT}], got {tuple(Pn.shape)}")
            full = torch.cat([Pn.to(torch.float32), Pn.new_zeros((b.F_frames[i] - T, N_SLOT), dtype=torch.float32)], 0)
            sd_tok = post_to_tokens_torch(full, n, b.dur_s[i])
            s = self.sd_proj(sd_tok[None].to(self.sd_proj.conv.weight.dtype))[0]
            ai = interleave(ai, s.to(emb.dtype))
            parts.append(ai)
            rec["P_in"].append(P)
            rec["P_out"].append(Pn)
            rec["enc"].append(enc)
        self.hook_record = rec
        feats = torch.cat(parts, 0)
        mask = (input_ids == AUDIO_PAD).unsqueeze(-1)
        assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
        return emb.masked_scatter(mask.to(emb.device), feats)

    def to_device(self, b: Batch) -> Batch:
        """Batch tensors to the model's device; audio features in the audio tower's dtype (as engine.audio_embeds and
        Megatron's Float16Module do), posteriors float32 (ThinkerGStarModel.embed casts them to the projector's)."""
        dev = self.new_emb.device
        b.input_ids, b.attention_mask = b.input_ids.to(dev), b.attention_mask.to(dev)
        b.input_features = b.input_features.to(dev, self.audio_dtype())
        b.feature_attention_mask = b.feature_attention_mask.to(dev)
        if b.sd_post is not None:
            b.sd_post = b.sd_post.to(dev)
        return b

    def hidden(self, b: Batch) -> torch.Tensor:
        b = self.to_device(b)
        emb = self.embed_batch(b)
        return self.thinker.model(inputs_embeds=emb, attention_mask=b.attention_mask, use_cache=False,
                                  return_dict=True).last_hidden_state

    def _lp_chunk(self, x, tt, et):
        lg = self.logits_at(x)
        ls = torch.log_softmax(lg, -1)
        return ls.gather(1, tt[:, None])[:, 0], ls.gather(1, et[:, None])[:, 0]

    def _lp_chunk_masked(self, x, tt, et):
        """(log p(tok), log p(e)) with <wait> barred before the softmax (engine.py:246); no grad."""
        lg = self.logits_at(x)
        lg[:, WAIT_ID] = float("-inf")
        lse = torch.logsumexp(lg, -1)
        return lg.gather(1, tt[:, None])[:, 0] - lse, lg.gather(1, et[:, None])[:, 0] - lse

    def pos_logprobs(self, b: Batch, chunk: int = 1024, masked_final: bool = False):
        """(tok_lp, end_lp, (tok_lp, end_lp) with <wait> barred or None) at the batch's read-out positions, [P] float32
        each. With grad, every chunk of logits is recomputed in backward (torch checkpoint), so only [P, d] hidden
        states are kept. The barred pair is computed without grad (pass 1 only)."""
        h = self.hidden(b)
        dev = h.device
        hs = h[b.sel_row.to(dev), b.sel_col.to(dev)]
        tt, et = b.tok_tgt.to(dev), b.end_tgt.to(dev)
        lt, le, mt, me = [], [], [], []
        grad = torch.is_grad_enabled() and hs.requires_grad
        for s in range(0, hs.shape[0], chunk):
            x, t_, e_ = hs[s: s + chunk], tt[s: s + chunk], et[s: s + chunk]
            if grad:
                a_, b_ = checkpoint(self._lp_chunk, x, t_, e_, use_reentrant=False)
            else:
                a_, b_ = self._lp_chunk(x, t_, e_)
            lt.append(a_)
            le.append(b_)
            if masked_final:
                with torch.no_grad():
                    a2, b2 = self._lp_chunk_masked(x.detach(), t_, e_)
                mt.append(a2)
                me.append(b2)
        cat = lambda v: torch.cat(v) if v else hs.new_zeros(0)   # noqa: E731
        return cat(lt), cat(le), ((cat(mt), cat(me)) if masked_final else None)


def load_thinker(path: str, dev: torch.device):
    """rl/train_w3po.load_policy's loading (the thinker on one card, talker / code2wav on CPU, then dropped)."""
    from transformers import Qwen3OmniMoeForConditionalGeneration

    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"thinker": dev.index, "talker": "cpu", "code2wav": "cpu"})
    thinker = omni.thinker
    del omni
    import gc

    gc.collect()
    return thinker


# ------------------------------------------------------------------------------------------------ passes
def scoring_rows(wins: list[Window]) -> list[Row]:
    """Pass 1 rows: read r cut at bmax(r), positions 0..bmax(r)."""
    return [Row(w, r, int(w.bmax[r]), np.arange(int(w.bmax[r]) + 1)) for w in wins for r in range(w.R)]


def score_pass(model: CrfThinker, wins: list[Window], ctx: dict, step: int, grad: bool = False) -> dict:
    """{window idx: (tok_lp [R, N], end_lp [R, N+1] (plain softmax, as the loss), tok_lp, end_lp for the lattice (the
    last read with <wait> barred when ctx["final_bar_wait"], i.e. the decoding distribution))}, float64 on CPU.
    grad=True keeps the plain pair differentiable (tests). Entries beyond bmax(r) (no feasible schedule reads them)
    stay 0."""
    parts = {w.idx: [] for w in wins}
    for mb in micro_batches(scoring_rows(wins), ctx["tok_budget"], ctx["n_prompt0"]):
        b = collate(mb, ctx["audio"], ctx["proc"], ctx["prompter"], model._post_transform is not None,
                    ctx["perturb"], ctx["seed"], step, "score")
        with torch.set_grad_enabled(grad):
            lt, le, lm = model.pos_logprobs(b, ctx["lp_chunk"], masked_final=ctx["final_bar_wait"])
        k = 0
        for rw in mb:
            n = len(rw.pos)
            parts[rw.win.idx].append((rw, lt[k: k + n], le[k: k + n],
                                      None if lm is None else (lm[0][k: k + n], lm[1][k: k + n])))
            k += n
    out = {}
    for w in wins:
        tok = torch.zeros((w.R, w.N), dtype=torch.float64)
        endp = torch.zeros((w.R, w.N + 1), dtype=torch.float64)
        tokl = torch.zeros((w.R, w.N), dtype=torch.float64)
        endl = torch.zeros((w.R, w.N + 1), dtype=torch.float64)
        for rw, a, e, m in parts[w.idx]:
            pos = torch.as_tensor(rw.pos)
            ok = pos < w.N
            ri = torch.full_like(pos, rw.r)
            tok = tok.index_put((ri[ok], pos[ok]), a[ok].double().cpu())     # out of place: differentiable
            endp = endp.index_put((ri, pos), e.double().cpu())
            la, le_ = m if (m is not None and rw.r == w.R - 1) else (a, e)
            tokl[rw.r, pos[ok]] = la[ok].detach().double().cpu()
            endl[rw.r, pos] = le_.detach().double().cpu()
        out[w.idx] = (tok, endp, tokl, endl)
    return out


def lattice_of(wins: list[Window], scores: dict, W: int, device) -> sl.Lattice:
    return sl.pack([{"tok_lp": scores[w.idx][2].detach(), "end_lp": scores[w.idx][3].detach(), "lo": w.lo,
                     "hi": w.hi, "lag": w.lag} for w in wins], W=W, device=device)


def posterior_stats(lat: sl.Lattice, mg: dict, wins: list[Window], lam: float) -> dict:
    """Per window: E_q[Lat], entropy H(q), q(heuristic schedule)."""
    adm, P, m, end, lag = sl._prep(lat, lam)
    lam_n = torch.as_tensor(lam, dtype=sl.DTYPE, device=lat.device) / lat.n_tok.clamp(min=1).to(sl.DTYPE)
    s = torch.where(adm, lat.tok_lp, torch.zeros((), dtype=sl.DTYPE, device=lat.device)) - lam_n[:, None, None] * lag
    e_score = (mg["w_tok"] * s).sum((1, 2)) + (mg["w_end"] * end).sum((1, 2))
    H = mg["logZ"] - e_score
    st = torch.as_tensor(np.stack([heur_states(w, lat.R) for w in wins]), device=lat.device)
    q_heur = torch.exp(sl.schedule_score(lat, st, lam) - mg["logZ"])
    return {"exp_lat": mg["exp_lat"].cpu().numpy(), "H": H.cpu().numpy(), "q_heur": q_heur.cpu().numpy(),
            "feasible": mg["feasible"].cpu().numpy()}


def train_rows(wins: list[Window], weights: dict, D: float, A: float, state_w: float) -> list[Row]:
    """Pass 2 rows with their coefficients; weights {idx: (w_tok [R,N], w_end [R,N+1]) numpy}."""
    rows = []
    for w in wins:
        wt, we = weights[w.idx]
        for r in range(w.R):
            ct = np.zeros(w.N + 1)
            ct[: w.N] = w.h * wt[r] / D
            cs = np.zeros(w.N + 1)
            if state_w and A > 0:
                cs[: w.N] = state_w * w.reach[r] / A
            ce = we[r] / D
            need = np.nonzero((ct != 0) | (cs != 0) | (ce != 0))[0]
            if len(need) == 0:
                continue
            K = int(need.max())
            pos = np.arange(K + 1)
            pos = pos[(ct[pos] != 0) | (cs[pos] != 0) | (ce[pos] != 0)]
            rows.append(Row(w, r, min(K, w.N), pos, ct[pos], cs[pos], ce[pos]))
    return rows


def grad_pass(model: CrfThinker, rows: list[Row], ctx: dict, step: int, extra_loss=None, n_rows_global: int = 0,
              n_windows_global: int = 0) -> dict:
    """Forward + backward of every micro-batch (gradients accumulate in .grad); returns the summed loss parts."""
    logs = {"loss_main": 0.0, "loss_state": 0.0, "loss_extra": 0.0, "pos": 0, "tokens": 0, "mbs": 0}
    hook = model._post_transform is not None
    for mb in micro_batches(rows, ctx["tok_budget"], ctx["n_prompt0"]):
        b = collate(mb, ctx["audio"], ctx["proc"], ctx["prompter"], hook, ctx["perturb"], ctx["seed"], step, "train")
        lt, le, _ = model.pos_logprobs(b, ctx["lp_chunk"])
        dev = lt.device
        ct = torch.as_tensor(np.concatenate([rw.c_tok for rw in mb]), dtype=torch.float32, device=dev)
        cs = torch.as_tensor(np.concatenate([rw.c_state for rw in mb]), dtype=torch.float32, device=dev)
        ce = torch.as_tensor(np.concatenate([rw.c_end for rw in mb]), dtype=torch.float32, device=dev)
        main = -(ct * lt).sum() - (ce * le).sum()
        state = -(cs * lt).sum()
        loss = main + state
        if extra_loss is not None:
            rec = model.hook_record if hook else {"P_in": [], "P_out": [], "enc": []}
            ex = extra_loss({"metas": b.metas, "P_in": rec["P_in"], "P_out": rec["P_out"], "enc": rec["enc"],
                             "rows_global": n_rows_global, "windows_global": n_windows_global, "step": step,
                             "device": dev})
            loss = loss + ex
            logs["loss_extra"] += float(ex.detach())
        loss.backward()
        logs["loss_main"] += float(main.detach())
        logs["loss_state"] += float(state.detach())
        logs["pos"] += int(len(lt))
        logs["tokens"] += int(b.input_ids.numel())
        logs["mbs"] += 1
        model.hook_record = None
    return logs


def step_totals(wins: list[Window]) -> tuple[float, float]:
    """D = sum (sum_j h_j + R) and |A| = number of reachable (r, j) over the (global) step's feasible windows."""
    D = float(sum(w.h.sum() + w.R for w in wins if w.feasible))
    A = float(sum(w.reach.sum() for w in wins if w.feasible))
    return D, A


def sft_loss_from_scores(wins: list[Window], scores: dict, D: float) -> float:
    """The C0 loss of these windows from pass-1 log-probs (sched mode logging)."""
    tot = 0.0
    for w in wins:
        if not w.feasible:
            continue
        wt, we = heur_weights(w)
        tok, endp = scores[w.idx][0].detach().numpy(), scores[w.idx][1].detach().numpy()
        tot -= float((w.h[None, :] * wt * tok).sum() + (we * endp).sum())
    return tot / D


# ------------------------------------------------------------------------------------------------ checkpoints
def save_ckpt(ck: Path, model: CrfThinker, opt, extra_params: list, lam: float, step: int, args) -> None:
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(lora.state_dict(model.adapters), ck / "lora.pt")
    torch.save({"sd_proj": {k: v.detach().cpu() for k, v in model.sd_proj.state_dict().items()},
                "new_emb": model.new_emb.detach().cpu(), "new_head": model.new_head.detach().cpu(),
                "extra_params": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in extra_params],
                "lam": lam, "step": step, "objective": args.objective, "base": args.base,
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha}, ck / "extra.pt")
    torch.save(opt.state_dict(), ck / "optim.pt")
    (ck / "DONE").write_text(json.dumps({"step": step, "time": time.time()}))


def load_ckpt(ck: Path, model: CrfThinker, opt, extra_params: list) -> dict:
    lora.load_state_dict(model.adapters, torch.load(ck / "lora.pt", map_location="cpu"))
    ex = torch.load(ck / "extra.pt", map_location="cpu")
    model.sd_proj.load_state_dict(ex["sd_proj"])
    with torch.no_grad():
        model.new_emb.copy_(ex["new_emb"])
        model.new_head.copy_(ex["new_head"])
    if len(ex["extra_params"]) != len(extra_params):
        raise SystemExit(f"checkpoint has {len(ex['extra_params'])} extra modules, the hook gives {len(extra_params)}")
    for m, st in zip(extra_params, ex["extra_params"]):
        m.load_state_dict(st)
    if opt is not None:
        opt.load_state_dict(torch.load(ck / "optim.pt", map_location="cpu"))
    return ex


def latest_ckpt(out: Path) -> Path | None:
    cks = [c for c in (out / "ckpt").glob("step_*") if (c / "DONE").exists()]
    return max(cks, key=lambda p: int(p.name.split("_")[1])) if cks else None


# ------------------------------------------------------------------------------------------------ main
def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", required=True, choices=["heuristic", "sched"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=V8, help="v8 merged Thinker (with sd_proj.pt)")
    ap.add_argument("--sched", default=SCHED)
    ap.add_argument("--dir", default="zh2en")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--rows-per-step", type=int, default=64, help="read rows per optimiser step (v8: GBS 64)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--perturb", type=float, default=0.2, help="v8's --gstar-perturb")
    ap.add_argument("--hier-weights", type=int, default=1, help="v8's time x1.5 / speaker x2 token weights (0: off)")
    ap.add_argument("--state-weight", type=float, default=1.0, help="sched: weight of L_state (pre-registered 1)")
    ap.add_argument("--lat-target", type=float, default=None, help="sched: target of the batch mean E_q[Lat] "
                    "(default: L_ref of sched_v1/summary.json, 0.826 s)")
    ap.add_argument("--lat-target-batch", action="store_true",
                    help="sched: target = the step's own mean heuristic Lat instead of the global L_ref (off)")
    ap.add_argument("--W", type=int, default=128, help="block-length cap of the lattice (tokens)")
    ap.add_argument("--final-bar-wait", type=int, default=1,
                    help="lattice scores of the last read with <wait> barred (the decoding distribution); 0: plain")
    ap.add_argument("--tok-budget", type=int, default=12000, help="padded tokens per micro-batch")
    ap.add_argument("--lp-chunk", type=int, default=1024, help="positions per logits chunk")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--mem-cap-gib", type=float, default=0.0, help="cap of this process's CUDA allocator (0: none)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="smoke tests: stop (and save a checkpoint) after this many steps of the --steps schedule")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


def main(argv=None, *, post_transform=None, extra_loss=None, extra_params=None) -> None:
    args = build_args(argv)
    extra_params = list(extra_params or [])
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import datetime as _dt

        torch.distributed.init_process_group("nccl", timeout=_dt.timedelta(hours=2))
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    if args.mem_cap_gib > 0:
        tot = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / tot), dev)
    out = Path(args.out)
    if rank() == 0:
        if out.exists() and not args.resume:
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    if is_dist():
        torch.distributed.barrier()
    summ = json.loads((Path(args.sched) / "summary.json").read_text())
    lat_target = args.lat_target if args.lat_target is not None else float(summ["L_ref_mean_lat_heur_windows"])

    wins = load_windows(args.sched, bool(args.hier_weights))
    plan = plan_steps([w.R for w in wins], args.seed, args.steps, args.rows_per_step)
    log0(f"{len(wins)} windows, {sum(w.R for w in wins)} read rows; {args.steps} steps of "
         f"{np.mean([sum(wins[i].R for i in s) for s in plan]):.1f} rows / {np.mean([len(s) for s in plan]):.2f} "
         f"windows on average; world {world()}; objective {args.objective}; lat target {lat_target:.4f}")

    from experiments.mist_w3po_v1.core.engine import Prompter

    prompter = Prompter(args.dir)
    proc = prompter.proc     # pretrain_thinker_gstar.processor: the release processor + the 259 new tokens
    for w in wins[:20]:
        if prompter.tok(w.rec["S"], add_special_tokens=False)["input_ids"] != w.S:
            raise SystemExit(f"tokenizer mismatch with sched_v1: {w.rec['id']}")

    torch.manual_seed(args.seed)
    thinker = load_thinker(args.base, dev)
    sd_state = torch.load(Path(args.base) / "sd_proj.pt", map_location="cpu")
    model = CrfThinker(thinker, sd_state, args.lora_rank, args.lora_alpha, post_transform)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for m in extra_params:
        m.to(dev)
    params = model.trainable() + [p for m in extra_params for p in m.parameters() if p.requires_grad]
    if is_dist():   # identical starting point on every rank
        for p in params:
            torch.distributed.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                            weight_decay=args.weight_decay)
    start, lam_prev = 0, 0.0
    if args.resume:
        ck = latest_ckpt(out)
        if ck is not None:
            ex = load_ckpt(ck, model, opt, extra_params)
            start, lam_prev = int(ex["step"]), float(ex.get("lam", 0.0))
            log0(f"resumed from {ck}")
    log0(json.dumps({"trainable_m": round(sum(p.numel() for p in params) / 1e6, 2), "adapters": len(model.adapters),
                     "extra_modules": len(extra_params), "start": start}))
    ctx = {"audio": None, "proc": proc, "prompter": prompter, "perturb": args.perturb, "seed": args.seed,
           "tok_budget": args.tok_budget, "lp_chunk": args.lp_chunk, "final_bar_wait": bool(args.final_bar_wait),
           "n_prompt0": len(prompter.a_ids) + len(prompter.b_ids) - 1}

    for step in range(start, args.steps):
        t0 = time.time()
        k = step + 1
        sw = [wins[i] for i in plan[step]]
        set_reach(sw, args.W)
        bad = [w for w in sw if not w.feasible]
        sw = [w for w in sw if w.feasible]
        D, A = step_totals(sw)
        mine = assign(sw, world())[rank()]
        ctx["audio"] = AudioCache()
        st = {"lam": None, "lam_bracketed": None, "Eq_lat": None, "H_q": None, "q_heur": None, "sft_loss": None}
        t1 = time.time()
        if args.objective == "sched":
            model.eval()
            for m in extra_params:
                m.eval()
            scores = score_pass(model, mine, ctx, step)
            t_p1 = time.time() - t1
            pay = [(w.idx,) + tuple(x.float() for x in scores[w.idx]) for w in mine]
            gat = [None] * world()
            if is_dist():
                torch.distributed.all_gather_object(gat, pay)
            else:
                gat = [pay]
            allsc = {x[0]: tuple(y.double() for y in x[1:]) for part in gat for x in part}
            t2 = time.time()
            res = [None]
            if rank() == 0:
                lat = lattice_of(sw, allsc, args.W, dev)
                tgt = float(np.mean([w.lat_heur for w in sw])) if args.lat_target_batch else lat_target
                sol = sl.solve_lambda([lat], tgt)
                lam = sol["lam"] if sol["bracketed"] else lam_prev
                mg = sl.marginals(lat, lam)
                ps = posterior_stats(lat, mg, sw, lam)
                wts = {w.idx: (mg["w_tok"][i, : w.R, : w.N].cpu().numpy(), mg["w_end"][i, : w.R, : w.N + 1].cpu().numpy())
                       for i, w in enumerate(sw)}
                res = [{"lam": lam, "bracketed": bool(sol["bracketed"]), "wts": wts, "ps": ps, "tgt": tgt,
                        "sft": sft_loss_from_scores(sw, allsc, D)}]
            if is_dist():
                torch.distributed.broadcast_object_list(res, 0)
            res = res[0]
            lam_prev = res["lam"]
            weights = res["wts"]
            ok = res["ps"]["feasible"]
            st.update(lam=res["lam"], lam_bracketed=res["bracketed"], lat_target=res["tgt"],
                      Eq_lat=float(np.mean(res["ps"]["exp_lat"][ok])), H_q=float(np.mean(res["ps"]["H"][ok])),
                      q_heur=float(np.mean(res["ps"]["q_heur"][ok])), sft_loss=res["sft"],
                      lat_heur=float(np.mean([w.lat_heur for w in sw])))
            t_lam = time.time() - t2
            state_w = args.state_weight
        else:
            weights = {w.idx: heur_weights(w) for w in mine}
            t_p1 = t_lam = 0.0
            state_w = 0.0
            st["lat_heur"] = float(np.mean([w.lat_heur for w in sw]))
        t3 = time.time()
        rows = train_rows(mine, weights, D, A, state_w)
        model.train()
        for m in extra_params:
            m.train()
        lr = lr_at(k, args)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        lg = grad_pass(model, rows, ctx, step, extra_loss, sum(w.R for w in sw), len(sw))
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if is_dist():
            buf = torch._utils._flatten_dense_tensors([p.grad for p in params])
            torch.distributed.all_reduce(buf)
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(buf, [p.grad for p in params])):
                p.grad.copy_(g)
            del buf
        gn = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        t_p2 = time.time() - t3
        vec = torch.tensor([lg["loss_main"], lg["loss_state"], lg["loss_extra"], lg["pos"], lg["tokens"], lg["mbs"],
                            len(rows), torch.cuda.max_memory_allocated(dev) / 2**30, t_p1, t_p2], dtype=torch.float64,
                           device=dev)
        if is_dist():
            torch.distributed.all_reduce(vec[:7])
            mx = vec[7:].clone()
            torch.distributed.all_reduce(mx, op=torch.distributed.ReduceOp.MAX)
            vec[7:] = mx
        v = vec.tolist()
        if rank() == 0:
            rec = {"step": k, "lr": lr, "objective": args.objective, "windows": len(sw), "infeasible": len(bad),
                   "rows": int(sum(w.R for w in sw)), "rows_trained": int(v[6]), "D": D, "A": A,
                   "loss": v[0] + v[1] + v[2], "loss_main": v[0], "loss_state": v[1], "loss_extra": v[2],
                   "sft_loss": st["sft_loss"] if args.objective == "sched" else v[0],
                   "lam": st["lam"], "lam_bracketed": st["lam_bracketed"], "Eq_lat": st["Eq_lat"],
                   "lat_target": st.get("lat_target"), "lat_heur": st["lat_heur"], "H_q": st["H_q"],
                   "q_heur": st["q_heur"], "gnorm": gn, "pos_pass2": int(v[3]), "tok_pass2": int(v[4]),
                   "micro_batches": int(v[5]), "mem_gib_max": round(v[7], 2), "t_pass1": round(v[8], 2),
                   "t_lambda": round(t_lam, 2), "t_pass2": round(v[9], 2), "t_step": round(time.time() - t0, 2)}
            with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            log0(json.dumps({a: (float(f"{b:.5g}") if isinstance(b, float) else b) for a, b in rec.items()}))
        stop = args.stop_after > 0 and k >= start + args.stop_after
        if k % args.save_every == 0 or k == args.steps or stop:
            if is_dist():   # every rank must hold the same parameters (identical start + all-reduced gradients)
                cs = torch.stack([p.detach().double().sum() for p in params]).sum().reshape(1)
                lo_, hi_ = cs.clone(), cs.clone()
                torch.distributed.all_reduce(lo_, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(hi_, op=torch.distributed.ReduceOp.MAX)
                log0(f"param checksum step {k}: min {float(lo_):.17g} max {float(hi_):.17g} "
                     + ("RANKS_IN_SYNC" if float(lo_) == float(hi_) else "RANK_DIVERGENCE"))
            if rank() == 0:
                save_ckpt(out / "ckpt" / f"step_{k:05d}", model, opt, extra_params, lam_prev, k, args)
            if is_dist():
                torch.distributed.barrier()
        torch.cuda.empty_cache()
        if stop:
            log0(f"--stop-after {args.stop_after}: stopping after step {k}")
            break
    log0("CRF_SFT_TRAIN_DONE")
    if is_dist():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
