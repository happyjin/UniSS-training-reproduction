"""Tests of train_crf_sft.py --final-read-full-post (C0-fp), CPU only, real sched_v1 AMI-train windows (audio and
posteriors), the real tokenizer / feature extractor / prompts (test_train_crf_sft's env), perturbation 0.2 (v8's):

  off      batches (both hook paths, score and train rows, every Batch field incl. metas) are bit-identical to the
           trainer BEFORE the option (the file at commit 8d289426, loaded from git) and to the default call; the
           option's default is 0 and score_pass / grad_pass pass final_full=False without the ctx key
  rows     on: non-final reads unchanged (wav, P, T); at the final read t = t_R only the 80 ms frames
           [max(0, t - 1040) // 80, t // 80) change, and they hold the real (perturbed, same draw) posteriors; frames
           >= t // 80 (the partial last frame) stay 0; the extended region is non-empty and non-zero
  batch    on vs off: every tensor except the posteriors identical; sd_post (v8 path) differs only in final rows, where it
           is post_to_tokens of the expected P; hook path: P_prefix / meta T likewise; embeddings (tiny model) differ
           only at the audio positions of final rows
  plumbing ctx["final_full_post"] reaches collate in score_pass and grad_pass; --final-read-full-post parses to 0 / 1

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_final_read_full_post
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T
from experiments.uniss_qwen3_omni_v1.training.pretrain_thinker_gstar import perturb
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import AUDIO_PAD, post_to_tokens

BEFORE = "8d289426"     # last commit of train_crf_sft.py before the option
PERTURB, SEED, STEP = 0.2, 7, 3
TENSORS = ("input_ids", "attention_mask", "input_features", "feature_attention_mask", "n_audio", "has_sd", "sel_row",
           "sel_col", "tok_tgt", "end_tgt", "final_pos")


def old_module():
    src = subprocess.run(["git", "show", f"{BEFORE}:experiments/mist_w3po_v1/crf/train_crf_sft.py"], check=True,
                         capture_output=True, text=True).stdout
    assert "final_full" not in src
    mod = types.ModuleType("train_crf_sft_before")
    sys.modules[mod.__name__] = mod      # dataclasses look the module up
    exec(compile(src, f"{BEFORE}:train_crf_sft.py", "exec"), mod.__dict__)
    return mod


def windows():
    """6 real windows (R = 2..5, distinct idx: AudioCache and the perturbation draw are keyed by idx)."""
    recs = [json.loads(l) for l in open(Path(T.SCHED) / "shard_000.jsonl", encoding="utf-8")]
    a = [r for r in recs if 2 <= r["n_read"] <= 3 and 8 <= r["n_tok"] <= 30][2:6]
    b = [r for r in recs if 4 <= r["n_read"] <= 5 and 8 <= r["n_tok"] <= 40][1:3]
    wins = [T.make_window(i, r) for i, r in enumerate(a + b)]
    T.set_reach(wins, 128)
    return wins


def rows_of(wins):
    weights = {w.idx: T.heur_weights(w) for w in wins}
    D, A = T.step_totals(wins)
    return {"score": T.scoring_rows(wins), "train": T.train_rows(wins, weights, D, A, 0.0)}


def same_batch(a, b, skip=()) -> None:
    for k in TENSORS + ("sd_post",):
        if k in skip:
            continue
        x, y = getattr(a, k), getattr(b, k)
        assert (x is None and y is None) or torch.equal(x, y), k
    if "P_prefix" not in skip:
        assert len(a.P_prefix) == len(b.P_prefix) and all(torch.equal(x, y) for x, y in zip(a.P_prefix, b.P_prefix))
    assert a.F_frames == b.F_frames and a.dur_s == b.dur_s
    if "metas" not in skip:
        assert a.metas == b.metas
    assert [(r.win.idx, r.r, r.K) for r in a.rows] == [(r.win.idx, r.r, r.K) for r in b.rows]


def test_off_bit_identical_to_before():
    old = old_module()
    wins = windows()
    ctx = TT.ctx_of(perturb=PERTURB)
    n = 0
    for phase, rows in rows_of(wins).items():
        for hook in (False, True):
            b_old = old.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], hook, PERTURB, SEED, STEP, phase)
            b_def = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], hook, PERTURB, SEED, STEP, phase)
            b_off = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], hook, PERTURB, SEED, STEP, phase,
                              final_full=False)
            same_batch(b_old, b_def)
            same_batch(b_old, b_off)
            n += len(rows)
    for w in wins:
        for r in range(w.R):
            xo, po, To = old.row_inputs(w, r, ctx["audio"], PERTURB, SEED, STEP)
            xn, pn, Tn = T.row_inputs(w, r, ctx["audio"], PERTURB, SEED, STEP)
            assert np.array_equal(xo, xn) and np.array_equal(po, pn) and To == Tn
    assert T.build_args(["--objective", "heuristic", "--out", "x"]).final_read_full_post == 0
    assert T.build_args(["--objective", "heuristic", "--out", "x", "--final-read-full-post", "1"]).final_read_full_post == 1
    print(f"  off: {n} rows (score/train x hook off/on) bit-identical to the trainer at {BEFORE} and to the default")


def expected_post(w, r, audio):
    """the real (perturbed, same draw as row_inputs) posteriors of read r, unmasked."""
    _, post = audio.get(w)
    return perturb(post, np.random.default_rng([SEED, STEP, w.idx, r]), PERTURB)


def test_rows_on():
    wins = windows()
    audio = T.AudioCache()
    n_ext, nz = 0, 0
    for w in wins:
        for r in range(w.R):
            xo, po, To = T.row_inputs(w, r, audio, PERTURB, SEED, STEP)
            xn, pn, Tn = T.row_inputs(w, r, audio, PERTURB, SEED, STEP, final_full=True)
            assert np.array_equal(xo, xn)
            if r < w.R - 1:
                assert np.array_equal(po, pn) and To == Tn
                continue
            t = int(w.t_ms[r])
            assert t == w.rec["dur_ms"]
            T0 = max(0, t - 1040) // 80
            assert To == min(T0, len(po)) and Tn == min(t // 80, len(pn)) and Tn > To, (To, Tn)
            full = expected_post(w, r, audio)
            assert np.array_equal(pn[:T0], po[:T0])                      # frames <= t - 1.04 s: unchanged
            assert np.array_equal(pn[T0:Tn], full[T0:Tn])                # extended frames: the real values
            assert not pn[Tn:].any() and not po[T0:].any()               # partial last frame / beyond: still 0
            diff = np.nonzero((pn != po).any(1))[0]
            assert len(diff) and diff.min() >= T0 and diff.max() < Tn
            n_ext += Tn - T0
            nz += int((full[T0:Tn] != 0).any())
    assert nz == len(wins), nz
    print(f"  rows: {len(wins)} final reads extended by {n_ext} frames in total (each region non-zero); "
          f"non-final reads unchanged")


def test_batch_on():
    wins = windows()
    ctx = TT.ctx_of(perturb=PERTURB)
    m = TT.tiny_model(seed=5)
    n_fin = 0
    for phase, rows in rows_of(wins).items():
        fin = [rw.r == rw.win.R - 1 for rw in rows]
        assert any(fin) and not all(fin)
        # v8 path (hook off)
        b0 = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], False, PERTURB, SEED, STEP, phase)
        b1 = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], False, PERTURB, SEED, STEP, phase,
                       final_full=True)
        same_batch(b0, b1, skip=("sd_post",))
        for i, rw in enumerate(rows):
            if not fin[i]:
                assert torch.equal(b0.sd_post[i], b1.sd_post[i])
                continue
            n_fin += 1
            t = int(rw.win.t_ms[rw.r])
            p = expected_post(rw.win, rw.r, ctx["audio"]).copy()
            p[t // 80:] = 0.0
            nA = int(b1.n_audio[i])
            dur = len(T.row_inputs(rw.win, rw.r, ctx["audio"], PERTURB, SEED, STEP)[0]) / 16000
            q = torch.as_tensor(post_to_tokens(p, nA, dur)).to(torch.float32)   # collate stores float32
            assert torch.equal(b1.sd_post[i, :nA], q) and not b1.sd_post[i, nA:].any()
            assert not torch.equal(b0.sd_post[i], b1.sd_post[i])
        # embeddings: only the audio positions of final rows move
        with torch.no_grad():
            e0 = m.embed_batch(m.to_device(b0))
            e1 = m.embed_batch(m.to_device(b1))
        aud = b0.input_ids == AUDIO_PAD
        dif = (e0 != e1).any(-1)
        assert not (dif & ~aud).any()
        for i in range(len(rows)):
            assert bool(dif[i].any()) == fin[i], (i, fin[i])
        # hook path
        h0 = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], True, PERTURB, SEED, STEP, phase)
        h1 = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], True, PERTURB, SEED, STEP, phase,
                       final_full=True)
        same_batch(h0, h1, skip=("P_prefix", "metas"))
        for i, rw in enumerate(rows):
            m0, m1 = dict(h0.metas[i]), dict(h1.metas[i])
            if not fin[i]:
                assert torch.equal(h0.P_prefix[i], h1.P_prefix[i]) and m0 == m1
                continue
            t = int(rw.win.t_ms[rw.r])
            full = expected_post(rw.win, rw.r, ctx["audio"])
            assert m1.pop("T") == t // 80 and m0.pop("T") == max(0, t - 1040) // 80 and m0 == m1
            assert torch.equal(h1.P_prefix[i], torch.as_tensor(full[: t // 80]))
    print(f"  batch: {n_fin} final rows differ only in their posteriors (sd_post = post_to_tokens of the full-window P; "
          f"hook P_prefix / T); embeddings differ only at final rows' audio positions")


def test_plumbing():
    wins = windows()[:3]
    seen = []
    orig = T.collate

    def rec(*a, **k):
        seen.append((a[8], k.get("final_full", False)))
        return orig(*a, **k)

    m = TT.tiny_model(seed=5)
    try:
        T.collate = rec
        for flag in (None, False, True):
            ctx = TT.ctx_of(perturb=PERTURB)
            if flag is not None:
                ctx["final_full_post"] = flag
            seen.clear()
            T.score_pass(m, wins, ctx, STEP)
            rows = rows_of(wins)["train"]
            TT.zero(m)
            T.grad_pass(m, rows, ctx, STEP)
            assert {p for p, _ in seen} == {"score", "train"}
            assert all(f == bool(flag) for _, f in seen), (flag, seen)
    finally:
        T.collate = orig
    print("  plumbing: ctx final_full_post (absent / False / True) reaches collate in score_pass and grad_pass")


def main():
    t0 = time.time()
    tests = [test_off_bit_identical_to_before, test_rows_on, test_batch_on, test_plumbing]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
