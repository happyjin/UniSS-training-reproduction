"""Tests of the joint bidirectional options of crf/train_crf_sft.py (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1
.zh-CN.md), CPU only, real tokenizer / feature extractor / prompts and real sched windows of both directions:

  (a) plan_joint: deterministic; its direction-1 part is plan_steps(set 1, seed, steps, rows / 2) and its direction-2
      part plan_steps(set 2, seed + 1, steps, rows / 2) offset by len(set 1); ~rows / 2 read rows of each direction per
      step; each direction's first pass is a permutation
  (b) JointPrompter + collate: a zh2en window's batch equals the batch of the single-direction trainer with
      Prompter("zh2en") bit for bit, an en2zh window's equals Prompter("en2zh")'s; a mixed batch is the two stacked;
      pick() / n_prompt0 as documented; the plain Prompter path is unchanged (no pick attribute is consulted)
  (c) build_args: the new options default off ("" each); invalid combinations are refused; defaults leave every
      pre-existing argument unchanged
  (e) lr_at with --lr-horizon: 0 (default) = the --steps cosine exactly; H: cosine to min_lr at H, then flat
  (d) stop_file_hit: "" -> False without touching the file system; a missing file -> False; an existing one -> True

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_joint_bidir
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import train_crf_sft as T

S_ZH = "eval_outputs/mist_w3po_v1/crf/sched_v1"
S_EN = "eval_outputs/mist_w3po_v1/crf/sched_en2zh_v1"


def test_plan_joint():
    Rs1 = list(np.random.default_rng(0).integers(1, 23, 1500))
    Rs2 = list(np.random.default_rng(1).integers(1, 30, 1100))
    p = T.plan_joint(Rs1, Rs2, 20261109, 1500, 64)
    assert p == T.plan_joint(Rs1, Rs2, 20261109, 1500, 64) and len(p) == 1500
    a = T.plan_steps(Rs1, 20261109, 1500, 32)
    b = T.plan_steps(Rs2, 20261110, 1500, 32)
    n1 = len(Rs1)
    for k in range(1500):
        assert p[k] == a[k] + [n1 + i for i in b[k]], k
    r1 = [sum(Rs1[i] for i in s if i < n1) for s in p]
    r2 = [sum(Rs2[i - n1] for i in s if i >= n1) for s in p]
    assert abs(np.mean(r1) - 32) < 3 and abs(np.mean(r2) - 32) < 3, (np.mean(r1), np.mean(r2))
    f1 = [i for s in p for i in s if i < n1]
    f2 = [i - n1 for s in p for i in s if i >= n1]
    assert sorted(f1[:n1]) == list(range(n1)) and sorted(f2[: len(Rs2)]) == list(range(len(Rs2)))
    print(f"  plan_joint: {np.mean(r1):.1f} + {np.mean(r2):.1f} rows / step, = the two halved single plans")


def windows(sched: str, n: int, d: str, offset: int = 0) -> list:
    recs = [json.loads(l) for l in open(Path(sched) / "shard_000.jsonl", encoding="utf-8")]
    good = [r for r in recs if 2 <= r["n_read"] <= 4 and 8 <= r["n_tok"] <= 40]
    ws = [T.make_window(offset + i, r) for i, r in enumerate(good[:n])]
    for w in ws:
        w.dir = d
    T.set_reach(ws, 128)
    return ws


def same_batch(x: T.Batch, y: T.Batch) -> None:
    for k in ("input_ids", "attention_mask", "input_features", "feature_attention_mask", "n_audio", "has_sd", "sd_post",
              "sel_row", "sel_col", "tok_tgt", "end_tgt"):
        a, b = getattr(x, k), getattr(y, k)
        if a is None or b is None:
            assert a is None and b is None, k
            continue
        assert a.shape == b.shape and torch.equal(a, b), k


def test_joint_prompter_collate():
    from experiments.mist_w3po_v1.core.engine import Prompter

    pz, pe = Prompter("zh2en"), Prompter("en2zh")
    assert pz.a_ids + pz.b_ids != pe.a_ids + pe.b_ids   # the two directions' prompts differ
    jp = T.JointPrompter({"zh2en": pz, "en2zh": pe}, "zh2en")
    assert jp.n_prompt0 == max(len(pz.a_ids) + len(pz.b_ids), len(pe.a_ids) + len(pe.b_ids)) - 1
    wz = windows(S_ZH, 3, "zh2en")
    we = windows(S_EN, 3, "en2zh", offset=100000)
    assert jp.pick(wz[0]) is pz and jp.pick(we[0]) is pe
    wz[0].dir = ""
    assert jp.pick(wz[0]) is pz   # empty dir -> the first direction
    wz[0].dir = "zh2en"
    audio = T.AudioCache()
    rz, re_ = T.scoring_rows(wz), T.scoring_rows(we)
    for rows, single in ((rz, pz), (re_, pe)):
        b1 = T.collate(rows, audio, pz.proc, jp, False, 0.0, 7, 0, "score")
        b0 = T.collate(rows, audio, pz.proc, single, False, 0.0, 7, 0, "score")
        same_batch(b1, b0)
        b1 = T.collate(rows, audio, pz.proc, jp, False, 0.2, 7, 3, "train")
        b0 = T.collate(rows, audio, pz.proc, single, False, 0.2, 7, 3, "train")
        same_batch(b1, b0)
    mixed = T.collate(rz[:2] + re_[:2], audio, pz.proc, jp, False, 0.0, 7, 0, "score")
    ids = mixed.input_ids.tolist()
    z0 = T.collate(rz[:1], audio, pz.proc, pz, False, 0.0, 7, 0, "score").input_ids[0].tolist()
    e0 = T.collate(re_[:1], audio, pz.proc, pe, False, 0.0, 7, 0, "score").input_ids[0].tolist()
    strip = lambda s: [t for t in s if t != pz.tok.pad_token_id]
    assert strip(ids[0]) == strip(z0) and strip(ids[2]) == strip(e0)
    print(f"  JointPrompter: per-direction batches = single-direction batches bit for bit ({len(rz)} + {len(re_)} rows)")


def test_build_args():
    base = ["--objective", "serial_v2p", "--out", "/nonexistent/x"]
    a = T.build_args(base)
    assert a.sched2 == "" and a.dir2 == "" and a.stop_file == ""
    assert a.dir == "zh2en" and a.steps == 3000 and a.save_every == 500 and a.rows_per_step == 64
    j = T.build_args(base + ["--sched2", S_EN, "--dir2", "en2zh", "--stop-file", "/tmp/x", "--steps", "6000"])
    assert j.sched2 == S_EN and j.dir2 == "en2zh" and j.stop_file == "/tmp/x" and j.steps == 6000
    for bad in (["--sched2", S_EN], ["--dir2", "en2zh"], ["--sched2", S_EN, "--dir2", "zh2en"],
                ["--sched2", S_EN, "--dir2", "fr2en"]):
        try:
            T.build_args(base + bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    try:
        T.build_args(["--objective", "latent_write", "--out", "/x", "--sched2", S_EN, "--dir2", "en2zh"])
        raise AssertionError("latent_write + --sched2 accepted")
    except SystemExit:
        pass
    print("  build_args: new options default off, invalid combinations refused")


def test_stop_file_hit():
    assert T.stop_file_hit("", torch.device("cpu")) is False
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "STOP"
        assert T.stop_file_hit(str(f), torch.device("cpu")) is False
        f.write_text("x")
        assert T.stop_file_hit(str(f), torch.device("cpu")) is True
    print("  stop_file_hit: off / missing / present")


def test_lr_horizon():
    from types import SimpleNamespace as NS
    a = NS(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000)
    b = NS(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000, lr_horizon=0)
    for k in (1, 50, 51, 1000, 2999, 3000):
        assert T.lr_at(k, a) == T.lr_at(k, b)
    h = NS(lr=1e-4, min_lr=1e-5, warmup=50, steps=15000, lr_horizon=12000)
    assert T.lr_at(12000, h) == 1e-5 and T.lr_at(15000, h) == 1e-5 and T.lr_at(50, h) == 1e-4
    assert abs(T.lr_at(3050, h) - (1e-5 + 9e-5 * 0.5 * (math.cos(math.pi * 3000 / 11950) + 1))) < 1e-15
    assert T.build_args(["--objective", "serial_v2p", "--out", "/x"]).lr_horizon == 0
    print("  lr horizon: 0 = unchanged; 12000 horizon flat at min_lr after it")


def main():
    t0 = time.time()
    tests = [test_plan_joint, test_build_args, test_stop_file_hit, test_lr_horizon, test_joint_prompter_collate]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
