"""Tests for the speech-gap batch-1 code: si_talker_fix (B2/B3 helpers), si_ctc_scorer.ctc_text (B1), si_infer's
new flags (defaults off = unchanged) and speech_var_orch's talker commands.

    conda_envs/qwen3omni/bin/python -m pytest -q experiments/mist_w3po_v1/evaluation/test_si_talker_fix.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))

from experiments.uniss_qwen3_omni_v1.evaluation.si_talker_fix import (  # noqa: E402
    forced_limit, is_forced, own_spans_ms, own_speaker_heard)
from experiments.mist_w3po_v1.evaluation.si_ctc_scorer import ctc_text  # noqa: E402

SI_INFER = REPO / "experiments/uniss_qwen3_omni_v1/evaluation/si_infer.py"


def test_forced_limit_matches_si_infer_rule():
    for exp in (0, 10, 50, 83, 84, 200):
        assert forced_limit(exp) == max(int(2.2 * exp), exp + 100)
        assert is_forced(forced_limit(exp), exp) and not is_forced(forced_limit(exp) - 1, exp)
    src = SI_INFER.read_text()
    assert src.count("max(int(2.2 * exp), exp + 100)") == 2   # the original rule and B2's copy agree


def test_own_spans_causal_and_swapped():
    blocks = [{"write_ms": 1000, "start_s": 0.1, "end_s": 0.5},
              {"write_ms": 2000, "start_s": 1.9, "end_s": 1.2},    # reversed by the model
              {"write_ms": 3000, "start_s": 0.6, "end_s": 0.8}]    # written later: invisible at 2000
    assert own_spans_ms(blocks, 2000, 0) == [(100, 500), (1200, 1900)]
    assert own_spans_ms(blocks, 2000, 250) == [(-150, 750), (950, 2150)]
    assert own_spans_ms(blocks, 1000, 0) == [(100, 500)]


def test_own_speaker_heard_masks_outside_and_keeps_length():
    x = np.ones(16 * 1000, dtype=np.float32)        # 1 s
    y = own_speaker_heard(x, [(-50, 200), (900, 5000)])
    assert len(y) == len(x) and y.dtype == x.dtype
    assert y[: 200 * 16].min() == 1 and y[200 * 16: 900 * 16].max() == 0 and y[900 * 16:].min() == 1
    assert x.min() == 1                              # input untouched
    assert own_speaker_heard(x, []).max() == 0


def test_ctc_text_alphabet():
    assert ctc_text("In the time,", "you’re welcome to move it.") == "IN THE TIME YOU'RE WELCOME TO MOVE IT"
    assert ctc_text("", "1990s -- ok!") == "S OK"
    assert ctc_text("", "...") == ""
    assert re.fullmatch(r"[A-Z' ]*", ctc_text("Ünïcode", "x"))


def test_si_infer_new_flags_default_off():
    src = SI_INFER.read_text()
    assert '"--rerun-forced", type=int, default=0' in src
    assert '"--src-mask-own-ms", type=int, default=-1' in src
    assert "for a in range(1, args.rerun_forced + 1):" in src            # no iteration at the default
    assert "if args.src_mask_own_ms >= 0:" in src
    assert '**({"rerun_forced": args.rerun_forced} if args.rerun_forced else {})' in src
    assert '**({"src_mask_own_ms": args.src_mask_own_ms} if args.src_mask_own_ms >= 0 else {})' in src


def test_speech_var_talker_cmds():
    import speech_var_orch as sv
    base = ("X -m m.si_infer --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 "
            "--shard 3 --num-shards 8 --seed 20261109 > log 2>&1")
    assert sv.talker_cmd(base, "c1", 3, 4) == base.replace("--speed-policy fixed", "--speed-policy queue")
    assert "--gpu-mem 0.1 --src-mask-own-ms 250 --shard 3 " in sv.talker_cmd(base, "b3", 3, 4)
    assert "--gpu-mem 0.1 --rerun-forced 3 --shard 3 " in sv.talker_cmd(base, "b2", 3, 4)
    assert "--select-k 2 --scorer-url http://127.0.0.1:18768/score --shard 3 " in sv.talker_cmd(base, "b1", 3, 4)
    assert "127.0.0.1:18765/score" in sv.talker_cmd(base.replace("--shard 3", "--shard 4"), "b1", 4, 4)


def test_speech_var_jobs_only_talker_changes():
    import crf_pair_dev_orch as cp
    import finalpost_dev_orch as fo
    import speech_var_orch as sv
    jobs = sv.build_jobs(["b3"], 4)
    ref: list = []

    def add(*a, **k):
        ref.append(fo.Job(*a, **k))
    for name, seed in ((f"{sv.BASE}__b3", None), (f"{sv.BASE}__b3_tts2", fo.SEED2)):
        cp.speech(add, name, sv.BASE, seed, 2, [])
    assert [j.id for j in jobs] == [j.id for j in ref]
    for j, r in zip(jobs, ref):
        a, b = j.cmd(1), r.cmd(1)
        if j.kind == "talk":
            assert a == b.replace(" --shard ", " --src-mask-own-ms 250 --shard ", 1) and a != b
        else:
            assert a == b
    assert sum(j.kind == "talk" for j in jobs) == 16
    assert any("--seed 20261109" in j.cmd(1) for j in jobs if j.kind == "talk")
