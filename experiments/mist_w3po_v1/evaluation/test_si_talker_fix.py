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


def test_speech_var_combos_and_base():
    import speech_var_orch as sv
    base = ("X -m m.si_infer --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 "
            "--shard 3 --num-shards 8 --seed 20261109 > log 2>&1")
    c = sv.talker_cmd(base, "c1+b1", 3, 4)
    assert c == sv.talker_cmd(base, "b1", 3, 4).replace("--speed-policy fixed", "--speed-policy queue")
    assert sv.tag("c1+b1") == "c1_b1" and sv.tag("b3") == "b3"
    assert sv.split_var("c1+b1", "B0") == ("c1+b1", "B0") and sv.split_var("c1@V2", "B0") == ("c1", "V2")
    jobs = sv.build_jobs(["c1+b1", "c1@dev_crf_v2p_s3000_finalpost_s8"], 4)
    ids = [j.id for j in jobs]
    assert f"{sv.BASE}__c1_b1__talk0" in ids and "dev_crf_v2p_s3000_finalpost_s8__c1_tts2__talk7" in ids
    prep = [j.cmd(1) for j in jobs if j.id == "dev_crf_v2p_s3000_finalpost_s8__c1__sprep"][0]
    assert "decode/dev_crf_v2p_s3000_finalpost_s8/hyp.jsonl" in prep
    assert sum(j.kind == "talk" for j in jobs) == 32
    # batch-1 job list unchanged by the new arguments
    assert [j.id for j in sv.build_jobs(["b1", "b3", "b2", "c1"], 4)] == \
        [j.id for j in sv.build_jobs(["b1", "b3", "b2", "c1"], 4, sv.BASE)]


def test_online_windows_gap_default_unchanged():
    import importlib.util
    from experiments.mist_w3po_v1.data.build_windows_zh2en import online_windows
    spec = importlib.util.spec_from_file_location("bow6", REPO / "reports/gstar_si_v6/build_online_windows.py")
    rng = np.random.default_rng(0)
    post = (rng.random((3000, 4)) < 0.04).astype(np.float32) * 0.9   # sparse activity, gaps of all lengths
    w = online_windows(post)
    assert w == online_windows(post, 1.0)
    try:
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert w == m.online_windows(post)
    except ImportError:   # the v6 script's own imports may be absent; the default-equality above still holds
        pass
    w5 = online_windows(post, 0.5)   # a shorter gap only splits: each window's active span lies inside one at 1.0 s
    assert len(w5) > len(w) and all(any(s0 <= s and e <= e0 for s0, e0, _ in w) for s, e, _ in w5)


def test_c2_orch_windows_and_talker():
    import crf_pair_dev_orch as cp
    import c2_dev_orch as c2
    saved = (cp.WIN, cp.CANDS, cp.DIAG)
    try:
        jobs = c2.build_jobs(4)
        cmds = [j.cmd(1) for j in jobs]
        assert not any(j.id.endswith("__diagA") for j in jobs)
        assert not any("windows_zh2en_v1/windows_dev_online" in c for c in cmds)
        assert sum(c2.WIN_C2 in c for c in cmds) == 13          # 8 decodes, text score, 2 preps, 2 speech scores
        talk = [c for j, c in zip(jobs, cmds) if j.kind == "talk"]
        assert len(talk) == 16 and all("--speed-policy queue" in c and "--select-k 2" in c for c in talk)
        assert "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl" in \
            [c for j, c in zip(jobs, cmds) if j.id.endswith("__score")][0]
    finally:
        cp.WIN, cp.CANDS, cp.DIAG = saved


def test_speech_var_second_points():
    import speech_var_orch as sv
    base = ("X -m m.si_infer --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 "
            "--shard 5 --num-shards 8 --seed 20261109 > log 2>&1")
    p0 = sv.SCORER_PORT0
    try:
        sv.SCORER_PORT0 = 18769
        c = sv.talker_cmd(base, "c1+b1k4", 5, 4)
        assert "--speed-policy queue" in c and "--select-k 4 --scorer-url http://127.0.0.1:18770/score --shard 5" in c
        c = sv.talker_cmd(base, "c1+b1+s2", 5, 4)
        assert "--select-k 2 --scorer-url http://127.0.0.1:18770/score --speed-shift 2 --shard 5" in c
    finally:
        sv.SCORER_PORT0 = p0
    assert sv.SCORER_PORT0 == 18765 and "18766" in sv.talker_cmd(base, "b1", 5, 4)


def test_speech_latency_win_default_unchanged():
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M
    from experiments.mist_w3po_v1.evaluation import speech_latency_lit as sl
    a = sl.build_parser().parse_args(["--run", "r", "--talker", "t", "--out", "o"])
    assert a.win == M.WIN == "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
    w = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g05_v1/windows_dev_online.jsonl"
    assert sl.build_parser().parse_args(["--run", "r", "--talker", "t", "--out", "o", "--win", w]).win == w


def test_c2_second_point_and_speech_var_win():
    import crf_pair_dev_orch as cp
    import c2_dev_orch as c2
    import speech_var_orch as sv
    saved = (cp.WIN, cp.CANDS, cp.DIAG)
    w6 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
    n6 = "dev_crf_v2p_s3000_finalpost_s8_c2g06"
    try:
        jobs = c2.build_jobs(4, w6, n6, "c1+b1+s2")
        cmds = [j.cmd(1) for j in jobs]
        assert sum(w6 in c for c in cmds) == 13 and not any(c2.WIN_C2 in c for c in cmds)
        assert list(cp.CANDS) == ["v2p_c2g06"] and cp.CANDS["v2p_c2g06"][0] == n6 and "c2g05" not in cp.DIAG
        assert all(f"/{n6}" in c or f"/{n6}_tts2" in c for j, c in zip(jobs, cmds) if j.kind in ("talk", "dec"))
        talk = [c for j, c in zip(jobs, cmds) if j.kind == "talk"]
        assert len(talk) == 16 and all("--speed-policy queue" in c and "--select-k 2" in c and "--speed-shift 2" in c
                                       for c in talk)
        # defaults unchanged: the gap-0.5 run's names / windows / talker
        jobs = c2.build_jobs(4)
        cmds = [j.cmd(1) for j in jobs]
        assert list(cp.CANDS) == ["v2p_c2g05"] and cp.DIAG.endswith("v2p_c2g05_dev_diagA")
        assert sum(c2.WIN_C2 in c for c in cmds) == 13 and not any("--speed-shift" in c for c in cmds)
        # speech_var_orch on a C2 base: its speech chain reads cp.WIN at call time (main sets it from --win)
        cp.WIN = c2.WIN_C2
        jobs = sv.build_jobs(["c1+b1+s2@dev_crf_v2p_s3000_finalpost_s8_c2g05"], 4)
        cmds = [j.cmd(1) for j in jobs]
        assert sum(c2.WIN_C2 in c for c in cmds) == 4 and not any("windows_zh2en_v1/" in c for c in cmds)
    finally:
        cp.WIN, cp.CANDS, cp.DIAG = saved
    assert cp.WIN == "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"


def test_speech_gap_decomp_win_default_unchanged():
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M
    from experiments.mist_w3po_v1.evaluation import speech_gap_decomp as sg
    a = sg.build_parser().parse_args(["--run", "r", "--speech", "s", "--out", "o"])
    assert a.win == M.WIN and a.sem == "" and a.asr_override == "" and a.asr_override_field == "asr_text"
    w = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
    assert sg.build_parser().parse_args(["--run", "r", "--speech", "s", "--out", "o", "--win", w]).win == w
