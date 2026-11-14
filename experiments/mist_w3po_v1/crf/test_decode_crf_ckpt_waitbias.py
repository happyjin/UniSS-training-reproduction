"""Tests of plan A's write bias (CPU): argv handling, the Engine.read wrapper (bias reaches Engine.read's wait_bias and
composes with online_decode_finalpost's wrapper), and joint_orch_wb's command rewrite.

    python -m experiments.mist_w3po_v1.crf.test_decode_crf_ckpt_waitbias
"""
from __future__ import annotations

import sys
from pathlib import Path

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.crf import decode_crf_ckpt_waitbias as W
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import joint_orch_wb as JW  # noqa: E402


def test_take_bias():
    b, r = W.take_bias(["--wait-bias", "-2.5", "--crf-ckpt", "x", "--policy", "wait"])
    assert b == -2.5 and r == ["--crf-ckpt", "x", "--policy", "wait"], (b, r)
    b, r = W.take_bias(["--a", "1", "--wait-bias=-1"])
    assert b == -1.0 and r == ["--a", "1"]
    try:
        W.take_bias(["--a"])
        raise AssertionError("missing --wait-bias accepted")
    except SystemExit:
        pass


def test_wrapper_composes():
    seen = []
    saved = E.Engine.read

    def fake(self, items, max_new=300, temp=0.0, temp_when=None, gen=None, record=False, wait_bias=0.0, cache=None):
        seen.append((list(items), max_new, wait_bias))
        return []

    try:
        E.Engine.read = fake
        W.install(-2.0)
        fp_orig = fp.install({})   # finalpost wraps ours (its install runs later in the real decoder)
        assert fp_orig is not fake
        fp.transform = lambda items, cd: items   # identity for the test
        E.Engine.read(object(), ["it"], max_new=7)
        E.Engine.read(object(), ["it"], wait_bias=0.5)
        assert seen[0] == (["it"], 7, -2.0), seen
        assert seen[1][2] == -1.5, seen
    finally:
        E.Engine.read = saved


def test_rewrite():
    c = "env X python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt c --thinker t --out o"
    r = JW.rewrite(c, -2.0)
    assert "decode_crf_ckpt_waitbias --wait-bias -2.0 --crf-ckpt c" in r and "finalpost " not in r, r
    assert JW.rewrite("python -m experiments.mist_w3po_v1.evaluation.score_run --run x", -2.0).endswith("--run x")
    try:
        JW.rewrite("python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt --crf-ckpt c", -2.0)
        raise AssertionError("plain decode_crf_ckpt accepted")
    except SystemExit:
        pass


if __name__ == "__main__":
    for f in (test_take_bias, test_wrapper_composes, test_rewrite):
        f()
        print("ok", f.__name__)
