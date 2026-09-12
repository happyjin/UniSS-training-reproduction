"""The bootstrap must find a real difference and refuse an imaginary one."""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.paired_bleu import (
    paired_bootstrap,
)

REFS = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(120)]


def _arm(hyps):
    return {f"s{i}": (h, r) for i, (h, r) in enumerate(zip(hyps, REFS))}


def test_identical_systems_give_an_interval_that_contains_zero():
    same = _arm(list(REFS))
    got = paired_bootstrap(same, same, tokenize="13a", samples=200)
    assert got["delta"] == 0.0
    assert got["ci_low"] <= 0 <= got["ci_high"]


def test_a_system_that_is_right_everywhere_beats_one_that_is_wrong_everywhere():
    good = _arm(list(REFS))
    bad = _arm(["completely unrelated words here" for _ in REFS])
    got = paired_bootstrap(good, bad, tokenize="13a", samples=200)
    assert got["delta"] > 30.0
    assert got["ci_low"] > 0, "the interval must exclude zero"
    assert got["p"] == 0.0


def test_the_pairing_uses_only_shared_utterances():
    a = _arm(list(REFS))
    b = {k: v for k, v in _arm(list(REFS)).items() if k != "s0"}
    got = paired_bootstrap(a, b, tokenize="13a", samples=50)
    assert got["n"] == len(REFS) - 1
