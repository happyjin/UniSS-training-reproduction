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


def test_the_repo_protocol_normalises_before_scoring():
    """Punctuation and spacing must not count, because ASR emits neither.

    This is the project's published protocol and it differs from bare
    sacrebleu by enough to flip the sign of a measured difference, so the two
    are kept separate rather than blended.
    """
    from experiments.uniss_streaming_p2st_traj_v1.evaluation.paired_bleu import (
        corpus_bleu,
    )

    hyp = ["我 昨天 去 了 公园。"]
    ref = ["我昨天去了公园"]
    # A perfect match after normalisation, up to floating point.
    assert corpus_bleu(hyp, ref, "zh", protocol="repo", language="cmn") > 99.99
    assert corpus_bleu(hyp, ref, "zh", protocol="raw", language="cmn") < 90.0


def test_an_unknown_protocol_is_refused():
    import pytest

    from experiments.uniss_streaming_p2st_traj_v1.evaluation.paired_bleu import (
        corpus_bleu,
    )

    with pytest.raises(ValueError):
        corpus_bleu(["a"], ["a"], "13a", protocol="made-up", language="eng")
