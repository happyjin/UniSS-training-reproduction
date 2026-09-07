"""The Silver-Medal rule, pinned on the property that makes it work.

NaturalFlow's ablations are the reason this rule exists: taking the candidate
with the lowest silence ratio collapsed their model to BLEU 1.50, and removing
the penalty on the extreme band gave unintelligible fast speech at BLEU 0.92.
So the test that matters is not "does it lower silence" -- it is "does it
refuse to take the extreme", which is what our own beam-search attempt lacked.
"""

from __future__ import annotations

import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.silver_medal import (
    SAMPLE_RATE,
    choose,
    silence_ratio,
)


def _cand(sr, chrf=100.0, arm=None):
    return {"arm": arm or f"a{sr}", "silence_ratio": sr, "chrf": chrf}


def test_the_lowest_candidate_is_never_chosen():
    pool = [_cand(s) for s in (0.05, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60)]
    pick, reason = choose(pool, min_chrf=0.0)
    assert pick["silence_ratio"] != 0.05, "the extreme band must be rejected"
    assert reason == "silver medal"


def test_the_pick_lands_in_the_second_quintile():
    pool = [_cand(round(0.05 * i, 2)) for i in range(1, 11)]
    pick, _ = choose(pool, min_chrf=0.0)
    ordered = sorted(c["silence_ratio"] for c in pool)
    second = ordered[2:4]
    assert pick["silence_ratio"] in second


def test_the_pick_still_beats_the_median():
    """Rejecting the extreme must not throw away the benefit entirely."""
    pool = [_cand(round(0.05 * i, 2)) for i in range(1, 11)]
    pick, _ = choose(pool, min_chrf=0.0)
    median = sorted(c["silence_ratio"] for c in pool)[len(pool) // 2]
    assert pick["silence_ratio"] < median


def test_a_drifted_candidate_is_dropped_before_selection():
    pool = [_cand(0.05, chrf=10.0), _cand(0.30, chrf=90.0), _cand(0.40, chrf=95.0)]
    pick, _ = choose(pool, min_chrf=60.0)
    assert pick["chrf"] >= 60.0


def test_no_candidate_clearing_quality_is_reported_not_forced():
    pool = [_cand(0.05, chrf=10.0), _cand(0.10, chrf=20.0)]
    pick, reason = choose(pool, min_chrf=60.0)
    assert pick is None
    assert "chrF" in reason


def test_a_small_pool_still_avoids_the_extreme():
    pool = [_cand(0.10), _cand(0.20), _cand(0.30)]
    pick, reason = choose(pool, min_chrf=0.0)
    assert pick["silence_ratio"] == 0.20
    assert reason == "too few to stratify"


def test_a_single_candidate_is_returned_unchanged():
    pick, _ = choose([_cand(0.25)], min_chrf=0.0)
    assert pick["silence_ratio"] == 0.25


def test_unmeasurable_candidates_are_ignored():
    pool = [{"arm": "x", "silence_ratio": None, "chrf": 100.0}, _cand(0.30)]
    pick, _ = choose(pool, min_chrf=0.0)
    assert pick["silence_ratio"] == 0.30


def _tone(ms, amp=0.3):
    n = SAMPLE_RATE * ms // 1000
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_silence_ratio_counts_only_gaps_over_100ms():
    speech = _tone(400)
    short = np.zeros(SAMPLE_RATE * 60 // 1000, dtype=np.float32)   # 60 ms
    long = np.zeros(SAMPLE_RATE * 400 // 1000, dtype=np.float32)   # 400 ms
    only_short = np.concatenate([speech, short, speech])
    with_long = np.concatenate([speech, long, speech])
    assert silence_ratio(only_short) < 0.02
    assert silence_ratio(with_long) > 0.25


def test_silence_ratio_ignores_leading_and_trailing_silence():
    """The span runs from first voiced frame to last, as their D_span does."""
    speech = np.concatenate([_tone(300), np.zeros(SAMPLE_RATE // 2, dtype=np.float32), _tone(300)])
    padded = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), speech,
                             np.zeros(SAMPLE_RATE, dtype=np.float32)])
    assert abs(silence_ratio(speech) - silence_ratio(padded)) < 0.02
