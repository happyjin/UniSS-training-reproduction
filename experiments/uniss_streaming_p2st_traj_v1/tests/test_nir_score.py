"""The NIR statistic is the paper's, so it is checked against its definition.

Appendix A.3: NIR = 2I / (|A|(|A|-1)) * 100%, where I is the inversion number of
the source positions aligned to target words.  A merge-sort counter replaces the
quadratic one, so the equivalence is asserted on random sequences rather than
assumed.
"""
from __future__ import annotations

import random

import pytest

from experiments.uniss_streaming_p2st_traj_v1.data.nir_score import (
    inversion_count,
    normalized_inversion_rate,
    score_record,
)


def quadratic_inversions(values: list[int]) -> int:
    return sum(
        1
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if values[i] > values[j]
    )


@pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 8, 9, 64, 129])
def test_merge_sort_matches_the_quadratic_count(size: int) -> None:
    rng = random.Random(20260904 + size)
    for _ in range(20):
        values = [rng.randrange(0, max(2, size)) for _ in range(size)]
        assert inversion_count(values) == quadratic_inversions(values)


def test_inversion_count_does_not_mutate_its_input() -> None:
    values = [3, 1, 2]
    inversion_count(values)
    assert values == [3, 1, 2]


def test_known_extremes() -> None:
    # Perfectly monotone alignment: no reordering at all.
    assert normalized_inversion_rate([0, 1, 2, 3, 4]) == 0.0
    # Fully reversed: every pair is an inversion, so the rate saturates at 100%.
    assert normalized_inversion_rate([4, 3, 2, 1, 0]) == 100.0
    # Ties are not inversions -- many target words can align to one source word.
    assert normalized_inversion_rate([2, 2, 2]) == 0.0
    # One swap out of three pairs.
    assert normalized_inversion_rate([0, 2, 1]) == pytest.approx(100.0 / 3.0)


def test_rate_is_undefined_below_two_words() -> None:
    assert normalized_inversion_rate([]) is None
    assert normalized_inversion_rate([5]) is None


def _gold(sample_id: str = "s0") -> dict:
    return {
        "sample_id": sample_id,
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_duration_ms": 4000,
        "source_manifest_record": 0,
    }


def _stage_a(sample_id: str = "s0") -> dict:
    return {
        "id": sample_id,
        "target_support": [
            {
                "source_links": [{"source_index": 0, "confidence": 0.9}],
                "raw_support_end_ms": 400,
                "support_end_ms": 400,
            },
            {
                "source_links": [{"source_index": 2, "confidence": 0.5}],
                "raw_support_end_ms": 300,
                "support_end_ms": 900,
            },
            {
                "source_links": [{"source_index": 1, "confidence": 0.7}],
                "raw_support_end_ms": 800,
                "support_end_ms": 900,
            },
        ],
    }


def test_score_record_reads_the_alignment_and_the_monotonisation_shift() -> None:
    row = score_record(_gold(), _stage_a())
    assert row["target_words"] == 3
    # positions [0, 2, 1] hold one inversion out of three pairs
    assert row["nir"] == pytest.approx(100.0 / 3.0)
    assert row["direction"] == "en2zh"
    # the second word's boundary moved 600 ms when monotonised
    assert row["monotonisation_shift_max_ms"] == 600
    assert row["alignment_confidence_mean"] == pytest.approx(0.7)


def test_score_record_refuses_a_mismatched_join() -> None:
    with pytest.raises(ValueError, match="stage-A join mismatch"):
        score_record(_gold("wanted"), _stage_a("other"))


def test_words_without_links_are_excluded_from_the_alignment_sequence() -> None:
    stage_a = _stage_a()
    stage_a["target_support"].append({"source_links": []})
    row = score_record(_gold(), stage_a)
    assert row["target_words"] == 3
    assert row["aligned_words"] == 4
