"""Bucketing and weights follow Appendix A.3, so both are asserted directly."""
from __future__ import annotations

import pytest

from experiments.uniss_streaming_p2st_traj_v1.data.nir_stratify import (
    LENGTH_ORDER,
    LENGTH_WEIGHTS,
    NIR_ORDER,
    NIR_WEIGHTS,
    bucket_of,
    quantile,
)


def test_weights_are_the_papers_and_sum_to_one() -> None:
    assert NIR_WEIGHTS == {"high": 0.1, "mid_high": 0.3, "mid_low": 0.4, "low": 0.2}
    assert LENGTH_WEIGHTS == {"short": 0.1, "medium": 0.5, "long": 0.4}
    assert sum(NIR_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(LENGTH_WEIGHTS.values()) == pytest.approx(1.0)


def test_bucket_names_run_easy_to_hard_and_short_to_long() -> None:
    # Order matters: bucket_of walks the edges in this order.
    assert NIR_ORDER == ("low", "mid_low", "mid_high", "high")
    assert LENGTH_ORDER == ("short", "medium", "long")


def test_bucket_of_is_inclusive_on_the_upper_edge() -> None:
    edges = [10.0, 20.0, 30.0]
    assert bucket_of(0.0, edges, NIR_ORDER) == "low"
    assert bucket_of(10.0, edges, NIR_ORDER) == "low"
    assert bucket_of(10.1, edges, NIR_ORDER) == "mid_low"
    assert bucket_of(20.0, edges, NIR_ORDER) == "mid_low"
    assert bucket_of(30.0, edges, NIR_ORDER) == "mid_high"
    assert bucket_of(30.1, edges, NIR_ORDER) == "high"
    assert bucket_of(100.0, edges, NIR_ORDER) == "high"


def test_bucket_of_handles_two_edges_for_three_names() -> None:
    assert bucket_of(1000, [4000, 7000], LENGTH_ORDER) == "short"
    assert bucket_of(4000, [4000, 7000], LENGTH_ORDER) == "short"
    assert bucket_of(5000, [4000, 7000], LENGTH_ORDER) == "medium"
    assert bucket_of(9000, [4000, 7000], LENGTH_ORDER) == "long"


def test_quantile_indexes_a_sorted_sequence() -> None:
    values = [float(v) for v in range(101)]
    assert quantile(values, 0.0) == 0.0
    assert quantile(values, 0.5) == 50.0
    assert quantile(values, 1.0) == 100.0


def test_quantile_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_a_uniform_population_would_be_reweighted_not_kept() -> None:
    """The point of the policy: with equal-sized buckets the shares must move.

    Quartiles give each NIR bucket 25% and terciles give each length bucket
    33.3%, so the joint cell is 8.33%; the paper's target for mid_low/medium is
    0.4 * 0.5 = 20%.  That 2.4x shortfall is what caps the pool without
    duplication, which is why the composition is reached by downsampling.
    """
    uniform_cell = 0.25 * (1 / 3)
    target_cell = NIR_WEIGHTS["mid_low"] * LENGTH_WEIGHTS["medium"]
    assert target_cell > uniform_cell
    assert target_cell / uniform_cell == pytest.approx(2.4)
    # and the hardest, shortest cell is cut hard
    assert NIR_WEIGHTS["high"] * LENGTH_WEIGHTS["short"] == pytest.approx(0.01)
