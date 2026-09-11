"""The flow statistics must read the stream the way the objective does."""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.text_stream_stats import (
    compare,
    stream_stats,
)


def _steps(spec):
    return [{"produced": [0] * n, "committed": c} for n, c in spec]


def test_a_stream_that_waits_then_talks():
    got = stream_stats(_steps([(1, 0), (0, 0), (4, 3), (6, 6)]))
    assert got["steps"] == 4
    assert got["produced"] == 11
    assert got["per_step"] == 11 / 4
    assert got["first_commit"] == 2
    assert got["empty_share"] == 0.5


def test_a_stream_that_never_commits_has_no_first_commit():
    """It falls back to the stream length, so it sorts as the latest onset."""
    got = stream_stats(_steps([(1, 0), (1, 0)]))
    assert got["first_commit"] == 2
    assert got["empty_share"] == 1.0


def test_the_comparison_counts_which_side_is_higher():
    examples = [
        {
            "chosen": _steps([(4, 4), (4, 4)]),
            "rejected": _steps([(1, 0), (1, 1)]),
        }
    ]
    report = compare(examples)
    assert report["n"] == 1
    assert report["chosen"]["per_step"] == 4.0
    assert report["rejected"]["per_step"] == 1.0
    assert report["chosen_higher_share"]["per_step"] == 1.0
    # The chosen side starts earlier, so it must *not* count as higher here.
    assert report["chosen_higher_share"]["first_commit"] == 0.0
    assert report["chosen_higher_share"]["empty_share"] == 0.0
