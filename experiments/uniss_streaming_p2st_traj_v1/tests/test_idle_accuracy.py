"""The scoring half of the IDLE evaluator, which is where it can be wrong.

Running the cascade needs a GPU and a checkpoint, so what is tested here is
everything else: that the labels come from the same binning the pool used,
that IDLE is the positive class, and that the degenerate model the design is
afraid of -- one that terminates immediately on every tick -- is scored as the
failure it is rather than as perfect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_streaming_p2st_traj_v1.data.uniform_chunk_tasks import (
    chunk_windows,
)
from experiments.uniss_streaming_p2st_traj_v1.evaluation.idle_accuracy import (
    _counts,
    _rates,
    gold_idle_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl"
)


@pytest.fixture(scope="module")
def trajectories() -> list[E2ETrajectory]:
    if not GOLD.exists():
        pytest.skip(f"gold trajectories not present at {GOLD}")
    records = []
    with GOLD.open() as handle:
        for index, line in enumerate(handle):
            records.append(E2ETrajectory.from_mapping(json.loads(line)))
            if index >= 15:
                break
    return records


def test_labels_have_one_entry_per_tick(trajectories):
    for trajectory in trajectories:
        source, target = gold_idle_labels(trajectory, chunk_ms=640)
        assert len(source) == len(target) == len(
            chunk_windows(trajectory, chunk_ms=640)
        )


def test_labels_match_the_pool_binning(trajectories):
    """The label and the training target must come from the same function."""
    for trajectory in trajectories:
        source, target = gold_idle_labels(trajectory, chunk_ms=640)
        for index, window in enumerate(chunk_windows(trajectory, chunk_ms=640)):
            assert source[index] == (
                not any(e.gold_source_delta.strip() for e in window.events)
            )
            assert target[index] == (
                not any(e.target_text_delta.strip() for e in window.events)
            )


def test_target_is_never_less_idle_than_source(trajectories):
    for trajectory in trajectories:
        source, target = gold_idle_labels(trajectory, chunk_ms=640)
        assert sum(target) >= sum(source)


def test_perfect_prediction_scores_one():
    labels = [True, False, True, False, False]
    rates = _rates(_counts(labels, list(labels)))
    assert rates["idle_recall"] == 1.0
    assert rates["idle_precision"] == 1.0
    assert rates["accuracy"] == 1.0


def test_always_quiet_is_caught_by_precision():
    """The failure mode the terminator-as-IDLE design risks.

    A model that terminates on every tick has perfect IDLE recall.  Precision
    is what says it is useless, which is why the evaluator reports both.
    """
    labels = [True, False, False, False]
    rates = _rates(_counts(labels, [True] * 4))
    assert rates["idle_recall"] == 1.0
    assert rates["idle_precision"] == 0.25
    assert rates["content_recall"] == 0.0
    assert rates["predicted_idle_rate"] == 1.0


def test_never_quiet_scores_zero_recall():
    labels = [True, True, False]
    rates = _rates(_counts(labels, [False] * 3))
    assert rates["idle_recall"] == 0.0
    assert rates["content_recall"] == 1.0
    assert rates["predicted_idle_rate"] == 0.0


def test_rates_are_none_when_a_class_is_absent():
    rates = _rates(_counts([False, False], [False, False]))
    assert rates["idle_recall"] is None
    assert rates["idle_precision"] is None
    assert rates["accuracy"] == 1.0


def test_counts_partition_every_tick():
    labels = [True, False, True, True, False]
    predicted = [True, True, False, True, False]
    counts = _counts(labels, predicted)
    assert sum(counts.values()) == len(labels)
    assert _rates(counts)["ticks"] == len(labels)
