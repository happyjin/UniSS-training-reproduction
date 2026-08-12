from __future__ import annotations

import pytest
import torch

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.speaker_similarity import (
    aggregate,
    cosine_scores,
)


def test_cosine_similarity_has_expected_range_and_identity() -> None:
    scores = cosine_scores(
        torch.tensor([1.0, 0.0]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    assert scores == pytest.approx([1.0, 0.0, -1.0])


def test_speaker_aggregate_is_grouped_by_direction() -> None:
    report = aggregate(
        [
            {"mode": "exact_runtime", "src_lang": "eng", "tgt_lang": "cmn", "speaker_similarity": 0.8},
            {"mode": "exact_runtime", "src_lang": "eng", "tgt_lang": "cmn", "speaker_similarity": 0.6},
        ]
    )
    group = report["groups"]["exact_runtime:eng->cmn"]
    assert group["mean"] == pytest.approx(0.7)
    assert report["scored_count"] == 2
