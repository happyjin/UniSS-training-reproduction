"""What the fixed-chunk pool builder must keep from its pure-CE sibling.

The point of building this pool with the same packer, writer and merge is that
any measured difference between the two runs belongs to the read schedule and
nothing else.  These tests pin the parts of that claim that code can check:
the family set is identical, the replay families are byte-identical to the
ones the pure-CE builder produces, and the IDLE accounting the manifest
reports is the accounting the samples actually carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_NONE,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.build_p2st_pools import (
    build_trajectory_samples as build_event_level,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    POOL_FAMILIES,
)
from experiments.uniss_streaming_p2st_traj_v1.training.build_traj_pools import (
    _is_idle,
    build_trajectory_samples,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl"
)


def _encode(text: str) -> list[int]:
    return [ord(char) % 1000 + 1 for char in text]


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


def test_family_set_matches_the_pure_ce_pool(trajectories):
    for trajectory in trajectories:
        built = build_trajectory_samples(trajectory, encode_text=_encode)
        assert set(built) == set(POOL_FAMILIES)


def test_replay_families_are_identical_to_the_pure_ce_pool(trajectories):
    """The controlled part of the comparison, asserted rather than intended."""
    for trajectory in trajectories:
        chunked = build_trajectory_samples(trajectory, encode_text=_encode)
        event = build_event_level(trajectory, encode_text=_encode)
        for family in (FAMILY_PHASE3_QUALITY, FAMILY_PHASE3_PERFORMANCE):
            assert chunked[family] == event[family]


def test_is_idle_agrees_with_the_supervised_span(trajectories):
    for trajectory in trajectories:
        built = build_trajectory_samples(trajectory, encode_text=_encode)
        for samples in built.values():
            for sample in samples:
                target = [
                    (token, kind)
                    for token, kind in zip(sample.token_ids, sample.loss_kinds)
                    if kind != LOSS_NONE
                ]
                assert _is_idle(sample) == (len(target) == 2)
                if _is_idle(sample):
                    assert [kind for _, kind in target] == [
                        LOSS_BOUNDARY,
                        LOSS_EOS,
                    ]


def test_replay_families_are_never_idle(trajectories):
    """IDLE is a property of the read clock, and replay has no read clock."""
    for trajectory in trajectories:
        built = build_trajectory_samples(trajectory, encode_text=_encode)
        for family in (FAMILY_PHASE3_QUALITY, FAMILY_PHASE3_PERFORMANCE):
            assert not any(_is_idle(s) for s in built[family])


def test_chunk_size_reaches_the_builders(trajectories):
    for trajectory in trajectories:
        fine = build_trajectory_samples(
            trajectory, encode_text=_encode, chunk_ms=320
        )
        coarse = build_trajectory_samples(
            trajectory, encode_text=_encode, chunk_ms=1920
        )
        assert len(fine["p2st_streaming_asr"]) > len(coarse["p2st_streaming_asr"])
        # ... while the replay families ignore it entirely.
        assert fine[FAMILY_PHASE3_QUALITY] == coarse[FAMILY_PHASE3_QUALITY]


def test_idle_ratio_zero_leaves_only_content(trajectories):
    for trajectory in trajectories:
        built = build_trajectory_samples(
            trajectory, encode_text=_encode, idle_ratio=0.0
        )
        for samples in built.values():
            assert not any(_is_idle(s) for s in samples)


def test_asr_content_samples_still_supervise_transcript(trajectories):
    for trajectory in trajectories:
        built = build_trajectory_samples(trajectory, encode_text=_encode)
        content = [
            s for s in built["p2st_streaming_asr"] if not _is_idle(s)
        ]
        assert content
        for sample in content:
            assert sample.loss_kinds.count(LOSS_ASR) > 0
