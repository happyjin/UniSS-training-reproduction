from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.event_rollout import (
    GeneratedTick,
    RecoveryPoint,
    build_recovery_example,
    build_rollout_trace,
    oracle_sessions_from_pack,
    parse_write_outcome,
    recovery_points,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.training.dataset import (
    replace_trajectory_batch_with_recovery,
)
from training import constants_uniss as c


ROOT = Path(__file__).resolve().parents[3]
CANARY = (
    ROOT
    / "data/megatron/uniss_phase3_runtime_parity_streaming_v2/canary128/train.packed.jsonl"
)


def _session():
    value = json.loads(CANARY.open(encoding="utf-8").readline())
    return oracle_sessions_from_pack(value)[0]


def _first_write(session):
    return next(
        (index, event)
        for index, event in enumerate(session.events)
        if event.action == "WRITE"
    )


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_semantic_divergence_keeps_generated_prefix() -> None:
    session = _session()
    index, event = _first_write(session)
    oracle = parse_write_outcome(event.outcome_tokens)
    wrong = (oracle.semantic_codes[0] + 1) % c.BICODEC_SEMANTIC_SIZE
    ticks = [GeneratedTick("WAIT") for _ in range(index)]
    ticks.append(
        GeneratedTick(
            "WRITE",
            oracle.text_ids,
            (oracle.semantic_codes[0], wrong),
            natural_semantic_end=False,
        )
    )
    trace = build_rollout_trace(session, ticks)
    point = trace.first_divergence(session)
    assert point is not None
    assert point.kind == "semantic_token"
    assert point.generated_semantic_prefix == 1
    recovery = build_recovery_example(session, trace, point)
    assert recovery.divergence_kind == "semantic_token"
    assert recovery.tokens[recovery.recovery_position] == (
        c.BICODEC_SEMANTIC_OFFSET + oracle.semantic_codes[0]
    )
    assert recovery.labels[recovery.recovery_position] == (
        c.BICODEC_SEMANTIC_OFFSET + oracle.semantic_codes[1]
    )
    assert not recovery.action_supervised
    assert not recovery.continuation_supervised
    assert all(value == 0 for value in recovery.loss_mask[: recovery.recovery_position])


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_recovery_includes_corrupted_semantic_state() -> None:
    session = _session()
    index, event = _first_write(session)
    oracle = parse_write_outcome(event.outcome_tokens)
    wrong = tuple(
        (value + 17) % c.BICODEC_SEMANTIC_SIZE
        for value in oracle.semantic_codes[:3]
    )
    ticks = [GeneratedTick("WAIT") for _ in range(index)]
    ticks.append(
        GeneratedTick(
            "WRITE",
            oracle.text_ids,
            wrong,
            natural_semantic_end=False,
        )
    )
    trace = build_rollout_trace(session, ticks)
    points = recovery_points(session, trace)
    corrupted = points[-1]
    assert corrupted.contains_corruption
    recovery = build_recovery_example(session, trace, corrupted)
    encoded_wrong = tuple(c.encode_bicodec_semantic(wrong))
    assert tuple(recovery.tokens[recovery.recovery_position - len(wrong) + 1 : recovery.recovery_position + 1]) == encoded_wrong
    assert recovery.corrupted_prefix_tokens >= len(wrong)
    assert recovery.labels[recovery.recovery_position] in {
        c.TOKEN_END_SEMANTIC,
        c.BICODEC_SEMANTIC_OFFSET + oracle.semantic_codes[len(wrong)],
    }


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_action_recovery_supervises_only_action_and_suffix() -> None:
    session = _session()
    assert session.events[0].action == "WAIT"
    trace = build_rollout_trace(
        session,
        [GeneratedTick("WRITE", (42,), (1, 2), natural_semantic_end=False)],
    )
    recovery = build_recovery_example(session, trace, RecoveryPoint(0, "action"))
    assert recovery.action_supervised
    assert recovery.recovery_position == recovery.action_position
    assert recovery.labels[recovery.action_position] == c.TOKEN_WAIT_READ


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_recovery_batch_exposes_exact_supervision_masks() -> None:
    session = _session()
    index, event = _first_write(session)
    oracle = parse_write_outcome(event.outcome_tokens)
    ticks = [GeneratedTick("WAIT") for _ in range(index)]
    ticks.append(
        GeneratedTick(
            "WRITE",
            oracle.text_ids,
            oracle.semantic_codes[:-1],
            natural_semantic_end=True,
        )
    )
    trace = build_rollout_trace(session, ticks)
    recovery = build_recovery_example(session, trace)
    fake = {"tokens": torch.zeros(1, 1024, dtype=torch.long)}
    output = replace_trajectory_batch_with_recovery(
        fake, [recovery], seq_length=1024
    )
    assert output["event_rollout_recovery_position"].tolist() == [
        recovery.recovery_position
    ]
    assert output["action_supervised"].tolist() == [False]
    assert output["continuation_supervised"].tolist() == [False]
    assert int(output["frontend_mask"].sum()) == len(recovery.frontend_ids)

