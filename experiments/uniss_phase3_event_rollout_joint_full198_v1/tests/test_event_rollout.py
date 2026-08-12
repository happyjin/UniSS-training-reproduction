from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    GeneratedTick,
    build_recovery_example,
    build_rollout_trace,
    choose_recovery_event,
    generated_tick_matches_oracle,
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_SEMANTIC,
    ROLE_TEXT,
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


def _oracle_ticks(session):
    values = []
    for event in session.events:
        if event.action == "WAIT":
            values.append(GeneratedTick("WAIT", choose_eos=event.continuation_token == c.TOKEN_EOS))
        else:
            parsed = parse_write_outcome(event.outcome_tokens)
            values.append(
                GeneratedTick(
                    "WRITE",
                    parsed.text_ids,
                    parsed.semantic_codes,
                    choose_eos=event.continuation_token == c.TOKEN_EOS,
                )
            )
    return values


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_oracle_roundtrip_has_no_divergence() -> None:
    session = _session()
    trace = build_rollout_trace(session, _oracle_ticks(session))
    assert trace.first_divergence(session) is None
    assert trace.transcript[-1] == c.TOKEN_EOS
    assert not trace.stopped_early


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_wait_to_write_changes_grammar_and_recovery_uses_generated_history() -> None:
    session = _session()
    ticks = _oracle_ticks(session)
    assert session.events[0].action == "WAIT"
    ticks[0] = GeneratedTick("WRITE", (42,), (1, 2, 3), choose_eos=False)
    trace = build_rollout_trace(session, ticks)
    assert trace.first_divergence(session) == 0
    recovery = build_recovery_example(session, trace, choose_recovery_event(session, trace))
    assert recovery.labels[recovery.action_position] == c.TOKEN_WAIT_READ
    assert recovery.token_roles[recovery.action_position] == ROLE_ACTION
    assert sum(recovery.loss_mask[: recovery.action_position]) == 0
    assert all(recovery.loss_mask[recovery.action_position :])


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_recovery_contains_text_semantic_and_natural_continuation_roles() -> None:
    session = _session()
    write_index = next(i for i, event in enumerate(session.events) if event.action == "WRITE")
    trace = build_rollout_trace(session, _oracle_ticks(session))
    recovery = build_recovery_example(session, trace, write_index)
    assert ROLE_TEXT in recovery.token_roles
    assert ROLE_SEMANTIC in recovery.token_roles
    assert recovery.labels[recovery.continuation_position] in {
        c.TOKEN_START_GLM,
        c.TOKEN_EOS,
    }
    assert len(recovery.frontend_positions) == len(recovery.frontend_ids)


def test_generated_write_requires_semantic_content() -> None:
    with pytest.raises(ValueError, match="semantic"):
        GeneratedTick("WRITE", (1,), ())


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_exact_event_match_detects_first_recovery_state() -> None:
    session = _session()
    expected = session.events[0]
    assert generated_tick_matches_oracle(expected, _oracle_ticks(session)[0])
    opposite = "WRITE" if expected.action == "WAIT" else "WAIT"
    generated = (
        GeneratedTick(opposite, (42,), (1, 2, 3))
        if opposite == "WRITE"
        else GeneratedTick("WAIT")
    )
    assert not generated_tick_matches_oracle(expected, generated)
