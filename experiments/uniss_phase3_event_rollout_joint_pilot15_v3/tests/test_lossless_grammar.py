from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.event_rollout import (
    GeneratedTick,
    build_rollout_trace,
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.training.dataset import (
    canonical_runtime_pack,
)
from training import constants_uniss as c


ROOT = Path(__file__).resolve().parents[3]
CANARY = (
    ROOT
    / "data/megatron/uniss_phase3_runtime_parity_streaming_v2/canary128/train.packed.jsonl"
)


def _raw():
    return json.loads(CANARY.open(encoding="utf-8").readline())


def _raw_write_count(value) -> int:
    return sum(
        int(annotation["natural_action"])
        for session in value["sessions"]
        for annotation in session["annotations"]
    )


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_semantic_only_writes_remain_writes() -> None:
    sessions = oracle_sessions_from_pack(_raw())
    semantic_only = [
        event
        for session in sessions
        for event in session.events
        if event.action == "WRITE" and not parse_write_outcome(event.outcome_tokens).text_ids
    ]
    assert semantic_only
    assert all(parse_write_outcome(event.outcome_tokens).semantic_codes for event in semantic_only)


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_canonical_write_count_equals_raw_write_count() -> None:
    raw = _raw()
    canonical = canonical_runtime_pack(raw)
    assert _raw_write_count(canonical) == _raw_write_count(raw)


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_canonical_semantic_spans_equal_corresponding_raw_spans() -> None:
    raw_sessions = oracle_sessions_from_pack(_raw())
    canonical_sessions = oracle_sessions_from_pack(canonical_runtime_pack(_raw()))
    assert len(raw_sessions) == len(canonical_sessions)
    for raw, canonical in zip(raw_sessions, canonical_sessions):
        assert len(raw.events) == len(canonical.events)
        for expected, actual in zip(raw.events, canonical.events):
            assert actual.action == expected.action
            assert actual.outcome_tokens == expected.outcome_tokens
            if actual.action == "WRITE":
                assert len(parse_write_outcome(actual.outcome_tokens).semantic_codes) <= 24


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data unavailable")
def test_oracle_roundtrip_preserves_all_events() -> None:
    session = oracle_sessions_from_pack(_raw())[0]
    ticks = []
    for event in session.events:
        if event.action == "WAIT":
            ticks.append(
                GeneratedTick(
                    "WAIT", choose_eos=event.continuation_token == c.TOKEN_EOS
                )
            )
        else:
            parsed = parse_write_outcome(event.outcome_tokens)
            ticks.append(
                GeneratedTick(
                    "WRITE",
                    parsed.text_ids,
                    parsed.semantic_codes,
                    choose_eos=event.continuation_token == c.TOKEN_EOS,
                )
            )
    trace = build_rollout_trace(session, ticks)
    assert trace.first_divergence(session) is None
    assert len(trace.generated_ticks) == len(session.events)
    assert trace.transcript[-1] == c.TOKEN_EOS

