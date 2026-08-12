from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    canonical_runtime_pack,
)
from training import constants_uniss as c


ROOT = Path(__file__).resolve().parents[3]
CANARY = ROOT / "data/megatron/uniss_phase3_runtime_parity_streaming_v2/canary128/train.packed.jsonl"


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_clean_pack_is_materialized_in_deployed_microblock_grammar() -> None:
    raw = json.loads(CANARY.open(encoding="utf-8").readline())
    canonical = canonical_runtime_pack(raw)
    assert len(canonical["tokens"]) == len(raw["tokens"]) == 18_000
    assert canonical["source_ids"] == raw["source_ids"]
    sessions = oracle_sessions_from_pack(canonical)
    assert len(sessions) == len(raw["sessions"])
    for session, packed in zip(sessions, canonical["sessions"]):
        annotations = packed["annotations"]
        assert len(annotations) == len(session.events)
        for event, annotation in zip(session.events, annotations):
            expected = (
                c.TOKEN_WRITE_GENERATE
                if event.action == "WRITE"
                else c.TOKEN_WAIT_READ
            )
            assert canonical["labels"][annotation["action_position"]] == expected
            if event.action == "WRITE":
                assert parse_write_outcome(event.outcome_tokens).text_ids


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_canonical_pack_reduces_conflicting_semantic_only_top_level_writes() -> None:
    raw = json.loads(CANARY.open(encoding="utf-8").readline())
    original_writes = sum(
        int(annotation["natural_action"])
        for session in raw["sessions"]
        for annotation in session["annotations"]
    )
    canonical = canonical_runtime_pack(raw)
    runtime_writes = sum(
        int(annotation["natural_action"])
        for session in canonical["sessions"]
        for annotation in session["annotations"]
    )
    assert 0 < runtime_writes < original_writes
