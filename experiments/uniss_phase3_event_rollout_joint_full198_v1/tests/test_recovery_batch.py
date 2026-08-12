from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    GeneratedTick,
    build_recovery_example,
    build_rollout_trace,
    oracle_sessions_from_pack,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    replace_trajectory_batch_with_recovery,
)


ROOT = Path(__file__).resolve().parents[3]
CANARY = ROOT / "data/megatron/uniss_phase3_runtime_parity_streaming_v2/canary128/train.packed.jsonl"


@pytest.mark.skipif(not CANARY.is_file(), reason="runtime canary data is unavailable")
def test_dynamic_recovery_repack_keeps_fixed_megatron_shape() -> None:
    pack = json.loads(CANARY.open(encoding="utf-8").readline())
    session = oracle_sessions_from_pack(pack)[0]
    trace = build_rollout_trace(session, [GeneratedTick("WAIT")])
    example = build_recovery_example(session, trace, 0)
    sequence = 1024
    fake = {
        "tokens": torch.zeros(1, sequence, dtype=torch.long),
        "support_bucket": torch.zeros(1, dtype=torch.long),
        "deadline_loss_enabled": torch.ones(1, dtype=torch.bool),
        "chunk_end_ms": torch.full((1,), 160, dtype=torch.long),
        "soft_deadline_ms": torch.full((1,), 640, dtype=torch.long),
        "hard_deadline_ms": torch.full((1,), 800, dtype=torch.long),
        "previous_committed_length": torch.zeros(1, dtype=torch.long),
        "stable_target_length": torch.zeros(1, dtype=torch.long),
        "sample_group": torch.ones(1, dtype=torch.long),
        "playback_buffer_ms": torch.zeros(1, dtype=torch.long),
        "translation_ids": torch.ones(1, 3, dtype=torch.long),
        "translation_mask": torch.ones(1, 3, dtype=torch.bool),
        "safe_commit_targets": torch.zeros(1, 3),
    }
    output = replace_trajectory_batch_with_recovery(
        fake, [example], seq_length=sequence
    )
    assert output["tokens"].shape == (1, sequence)
    assert output["cu_seqlens"].shape == (1, sequence + 1)
    assert int(output["action_position"][0]) == example.action_position
    assert int(output["continuation_position"][0]) == example.continuation_position
    assert int(output["frontend_mask"].sum()) == len(example.frontend_ids)

