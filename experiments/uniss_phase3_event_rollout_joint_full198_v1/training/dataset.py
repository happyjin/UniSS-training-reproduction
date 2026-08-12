"""Dynamic recovery repacking while preserving Megatron packed geometry."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    RecoveryExample,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_OBSERVED,
)
from training import constants_uniss as c
from training.megatron_uniss_dataset import boundaries_to_padded_cu_seqlens


def replace_trajectory_batch_with_recovery(
    batch: Mapping[str, object],
    examples: Sequence[RecoveryExample],
    *,
    seq_length: int,
) -> dict[str, object]:
    """Return a fixed-shape CPU batch containing variable recovery sessions."""

    tokens = batch.get("tokens")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("source trajectory batch must expose [MBS,seq] tokens")
    micro_batch = int(tokens.shape[0])
    if len(examples) != micro_batch:
        raise ValueError("one recovery example is required for every local MBS lane")
    if any(len(example.tokens) > seq_length for example in examples):
        raise ValueError("a recovery transcript exceeds the training sequence length")

    def padded(values, fill, dtype):
        output = torch.full((micro_batch, seq_length), fill, dtype=dtype)
        for row, example in enumerate(examples):
            source = getattr(example, values)
            output[row, : len(source)] = torch.tensor(source, dtype=dtype)
        return output

    output: dict[str, object] = {
        "sample_kind": "trajectory",
        "tokens": padded("tokens", c.TOKEN_PAD, torch.long),
        "labels": padded("labels", c.TOKEN_PAD, torch.long),
        "loss_mask": padded("loss_mask", 0, torch.float32),
        "position_ids": padded("position_ids", 0, torch.long),
        "token_roles": padded("token_roles", ROLE_OBSERVED, torch.long),
        "source_ids": [[example.sample_id] for example in examples],
        "sample_boundaries": torch.tensor(
            [[[0, len(example.tokens)]] for example in examples], dtype=torch.long
        ),
        "event_rollout_recovery": True,
        "event_rollout_event_index": torch.tensor(
            [example.event_index for example in examples], dtype=torch.long
        ),
        "continuation_batch": torch.arange(micro_batch, dtype=torch.long),
        "continuation_position": torch.tensor(
            [example.continuation_position for example in examples], dtype=torch.long
        ),
        "continuation_target": torch.tensor(
            [example.continuation_target for example in examples], dtype=torch.long
        ),
    }
    cu_rows = []
    max_rows = []
    for example in examples:
        cu, maximum = boundaries_to_padded_cu_seqlens(
            [[0, len(example.tokens)]], seq_length
        )
        cu_rows.append(cu)
        max_rows.append(maximum)
    output["cu_seqlens"] = torch.stack(cu_rows)
    output["max_seqlen"] = torch.stack(max_rows)

    frontend_values = [
        (row, position, code)
        for row, example in enumerate(examples)
        for position, code in zip(example.frontend_positions, example.frontend_ids)
    ]
    count = micro_batch
    max_translation = max(
        int(batch["translation_mask"][row].sum())  # type: ignore[index]
        for row in range(micro_batch)
    )
    output.update(
        {
            "action_batch": torch.arange(count, dtype=torch.long),
            "action_position": torch.tensor(
                [example.action_position for example in examples], dtype=torch.long
            ),
            "support_bucket": batch["support_bucket"][:count],
            "natural_action": torch.tensor(
                [example.action_target for example in examples], dtype=torch.long
            ),
            "deadline_action": torch.tensor(
                [example.action_target for example in examples], dtype=torch.long
            ),
            "deadline_forced": torch.zeros(count, dtype=torch.bool),
            "deadline_loss_enabled": batch["deadline_loss_enabled"][:count],
            "chunk_end_ms": batch["chunk_end_ms"][:count],
            "soft_deadline_ms": batch["soft_deadline_ms"][:count],
            "hard_deadline_ms": batch["hard_deadline_ms"][:count],
            "previous_committed_length": batch["previous_committed_length"][:count],
            "stable_target_length": batch["stable_target_length"][:count],
            "sample_group": batch["sample_group"][:count],
            "playback_buffer_ms": batch["playback_buffer_ms"][:count],
            "translation_ids": batch["translation_ids"][:count, :max_translation],
            "translation_mask": batch["translation_mask"][:count, :max_translation],
            "safe_commit_targets": batch["safe_commit_targets"][:count, :max_translation],
            "frontend_ids": torch.zeros(count, seq_length, dtype=torch.long),
            "frontend_positions": torch.zeros(count, seq_length, dtype=torch.long),
            "frontend_mask": torch.zeros(count, seq_length, dtype=torch.bool),
            "teacher_indices": torch.empty(count, 4, 0, 1, dtype=torch.long),
            "teacher_probabilities": torch.empty(count, 4, 0, 1, dtype=torch.float32),
            "teacher_confidence": torch.empty(count, 4, 0, dtype=torch.float32),
            "teacher_mask": torch.empty(count, 4, 0, dtype=torch.bool),
            "kd_batch": torch.empty(0, dtype=torch.long),
            "kd_position": torch.empty(0, dtype=torch.long),
            "kd_annotation": torch.empty(0, dtype=torch.long),
            "kd_target_index": torch.empty(0, dtype=torch.long),
        }
    )
    for row, position, code in frontend_values:
        slot = int(output["frontend_mask"][row].sum())  # type: ignore[index]
        output["frontend_ids"][row, slot] = code  # type: ignore[index]
        output["frontend_positions"][row, slot] = position  # type: ignore[index]
        output["frontend_mask"][row, slot] = True  # type: ignore[index]
    return output


__all__ = ["replace_trajectory_batch_with_recovery"]

