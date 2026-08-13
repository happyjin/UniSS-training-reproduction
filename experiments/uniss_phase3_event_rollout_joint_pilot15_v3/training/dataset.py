"""Lossless v3 canonical packs and exact-state recovery repacking."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import torch

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_OBSERVED,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    IndexedDenseTrajectoryDataset,
    collate_dense_trajectory,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.event_rollout import (
    DIVERGENCE_KINDS,
    RecoveryExample,
    oracle_sessions_from_pack,
)
from training import constants_uniss as c
from training.megatron_uniss_dataset import (
    boundaries_to_padded_cu_seqlens,
    packed_json_to_megatron_item,
)


def canonical_runtime_pack(value: Mapping[str, object]) -> dict[str, object]:
    """Materialize the exact raw WRITE cadence in runtime grammar.

    No WRITE is merged, removed, or relabelled.  The function rebuilds the
    fixed-size tensors so the clean teacher-forced path and exact runtime use
    the same parser and the same event boundaries.
    """

    if value.get("schema_version") != PACK_SCHEMA:
        raise ValueError("unexpected dense pack schema")
    sequence = len(value["tokens"])  # type: ignore[arg-type]
    sessions = oracle_sessions_from_pack(value)
    tokens: list[int] = []
    labels: list[int] = []
    loss_mask: list[int] = []
    token_roles: list[int] = []
    position_ids: list[int] = []
    boundaries: list[list[int]] = []
    packed_sessions: list[dict[str, object]] = []
    source_ids: list[str] = []

    for session in sessions:
        conceptual = list(session.header)
        conceptual_roles = [ROLE_OBSERVED] * len(conceptual)
        local_annotations: list[dict[str, object]] = []
        for event in session.events:
            source_start = len(conceptual)
            conceptual.extend(event.source_block)
            conceptual_roles.extend([ROLE_OBSERVED] * len(event.source_block))
            action_position = len(conceptual) - 1
            frontend_positions = list(
                range(source_start + 1, source_start + 1 + len(event.source_codes))
            )
            conceptual.extend(event.outcome_tokens)
            conceptual_roles.extend(event.outcome_roles)
            local_annotations.append(
                {
                    "event_index": event.event_index,
                    "action_position": action_position,
                    "frontend_positions": frontend_positions,
                    "frontend_ids": list(event.source_codes),
                    "previous_committed_length": event.previous_committed_length,
                    "stable_target_length": event.stable_target_length,
                    "support_bucket": event.support_bucket,
                    "natural_action": 1 if event.action == "WRITE" else 0,
                    "deadline_action": 1 if event.action == "WRITE" else 0,
                    "deadline_forced": event.deadline_forced,
                    "deadline_loss_enabled": event.deadline_loss_enabled,
                    "chunk_end_ms": event.chunk_end_ms,
                    "soft_deadline_ms": event.soft_deadline_ms,
                    "hard_deadline_ms": event.hard_deadline_ms,
                    "sample_id": session.sample_id,
                    "source_finished": event.source_finished,
                    "playback_buffer_ms": event.playback_buffer_ms,
                }
            )
        conceptual.append(c.TOKEN_EOS)
        conceptual_roles.append(ROLE_BOUNDARY)
        local_tokens = conceptual[:-1]
        local_labels = conceptual[1:]
        shifted_roles = conceptual_roles[1:]
        start = len(tokens)
        end = start + len(local_tokens)
        if end > sequence:
            raise ValueError("canonical runtime pack exceeds fixed sequence length")
        tokens.extend(local_tokens)
        labels.extend(local_labels)
        loss_mask.extend(0 if role == ROLE_OBSERVED else 1 for role in shifted_roles)
        token_roles.extend(shifted_roles)
        position_ids.extend(range(len(local_tokens)))
        boundaries.append([start, end])
        source_ids.append(session.sample_id)
        annotations = []
        for annotation in local_annotations:
            annotation = dict(annotation)
            annotation["action_position"] = start + int(
                annotation["action_position"]
            )
            annotation["frontend_positions"] = [
                start + int(position)
                for position in annotation["frontend_positions"]  # type: ignore[index]
            ]
            annotations.append(annotation)
        packed_sessions.append(
            {
                "sample_id": session.sample_id,
                "boundary": [start, end],
                "translation_ids": list(session.full_translation_ids),
                "annotations": annotations,
            }
        )

    def pad(values, fill):
        if len(values) > sequence:
            raise ValueError("canonical runtime pack overflow")
        return [*values, *([fill] * (sequence - len(values)))]

    return {
        "schema_version": PACK_SCHEMA,
        "tokens": pad(tokens, c.TOKEN_PAD),
        "labels": pad(labels, c.TOKEN_PAD),
        "loss_mask": pad(loss_mask, 0),
        "token_roles": pad(token_roles, ROLE_OBSERVED),
        "position_ids": pad(position_ids, 0),
        "sample_boundaries": boundaries,
        "tasks": ["event_rollout_runtime_trajectory_v3"] * len(boundaries),
        "source_ids": source_ids,
        "sessions": packed_sessions,
    }


class IndexedEventRolloutV3TrajectoryDataset(IndexedDenseTrajectoryDataset):
    """Indexed trajectories retaining lossless ordered oracle sessions."""

    def __getitem__(self, index: int) -> dict[str, object]:
        canonical = canonical_runtime_pack(self._read(index))
        result: dict[str, object] = dict(
            packed_json_to_megatron_item(canonical, seq_length=self.seq_length)
        )
        result["sample_kind"] = "trajectory"
        roles = torch.tensor(canonical["token_roles"], dtype=torch.int64)
        result["token_roles"] = roles
        result["source_ids"] = [str(item) for item in canonical["source_ids"]]
        result["sample_boundaries"] = torch.tensor(
            canonical["sample_boundaries"], dtype=torch.int64
        )
        annotations: list[dict[str, object]] = []
        packed_tokens = canonical["tokens"]
        for raw_session in canonical["sessions"]:
            session = dict(raw_session)
            translation_ids = torch.tensor(
                session["translation_ids"], dtype=torch.int64
            )
            if translation_ids.numel() <= 0:
                raise ValueError("canonical runtime session has no target text")
            for raw in session["annotations"]:
                annotation = dict(raw)
                action_position = int(annotation["action_position"])
                if int(roles[action_position]) != ROLE_ACTION:
                    raise ValueError("canonical action annotation is not ROLE_ACTION")
                positions = [
                    int(item) for item in annotation["frontend_positions"]
                ]
                ids = [int(item) for item in annotation["frontend_ids"]]
                packed_ids = [
                    int(packed_tokens[position]) - c.GLM_SEMANTIC_OFFSET
                    for position in positions
                ]
                if packed_ids != ids:
                    raise ValueError("canonical inline source codes disagree")
                annotations.append(
                    {
                        **annotation,
                        "translation_ids": translation_ids,
                        "frontend_positions": torch.tensor(
                            positions, dtype=torch.int64
                        ),
                        "causal_ids": torch.tensor(ids, dtype=torch.int64),
                    }
                )
        result["annotations"] = annotations
        result["oracle_sessions"] = oracle_sessions_from_pack(canonical)
        return result


def collate_event_rollout_v3(
    batch: Sequence[dict[str, object]],
) -> dict[str, object]:
    kinds = {str(value.get("sample_kind")) for value in batch}
    if kinds == {"trajectory"}:
        result = collate_dense_trajectory(batch)
        result["oracle_sessions"] = [value["oracle_sessions"] for value in batch]
        return result
    if kinds == {"replay"}:
        from torch.utils.data._utils.collate import default_collate

        return default_collate(batch)
    raise ValueError(f"one microbatch cannot mix sample kinds: {sorted(kinds)}")


def _stable_group_id(sample_id: str) -> int:
    return (
        int.from_bytes(
            hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), "little"
        )
        & ((1 << 63) - 1)
    )


def replace_trajectory_batch_with_recovery(
    batch: Mapping[str, object],
    examples: Sequence[RecoveryExample],
    *,
    seq_length: int,
) -> dict[str, object]:
    """Repack exact generated histories into fixed Megatron geometry."""

    tokens = batch.get("tokens")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("source trajectory batch must expose [MBS,seq] tokens")
    micro_batch = int(tokens.shape[0])
    if len(examples) != micro_batch:
        raise ValueError("one recovery example is required for every local MBS lane")
    if any(len(example.tokens) > seq_length for example in examples):
        raise ValueError("a recovery transcript exceeds the training sequence length")

    def padded(attribute, fill, dtype):
        output = torch.full((micro_batch, seq_length), fill, dtype=dtype)
        for row, example in enumerate(examples):
            source = getattr(example, attribute)
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
        "event_rollout_recovery_position": torch.tensor(
            [example.recovery_position for example in examples], dtype=torch.long
        ),
        "event_rollout_generated_prefix_length": torch.tensor(
            [example.generated_prefix_length for example in examples],
            dtype=torch.long,
        ),
        "event_rollout_corrupted_prefix_tokens": torch.tensor(
            [example.corrupted_prefix_tokens for example in examples],
            dtype=torch.long,
        ),
        "event_rollout_divergence_kind": torch.tensor(
            [DIVERGENCE_KINDS.index(example.divergence_kind) for example in examples],
            dtype=torch.long,
        ),
        "action_batch": torch.arange(micro_batch, dtype=torch.long),
        "action_position": torch.tensor(
            [example.action_position for example in examples], dtype=torch.long
        ),
        "action_supervised": torch.tensor(
            [example.action_supervised for example in examples], dtype=torch.bool
        ),
        "natural_action": torch.tensor(
            [example.action_target for example in examples], dtype=torch.long
        ),
        "deadline_action": torch.tensor(
            [example.action_target for example in examples], dtype=torch.long
        ),
        "continuation_batch": torch.arange(micro_batch, dtype=torch.long),
        "continuation_position": torch.tensor(
            [example.continuation_position for example in examples], dtype=torch.long
        ),
        "continuation_target": torch.tensor(
            [example.continuation_target for example in examples], dtype=torch.long
        ),
        "continuation_supervised": torch.tensor(
            [example.continuation_supervised for example in examples], dtype=torch.bool
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

    count = micro_batch
    max_translation = max(len(example.translation_ids) for example in examples)
    max_frontend = max(1, max(len(example.frontend_ids) for example in examples))
    output.update(
        {
            "support_bucket": torch.tensor(
                [example.support_bucket for example in examples], dtype=torch.long
            ),
            "deadline_forced": torch.tensor(
                [example.deadline_forced for example in examples], dtype=torch.bool
            ),
            "deadline_loss_enabled": torch.tensor(
                [example.deadline_loss_enabled for example in examples],
                dtype=torch.bool,
            ),
            "chunk_end_ms": torch.tensor(
                [example.chunk_end_ms for example in examples], dtype=torch.long
            ),
            "soft_deadline_ms": torch.tensor(
                [example.soft_deadline_ms for example in examples], dtype=torch.long
            ),
            "hard_deadline_ms": torch.tensor(
                [example.hard_deadline_ms for example in examples], dtype=torch.long
            ),
            "previous_committed_length": torch.tensor(
                [example.previous_committed_length for example in examples],
                dtype=torch.long,
            ),
            "stable_target_length": torch.tensor(
                [example.stable_target_length for example in examples], dtype=torch.long
            ),
            "sample_group": torch.tensor(
                [_stable_group_id(example.sample_id) for example in examples],
                dtype=torch.long,
            ),
            "playback_buffer_ms": torch.tensor(
                [example.playback_buffer_ms for example in examples], dtype=torch.long
            ),
            "translation_ids": torch.full(
                (count, max_translation), c.TOKEN_PAD, dtype=torch.long
            ),
            "translation_mask": torch.zeros(
                count, max_translation, dtype=torch.bool
            ),
            "safe_commit_targets": torch.zeros(
                count, max_translation, dtype=torch.float32
            ),
            "frontend_ids": torch.zeros(count, max_frontend, dtype=torch.long),
            "frontend_positions": torch.zeros(
                count, max_frontend, dtype=torch.long
            ),
            "frontend_mask": torch.zeros(count, max_frontend, dtype=torch.bool),
            "teacher_indices": torch.empty(count, 4, 0, 1, dtype=torch.long),
            "teacher_probabilities": torch.empty(
                count, 4, 0, 1, dtype=torch.float32
            ),
            "teacher_confidence": torch.empty(count, 4, 0, dtype=torch.float32),
            "teacher_mask": torch.empty(count, 4, 0, dtype=torch.bool),
            "kd_batch": torch.empty(0, dtype=torch.long),
            "kd_position": torch.empty(0, dtype=torch.long),
            "kd_annotation": torch.empty(0, dtype=torch.long),
            "kd_target_index": torch.empty(0, dtype=torch.long),
        }
    )
    for row, example in enumerate(examples):
        length = len(example.translation_ids)
        output["translation_ids"][row, :length] = torch.tensor(  # type: ignore[index]
            example.translation_ids, dtype=torch.long
        )
        output["translation_mask"][row, :length] = True  # type: ignore[index]
        output["safe_commit_targets"][  # type: ignore[index]
            row, : min(length, example.stable_target_length)
        ] = 1.0
        for slot, (position, code) in enumerate(
            zip(example.frontend_positions, example.frontend_ids)
        ):
            output["frontend_ids"][row, slot] = code  # type: ignore[index]
            output["frontend_positions"][row, slot] = position  # type: ignore[index]
            output["frontend_mask"][row, slot] = True  # type: ignore[index]
    return output


__all__ = [
    "IndexedEventRolloutV3TrajectoryDataset",
    "canonical_runtime_pack",
    "collate_event_rollout_v3",
    "replace_trajectory_batch_with_recovery",
]
