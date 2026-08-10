"""Build and pack trajectory-token samples with explicit per-label loss roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    Action,
    TrajectoryRecord,
)
from training import constants_uniss as c


PACKED_TRAJECTORY_SCHEMA = "uniss_true_subsecond_packed_trajectory_v1"
SIDECAR_SCHEMA = "uniss_true_subsecond_training_sidecar_v1"

ROLE_OBSERVED = 0
ROLE_ACTION = 1
ROLE_TEXT = 2
ROLE_SEMANTIC = 3
ROLE_BOUNDARY = 4
VALID_ROLES = {ROLE_OBSERVED, ROLE_ACTION, ROLE_TEXT, ROLE_SEMANTIC, ROLE_BOUNDARY}


@dataclass(frozen=True)
class TrajectoryTokenSample:
    sample_id: str
    input_ids: tuple[int, ...]
    token_roles: tuple[int, ...]
    sidecar: dict[str, object]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.token_roles):
            raise ValueError("input_ids and token_roles must have identical lengths")
        if len(self.input_ids) < 2:
            raise ValueError("trajectory token sample is too short")
        if any(role not in VALID_ROLES for role in self.token_roles):
            raise ValueError("trajectory token sample contains an unknown loss role")
        for token in self.input_ids:
            c.validate_token_id(int(token))


@dataclass(frozen=True)
class ShiftedTrajectorySample:
    sample_id: str
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[float, ...]
    token_roles: tuple[int, ...]
    position_ids: tuple[int, ...]
    sidecar: dict[str, object]

    @property
    def length(self) -> int:
        return len(self.tokens)


def _append(
    tokens: list[int], roles: list[int], values: Sequence[int], role: int
) -> None:
    tokens.extend(int(value) for value in values)
    roles.extend([role] * len(values))


def training_sidecar(record: TrajectoryRecord) -> dict[str, object]:
    return {
        "schema_version": SIDECAR_SCHEMA,
        "trajectory_schema": record.schema_version,
        "trajectory_checksum": record.checksum,
        "sample_id": record.sample_id,
        "shard": record.shard,
        "row_index": record.row_index,
        "src_lang": record.src_lang,
        "tgt_lang": record.tgt_lang,
        "source_duration_ms": record.source_duration_ms,
        "chunk_end_ms": record.chunk_end_ms,
        "future_1_end_ms": record.future_1_end_ms,
        "future_2_end_ms": record.future_2_end_ms,
        "soft_deadline_ms": record.soft_deadline_ms,
        "hard_deadline_ms": record.hard_deadline_ms,
        "frontend_token_cache": record.frontend_token_cache,
        "teacher_prefix_topk_path": record.teacher_prefix_topk_path,
        "teacher_future_1_topk_path": record.teacher_future_1_topk_path,
        "teacher_future_2_topk_path": record.teacher_future_2_topk_path,
        "teacher_full_topk_path": record.teacher_full_topk_path,
        "translation_ids": list(record.translation_ids),
        "previous_committed_length": record.previous_committed_length,
        "stable_target_length": record.stable_target_length,
        "new_supported_count": record.new_supported_count,
        "support_bucket": record.support_bucket,
        "safe_commit_mask": list(record.safe_commit_mask),
        "natural_action_target": record.natural_action_target.value,
        "deadline_action_target": record.deadline_action_target.value,
        "deadline_forced_target": record.deadline_forced_target,
        "semantic_history_start": record.semantic_history_start,
        "semantic_history_end": record.semantic_history_end,
        "semantic_target_start": record.semantic_target_start,
        "semantic_target_end": record.semantic_target_end,
        "quality_flags": list(record.quality_flags),
    }


def build_trajectory_token_sample(
    record: TrajectoryRecord,
    target_bicodec: Sequence[int],
    *,
    speed: float = 1.0,
) -> TrajectoryTokenSample:
    semantic = [
        int(value)
        for value in target_bicodec[
            record.semantic_target_start : record.semantic_target_end
        ]
    ]
    expected_semantic = record.semantic_target_end - record.semantic_target_start
    if len(semantic) != expected_semantic:
        raise ValueError("target_bicodec is shorter than the trajectory semantic span")

    tokens: list[int] = []
    roles: list[int] = []
    header = [
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
        c.language_token_id(record.tgt_lang),
        c.speed_token_id(speed),
        *c.wrap_global_tokens(record.speaker_global),
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic(record.causal_source_glm),
        c.TOKEN_END_GLM,
    ]
    _append(tokens, roles, header, ROLE_OBSERVED)

    if record.deadline_action_target is Action.READ:
        _append(tokens, roles, [c.TOKEN_WAIT_READ], ROLE_ACTION)
    else:
        _append(tokens, roles, [c.TOKEN_WRITE_GENERATE], ROLE_ACTION)
        if not record.deadline_forced_target:
            _append(
                tokens,
                roles,
                [
                    c.language_token_id(record.tgt_lang),
                    c.speed_token_id(speed),
                    c.TOKEN_START_CONTENT,
                ],
                ROLE_BOUNDARY,
            )
            _append(tokens, roles, record.target_text_delta_ids, ROLE_TEXT)
            _append(
                tokens,
                roles,
                [c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC],
                ROLE_BOUNDARY,
            )
            _append(tokens, roles, c.encode_bicodec_semantic(semantic), ROLE_SEMANTIC)
            _append(tokens, roles, [c.TOKEN_END_SEMANTIC], ROLE_BOUNDARY)
    _append(tokens, roles, [c.TOKEN_EOS], ROLE_BOUNDARY)
    return TrajectoryTokenSample(
        sample_id=f"{record.sample_id}:{record.chunk_end_ms}",
        input_ids=tuple(tokens),
        token_roles=tuple(roles),
        sidecar=training_sidecar(record),
    )


def shift_trajectory_sample(sample: TrajectoryTokenSample) -> ShiftedTrajectorySample:
    roles = sample.token_roles[1:]
    return ShiftedTrajectorySample(
        sample_id=sample.sample_id,
        tokens=sample.input_ids[:-1],
        labels=sample.input_ids[1:],
        loss_mask=tuple(0.0 if role == ROLE_OBSERVED else 1.0 for role in roles),
        token_roles=roles,
        position_ids=tuple(range(len(sample.input_ids) - 1)),
        sidecar=sample.sidecar,
    )


def _pad(values: list, length: int, fill):
    return [*values, *([fill] * (length - len(values)))]


def pack_trajectory_samples(
    samples: Iterable[ShiftedTrajectorySample],
    seq_length: int,
) -> Iterator[dict[str, object]]:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    current: list[ShiftedTrajectorySample] = []
    current_length = 0

    def emit() -> dict[str, object] | None:
        if not current:
            return None
        tokens: list[int] = []
        labels: list[int] = []
        loss_mask: list[float] = []
        token_roles: list[int] = []
        position_ids: list[int] = []
        boundaries: list[list[int]] = []
        source_ids: list[str] = []
        sidecars: list[dict[str, object]] = []
        for sample in current:
            start = len(tokens)
            tokens.extend(sample.tokens)
            labels.extend(sample.labels)
            loss_mask.extend(sample.loss_mask)
            token_roles.extend(sample.token_roles)
            position_ids.extend(sample.position_ids)
            boundaries.append([start, len(tokens)])
            source_ids.append(sample.sample_id)
            sidecars.append(sample.sidecar)
        return {
            "schema_version": PACKED_TRAJECTORY_SCHEMA,
            "tokens": _pad(tokens, seq_length, c.TOKEN_PAD),
            "labels": _pad(labels, seq_length, c.TOKEN_PAD),
            "loss_mask": _pad(loss_mask, seq_length, 0.0),
            "token_roles": _pad(token_roles, seq_length, ROLE_OBSERVED),
            "position_ids": _pad(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "tasks": ["trajectory"] * len(current),
            "source_ids": source_ids,
            "trajectory_sidecars": sidecars,
        }

    for sample in samples:
        if sample.length > seq_length:
            raise ValueError(
                f"trajectory sample {sample.sample_id} length {sample.length} exceeds {seq_length}"
            )
        if current and current_length + sample.length > seq_length:
            packed = emit()
            if packed is not None:
                yield packed
            current = []
            current_length = 0
        current.append(sample)
        current_length += sample.length
    packed = emit()
    if packed is not None:
        yield packed
