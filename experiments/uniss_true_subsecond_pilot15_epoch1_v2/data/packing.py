"""History-conditioned v2 packing while retaining the native dataset format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    PACKED_TRAJECTORY_SCHEMA,
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_KD,
    ROLE_OBSERVED,
    ROLE_SEMANTIC,
    ROLE_TEXT,
    ShiftedTrajectorySample,
    TrajectoryTokenSample,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import (
    Action,
    TrajectoryRecord,
)
from training import constants_uniss as c


SIDECAR_SCHEMA = "uniss_true_subsecond_pilot15_training_sidecar_v2"


def _append(tokens: list[int], roles: list[int], values: Sequence[int], role: int) -> None:
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
        "deadline_loss_enabled": record.deadline_loss_enabled,
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
        "history_context_version": record.history_context_version,
        "quality_flags": list(record.quality_flags),
    }


def build_token_sample(
    record: TrajectoryRecord,
    target_bicodec: Sequence[int],
    *,
    speed: float = 1.0,
    anticipation_ids: Sequence[int] = (),
) -> TrajectoryTokenSample:
    semantic_history = [
        int(value)
        for value in target_bicodec[
            record.semantic_history_start : record.semantic_history_end
        ]
    ]
    semantic_delta = [
        int(value)
        for value in target_bicodec[
            record.semantic_target_start : record.semantic_target_end
        ]
    ]
    if len(semantic_history) != record.semantic_history_end - record.semantic_history_start:
        raise ValueError("target BiCodec is shorter than semantic history")
    if len(semantic_delta) != record.semantic_target_end - record.semantic_target_start:
        raise ValueError("target BiCodec is shorter than semantic delta")

    tokens: list[int] = []
    roles: list[int] = []
    _append(
        tokens,
        roles,
        [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(record.tgt_lang),
            c.speed_token_id(speed),
            *c.wrap_global_tokens(record.speaker_global),
            c.TOKEN_START_GLM,
            *c.encode_glm_semantic(record.causal_source_glm),
            c.TOKEN_END_GLM,
        ],
        ROLE_OBSERVED,
    )

    # Make the append-only state visible to the policy. This is intentionally
    # observed-only: previously committed content must never receive CE again.
    if record.previous_committed_length or semantic_history:
        _append(
            tokens,
            roles,
            [
                c.language_token_id(record.tgt_lang),
                c.speed_token_id(speed),
                c.TOKEN_START_CONTENT,
                *record.translation_ids[: record.previous_committed_length],
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_SEMANTIC,
                *c.encode_bicodec_semantic(semantic_history),
                c.TOKEN_END_SEMANTIC,
            ],
            ROLE_OBSERVED,
        )

    if record.deadline_action_target is Action.READ:
        _append(tokens, roles, [c.TOKEN_WAIT_READ], ROLE_ACTION)
    else:
        _append(tokens, roles, [c.TOKEN_WRITE_GENERATE], ROLE_ACTION)
        _append(
            tokens,
            roles,
            [
                c.language_token_id(record.tgt_lang),
                c.speed_token_id(speed),
                c.TOKEN_START_CONTENT,
            ],
            ROLE_OBSERVED if record.deadline_forced_target else ROLE_BOUNDARY,
        )
        if record.deadline_forced_target:
            if not anticipation_ids:
                raise ValueError("forced WRITE requires soft teacher anticipation")
            _append(tokens, roles, anticipation_ids, ROLE_KD)
            _append(tokens, roles, [c.TOKEN_END_CONTENT], ROLE_OBSERVED)
        else:
            _append(tokens, roles, record.target_text_delta_ids, ROLE_TEXT)
            _append(
                tokens,
                roles,
                [c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC],
                ROLE_BOUNDARY,
            )
            _append(tokens, roles, c.encode_bicodec_semantic(semantic_delta), ROLE_SEMANTIC)
            _append(tokens, roles, [c.TOKEN_END_SEMANTIC], ROLE_BOUNDARY)
    _append(tokens, roles, [c.TOKEN_EOS], ROLE_BOUNDARY)
    return TrajectoryTokenSample(
        sample_id=f"{record.sample_id}:{record.chunk_end_ms}",
        input_ids=tuple(tokens),
        token_roles=tuple(roles),
        sidecar=training_sidecar(record),
    )


def shift_sample(sample: TrajectoryTokenSample) -> ShiftedTrajectorySample:
    roles = sample.token_roles[1:]
    forced = bool(sample.sidecar["deadline_forced_target"])
    return ShiftedTrajectorySample(
        sample_id=sample.sample_id,
        tokens=sample.input_ids[:-1],
        labels=sample.input_ids[1:],
        loss_mask=tuple(
            0.0
            if role in {ROLE_OBSERVED, ROLE_KD} or (forced and role == ROLE_ACTION)
            else 1.0
            for role in roles
        ),
        token_roles=roles,
        position_ids=tuple(range(len(sample.input_ids) - 1)),
        sidecar=sample.sidecar,
    )


@dataclass(frozen=True)
class SessionSamples:
    sample_id: str
    events: tuple[ShiftedTrajectorySample, ...]

    @property
    def length(self) -> int:
        return sum(event.length for event in self.events)


def _pad(values: list, length: int, fill):
    return [*values, *([fill] * (length - len(values)))]


def pack_sessions(
    sessions: Iterable[SessionSamples], seq_length: int
) -> Iterator[dict[str, object]]:
    """Pack complete sessions atomically so grouped deadline loss sees all ticks."""

    current: list[SessionSamples] = []
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
        for session in current:
            for event in session.events:
                start = len(tokens)
                tokens.extend(event.tokens)
                labels.extend(event.labels)
                loss_mask.extend(event.loss_mask)
                token_roles.extend(event.token_roles)
                position_ids.extend(event.position_ids)
                boundaries.append([start, len(tokens)])
                source_ids.append(event.sample_id)
                sidecars.append(event.sidecar)
        return {
            "schema_version": PACKED_TRAJECTORY_SCHEMA,
            "tokens": _pad(tokens, seq_length, c.TOKEN_PAD),
            "labels": _pad(labels, seq_length, c.TOKEN_PAD),
            "loss_mask": _pad(loss_mask, seq_length, 0.0),
            "token_roles": _pad(token_roles, seq_length, ROLE_OBSERVED),
            "position_ids": _pad(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "tasks": ["trajectory"] * len(sidecars),
            "source_ids": source_ids,
            "trajectory_sidecars": sidecars,
        }

    for session in sessions:
        if session.length > seq_length:
            raise ValueError(
                f"session {session.sample_id} length {session.length} exceeds {seq_length}"
            )
        if current and current_length + session.length > seq_length:
            value = emit()
            if value is not None:
                yield value
            current = []
            current_length = 0
        current.append(session)
        current_length += session.length
    value = emit()
    if value is not None:
        yield value


__all__ = [
    "SIDECAR_SCHEMA",
    "SessionSamples",
    "build_token_sample",
    "pack_sessions",
    "shift_sample",
    "training_sidecar",
]
