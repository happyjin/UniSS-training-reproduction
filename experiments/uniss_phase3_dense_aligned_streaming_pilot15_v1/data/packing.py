"""Pack complete ordered dense sessions into 18k Megatron THD records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseSession,
)
from training import constants_uniss as c


PACK_SCHEMA = "uniss_dense_aligned_streaming_pack_v3"
ROLE_OBSERVED = 0
ROLE_ACTION = 1
ROLE_TEXT = 2
ROLE_SEMANTIC = 3
ROLE_BOUNDARY = 4
VALID_ROLES = {
    ROLE_OBSERVED,
    ROLE_ACTION,
    ROLE_TEXT,
    ROLE_SEMANTIC,
    ROLE_BOUNDARY,
}


def _append(
    tokens: list[int], roles: list[int], values: Sequence[int], role: int
) -> tuple[int, int]:
    start = len(tokens)
    tokens.extend(int(value) for value in values)
    roles.extend([role] * len(values))
    return start, len(tokens)


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_value, right_value in zip(left, right):
        if int(left_value) != int(right_value):
            break
        length += 1
    return length


@dataclass(frozen=True)
class DenseSessionTokenSample:
    sample_id: str
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    token_roles: tuple[int, ...]
    position_ids: tuple[int, ...]
    annotations: tuple[dict[str, object], ...]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def __post_init__(self) -> None:
        lengths = {
            len(self.tokens),
            len(self.labels),
            len(self.loss_mask),
            len(self.token_roles),
            len(self.position_ids),
        }
        if len(lengths) != 1 or self.length <= 0:
            raise ValueError("shifted dense session tensors have inconsistent lengths")
        if any(role not in VALID_ROLES for role in self.token_roles):
            raise ValueError("dense session contains an unknown token role")
        if not self.annotations:
            raise ValueError("dense session contains no action annotations")


def build_session_token_sample(
    session: DenseSession,
    formal: Mapping[str, object],
    encode_text: Callable[[str], Sequence[int]],
    *,
    speed: float = 1.0,
    soft_deadline_ms: int = 640,
    hard_deadline_ms: int = 800,
) -> DenseSessionTokenSample:
    """Build one causal interleaved sequence, appending source codes only once."""

    if str(formal.get("id")) != session.sample_id:
        raise ValueError("dense/formal sample IDs differ")
    source_glm = [int(value) for value in formal["source_glm"]]  # type: ignore[index]
    target_semantic = [int(value) for value in formal["target_bicodec"]]  # type: ignore[index]
    if len(source_glm) != session.source_glm_length:
        raise ValueError("dense/formal source GLM lengths differ")
    if len(target_semantic) != session.target_semantic_length:
        raise ValueError("dense/formal target semantic lengths differ")
    full_translation_ids = [int(value) for value in encode_text(session.target_text)]
    if not full_translation_ids:
        raise ValueError("target text encoded to an empty sequence")

    tokens: list[int] = []
    roles: list[int] = []
    _append(
        tokens,
        roles,
        [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(session.tgt_lang),
            c.speed_token_id(speed),
            *c.wrap_global_tokens(session.speaker_global),
        ],
        ROLE_OBSERVED,
    )
    source_cursor = 0
    committed_text = ""
    committed_token_end = 0
    annotations: list[dict[str, object]] = []
    first_write = next(event for event in session.events if event.action == "WRITE")
    deadline_enabled = int(first_write.earliest_safe_ms) <= hard_deadline_ms

    for event in session.events:
        visible = int(event.visible_source_token_end)
        if not source_cursor <= visible <= len(source_glm):
            raise ValueError("dense visible source prefix moved backwards or out of range")
        delta_codes = source_glm[source_cursor:visible]
        _append(tokens, roles, [c.TOKEN_START_GLM], ROLE_OBSERVED)
        glm_start, glm_end = _append(
            tokens, roles, c.encode_glm_semantic(delta_codes), ROLE_OBSERVED
        )
        _append(tokens, roles, [c.TOKEN_END_GLM], ROLE_OBSERVED)
        source_cursor = visible

        action_original = len(tokens)
        action_token = (
            c.TOKEN_WRITE_GENERATE if event.action == "WRITE" else c.TOKEN_WAIT_READ
        )
        _append(tokens, roles, [action_token], ROLE_ACTION)
        previous_token_end = committed_token_end
        if event.action == "WRITE":
            _append(
                tokens,
                roles,
                [
                    c.language_token_id(session.tgt_lang),
                    c.speed_token_id(speed),
                    c.TOKEN_START_CONTENT,
                ],
                ROLE_BOUNDARY,
            )
            delta_text_ids = [int(value) for value in encode_text(event.text_delta)]
            _append(tokens, roles, delta_text_ids, ROLE_TEXT)
            _append(
                tokens,
                roles,
                [c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC],
                ROLE_BOUNDARY,
            )
            semantic = target_semantic[event.semantic_start : event.semantic_end]
            if len(semantic) != event.semantic_end - event.semantic_start:
                raise ValueError("target semantic span exceeds the formal sequence")
            _append(
                tokens, roles, c.encode_bicodec_semantic(semantic), ROLE_SEMANTIC
            )
            _append(tokens, roles, [c.TOKEN_END_SEMANTIC], ROLE_BOUNDARY)
            committed_text += event.text_delta
            prefix_ids = [int(value) for value in encode_text(committed_text)]
            committed_token_end = max(
                committed_token_end,
                _longest_common_prefix(prefix_ids, full_translation_ids),
            )
            if committed_text == session.target_text:
                committed_token_end = len(full_translation_ids)

        # The action label is predicted by the token immediately before the
        # action token after the standard next-token shift.
        annotations.append(
            {
                "event_index": event.event_index,
                "action_position": action_original - 1,
                "frontend_positions": list(range(glm_start, glm_end)),
                "frontend_ids": delta_codes,
                "translation_ids": full_translation_ids,
                "previous_committed_length": previous_token_end,
                "stable_target_length": committed_token_end,
                "support_bucket": event.support_bucket,
                "natural_action": 1 if event.action == "WRITE" else 0,
                "deadline_action": 1 if event.action == "WRITE" else 0,
                "deadline_forced": False,
                "deadline_loss_enabled": deadline_enabled,
                "chunk_end_ms": event.wall_time_ms,
                "soft_deadline_ms": soft_deadline_ms,
                "hard_deadline_ms": hard_deadline_ms,
                "sample_id": session.sample_id,
                "source_finished": event.source_finished,
                "playback_buffer_ms": event.playback_buffer_after_ms,
            }
        )

    if source_cursor != len(source_glm):
        raise ValueError("dense sequence did not consume the complete source GLM")
    if committed_text != session.target_text:
        raise ValueError("dense sequence did not commit the complete target text")
    if committed_token_end != len(full_translation_ids):
        raise ValueError("final committed target tokens do not cover the translation")
    _append(tokens, roles, [c.TOKEN_EOS], ROLE_BOUNDARY)
    for token in tokens:
        c.validate_token_id(int(token))

    shifted_roles = roles[1:]
    return DenseSessionTokenSample(
        sample_id=session.sample_id,
        tokens=tuple(tokens[:-1]),
        labels=tuple(tokens[1:]),
        loss_mask=tuple(
            0 if role == ROLE_OBSERVED else 1 for role in shifted_roles
        ),
        token_roles=tuple(shifted_roles),
        position_ids=tuple(range(len(tokens) - 1)),
        annotations=tuple(annotations),
    )


def _pad(values: Sequence[int], length: int, fill: int) -> list[int]:
    if len(values) > length:
        raise ValueError("cannot pad a sequence that already exceeds the target")
    return [*values, *([fill] * (length - len(values)))]


def pack_session_samples(
    samples: Iterable[DenseSessionTokenSample], *, seq_length: int
) -> Iterator[dict[str, object]]:
    """Pack sessions densely while retaining each full session as one THD boundary."""

    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    current: list[DenseSessionTokenSample] = []
    current_length = 0

    def emit() -> dict[str, object] | None:
        if not current:
            return None
        tokens: list[int] = []
        labels: list[int] = []
        loss_mask: list[int] = []
        token_roles: list[int] = []
        position_ids: list[int] = []
        boundaries: list[list[int]] = []
        sessions: list[dict[str, object]] = []
        source_ids: list[str] = []
        for sample in current:
            start = len(tokens)
            tokens.extend(sample.tokens)
            labels.extend(sample.labels)
            loss_mask.extend(sample.loss_mask)
            token_roles.extend(sample.token_roles)
            position_ids.extend(sample.position_ids)
            end = len(tokens)
            boundaries.append([start, end])
            source_ids.append(sample.sample_id)
            annotations: list[dict[str, object]] = []
            translation_ids = list(sample.annotations[0]["translation_ids"])
            for value in sample.annotations:
                annotation = dict(value)
                if list(annotation.pop("translation_ids")) != translation_ids:
                    raise ValueError("one session contains inconsistent target token IDs")
                annotation["action_position"] = start + int(value["action_position"])
                annotation["frontend_positions"] = [
                    start + int(position)
                    for position in value["frontend_positions"]  # type: ignore[index]
                ]
                annotations.append(annotation)
            sessions.append(
                {
                    "sample_id": sample.sample_id,
                    "boundary": [start, end],
                    "translation_ids": translation_ids,
                    "annotations": annotations,
                }
            )
        return {
            "schema_version": PACK_SCHEMA,
            "tokens": _pad(tokens, seq_length, c.TOKEN_PAD),
            "labels": _pad(labels, seq_length, c.TOKEN_PAD),
            "loss_mask": _pad(loss_mask, seq_length, 0),
            "token_roles": _pad(token_roles, seq_length, ROLE_OBSERVED),
            "position_ids": _pad(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "tasks": ["dense_trajectory"] * len(current),
            "source_ids": source_ids,
            "sessions": sessions,
        }

    for sample in samples:
        if sample.length > seq_length:
            raise ValueError(
                f"dense session {sample.sample_id} length {sample.length} exceeds {seq_length}"
            )
        if current and current_length + sample.length > seq_length:
            value = emit()
            if value is not None:
                yield value
            current = []
            current_length = 0
        current.append(sample)
        current_length += sample.length
    value = emit()
    if value is not None:
        yield value


__all__ = [
    "DenseSessionTokenSample",
    "PACK_SCHEMA",
    "ROLE_ACTION",
    "ROLE_BOUNDARY",
    "ROLE_OBSERVED",
    "ROLE_SEMANTIC",
    "ROLE_TEXT",
    "build_session_token_sample",
    "pack_session_samples",
]
