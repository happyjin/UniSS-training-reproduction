"""Build future-safe V1 ASR teacher sequences for same-prefix KL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from training import constants_uniss as c


V1HistoryKind = Literal["gold_asr", "v1_asr"]


def _hash_tokens(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True)
class V1TeacherRequest:
    event_index: int
    history_kind: V1HistoryKind
    visible_glm_tokens: int
    visible_source_prefix: str
    predictor_positions: tuple[int, ...]
    target_indices: tuple[int, ...]
    reference_labels: tuple[int, ...]
    prefix_sha256: str
    target_sha256: str
    final: bool

    def __post_init__(self) -> None:
        count = len(self.reference_labels)
        if count <= 0 or len(self.predictor_positions) != count:
            raise ValueError("V1 teacher request denominator differs")
        if self.target_indices != tuple(range(count)):
            raise ValueError("V1 teacher target positions are not contiguous")
        if any(left + 1 != right for left, right in zip(
            self.predictor_positions, self.predictor_positions[1:]
        )):
            raise ValueError("V1 teacher predictor positions are not contiguous")
        if self.predictor_positions[0] < 0 or self.visible_glm_tokens < 0:
            raise ValueError("V1 teacher request contains a negative boundary")
        if any(not 0 <= value < c.VOCAB_SIZE for value in self.reference_labels):
            raise ValueError("V1 teacher reference label is outside vocabulary")


@dataclass(frozen=True)
class V1TeacherSequence:
    history_kind: V1HistoryKind
    token_ids: tuple[int, ...]
    speech_indices: tuple[int | None, ...]
    requests: tuple[V1TeacherRequest, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.speech_indices) or len(self.token_ids) < 2:
            raise ValueError("V1 teacher sequence geometry differs")
        if not self.requests:
            raise ValueError("V1 teacher sequence has no requests")
        for request in self.requests:
            if request.history_kind != self.history_kind:
                raise ValueError("V1 teacher request history differs from sequence")
            for position, label in zip(
                request.predictor_positions, request.reference_labels
            ):
                if position + 1 >= len(self.token_ids):
                    raise ValueError("V1 teacher predictor position exceeds sequence")
                if self.token_ids[position + 1] != label:
                    raise ValueError("V1 teacher label differs from sequence target")

    @property
    def selected_predictor_positions(self) -> tuple[int, ...]:
        return tuple(
            position
            for request in self.requests
            for position in request.predictor_positions
        )


def _append_source(
    token_ids: list[int],
    speech_indices: list[int | None],
    start: int,
    stop: int,
) -> None:
    if not 0 <= start <= stop:
        raise ValueError("V1 teacher source interval is invalid")
    if start == stop:
        return
    token_ids.extend(
        [
            c.TOKEN_START_GLM,
            *([c.glm_semantic_id(0)] * (stop - start)),
            c.TOKEN_END_GLM,
        ]
    )
    speech_indices.extend([None, *range(start, stop), None])


def _append_request(
    token_ids: list[int],
    speech_indices: list[int | None],
    requests: list[V1TeacherRequest],
    *,
    event_index: int,
    history_kind: V1HistoryKind,
    visible_glm_tokens: int,
    visible_source_prefix: str,
    targets: Sequence[int],
    final: bool,
) -> None:
    values = tuple(int(value) for value in targets)
    if not values:
        raise ValueError("V1 teacher request target is empty")
    prefix_sha256 = _hash_tokens(token_ids)
    target_start = len(token_ids)
    token_ids.extend(values)
    speech_indices.extend([None] * len(values))
    requests.append(
        V1TeacherRequest(
            event_index=event_index,
            history_kind=history_kind,
            visible_glm_tokens=visible_glm_tokens,
            visible_source_prefix=visible_source_prefix,
            predictor_positions=tuple(range(target_start - 1, len(token_ids) - 1)),
            target_indices=tuple(range(len(values))),
            reference_labels=values,
            prefix_sha256=prefix_sha256,
            target_sha256=_hash_tokens(values),
            final=final,
        )
    )


def _build_sequence(
    trajectory: E2ETrajectory,
    rollout: V1Rollout,
    *,
    history_kind: V1HistoryKind,
    encode_text: Callable[[str], Sequence[int]],
) -> V1TeacherSequence:
    token_ids = [
        c.TOKEN_TASK_STREAMING_ASR,
        c.TOKEN_STREAMING_MODE,
        c.language_token_id(trajectory.src_lang),
        *c.wrap_global_tokens(trajectory.speaker_global),
    ]
    speech_indices: list[int | None] = [None] * len(token_ids)
    requests: list[V1TeacherRequest] = []
    visible_glm = 0
    for event, rollout_event in zip(trajectory.events, rollout.events):
        if not event.gold_source_delta:
            continue
        _append_source(
            token_ids,
            speech_indices,
            visible_glm,
            event.source_glm_end,
        )
        visible_glm = event.source_glm_end
        if history_kind == "gold_asr":
            content = tuple(int(value) for value in encode_text(event.gold_source_delta))
            if not content:
                raise ValueError("V1 gold ASR delta encoded empty")
            targets = (
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.src_lang),
                c.TOKEN_START_CONTENT,
                *content,
                c.TOKEN_END_CONTENT,
            )
            visible_prefix = event.gold_source_prefix
        else:
            targets = rollout_event.generated_tokens
            visible_prefix = rollout_event.v1_source_prefix
        _append_request(
            token_ids,
            speech_indices,
            requests,
            event_index=event.event_index,
            history_kind=history_kind,
            visible_glm_tokens=visible_glm,
            visible_source_prefix=visible_prefix,
            targets=targets,
            final=False,
        )
    _append_source(
        token_ids,
        speech_indices,
        visible_glm,
        trajectory.source_glm_length,
    )
    visible_glm = trajectory.source_glm_length
    final_targets = (
        (c.TOKEN_EOS,)
        if history_kind == "gold_asr"
        else rollout.final_generated_tokens
    )
    _append_request(
        token_ids,
        speech_indices,
        requests,
        event_index=len(trajectory.events),
        history_kind=history_kind,
        visible_glm_tokens=visible_glm,
        visible_source_prefix=(
            trajectory.normalized_transcription
            if history_kind == "gold_asr"
            else rollout.full_text
        ),
        targets=final_targets,
        final=True,
    )
    sequence = V1TeacherSequence(
        history_kind=history_kind,
        token_ids=tuple(token_ids),
        speech_indices=tuple(speech_indices),
        requests=tuple(requests),
    )
    selected_speech = [
        int(value) for value in sequence.speech_indices if value is not None
    ]
    if selected_speech != list(range(trajectory.source_glm_length)):
        raise ValueError("V1 teacher sequence does not cover source GLM exactly once")
    return sequence


def build_v1_teacher_sequences(
    trajectory: E2ETrajectory,
    rollout: V1Rollout,
    *,
    encode_text: Callable[[str], Sequence[int]],
) -> tuple[V1TeacherSequence, V1TeacherSequence]:
    if (
        trajectory.sample_id != rollout.sample_id
        or trajectory.split != rollout.split
        or len(trajectory.events) != len(rollout.events)
    ):
        raise ValueError("gold/rollout geometry differs for V1 teacher requests")
    return (
        _build_sequence(
            trajectory,
            rollout,
            history_kind="gold_asr",
            encode_text=encode_text,
        ),
        _build_sequence(
            trajectory,
            rollout,
            history_kind="v1_asr",
            encode_text=encode_text,
        ),
    )


__all__ = [
    "V1HistoryKind",
    "V1TeacherRequest",
    "V1TeacherSequence",
    "build_v1_teacher_sequences",
]
