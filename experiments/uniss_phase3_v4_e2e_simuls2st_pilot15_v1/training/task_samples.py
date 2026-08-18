"""Build the five immutable E2E task-family sample types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.requests import (
    build_phase3_requests,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_requests import (
    build_v1_teacher_sequences,
)
from training import constants_uniss as c
from training.sample_builders import build_performance_sample, build_quality_sample


TASK_SAMPLE_SCHEMA = "uniss_phase3_v4_e2e_task_sample_v1"

FAMILY_STREAMING_ASR = "streaming_asr_event"
FAMILY_INCREMENTAL_MT = "incremental_mt_event"
FAMILY_INTERLEAVED = "interleaved_e2e_s2st"
FAMILY_PHASE3_QUALITY = "phase3_quality_replay"
FAMILY_PHASE3_PERFORMANCE = "phase3_performance_replay"
TASK_FAMILIES = (
    FAMILY_STREAMING_ASR,
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_QUALITY,
    FAMILY_PHASE3_PERFORMANCE,
)

LOSS_NONE = 0
LOSS_ASR = 1
LOSS_MT = 2
LOSS_SEMANTIC = 3
LOSS_BOUNDARY = 4
LOSS_EOS = 5
LOSS_REPLAY = 6
LOSS_KIND_NAMES = {
    LOSS_NONE: "none",
    LOSS_ASR: "asr_ce",
    LOSS_MT: "mt_ce",
    LOSS_SEMANTIC: "semantic_ce",
    LOSS_BOUNDARY: "boundary_ce",
    LOSS_EOS: "eos_ce",
    LOSS_REPLAY: "phase3_replay_ce",
}


@dataclass(frozen=True)
class TeacherBinding:
    cache_kind: str
    request_id: int
    cache_position_start: int
    cache_position_stop: int
    target_start: int
    target_stop: int

    def __post_init__(self) -> None:
        if self.cache_kind not in {"v1_asr", "phase3"}:
            raise ValueError("unknown E2E teacher cache kind")
        if (
            self.request_id < 0
            or not 0 <= self.cache_position_start < self.cache_position_stop
            or not 0 <= self.target_start < self.target_stop
            or self.cache_position_stop - self.cache_position_start
            != self.target_stop - self.target_start
        ):
            raise ValueError("invalid E2E teacher binding geometry")


@dataclass(frozen=True)
class E2ETaskSample:
    sample_id: str
    sequence_id: str
    source_manifest_record: int
    family: str
    token_ids: tuple[int, ...]
    loss_kinds: tuple[int, ...]
    speech_indices: tuple[int | None, ...]
    source_audio: str | None
    source_glm_length: int
    teacher_bindings: tuple[TeacherBinding, ...] = ()
    commit_key: str | None = None
    commit_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in TASK_FAMILIES:
            raise ValueError("unknown E2E task family")
        if (
            not self.sample_id
            or not self.sequence_id
            or self.source_manifest_record < 0
            or len(self.token_ids) < 2
        ):
            raise ValueError("E2E task sample identity or sequence is incomplete")
        if not (
            len(self.token_ids)
            == len(self.loss_kinds)
            == len(self.speech_indices)
        ):
            raise ValueError("E2E task token/loss/acoustic geometry differs")
        if any(value not in LOSS_KIND_NAMES for value in self.loss_kinds):
            raise ValueError("E2E task sample contains an unknown loss kind")
        if not any(value != LOSS_NONE for value in self.loss_kinds):
            raise ValueError("E2E task sample has no supervised tokens")
        for token in self.token_ids:
            c.validate_token_id(int(token))
        speech = [int(value) for value in self.speech_indices if value is not None]
        if speech:
            if self.source_audio is None or speech != list(range(self.source_glm_length)):
                raise ValueError("E2E acoustic sample does not cover source GLM exactly")
        elif self.source_audio is not None or self.source_glm_length != 0:
            raise ValueError("E2E text/discrete sample has an acoustic sidecar")
        for binding in self.teacher_bindings:
            if binding.target_stop > len(self.token_ids):
                raise ValueError("E2E teacher binding exceeds the token sequence")
            if any(
                self.loss_kinds[index] == LOSS_NONE
                for index in range(binding.target_start, binding.target_stop)
            ):
                raise ValueError("E2E teacher binding covers an unsupervised token")
        if (self.commit_key is None) != (not self.commit_positions):
            raise ValueError("E2E commit-consistency identity/positions differ")
        if self.commit_positions:
            if any(
                not 0 < position < len(self.token_ids)
                for position in self.commit_positions
            ) or any(
                left + 1 != right
                for left, right in zip(
                    self.commit_positions, self.commit_positions[1:]
                )
            ):
                raise ValueError("E2E commit-consistency positions are not contiguous")

    @property
    def shifted_length(self) -> int:
        return len(self.token_ids) - 1

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": TASK_SAMPLE_SCHEMA,
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "source_manifest_record": self.source_manifest_record,
            "family": self.family,
            "token_ids": list(self.token_ids),
            "loss_kinds": list(self.loss_kinds),
            "speech_indices": [
                -1 if value is None else int(value) for value in self.speech_indices
            ],
            "source_audio": self.source_audio,
            "source_glm_length": self.source_glm_length,
            "teacher_bindings": [
                {
                    "cache_kind": value.cache_kind,
                    "request_id": value.request_id,
                    "cache_position_start": value.cache_position_start,
                    "cache_position_stop": value.cache_position_stop,
                    "target_start": value.target_start,
                    "target_stop": value.target_stop,
                }
                for value in self.teacher_bindings
            ],
            "commit_key": self.commit_key,
            "commit_positions": list(self.commit_positions),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "E2ETaskSample":
        if value.get("schema_version") != TASK_SAMPLE_SCHEMA:
            raise ValueError("unexpected E2E task sample schema")
        raw_speech = value.get("speech_indices")
        raw_bindings = value.get("teacher_bindings")
        if not isinstance(raw_speech, list) or not isinstance(raw_bindings, list):
            raise TypeError("E2E task sample sidecars are malformed")
        return cls(
            sample_id=str(value["sample_id"]),
            sequence_id=str(value["sequence_id"]),
            source_manifest_record=int(value["source_manifest_record"]),
            family=str(value["family"]),
            token_ids=tuple(int(item) for item in value["token_ids"]),  # type: ignore[arg-type]
            loss_kinds=tuple(int(item) for item in value["loss_kinds"]),  # type: ignore[arg-type]
            speech_indices=tuple(
                None if int(item) < 0 else int(item) for item in raw_speech
            ),
            source_audio=(
                None if value.get("source_audio") is None else str(value["source_audio"])
            ),
            source_glm_length=int(value["source_glm_length"]),
            teacher_bindings=tuple(
                TeacherBinding(
                    cache_kind=str(item["cache_kind"]),
                    request_id=int(item["request_id"]),
                    cache_position_start=int(item["cache_position_start"]),
                    cache_position_stop=int(item["cache_position_stop"]),
                    target_start=int(item["target_start"]),
                    target_stop=int(item["target_stop"]),
                )
                for item in raw_bindings
            ),
            commit_key=(
                None if value.get("commit_key") is None else str(value["commit_key"])
            ),
            commit_positions=tuple(
                int(item) for item in value.get("commit_positions", [])  # type: ignore[arg-type]
            ),
        )


def _mark_fragment(
    loss_kinds: list[int],
    start: int,
    tokens: Sequence[int],
    content_kind: int,
) -> None:
    for offset, token in enumerate(tokens):
        position = start + offset
        if token == c.TOKEN_EOS:
            loss_kinds[position] = LOSS_EOS
        elif token in {
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_ASR,
            c.TOKEN_TASK_S2T_TRANSLATION,
            c.TOKEN_TASK_TTS,
            c.TOKEN_START_CONTENT,
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_WAIT_READ,
            c.TOKEN_CMN,
            c.TOKEN_ENG,
        } or c.SPEED_OFFSET <= int(token) < c.SPEED_OFFSET + c.SPEED_SIZE:
            loss_kinds[position] = LOSS_BOUNDARY
        else:
            loss_kinds[position] = content_kind


def build_streaming_asr_task(
    trajectory: E2ETrajectory,
    rollout: V1Rollout,
    *,
    encode_text: Callable[[str], Sequence[int]],
) -> E2ETaskSample:
    sequence = build_v1_teacher_sequences(
        trajectory, rollout, encode_text=encode_text
    )[0]
    loss_kinds = [LOSS_NONE] * len(sequence.token_ids)
    bindings: list[TeacherBinding] = []
    for request_id, request in enumerate(sequence.requests):
        target_start = request.predictor_positions[0] + 1
        target_stop = target_start + len(request.reference_labels)
        _mark_fragment(
            loss_kinds,
            target_start,
            request.reference_labels,
            LOSS_ASR,
        )
        bindings.append(
            TeacherBinding(
                cache_kind="v1_asr",
                request_id=request_id,
                cache_position_start=0,
                cache_position_stop=len(request.reference_labels),
                target_start=target_start,
                target_stop=target_stop,
            )
        )
    return E2ETaskSample(
        sample_id=trajectory.sample_id,
        sequence_id=f"{trajectory.sample_id}:asr:gold",
        source_manifest_record=trajectory.source_manifest_record,
        family=FAMILY_STREAMING_ASR,
        token_ids=sequence.token_ids,
        loss_kinds=tuple(loss_kinds),
        speech_indices=sequence.speech_indices,
        source_audio=trajectory.source_audio,
        source_glm_length=trajectory.source_glm_length,
        teacher_bindings=tuple(bindings),
    )


def build_incremental_mt_tasks(
    trajectory: E2ETrajectory,
    rollout: V1Rollout,
    *,
    encode_text: Callable[[str], Sequence[int]],
) -> list[E2ETaskSample]:
    requests = build_phase3_requests(
        trajectory, rollout, encode_text=encode_text
    )
    output: list[E2ETaskSample] = []
    for request_id, request in enumerate(requests):
        if request.family != "phase3_mt":
            continue
        tokens = (*request.prompt_ids, *request.target_ids)
        loss_kinds = [LOSS_NONE] * len(tokens)
        selected_positions: list[int] = []
        for target_index, label in zip(
            request.selected_target_indices, request.reference_labels
        ):
            position = len(request.prompt_ids) + target_index
            selected_positions.append(position)
            loss_kinds[position] = (
                LOSS_BOUNDARY if label == c.TOKEN_END_CONTENT else LOSS_MT
            )
        if any(left + 1 != right for left, right in zip(
            selected_positions, selected_positions[1:]
        )):
            # BPE retokenization can select a sparse LCS. Each contiguous run
            # gets its own cache binding while preserving one task sample.
            runs: list[tuple[int, int]] = []
            run_start = run_stop = selected_positions[0]
            for position in selected_positions[1:]:
                if position == run_stop + 1:
                    run_stop = position
                else:
                    runs.append((run_start, run_stop + 1))
                    run_start = run_stop = position
            runs.append((run_start, run_stop + 1))
        else:
            runs = [(selected_positions[0], selected_positions[-1] + 1)]
        bindings_list: list[TeacherBinding] = []
        cache_cursor = 0
        for start, stop in runs:
            count = stop - start
            bindings_list.append(
                TeacherBinding(
                    cache_kind="phase3",
                    request_id=request_id,
                    cache_position_start=cache_cursor,
                    cache_position_stop=cache_cursor + count,
                    target_start=start,
                    target_stop=stop,
                )
            )
            cache_cursor += count
        bindings = tuple(bindings_list)
        output.append(
            E2ETaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=(
                    f"{trajectory.sample_id}:mt:{request.event_index}:"
                    f"{request.history_kind}"
                ),
                source_manifest_record=trajectory.source_manifest_record,
                family=FAMILY_INCREMENTAL_MT,
                token_ids=tuple(tokens),
                loss_kinds=tuple(loss_kinds),
                speech_indices=(None,) * len(tokens),
                source_audio=None,
                source_glm_length=0,
                teacher_bindings=bindings,
                commit_key=f"{trajectory.sample_id}:{request.history_kind}",
                commit_positions=tuple(
                    range(
                        len(request.prompt_ids),
                        len(request.prompt_ids) + len(request.target_ids) - 2,
                    )
                ),
            )
        )
    if not output:
        raise ValueError("trajectory produced no incremental MT task samples")
    return output


def _append_observed(
    tokens: list[int],
    loss_kinds: list[int],
    speech_indices: list[int | None],
    values: Sequence[int],
    speech: Sequence[int | None] | None = None,
) -> None:
    tokens.extend(int(value) for value in values)
    loss_kinds.extend([LOSS_NONE] * len(values))
    speech_indices.extend(
        [None] * len(values) if speech is None else [value for value in speech]
    )


def _append_generated(
    tokens: list[int],
    loss_kinds: list[int],
    speech_indices: list[int | None],
    values: Sequence[int],
    content_kind: int,
) -> None:
    start = len(tokens)
    tokens.extend(int(value) for value in values)
    loss_kinds.extend([LOSS_NONE] * len(values))
    speech_indices.extend([None] * len(values))
    _mark_fragment(loss_kinds, start, values, content_kind)


def build_interleaved_task(
    trajectory: E2ETrajectory,
    *,
    encode_text: Callable[[str], Sequence[int]],
    speed: float = 1.0,
    rollout: V1Rollout | None = None,
    semantic_stride: int = 8,
) -> E2ETaskSample:
    tokens: list[int] = []
    loss_kinds: list[int] = []
    speech_indices: list[int | None] = []
    semantic_positions: dict[tuple[int, int], int] = {}
    semantic_boundaries: dict[int, int] = {}
    _append_observed(
        tokens,
        loss_kinds,
        speech_indices,
        [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(trajectory.tgt_lang),
            c.speed_token_id(speed),
            *c.wrap_global_tokens(trajectory.speaker_global),
        ],
    )
    source_cursor = 0
    for event in trajectory.events:
        if event.source_glm_delta:
            source_stop = source_cursor + len(event.source_glm_delta)
            _append_observed(
                tokens,
                loss_kinds,
                speech_indices,
                [
                    c.TOKEN_START_GLM,
                    *([c.glm_semantic_id(0)] * len(event.source_glm_delta)),
                    c.TOKEN_END_GLM,
                ],
                [None, *range(source_cursor, source_stop), None],
            )
            source_cursor = source_stop
        if event.gold_source_delta:
            content = tuple(int(value) for value in encode_text(event.gold_source_delta))
            if not content:
                raise ValueError("interleaved ASR delta encoded empty")
            _append_generated(
                tokens,
                loss_kinds,
                speech_indices,
                (
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_TASK_ASR,
                    c.language_token_id(trajectory.src_lang),
                    c.TOKEN_START_CONTENT,
                    *content,
                    c.TOKEN_END_CONTENT,
                ),
                LOSS_ASR,
            )
        target_write = bool(event.target_text_delta or event.target_semantic_delta)
        if event.target_text_delta:
            content = tuple(int(value) for value in encode_text(event.target_text_delta))
            if not content:
                raise ValueError("interleaved MT delta encoded empty")
            _append_generated(
                tokens,
                loss_kinds,
                speech_indices,
                (
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_TASK_S2T_TRANSLATION,
                    c.language_token_id(trajectory.tgt_lang),
                    c.TOKEN_START_CONTENT,
                    *content,
                    c.TOKEN_END_CONTENT,
                ),
                LOSS_MT,
            )
        if event.target_semantic_delta:
            fragment_start = len(tokens)
            _append_generated(
                tokens,
                loss_kinds,
                speech_indices,
                (
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_TASK_TTS,
                    c.language_token_id(trajectory.tgt_lang),
                    c.speed_token_id(speed),
                    c.TOKEN_START_SEMANTIC,
                    *c.encode_bicodec_semantic(event.target_semantic_delta),
                    c.TOKEN_END_SEMANTIC,
                ),
                LOSS_SEMANTIC,
            )
            content_start = fragment_start + 5
            for local_index in range(len(event.target_semantic_delta)):
                semantic_positions[
                    (event.event_index, event.target_semantic_start + local_index)
                ] = content_start + local_index
            semantic_boundaries[event.event_index] = (
                content_start + len(event.target_semantic_delta)
            )
        if not target_write and not event.source_final:
            _append_generated(
                tokens,
                loss_kinds,
                speech_indices,
                (c.TOKEN_WAIT_READ,),
                LOSS_BOUNDARY,
            )
    if source_cursor != trajectory.source_glm_length:
        raise ValueError("interleaved task did not cover source GLM")
    eos_position = len(tokens)
    _append_generated(
        tokens,
        loss_kinds,
        speech_indices,
        (c.TOKEN_EOS,),
        LOSS_EOS,
    )
    bindings: list[TeacherBinding] = []
    if rollout is not None:
        requests = build_phase3_requests(
            trajectory,
            rollout,
            encode_text=encode_text,
            semantic_stride=semantic_stride,
        )
        for request_id, request in enumerate(requests):
            if request.family != "phase3_semantic":
                continue
            event = trajectory.events[request.event_index]
            mapped: list[tuple[int, int]] = []
            for cache_position, (target_index, label) in enumerate(
                zip(request.selected_target_indices, request.reference_labels)
            ):
                if target_index < event.target_semantic_end:
                    target_position = semantic_positions[
                        (event.event_index, target_index)
                    ]
                elif target_index == event.target_semantic_end:
                    target_position = semantic_boundaries[event.event_index]
                elif event.target_final and target_index == event.target_semantic_end + 1:
                    target_position = eos_position
                else:
                    raise ValueError("semantic teacher target cannot map to interleaved task")
                if tokens[target_position] != label:
                    raise ValueError("semantic teacher label differs from interleaved token")
                mapped.append((cache_position, target_position))
            run_cache_start, run_target_start = mapped[0]
            previous_cache, previous_target = mapped[0]
            for cache_position, target_position in mapped[1:]:
                if (
                    cache_position == previous_cache + 1
                    and target_position == previous_target + 1
                ):
                    previous_cache = cache_position
                    previous_target = target_position
                    continue
                bindings.append(
                    TeacherBinding(
                        cache_kind="phase3",
                        request_id=request_id,
                        cache_position_start=run_cache_start,
                        cache_position_stop=previous_cache + 1,
                        target_start=run_target_start,
                        target_stop=previous_target + 1,
                    )
                )
                run_cache_start, run_target_start = cache_position, target_position
                previous_cache, previous_target = cache_position, target_position
            bindings.append(
                TeacherBinding(
                    cache_kind="phase3",
                    request_id=request_id,
                    cache_position_start=run_cache_start,
                    cache_position_stop=previous_cache + 1,
                    target_start=run_target_start,
                    target_stop=previous_target + 1,
                )
            )
    return E2ETaskSample(
        sample_id=trajectory.sample_id,
        sequence_id=f"{trajectory.sample_id}:e2e",
        source_manifest_record=trajectory.source_manifest_record,
        family=FAMILY_INTERLEAVED,
        token_ids=tuple(tokens),
        loss_kinds=tuple(loss_kinds),
        speech_indices=tuple(speech_indices),
        source_audio=trajectory.source_audio,
        source_glm_length=trajectory.source_glm_length,
        teacher_bindings=tuple(bindings),
    )


def _reconstructed_source_glm(trajectory: E2ETrajectory) -> tuple[int, ...]:
    return tuple(
        int(value)
        for event in trajectory.events
        for value in event.source_glm_delta
    )


def _reconstructed_target_semantic(trajectory: E2ETrajectory) -> tuple[int, ...]:
    return tuple(
        int(value)
        for event in trajectory.events
        for value in event.target_semantic_delta
    )


def build_phase3_replay_tasks(
    trajectory: E2ETrajectory,
    *,
    encode_text: Callable[[str], Sequence[int]],
) -> tuple[E2ETaskSample, E2ETaskSample]:
    common: dict[str, object] = {
        "source_glm": _reconstructed_source_glm(trajectory),
        "bicodec_global": trajectory.speaker_global,
        "tgt_lang": trajectory.tgt_lang,
        "translation": trajectory.full_translation,
        "target_bicodec": _reconstructed_target_semantic(trajectory),
        "text_encoder": lambda text: [int(value) for value in encode_text(text)],
        "source_id": trajectory.sample_id,
    }
    quality = build_quality_sample(
        **common,  # type: ignore[arg-type]
        src_lang=trajectory.src_lang,
        transcription=trajectory.full_transcription,
    )
    performance = build_performance_sample(**common)  # type: ignore[arg-type]

    def convert(family: str, sample) -> E2ETaskSample:
        tokens = (*sample.prompt_ids, *sample.target_ids)
        kinds = [LOSS_NONE] * len(sample.prompt_ids) + [
            LOSS_REPLAY
        ] * len(sample.target_ids)
        return E2ETaskSample(
            sample_id=trajectory.sample_id,
            sequence_id=f"{trajectory.sample_id}:{family}",
            source_manifest_record=trajectory.source_manifest_record,
            family=family,
            token_ids=tuple(tokens),
            loss_kinds=tuple(kinds),
            speech_indices=(None,) * len(tokens),
            source_audio=None,
            source_glm_length=0,
        )

    return (
        convert(FAMILY_PHASE3_QUALITY, quality),
        convert(FAMILY_PHASE3_PERFORMANCE, performance),
    )


def loss_counts(sample: E2ETaskSample) -> dict[str, int]:
    output = {name: 0 for name in LOSS_KIND_NAMES.values()}
    for value in sample.loss_kinds:
        output[LOSS_KIND_NAMES[value]] += 1
    return output


__all__ = [
    "E2ETaskSample",
    "FAMILY_INCREMENTAL_MT",
    "FAMILY_INTERLEAVED",
    "FAMILY_PHASE3_PERFORMANCE",
    "FAMILY_PHASE3_QUALITY",
    "FAMILY_STREAMING_ASR",
    "LOSS_ASR",
    "LOSS_BOUNDARY",
    "LOSS_EOS",
    "LOSS_KIND_NAMES",
    "LOSS_MT",
    "LOSS_NONE",
    "LOSS_REPLAY",
    "LOSS_SEMANTIC",
    "TASK_FAMILIES",
    "TASK_SAMPLE_SCHEMA",
    "TeacherBinding",
    "build_incremental_mt_tasks",
    "build_interleaved_task",
    "build_phase3_replay_tasks",
    "build_streaming_asr_task",
    "loss_counts",
]
