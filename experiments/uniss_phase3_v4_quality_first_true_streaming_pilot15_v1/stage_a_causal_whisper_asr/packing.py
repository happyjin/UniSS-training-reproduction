"""Build isolated Stage A 18k Megatron packs with causal-ASR sidecars."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.ctc_targets import (
    UTF8ByteCTCMap,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.events import (
    ASREventSession,
    build_asr_event_session,
)
from training import constants_uniss as c
from training.sample_builders import (
    TrainingSample,
    build_asr_sample,
    build_performance_sample,
    build_quality_sample,
)


PACK_SCHEMA = "uniss_quality_first_stage_a_pack_v1"
LOSS_NONE = 0
LOSS_STREAMING_ASR = 1
LOSS_CAUSAL_FULL_ASR = 2
LOSS_OFFLINE_ASR_REPLAY = 3
LOSS_PHASE3_REPLAY = 4
LOSS_KIND_NAMES = {
    LOSS_STREAMING_ASR: "streaming_asr",
    LOSS_CAUSAL_FULL_ASR: "causal_full_asr",
    LOSS_OFFLINE_ASR_REPLAY: "offline_asr_replay",
    LOSS_PHASE3_REPLAY: "phase3_replay",
}


@dataclass(frozen=True)
class StageASample:
    sample_id: str
    task: str
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    loss_kinds: tuple[int, ...]
    position_ids: tuple[int, ...]
    acoustic: dict[str, object] | None

    @property
    def length(self) -> int:
        return len(self.tokens)

    def __post_init__(self) -> None:
        lengths = {
            len(self.tokens),
            len(self.labels),
            len(self.loss_mask),
            len(self.loss_kinds),
            len(self.position_ids),
        }
        if len(lengths) != 1 or self.length <= 0:
            raise ValueError("Stage A shifted sample tensors differ in length")
        if any(kind not in LOSS_KIND_NAMES and kind != LOSS_NONE for kind in self.loss_kinds):
            raise ValueError("Stage A sample contains an unknown loss kind")


def _stable_bucket(sample_id: str, modulus: int, salt: str = "kind") -> int:
    digest = hashlib.blake2b(
        f"{salt}:{sample_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % modulus


def supervision_kind(sample_id: str) -> int:
    bucket = _stable_bucket(sample_id, 20)
    if bucket < 12:
        return LOSS_STREAMING_ASR
    if bucket < 16:
        return LOSS_CAUSAL_FULL_ASR
    if bucket < 19:
        return LOSS_OFFLINE_ASR_REPLAY
    return LOSS_PHASE3_REPLAY


def _shift(
    sample_id: str,
    task: str,
    conceptual: Sequence[int],
    generated: Sequence[bool],
    loss_kind: int,
    acoustic: dict[str, object] | None,
) -> StageASample:
    if len(conceptual) != len(generated) or len(conceptual) < 2:
        raise ValueError("conceptual tokens/roles are malformed")
    for token in conceptual:
        c.validate_token_id(int(token))
    return StageASample(
        sample_id=sample_id,
        task=task,
        tokens=tuple(int(value) for value in conceptual[:-1]),
        labels=tuple(int(value) for value in conceptual[1:]),
        loss_mask=tuple(int(value) for value in generated[1:]),
        loss_kinds=tuple(loss_kind if value else LOSS_NONE for value in generated[1:]),
        position_ids=tuple(range(len(conceptual) - 1)),
        acoustic=acoustic,
    )


def _from_training_sample(
    sample: TrainingSample,
    loss_kind: int,
    acoustic: dict[str, object] | None = None,
) -> StageASample:
    conceptual = [*sample.prompt_ids, *sample.target_ids]
    generated = [False] * len(sample.prompt_ids) + [True] * len(sample.target_ids)
    return _shift(
        sample.source_id or "unknown",
        sample.task,
        conceptual,
        generated,
        loss_kind,
        acoustic,
    )


def _acoustic_sidecar(
    record: Mapping[str, object],
    session: ASREventSession,
    glm_positions: Sequence[int],
    task: str,
) -> dict[str, object]:
    source_glm = [int(value) for value in record["source_glm"]]  # type: ignore[index]
    if len(glm_positions) != len(source_glm):
        raise ValueError("acoustic GLM positions do not cover the source sequence")
    ctc = UTF8ByteCTCMap(session.src_lang).encode_text(session.normalized_transcript)
    return {
        "sample_id": session.sample_id,
        "task": task,
        "source_audio": str(record["source_audio"]),
        "source_duration_ms": session.source_duration_ms,
        "src_lang": session.src_lang,
        "canonical_transcript": session.normalized_transcript,
        "ctc_ids": ctc,
        "source_glm": source_glm,
        "glm_positions": [int(value) for value in glm_positions],
    }


def build_streaming_asr_sample(
    record: Mapping[str, object],
    encode_text: Callable[[str], Sequence[int]],
    fixed_speaker: Sequence[int],
) -> StageASample:
    session = build_asr_event_session(record)
    conceptual: list[int] = [
        c.TOKEN_TASK_STREAMING_ASR,
        c.TOKEN_STREAMING_MODE,
        c.language_token_id(session.src_lang),
        *c.wrap_global_tokens(fixed_speaker),
    ]
    generated: list[bool] = [False] * len(conceptual)
    glm_positions: list[int] = []
    glm_cursor = 0
    for event in session.events:
        if not event.has_text_delta:
            continue
        conceptual.append(c.TOKEN_START_GLM)
        generated.append(False)
        values = [
            c.glm_semantic_id(int(value))
            for value in record["source_glm"][glm_cursor : event.glm_end]  # type: ignore[index]
        ]
        glm_positions.extend(range(len(conceptual), len(conceptual) + len(values)))
        conceptual.extend(values)
        generated.extend([False] * len(values))
        conceptual.append(c.TOKEN_END_GLM)
        generated.append(False)
        glm_cursor = event.glm_end
        delta = [int(value) for value in encode_text(event.delta_text)]
        if not delta:
            raise ValueError("non-empty streaming text delta encoded empty")
        output = [
            c.TOKEN_WRITE_GENERATE,
            c.language_token_id(session.src_lang),
            c.TOKEN_START_CONTENT,
            *delta,
            c.TOKEN_END_CONTENT,
        ]
        conceptual.extend(output)
        generated.extend([True] * len(output))
    source_glm = record["source_glm"]  # type: ignore[index]
    if glm_cursor < len(source_glm):
        conceptual.append(c.TOKEN_START_GLM)
        generated.append(False)
        values = [c.glm_semantic_id(int(value)) for value in source_glm[glm_cursor:]]
        glm_positions.extend(range(len(conceptual), len(conceptual) + len(values)))
        conceptual.extend(values)
        generated.extend([False] * len(values))
        conceptual.append(c.TOKEN_END_GLM)
        generated.append(False)
    conceptual.append(c.TOKEN_EOS)
    generated.append(True)
    acoustic = _acoustic_sidecar(record, session, glm_positions, "streaming_asr")
    return _shift(
        session.sample_id,
        "streaming_asr",
        conceptual,
        generated,
        LOSS_STREAMING_ASR,
        acoustic,
    )


def build_stage_a_sample(
    record: Mapping[str, object],
    encode_text: Callable[[str], Sequence[int]],
    fixed_speaker: Sequence[int],
) -> StageASample:
    sample_id = str(record["id"])
    kind = supervision_kind(sample_id)
    if kind == LOSS_STREAMING_ASR:
        return build_streaming_asr_sample(record, encode_text, fixed_speaker)
    if kind == LOSS_CAUSAL_FULL_ASR:
        session = build_asr_event_session(record)
        training = build_asr_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=fixed_speaker,
            src_lang=session.src_lang,
            transcription=session.normalized_transcript,
            text_encoder=lambda text: [int(value) for value in encode_text(text)],
            source_id=sample_id,
        )
        glm_positions = [
            index
            for index, token in enumerate(training.input_ids)
            if c.GLM_SEMANTIC_OFFSET <= int(token) <= c.GLM_SEMANTIC_SPAN.last_id
        ]
        acoustic = _acoustic_sidecar(record, session, glm_positions, "causal_full_asr")
        return _from_training_sample(training, kind, acoustic)
    if kind == LOSS_OFFLINE_ASR_REPLAY:
        training = build_asr_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            src_lang=str(record["src_lang"]),
            transcription=str(record["transcription"]),
            text_encoder=lambda text: [int(value) for value in encode_text(text)],
            source_id=sample_id,
        )
        return _from_training_sample(training, kind)
    common = {
        "source_glm": record["source_glm"],
        "bicodec_global": record["bicodec_global"],
        "tgt_lang": str(record["tgt_lang"]),
        "translation": str(record["translation"]),
        "target_bicodec": record["target_bicodec"],
        "text_encoder": lambda text: [int(value) for value in encode_text(text)],
        "source_id": sample_id,
    }
    if _stable_bucket(sample_id, 2, salt="phase3-mode") == 0:
        training = build_quality_sample(
            **common,  # type: ignore[arg-type]
            src_lang=str(record["src_lang"]),
            transcription=str(record["transcription"]),
        )
    else:
        training = build_performance_sample(**common)  # type: ignore[arg-type]
    return _from_training_sample(training, kind)


def _pad(values: Sequence[int], length: int, fill: int) -> list[int]:
    if len(values) > length:
        raise ValueError("cannot pad an overlong Stage A pack")
    return [*values, *([fill] * (length - len(values)))]


def pack_stage_a_samples(
    samples: Iterable[StageASample],
    *,
    seq_length: int,
) -> Iterable[dict[str, object]]:
    current: list[StageASample] = []
    current_length = 0

    def emit() -> dict[str, object] | None:
        if not current:
            return None
        tokens: list[int] = []
        labels: list[int] = []
        loss_mask: list[int] = []
        loss_kinds: list[int] = []
        position_ids: list[int] = []
        boundaries: list[list[int]] = []
        acoustics: list[dict[str, object]] = []
        tasks: list[str] = []
        source_ids: list[str] = []
        for sample in current:
            start = len(tokens)
            tokens.extend(sample.tokens)
            labels.extend(sample.labels)
            loss_mask.extend(sample.loss_mask)
            loss_kinds.extend(sample.loss_kinds)
            position_ids.extend(sample.position_ids)
            end = len(tokens)
            boundaries.append([start, end])
            tasks.append(sample.task)
            source_ids.append(sample.sample_id)
            if sample.acoustic is not None:
                value = dict(sample.acoustic)
                value["batch_boundary_index"] = len(boundaries) - 1
                value["glm_positions"] = [
                    start + int(position)
                    for position in value["glm_positions"]  # type: ignore[index]
                ]
                acoustics.append(value)
        return {
            "schema_version": PACK_SCHEMA,
            "tokens": _pad(tokens, seq_length, c.TOKEN_PAD),
            "labels": _pad(labels, seq_length, c.TOKEN_PAD),
            "loss_mask": _pad(loss_mask, seq_length, 0),
            "loss_kinds": _pad(loss_kinds, seq_length, LOSS_NONE),
            "position_ids": _pad(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "tasks": tasks,
            "source_ids": source_ids,
            "acoustics": acoustics,
            "used_tokens": len(tokens),
        }

    for sample in samples:
        if sample.length > seq_length:
            raise ValueError(
                f"Stage A sample {sample.sample_id} length {sample.length} exceeds {seq_length}"
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
    "LOSS_CAUSAL_FULL_ASR",
    "LOSS_KIND_NAMES",
    "LOSS_NONE",
    "LOSS_OFFLINE_ASR_REPLAY",
    "LOSS_PHASE3_REPLAY",
    "LOSS_STREAMING_ASR",
    "PACK_SCHEMA",
    "StageASample",
    "build_stage_a_sample",
    "build_streaming_asr_sample",
    "pack_stage_a_samples",
    "supervision_kind",
]
