"""Versioned, self-auditing trajectory schema for E2E simultaneous S2ST.

The serialized trajectory stores every target semantic code exactly once in
event deltas.  A stable digest of the original full semantic sequence makes
the lossless-concatenation invariant independently verifiable without
duplicating the long sequence at the top level.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from training.constants_uniss import normalize_language


TRAJECTORY_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_trajectory_v1"
SOURCE_SAMPLE_RATE = 16_000
ROLLOUT_PENDING = "pending"


def hash_int_sequence(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        integer = int(value)
        if not 0 <= integer < 2**32:
            raise ValueError(f"integer sequence value is outside uint32: {integer}")
        digest.update(integer.to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def append_text(prefix: str, delta: str, language: str) -> str:
    prefix = " ".join(str(prefix).strip().split())
    delta = " ".join(str(delta).strip().split())
    if not delta:
        return prefix
    if not prefix:
        return delta
    if normalize_language(language) == "cmn":
        return f"{prefix}{delta}"
    return f"{prefix} {delta}"


@dataclass(frozen=True)
class TrajectoryEvent:
    event_index: int
    source_start_ms: int
    source_end_ms: int
    source_pcm_start: int
    source_pcm_end: int
    source_glm_start: int
    source_glm_end: int
    source_glm_delta: tuple[int, ...]
    gold_source_word_start: int
    gold_source_word_end: int
    gold_source_delta: str
    gold_source_prefix: str
    v1_source_delta: str | None
    v1_source_prefix: str | None
    target_text_delta: str
    target_text_prefix: str
    target_semantic_start: int
    target_semantic_end: int
    target_semantic_delta: tuple[int, ...]
    target_support_end_ms: int | None
    source_final: bool
    target_final: bool
    alignment_confidence: float
    noise_severity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrajectoryEvent":
        fields = dict(value)
        fields["source_glm_delta"] = tuple(int(item) for item in fields["source_glm_delta"])  # type: ignore[index]
        fields["target_semantic_delta"] = tuple(
            int(item) for item in fields["target_semantic_delta"]  # type: ignore[index]
        )
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class E2ETrajectory:
    sample_id: str
    split: str
    src_lang: str
    tgt_lang: str
    source_audio: str
    source_audio_sha256: str | None
    source_audio_hash_status: str
    source_sample_rate: int
    source_duration_ms: int
    speaker_global: tuple[int, ...]
    full_transcription: str
    normalized_transcription: str
    full_translation: str
    normalized_translation: str
    source_glm_length: int
    source_glm_sha256: str
    target_semantic_length: int
    target_semantic_sha256: str
    source_manifest: str
    source_manifest_record: int
    v1_checkpoint_sha256: str
    phase3_teacher_sha256: str
    v1_rollout_status: str
    events: tuple[TrajectoryEvent, ...]
    schema_version: str = TRAJECTORY_SCHEMA

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "E2ETrajectory":
        fields = dict(value)
        fields["speaker_global"] = tuple(int(item) for item in fields["speaker_global"])  # type: ignore[index]
        fields["events"] = tuple(
            TrajectoryEvent.from_mapping(item)
            for item in fields["events"]  # type: ignore[index]
        )
        return cls(**fields)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, separators=(",", ":"))


def _validate_hex_digest(value: str | None, *, label: str, allow_deferred: bool) -> None:
    if value is None and allow_deferred:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} is not hexadecimal") from exc


def validate_trajectory(
    trajectory: E2ETrajectory,
    *,
    require_audio_hash: bool = False,
    require_v1_rollout: bool = False,
) -> dict[str, int | float | str]:
    if trajectory.schema_version != TRAJECTORY_SCHEMA:
        raise ValueError("unexpected trajectory schema")
    if not trajectory.sample_id or not trajectory.events:
        raise ValueError("trajectory is missing sample ID or events")
    src_lang = normalize_language(trajectory.src_lang)
    tgt_lang = normalize_language(trajectory.tgt_lang)
    if src_lang == tgt_lang:
        raise ValueError("trajectory source and target languages must differ")
    if trajectory.source_sample_rate != SOURCE_SAMPLE_RATE:
        raise ValueError("trajectory source sample rate is not 16 kHz")
    if trajectory.source_duration_ms <= 0:
        raise ValueError("trajectory source duration must be positive")
    if len(trajectory.speaker_global) != 32:
        raise ValueError("speaker condition must contain 32 global codes")
    if any(not 0 <= int(value) < 4096 for value in trajectory.speaker_global):
        raise ValueError("speaker condition escaped the BiCodec global range")
    _validate_hex_digest(
        trajectory.source_audio_sha256,
        label="source audio SHA256",
        allow_deferred=not require_audio_hash,
    )
    if require_audio_hash and trajectory.source_audio_hash_status != "complete":
        raise ValueError("source audio hash was deferred")
    _validate_hex_digest(trajectory.source_glm_sha256, label="source GLM SHA256", allow_deferred=False)
    _validate_hex_digest(
        trajectory.target_semantic_sha256,
        label="target semantic SHA256",
        allow_deferred=False,
    )
    _validate_hex_digest(
        trajectory.v1_checkpoint_sha256,
        label="V1 checkpoint SHA256",
        allow_deferred=False,
    )
    _validate_hex_digest(
        trajectory.phase3_teacher_sha256,
        label="Phase3 checkpoint SHA256",
        allow_deferred=False,
    )

    source_text = ""
    v1_text = ""
    target_text = ""
    source_glm: list[int] = []
    target_semantic: list[int] = []
    previous_source_ms = 0
    previous_source_pcm = 0
    previous_source_word = 0
    previous_source_glm = 0
    previous_target_semantic = 0
    for index, event in enumerate(trajectory.events):
        if event.event_index != index:
            raise ValueError("event indices are not contiguous")
        if event.source_start_ms != previous_source_ms or event.source_end_ms <= event.source_start_ms:
            raise ValueError("source event times contain a gap, overlap, or empty interval")
        expected_start = event.source_start_ms * SOURCE_SAMPLE_RATE // 1000
        expected_end = event.source_end_ms * SOURCE_SAMPLE_RATE // 1000
        if event.source_pcm_start != expected_start or event.source_pcm_end != expected_end:
            raise ValueError("PCM offsets do not match millisecond boundaries")
        if event.source_pcm_start != previous_source_pcm:
            raise ValueError("source PCM intervals contain a gap or overlap")
        if event.gold_source_word_start != previous_source_word:
            raise ValueError("source word intervals contain a gap or overlap")
        if event.source_glm_start != previous_source_glm:
            raise ValueError("source GLM intervals contain a gap or overlap")
        if event.source_glm_end - event.source_glm_start != len(event.source_glm_delta):
            raise ValueError("source GLM span length differs from its delta")
        if event.target_semantic_start != previous_target_semantic:
            raise ValueError("target semantic intervals contain a gap or overlap")
        if event.target_semantic_end - event.target_semantic_start != len(event.target_semantic_delta):
            raise ValueError("target semantic span length differs from its delta")
        source_text = append_text(source_text, event.gold_source_delta, src_lang)
        if source_text != event.gold_source_prefix:
            raise ValueError("gold source prefix rolled back or is not its delta concatenation")
        target_text = append_text(target_text, event.target_text_delta, tgt_lang)
        if target_text != event.target_text_prefix:
            raise ValueError("target text prefix rolled back or is not its delta concatenation")
        if event.target_support_end_ms is not None:
            if event.target_support_end_ms < 0:
                raise ValueError("target support time is negative")
            if not event.source_final and event.source_end_ms < event.target_support_end_ms:
                raise ValueError("future target content leaked before its source support")
        if not 0.0 <= float(event.alignment_confidence) <= 1.0:
            raise ValueError("alignment confidence is outside [0, 1]")
        if require_v1_rollout:
            if event.v1_source_delta is None or event.v1_source_prefix is None:
                raise ValueError("required V1 free-running rollout is missing")
            v1_text = append_text(v1_text, event.v1_source_delta, src_lang)
            if v1_text != event.v1_source_prefix:
                raise ValueError("V1 source prefix is not append-only")
        elif event.v1_source_delta is not None or event.v1_source_prefix is not None:
            if event.v1_source_delta is None or event.v1_source_prefix is None:
                raise ValueError("V1 rollout delta/prefix must both be present or absent")
        if event.source_final != (index == len(trajectory.events) - 1):
            raise ValueError("only the final event may set source_final")
        if event.target_final != (index == len(trajectory.events) - 1):
            raise ValueError("only the final event may set target_final")
        source_glm.extend(int(value) for value in event.source_glm_delta)
        target_semantic.extend(int(value) for value in event.target_semantic_delta)
        previous_source_ms = event.source_end_ms
        previous_source_pcm = event.source_pcm_end
        previous_source_word = event.gold_source_word_end
        previous_source_glm = event.source_glm_end
        previous_target_semantic = event.target_semantic_end

    if previous_source_ms != trajectory.source_duration_ms:
        raise ValueError("event timeline does not cover the full source duration")
    if source_text != trajectory.normalized_transcription:
        raise ValueError("gold source deltas do not reconstruct normalized transcription")
    if target_text != trajectory.normalized_translation:
        raise ValueError("target deltas do not reconstruct normalized translation")
    if len(source_glm) != trajectory.source_glm_length:
        raise ValueError("source GLM events do not provide full coverage")
    if hash_int_sequence(source_glm) != trajectory.source_glm_sha256:
        raise ValueError("source GLM concatenation differs from the source record")
    if len(target_semantic) != trajectory.target_semantic_length:
        raise ValueError("target semantic events do not provide full coverage")
    if hash_int_sequence(target_semantic) != trajectory.target_semantic_sha256:
        raise ValueError("target semantic concatenation differs from the source record")
    return {
        "sample_id": trajectory.sample_id,
        "events": len(trajectory.events),
        "source_glm_tokens": len(source_glm),
        "target_semantic_tokens": len(target_semantic),
        "prefinal_target_writes": sum(
            bool(event.target_semantic_delta) and not event.source_final
            for event in trajectory.events
        ),
    }


__all__ = [
    "E2ETrajectory",
    "ROLLOUT_PENDING",
    "SOURCE_SAMPLE_RATE",
    "TRAJECTORY_SCHEMA",
    "TrajectoryEvent",
    "append_text",
    "hash_int_sequence",
    "validate_trajectory",
]
