"""Compact, versioned dense-session schema with strict continuity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "uniss_dense_aligned_streaming_session_v1"
TICK_MS = 160


def canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("checksum", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DenseEvent:
    event_index: int
    wall_time_ms: int
    source_end_ms: int
    visible_source_token_end: int
    action: str
    playback_buffer_before_ms: int
    playback_buffer_after_ms: int
    support_bucket: int
    safe_pending_count: int
    text_delta: str = ""
    target_word_start: int = 0
    target_word_end: int = 0
    semantic_start: int = 0
    semantic_end: int = 0
    target_audio_start_ms: int = 0
    target_audio_end_ms: int = 0
    earliest_safe_ms: int = 0
    final_write: bool = False
    source_finished: bool = False

    def __post_init__(self) -> None:
        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")
        if self.wall_time_ms != (self.event_index + 1) * TICK_MS:
            raise ValueError("events must occur at exact 160ms ticks")
        if not 0 <= self.source_end_ms <= self.wall_time_ms:
            raise ValueError("source_end_ms is outside the observed wall time")
        if self.visible_source_token_end < 0:
            raise ValueError("visible source token count must be non-negative")
        if self.action not in {"READ", "WRITE"}:
            raise ValueError("action must be READ or WRITE")
        if min(
            self.playback_buffer_before_ms,
            self.playback_buffer_after_ms,
            self.support_bucket,
            self.safe_pending_count,
        ) < 0:
            raise ValueError("buffer/support fields must be non-negative")
        if self.support_bucket != min(self.safe_pending_count, 4):
            raise ValueError("support_bucket must be min(safe_pending_count, 4)")
        if self.action == "READ":
            if self.text_delta or self.semantic_end != self.semantic_start:
                raise ValueError("READ cannot carry target text or semantic audio")
            if self.final_write:
                raise ValueError("READ cannot be final_write")
        else:
            if self.semantic_end <= self.semantic_start:
                raise ValueError("WRITE must carry a non-empty semantic span")
            if self.target_audio_end_ms < self.target_audio_start_ms:
                raise ValueError("target audio span is reversed")
            if self.target_word_end < self.target_word_start:
                raise ValueError("target word span is reversed")
        if self.source_finished and self.source_end_ms <= 0:
            raise ValueError("a finished source must expose a positive prefix")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenseEvent":
        return cls(**dict(value))


@dataclass(frozen=True)
class DenseSession:
    sample_id: str
    source_manifest: str
    source_index: int
    split: str
    src_lang: str
    tgt_lang: str
    source_duration_ms: int
    target_duration_ms: int
    source_glm_length: int
    target_semantic_length: int
    target_word_count: int
    target_text: str
    speaker_global: tuple[int, ...]
    events: tuple[DenseEvent, ...]
    low_watermark_ms: int = 240
    target_buffer_ms: int = 400
    semantic_history_tokens: int = 200
    checksum: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported dense session schema")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if self.split not in {"train", "valid"}:
            raise ValueError("split must be train or valid")
        if self.src_lang not in {"eng", "cmn"} or self.tgt_lang not in {"eng", "cmn"}:
            raise ValueError("only eng/cmn are supported")
        if self.src_lang == self.tgt_lang:
            raise ValueError("source and target languages must differ")
        if min(
            self.source_duration_ms,
            self.target_duration_ms,
            self.source_glm_length,
            self.target_semantic_length,
            self.target_word_count,
        ) <= 0:
            raise ValueError("session geometry must be positive")
        if len(self.speaker_global) != 32:
            raise ValueError("speaker_global must contain exactly 32 tokens")
        if any(int(value) < 0 for value in self.speaker_global):
            raise ValueError("speaker_global contains a negative token")
        if not self.events:
            raise ValueError("dense session contains no events")
        self._validate_events()
        if self.checksum and self.checksum != canonical_checksum(self.to_dict()):
            raise ValueError("dense session checksum mismatch")

    def _validate_events(self) -> None:
        previous_source = 0
        previous_visible = 0
        previous_semantic = 0
        writes = 0
        final_writes = 0
        pieces: list[str] = []
        for index, event in enumerate(self.events):
            if event.event_index != index:
                raise ValueError("event indices are not contiguous")
            if event.source_end_ms < previous_source:
                raise ValueError("source observation moved backwards")
            if event.visible_source_token_end < previous_visible:
                raise ValueError("visible source token prefix moved backwards")
            if event.visible_source_token_end > self.source_glm_length:
                raise ValueError("visible source token prefix exceeds full source")
            if event.action == "WRITE":
                writes += 1
                if event.semantic_start != previous_semantic:
                    raise ValueError("WRITE semantic coverage has a gap or overlap")
                previous_semantic = event.semantic_end
                pieces.append(event.text_delta)
                final_writes += int(event.final_write)
            previous_source = event.source_end_ms
            previous_visible = event.visible_source_token_end
        if writes <= 0:
            raise ValueError("dense session contains no WRITE")
        if previous_semantic != self.target_semantic_length:
            raise ValueError("WRITE events do not cover the complete target semantic sequence")
        if "".join(pieces) != self.target_text:
            raise ValueError("WRITE text deltas do not reconstruct target_text exactly")
        if final_writes != 1 or not self.events[-1].final_write:
            raise ValueError("the last event must be the unique final WRITE")
        if self.events[-1].visible_source_token_end != self.source_glm_length:
            raise ValueError("final event must observe the complete source token sequence")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["speaker_global"] = list(self.speaker_global)
        value["events"] = [event.to_dict() for event in self.events]
        return value

    def with_checksum(self) -> "DenseSession":
        value = self.to_dict()
        value["checksum"] = canonical_checksum(value)
        return DenseSession.from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenseSession":
        payload = dict(value)
        payload["speaker_global"] = tuple(int(item) for item in payload["speaker_global"])
        payload["events"] = tuple(
            DenseEvent.from_dict(item) for item in payload.get("events", ())
        )
        return cls(**payload)


def visible_prefix_length(end_times_ms: Sequence[object], source_end_ms: int) -> int:
    """Return the number of source GLM tokens causally visible by source_end_ms."""

    low = 0
    high = len(end_times_ms)
    while low < high:
        middle = (low + high) // 2
        if int(end_times_ms[middle]) <= int(source_end_ms):
            low = middle + 1
        else:
            high = middle
    return low


__all__ = [
    "DenseEvent",
    "DenseSession",
    "SCHEMA_VERSION",
    "TICK_MS",
    "canonical_checksum",
    "visible_prefix_length",
]
