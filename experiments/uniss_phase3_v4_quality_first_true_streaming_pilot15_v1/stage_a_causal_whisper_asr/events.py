"""Deterministic append-only source-ASR event construction.

The source word alignment is used only to decide when already spoken words may
be committed.  The canonical ASR target is the aligned word sequence itself,
so punctuation differences from the forced aligner cannot create a rollback.
Every source GLM token is assigned to exactly one event and the final event
always closes the utterance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_MS,
)
from training.constants_uniss import normalize_language


MAX_EMPTY_EVENT_GAP_MS = 1280


@dataclass(frozen=True)
class SourceWord:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ASREvent:
    event_index: int
    source_end_ms: int
    word_start: int
    word_end: int
    glm_start: int
    glm_end: int
    delta_text: str
    committed_text: str
    is_final: bool

    @property
    def has_text_delta(self) -> bool:
        return bool(self.delta_text)


@dataclass(frozen=True)
class ASREventSession:
    sample_id: str
    src_lang: str
    source_duration_ms: int
    normalized_transcript: str
    words: tuple[SourceWord, ...]
    events: tuple[ASREvent, ...]
    source_glm_tokens: int

    @property
    def prefinal_text_commit(self) -> bool:
        return any(event.has_text_delta and not event.is_final for event in self.events)


def _word_text(value: Mapping[str, object]) -> str:
    text = " ".join(str(value.get("text") or "").strip().split())
    if not text:
        raise ValueError("source word has empty text")
    return text


def parse_source_words(values: Sequence[Mapping[str, object]], duration_ms: int) -> tuple[SourceWord, ...]:
    if duration_ms <= 0:
        raise ValueError("source duration must be positive")
    if not values:
        raise ValueError("source word alignment is empty")
    words: list[SourceWord] = []
    previous_start = -1
    previous_end = -1
    for index, value in enumerate(values):
        start = int(value.get("start_ms", -1))
        end = int(value.get("end_ms", -1))
        if start < 0 or end <= start:
            raise ValueError(f"invalid source word span at {index}: {start}..{end}")
        if start < previous_start or end < previous_end:
            raise ValueError(f"source word timestamps are non-monotonic at {index}")
        if end > duration_ms + BLOCK_MS:
            raise ValueError(f"source word ends beyond audio at {index}: {end}>{duration_ms}")
        words.append(SourceWord(_word_text(value), start, end))
        previous_start = start
        previous_end = end
    return tuple(words)


def join_words(words: Sequence[SourceWord], language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "cmn":
        return "".join(word.text.replace(" ", "") for word in words)
    return " ".join(word.text for word in words)


def _aligned_tick(milliseconds: int) -> int:
    return max(BLOCK_MS, int(math.ceil(milliseconds / BLOCK_MS)) * BLOCK_MS)


def _event_ticks(words: Sequence[SourceWord], duration_ms: int) -> tuple[int, ...]:
    final_tick = _aligned_tick(duration_ms)
    natural = {_aligned_tick(word.end_ms) for word in words}
    natural.add(final_tick)
    ticks: set[int] = set()
    previous = 0
    for current in sorted(natural):
        while current - previous > MAX_EMPTY_EVENT_GAP_MS:
            previous += MAX_EMPTY_EVENT_GAP_MS
            ticks.add(previous)
        ticks.add(current)
        previous = current
    return tuple(sorted(ticks))


def _validate_glm_times(source_glm: Sequence[object], source_glm_end_ms: Sequence[object], duration_ms: int) -> tuple[int, ...]:
    if not source_glm or len(source_glm) != len(source_glm_end_ms):
        raise ValueError("source GLM tokens/timestamps are empty or differ in length")
    times = tuple(int(value) for value in source_glm_end_ms)
    if any(value <= 0 for value in times):
        raise ValueError("source GLM end times must be positive")
    if any(left > right for left, right in zip(times, times[1:])):
        raise ValueError("source GLM end times are non-monotonic")
    if times[-1] > duration_ms + BLOCK_MS:
        raise ValueError("source GLM extends beyond the final causal block")
    return times


def build_asr_event_session(record: Mapping[str, object]) -> ASREventSession:
    sample_id = str(record.get("id") or "")
    if not sample_id:
        raise ValueError("record is missing id")
    language = normalize_language(str(record.get("src_lang") or ""))
    duration_ms = int(record.get("source_duration_ms") or 0)
    raw_words = record.get("source_words")
    if not isinstance(raw_words, list):
        raise ValueError("record is missing source_words")
    words = parse_source_words(raw_words, duration_ms)
    source_glm = record.get("source_glm")
    source_glm_end_ms = record.get("source_glm_end_ms")
    if not isinstance(source_glm, list) or not isinstance(source_glm_end_ms, list):
        raise ValueError("record is missing causal source GLM supervision")
    glm_times = _validate_glm_times(source_glm, source_glm_end_ms, duration_ms)
    transcript = join_words(words, language)
    if not transcript:
        raise ValueError("aligned transcript is empty")

    events: list[ASREvent] = []
    word_cursor = 0
    glm_cursor = 0
    committed_words: list[SourceWord] = []
    final_tick = _aligned_tick(duration_ms)
    for event_index, tick in enumerate(_event_ticks(words, duration_ms)):
        word_start = word_cursor
        while word_cursor < len(words) and words[word_cursor].end_ms <= tick:
            committed_words.append(words[word_cursor])
            word_cursor += 1
        glm_start = glm_cursor
        while glm_cursor < len(glm_times) and (glm_times[glm_cursor] <= tick or tick == final_tick):
            glm_cursor += 1
        delta = join_words(words[word_start:word_cursor], language)
        events.append(
            ASREvent(
                event_index=event_index,
                source_end_ms=tick,
                word_start=word_start,
                word_end=word_cursor,
                glm_start=glm_start,
                glm_end=glm_cursor,
                delta_text=delta,
                committed_text=join_words(committed_words, language),
                is_final=tick == final_tick,
            )
        )

    if not events or not events[-1].is_final:
        raise AssertionError("event construction did not create a final event")
    if word_cursor != len(words) or events[-1].committed_text != transcript:
        raise ValueError("event text deltas do not reconstruct the aligned transcript")
    if glm_cursor != len(source_glm):
        raise ValueError("event GLM spans do not cover the complete source sequence")
    if events[0].glm_start != 0 or events[-1].glm_end != len(source_glm):
        raise ValueError("event GLM coverage endpoints are invalid")
    for previous, current in zip(events, events[1:]):
        if previous.word_end != current.word_start or previous.glm_end != current.glm_start:
            raise ValueError("event spans contain a gap or overlap")
        if not current.committed_text.startswith(previous.committed_text):
            raise ValueError("committed ASR text rolled back")
    return ASREventSession(
        sample_id=sample_id,
        src_lang=language,
        source_duration_ms=duration_ms,
        normalized_transcript=transcript,
        words=words,
        events=tuple(events),
        source_glm_tokens=len(source_glm),
    )


__all__ = [
    "ASREvent",
    "ASREventSession",
    "MAX_EMPTY_EVENT_GAP_MS",
    "SourceWord",
    "build_asr_event_session",
    "join_words",
    "parse_source_words",
]
