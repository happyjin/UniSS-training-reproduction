"""Convert an already aligned Stage-A record into a lossless E2E trajectory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    ROLLOUT_PENDING,
    SOURCE_SAMPLE_RATE,
    TrajectoryEvent,
    append_text,
    hash_int_sequence,
    validate_trajectory,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.events import (
    ASREvent,
    build_asr_event_session,
)
from training.constants_uniss import normalize_language


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def audit_pcm_audio(path: Path, *, expected_frames: int) -> tuple[int, int, bool]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    if int(info.samplerate) != SOURCE_SAMPLE_RATE:
        raise ValueError(f"source audio sample rate is not 16 kHz: {path}")
    if int(info.channels) != 1:
        raise ValueError(f"source audio is not mono: {path}")
    if int(info.frames) != int(expected_frames):
        raise ValueError(
            f"source audio frames differ from duration: {info.frames}!={expected_frames}: {path}"
        )
    finite = True
    for block in sf.blocks(str(path), blocksize=262_144, dtype="float32", always_2d=False):
        if not bool(np.isfinite(block).all()):
            finite = False
            break
    if not finite:
        raise ValueError(f"source audio contains NaN/Inf: {path}")
    return int(info.frames), int(info.channels), finite


def _normal_target_text(record: Mapping[str, object]) -> str:
    language = normalize_language(str(record["tgt_lang"]))
    raw_words = record.get("target_words")
    if not isinstance(raw_words, list) or not raw_words:
        raise ValueError("record has no aligned target words")
    value = ""
    for item in raw_words:
        if not isinstance(item, Mapping):
            raise ValueError("target word entry is malformed")
        text = " ".join(str(item.get("text") or "").strip().split())
        if not text:
            raise ValueError("target word has empty text")
        value = append_text(value, text, language)
    return value


def _validate_micro_writes(record: Mapping[str, object]) -> list[dict[str, object]]:
    raw = record.get("micro_write_events")
    target = record.get("target_bicodec")
    if not isinstance(raw, list) or not raw:
        raise ValueError("record has no micro-WRITE events")
    if not isinstance(target, list) or not target:
        raise ValueError("record has no target BiCodec semantic sequence")
    events = [dict(value) for value in raw]
    semantic_cursor = 0
    target_word_cursor = 0
    for index, event in enumerate(events):
        if int(event.get("micro_write_index", -1)) != index:
            raise ValueError("micro-WRITE indices are not contiguous")
        start = int(event.get("semantic_start", -1))
        end = int(event.get("semantic_end", -1))
        if start != semantic_cursor or end <= start or end > len(target):
            raise ValueError("micro-WRITE semantic spans contain a gap, overlap, or invalid end")
        if int(event.get("semantic_count", -1)) != end - start:
            raise ValueError("micro-WRITE semantic_count differs from its span")
        word_start = int(event.get("target_word_start", -1))
        word_end = int(event.get("target_word_end", -1))
        continuation = bool(event.get("semantic_continuation", False))
        if continuation:
            if word_start > target_word_cursor or word_end < target_word_cursor:
                raise ValueError("semantic continuation references a non-committed target word")
        elif word_start != target_word_cursor or word_end <= word_start:
            raise ValueError("micro-WRITE target word spans contain a gap or overlap")
        target_word_cursor = max(target_word_cursor, word_end)
        if not bool(event.get("future_monotonic_support", False)):
            raise ValueError("micro-WRITE event is not future-monotonic")
        semantic_cursor = end
    if semantic_cursor != len(target):
        raise ValueError("micro-WRITE events do not cover the full target semantic sequence")
    target_words = record.get("target_words")
    if not isinstance(target_words, list) or target_word_cursor != len(target_words):
        raise ValueError("micro-WRITE events do not cover every aligned target word")
    return events


def _alignment_confidence(record: Mapping[str, object], writes: Sequence[Mapping[str, object]]) -> float:
    support = record.get("target_support")
    if not isinstance(support, list):
        return 1.0
    word_indices: set[int] = set()
    for write in writes:
        word_indices.update(
            range(int(write["target_word_start"]), int(write["target_word_end"]))
        )
    values = [
        float(dict(support[index]).get("alignment_confidence", 1.0))
        for index in sorted(word_indices)
        if 0 <= index < len(support)
    ]
    return min(values) if values else 1.0


def _source_events_until(
    source_events: Sequence[ASREvent],
    cursor: int,
    source_end_ms: int,
    source_final: bool,
) -> tuple[list[ASREvent], int]:
    selected: list[ASREvent] = []
    while cursor < len(source_events):
        event = source_events[cursor]
        if not source_final and event.source_end_ms > source_end_ms:
            break
        selected.append(event)
        cursor += 1
    return selected, cursor


def build_gold_trajectory(
    record: Mapping[str, object],
    *,
    split: str,
    source_manifest: str,
    source_manifest_record: int,
    v1_checkpoint_sha256: str,
    phase3_teacher_sha256: str,
    hash_audio: bool = False,
    audit_audio: bool = False,
) -> E2ETrajectory:
    if not bool(record.get("formal_a45_pass")) or not bool(record.get("formal_a68_pass")):
        raise ValueError("record did not pass the formal A4-A8 alignment gates")
    session = build_asr_event_session(record)
    writes = _validate_micro_writes(record)
    source_audio = Path(str(record["source_audio"]))
    if not source_audio.is_file():
        raise FileNotFoundError(source_audio)
    source_audio_sha256 = sha256_file(source_audio) if hash_audio else None
    source_glm = tuple(int(value) for value in record["source_glm"])  # type: ignore[index]
    target_semantic = tuple(int(value) for value in record["target_bicodec"])  # type: ignore[index]
    duration_ms = int(record["source_duration_ms"])
    if duration_ms <= 0:
        raise ValueError("source duration must be positive")
    expected_frames = duration_ms * SOURCE_SAMPLE_RATE // 1000
    if audit_audio:
        source_audio_frames, source_audio_channels, source_audio_finite = audit_pcm_audio(
            source_audio, expected_frames=expected_frames
        )
    else:
        source_audio_frames = expected_frames
        source_audio_channels = None
        source_audio_finite = None

    ticks = {
        min(duration_ms, int(event.source_end_ms)) for event in session.events
    }
    ticks.update(
        min(duration_ms, int(event["safe_if_source_ms_gte"])) for event in writes
    )
    ticks.add(duration_ms)
    ordered_ticks = sorted(value for value in ticks if value > 0)
    if not ordered_ticks or ordered_ticks[-1] != duration_ms:
        raise AssertionError("trajectory ticks do not end at source duration")

    events: list[TrajectoryEvent] = []
    source_cursor = 0
    source_glm_cursor = 0
    source_word_cursor = 0
    source_prefix = ""
    write_cursor = 0
    target_semantic_cursor = 0
    target_prefix = ""
    previous_ms = 0
    for event_index, tick in enumerate(ordered_ticks):
        source_final = event_index == len(ordered_ticks) - 1
        selected_source, source_cursor = _source_events_until(
            session.events, source_cursor, tick, source_final
        )
        gold_delta = ""
        for source_event in selected_source:
            gold_delta = append_text(gold_delta, source_event.delta_text, session.src_lang)
        source_prefix = append_text(source_prefix, gold_delta, session.src_lang)
        source_word_end = (
            selected_source[-1].word_end if selected_source else source_word_cursor
        )
        source_glm_end = (
            selected_source[-1].glm_end if selected_source else source_glm_cursor
        )

        selected_writes: list[dict[str, object]] = []
        while write_cursor < len(writes):
            write = writes[write_cursor]
            if not source_final and int(write["safe_if_source_ms_gte"]) > tick:
                break
            selected_writes.append(write)
            write_cursor += 1
        target_delta = ""
        target_support_end_ms: int | None = None
        semantic_end = target_semantic_cursor
        for write in selected_writes:
            target_delta = append_text(target_delta, str(write.get("text") or ""), str(record["tgt_lang"]))
            start = int(write["semantic_start"])
            end = int(write["semantic_end"])
            if start != semantic_end:
                raise ValueError("selected micro-WRITEs lost semantic contiguity")
            semantic_end = end
            support_ms = int(write["support_end_ms"])
            target_support_end_ms = (
                support_ms
                if target_support_end_ms is None
                else max(target_support_end_ms, support_ms)
            )
        target_prefix = append_text(target_prefix, target_delta, str(record["tgt_lang"]))
        semantic_delta = target_semantic[target_semantic_cursor:semantic_end]
        events.append(
            TrajectoryEvent(
                event_index=event_index,
                source_start_ms=previous_ms,
                source_end_ms=tick,
                source_pcm_start=previous_ms * SOURCE_SAMPLE_RATE // 1000,
                source_pcm_end=tick * SOURCE_SAMPLE_RATE // 1000,
                source_glm_start=source_glm_cursor,
                source_glm_end=source_glm_end,
                source_glm_delta=source_glm[source_glm_cursor:source_glm_end],
                gold_source_word_start=source_word_cursor,
                gold_source_word_end=source_word_end,
                gold_source_delta=gold_delta,
                gold_source_prefix=source_prefix,
                v1_source_delta=None,
                v1_source_prefix=None,
                target_text_delta=target_delta,
                target_text_prefix=target_prefix,
                target_semantic_start=target_semantic_cursor,
                target_semantic_end=semantic_end,
                target_semantic_delta=semantic_delta,
                target_support_end_ms=target_support_end_ms,
                source_final=source_final,
                target_final=source_final,
                alignment_confidence=_alignment_confidence(record, selected_writes),
                noise_severity=ROLLOUT_PENDING,
            )
        )
        previous_ms = tick
        source_word_cursor = source_word_end
        source_glm_cursor = source_glm_end
        target_semantic_cursor = semantic_end

    if source_cursor != len(session.events) or write_cursor != len(writes):
        raise ValueError("final event did not flush all source/target events")
    trajectory = E2ETrajectory(
        sample_id=session.sample_id,
        split=str(split),
        src_lang=session.src_lang,
        tgt_lang=normalize_language(str(record["tgt_lang"])),
        source_audio=str(source_audio.resolve()),
        source_audio_sha256=source_audio_sha256,
        source_audio_hash_status="complete" if hash_audio else "deferred",
        source_audio_frames=source_audio_frames,
        source_audio_channels=source_audio_channels,
        source_audio_finite=source_audio_finite,
        source_audio_audit_status="complete" if audit_audio else "deferred",
        source_sample_rate=SOURCE_SAMPLE_RATE,
        source_duration_ms=duration_ms,
        speaker_global=tuple(int(value) for value in record["bicodec_global"]),  # type: ignore[index]
        full_transcription=str(record["transcription"]),
        normalized_transcription=session.normalized_transcript,
        full_translation=str(record["translation"]),
        normalized_translation=_normal_target_text(record),
        source_glm_length=len(source_glm),
        source_glm_sha256=hash_int_sequence(source_glm),
        target_semantic_length=len(target_semantic),
        target_semantic_sha256=hash_int_sequence(target_semantic),
        source_manifest=str(Path(source_manifest).resolve()),
        source_manifest_record=int(source_manifest_record),
        v1_checkpoint_sha256=str(v1_checkpoint_sha256),
        phase3_teacher_sha256=str(phase3_teacher_sha256),
        v1_rollout_status=ROLLOUT_PENDING,
        events=tuple(events),
    )
    validate_trajectory(
        trajectory,
        require_audio_hash=hash_audio,
        require_audio_audit=audit_audio,
    )
    return trajectory


__all__ = ["audit_pcm_audio", "build_gold_trajectory", "sha256_file"]
