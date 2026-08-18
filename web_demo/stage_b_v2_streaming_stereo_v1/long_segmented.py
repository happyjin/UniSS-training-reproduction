"""Adaptive segmented long-audio inference for the Student-v2 demo.

This path intentionally keeps long-upload recovery separate from the audited
single-session streaming path.  Each source window is processed by the frozen
Student-v2/R2 engine, failed windows are bisected, and successful target clips
are scheduled without overlap on one auditable timeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    cleanup_expired,
    concatenate_audio,
    create_request_directory,
    normalize_uploaded_audio,
    write_json,
)


@dataclass
class SegmentRecord:
    index: int
    depth: int
    source_start_seconds: float
    source_end_seconds: float
    status: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    first_write_ms: float | None = None
    first_audio_ms: float | None = None
    target_start_seconds: float | None = None
    target_end_seconds: float | None = None
    translation: str = ""
    translation_audio_path: str | None = None
    segment_result_json_path: str | None = None
    error: str | None = None


@dataclass
class SegmentedLongResult:
    request_dir: str
    source_audio_path: str
    translation_audio_path: str
    timeline_audio_path: str
    aligned_stereo_path: str
    result_json_path: str
    translation: str
    source_duration_seconds: float
    translation_duration_seconds: float
    timeline_duration_seconds: float
    total_seconds: float
    base_segment_seconds: float
    minimum_segment_seconds: float
    completed_segments: int
    failed_segments: int
    fallback_segments: int
    records: list[SegmentRecord]


@dataclass
class SegmentedLongUpdate:
    status: str
    progress: float
    result: SegmentedLongResult | None = None


def place_without_overlap(
    placements: list[tuple[int, np.ndarray]],
    waveform: np.ndarray,
    *,
    available_sample: int,
    cursor: int,
) -> tuple[int, int]:
    """Schedule one target clip no earlier than availability or prior output."""

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    start = max(int(available_sample), int(cursor))
    end = start + len(values)
    if values.size:
        placements.append((start, values))
    return start, end


def render_placements(
    placements: list[tuple[int, np.ndarray]], minimum_samples: int
) -> np.ndarray:
    total = max(
        [int(minimum_samples), *[start + len(values) for start, values in placements]]
    )
    timeline = np.zeros(total, dtype=np.float32)
    for start, values in placements:
        timeline[start : start + len(values)] = values
    return timeline


class AdaptiveLongAudioRunner:
    def __init__(
        self,
        engine,
        *,
        base_segment_seconds: float = 15.0,
        minimum_segment_seconds: float = 3.75,
    ) -> None:
        if base_segment_seconds <= 0:
            raise ValueError("base_segment_seconds must be positive")
        if not 0 < minimum_segment_seconds <= base_segment_seconds:
            raise ValueError("invalid minimum_segment_seconds")
        self.engine = engine
        self.config = engine.config
        self.base_segment_seconds = float(base_segment_seconds)
        self.minimum_segment_seconds = float(minimum_segment_seconds)

    def _run_piece(
        self,
        waveform: np.ndarray,
        *,
        direction: str,
        request_dir: Path,
        source_start_sample: int,
        source_end_sample: int,
        index_counter: list[int],
        depth: int,
    ) -> list[tuple[SegmentRecord, np.ndarray]]:
        index = index_counter[0]
        index_counter[0] += 1
        source_start_seconds = source_start_sample / SAMPLE_RATE
        source_end_seconds = source_end_sample / SAMPLE_RATE
        segment = np.asarray(
            waveform[source_start_sample:source_end_sample], dtype=np.float32
        )
        segment_dir = request_dir / "segments" / f"piece_{index:04d}_d{depth}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_path = segment_dir / "source.wav"
        sf.write(segment_path, segment, SAMPLE_RATE, subtype="PCM_16")
        try:
            final_result = None
            for update in self.engine.stream_upload(segment_path, direction=direction):
                if update.result is not None:
                    final_result = update.result
            if final_result is None:
                raise RuntimeError("segment engine returned no final result")
            target, sample_rate = sf.read(
                final_result.translation_audio_path,
                dtype="float32",
                always_2d=False,
            )
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected target sample rate: {sample_rate}")
            target = np.asarray(target, dtype=np.float32).reshape(-1)
            if target.size == 0 or not np.isfinite(target).all():
                raise RuntimeError("segment target audio is empty or invalid")
            return [
                (
                    SegmentRecord(
                        index=index,
                        depth=depth,
                        source_start_seconds=source_start_seconds,
                        source_end_seconds=source_end_seconds,
                        status="completed",
                        fallback_used=bool(final_result.fallback_used),
                        fallback_reason=final_result.fallback_reason,
                        first_write_ms=final_result.first_write_ms,
                        first_audio_ms=final_result.first_audio_ms,
                        translation=final_result.translation,
                        translation_audio_path=final_result.translation_audio_path,
                        segment_result_json_path=final_result.result_json_path,
                    ),
                    target,
                )
            ]
        except Exception as exc:
            duration = len(segment) / SAMPLE_RATE
            half = len(segment) // 2
            can_split = duration / 2 >= self.minimum_segment_seconds and half > 0
            if can_split:
                midpoint = source_start_sample + half
                return [
                    *self._run_piece(
                        waveform,
                        direction=direction,
                        request_dir=request_dir,
                        source_start_sample=source_start_sample,
                        source_end_sample=midpoint,
                        index_counter=index_counter,
                        depth=depth + 1,
                    ),
                    *self._run_piece(
                        waveform,
                        direction=direction,
                        request_dir=request_dir,
                        source_start_sample=midpoint,
                        source_end_sample=source_end_sample,
                        index_counter=index_counter,
                        depth=depth + 1,
                    ),
                ]
            return [
                (
                    SegmentRecord(
                        index=index,
                        depth=depth,
                        source_start_seconds=source_start_seconds,
                        source_end_seconds=source_end_seconds,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    np.zeros(0, dtype=np.float32),
                )
            ]

    def run(self, input_audio: str | Path, *, direction: str) -> Iterator[SegmentedLongUpdate]:
        cleanup_expired(self.config.output_root, self.config.output_ttl_hours)
        request_dir = create_request_directory(self.config.output_root)
        source_path = request_dir / "source_16k.wav"
        metadata = normalize_uploaded_audio(
            input_audio,
            source_path,
            max_upload_bytes=self.config.max_upload_bytes,
            min_audio_seconds=self.config.min_audio_seconds,
            max_audio_seconds=self.config.max_audio_seconds,
        )
        source, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"unexpected normalized sample rate: {sample_rate}")
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        segment_samples = max(1, int(round(self.base_segment_seconds * SAMPLE_RATE)))
        spans = [
            (start, min(len(source), start + segment_samples))
            for start in range(0, len(source), segment_samples)
        ]
        minimum_upload_samples = int(round(self.config.min_audio_seconds * SAMPLE_RATE))
        if (
            len(spans) >= 2
            and spans[-1][1] - spans[-1][0] < minimum_upload_samples
        ):
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()
        started = time.perf_counter()
        records_and_audio: list[tuple[SegmentRecord, np.ndarray]] = []
        index_counter = [0]
        yield SegmentedLongUpdate(
            status=f"已规范化 {metadata['duration_seconds']:.1f}s 音频，共 {len(spans)} 个基础窗口。",
            progress=0.0,
        )
        for span_index, (start, end) in enumerate(spans):
            records_and_audio.extend(
                self._run_piece(
                    source,
                    direction=direction,
                    request_dir=request_dir,
                    source_start_sample=start,
                    source_end_sample=end,
                    index_counter=index_counter,
                    depth=0,
                )
            )
            yield SegmentedLongUpdate(
                status=(
                    f"已完成基础窗口 {span_index + 1}/{len(spans)} "
                    f"（源时间 {start / SAMPLE_RATE:.1f}–{end / SAMPLE_RATE:.1f}s）"
                ),
                progress=(span_index + 1) / len(spans),
            )

        placements: list[tuple[int, np.ndarray]] = []
        continuous_chunks: list[np.ndarray] = []
        target_cursor = 0
        records: list[SegmentRecord] = []
        translations: list[str] = []
        for record, target in records_and_audio:
            records.append(record)
            if record.status != "completed" or target.size == 0:
                continue
            source_start_sample = int(round(record.source_start_seconds * SAMPLE_RATE))
            local_available_ms = record.first_audio_ms
            if local_available_ms is None:
                local_available_ms = (
                    record.source_end_seconds - record.source_start_seconds
                ) * 1000.0
            available_sample = source_start_sample + int(
                round(float(local_available_ms) * SAMPLE_RATE / 1000.0)
            )
            target_start, target_end = place_without_overlap(
                placements,
                target,
                available_sample=available_sample,
                cursor=target_cursor,
            )
            target_cursor = target_end
            record.target_start_seconds = target_start / SAMPLE_RATE
            record.target_end_seconds = target_end / SAMPLE_RATE
            continuous_chunks.append(target)
            if record.translation.strip():
                translations.append(record.translation.strip())

        continuous = concatenate_audio(continuous_chunks)
        if continuous.size == 0:
            raise RuntimeError("all adaptive segments failed; no target audio was produced")
        timeline = render_placements(placements, len(source))
        stereo_total = max(len(source), len(timeline))
        stereo = np.zeros((stereo_total, 2), dtype=np.float32)
        stereo[: len(source), 0] = source
        stereo[: len(timeline), 1] = timeline
        translation_path = request_dir / "segmented_translation.wav"
        timeline_path = request_dir / "segmented_timeline.wav"
        stereo_path = request_dir / "segmented_aligned_stereo.wav"
        result_path = request_dir / "segmented_session_summary.json"
        sf.write(translation_path, continuous, SAMPLE_RATE, subtype="PCM_16")
        sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
        sf.write(stereo_path, stereo, SAMPLE_RATE, subtype="PCM_16")
        completed = sum(record.status == "completed" for record in records)
        failed = sum(record.status == "failed" for record in records)
        fallback = sum(
            record.status == "completed" and record.fallback_used for record in records
        )
        result = SegmentedLongResult(
            request_dir=str(request_dir.resolve()),
            source_audio_path=str(source_path.resolve()),
            translation_audio_path=str(translation_path.resolve()),
            timeline_audio_path=str(timeline_path.resolve()),
            aligned_stereo_path=str(stereo_path.resolve()),
            result_json_path=str(result_path.resolve()),
            translation="\n\n".join(translations),
            source_duration_seconds=len(source) / SAMPLE_RATE,
            translation_duration_seconds=len(continuous) / SAMPLE_RATE,
            timeline_duration_seconds=len(timeline) / SAMPLE_RATE,
            total_seconds=time.perf_counter() - started,
            base_segment_seconds=self.base_segment_seconds,
            minimum_segment_seconds=self.minimum_segment_seconds,
            completed_segments=completed,
            failed_segments=failed,
            fallback_segments=fallback,
            records=records,
        )
        payload = asdict(result)
        payload["mode"] = "adaptive segmented long-audio recovery"
        payload["warning"] = (
            "This upload recovery path is segmented and may use Phase3 fallback; "
            "it is not a claim of continuous low-latency simultaneous inference."
        )
        write_json(result_path, payload)
        yield SegmentedLongUpdate(
            status=(
                f"完成：{completed} 个子段成功，{failed} 个失败，"
                f"{fallback} 个使用 Phase3 fallback。"
            ),
            progress=1.0,
            result=result,
        )
