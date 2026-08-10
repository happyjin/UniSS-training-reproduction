"""Adaptive bounded-window long-form inference around the audited V3 engine.

Each source window is at most 30 seconds, matching the native WhisperVQ
segmentation geometry.  The frozen ``iter_0008000`` short-form engine remains
unchanged and is invoked independently for every window.  Successful target
clips are placed on one monotonic global timeline; a failed window is bisected
down to a safe minimum so one difficult span does not discard a five-minute
request.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.streaming_engine import (
    EngineConfig,
    PrefixStreamingEngine,
)
from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    cleanup_expired,
    create_request_directory,
    normalize_uploaded_audio,
    write_json,
)

from .config import LongFormDemoConfig
from .windowing import (
    WindowSpan,
    place_target_without_overlap,
    plan_bounded_windows,
    render_target_timeline,
    stereo_waveform,
)


@dataclass
class LongFormWindowRecord:
    index: int
    plan_index: int
    depth: int
    source_start_seconds: float
    source_end_seconds: float
    status: str
    boundary_rms: float
    chunk_ms: int
    first_write_local_ms: float | None = None
    first_audio_local_ms: float | None = None
    first_audio_global_ms: float | None = None
    target_start_seconds: float | None = None
    target_end_seconds: float | None = None
    processing_seconds: float | None = None
    rtf: float | None = None
    wait_events: int = 0
    write_events: int = 0
    committed_text_tokens: int = 0
    semantic_tokens: int = 0
    semantic_per_text_token: float = 0.0
    semantic_rejections: list[str] = field(default_factory=list)
    retry_reason: str | None = None
    translation: str = ""
    translation_audio_path: str | None = None
    window_result_path: str | None = None
    error: str | None = None

    @property
    def source_duration_seconds(self) -> float:
        return self.source_end_seconds - self.source_start_seconds


@dataclass
class LongFormResult:
    request_dir: str
    source_path: str
    translation_path: str
    timeline_path: str
    stereo_path: str
    result_path: str
    direction: str
    chunk_ms: int
    selected_iteration: int
    source_duration_seconds: float
    translation_duration_seconds: float
    timeline_duration_seconds: float
    processing_seconds: float
    rtf: float
    first_audio_global_ms: float | None
    translation: str
    planned_windows: int
    completed_windows: int
    failed_windows: int
    retry_windows: int
    maximum_observed_window_seconds: float
    bounded_window: bool = True
    pseudo_streaming: bool = True
    records: list[LongFormWindowRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LongFormUpdate:
    status: str
    progress: float
    translation: str = ""
    result: LongFormResult | None = None


class BoundedLongFormEngine:
    def __init__(
        self,
        config: LongFormDemoConfig,
        base_engine: PrefixStreamingEngine | object | None = None,
    ) -> None:
        self.config = config
        self.base_engine = base_engine or PrefixStreamingEngine(
            EngineConfig(
                adapter_dir=config.adapter_dir,
                speech_tokenizer_dir=config.speech_tokenizer_dir,
                output_root=config.output_root / "window_runs",
                device=config.device,
                chunk_ms=480,
                max_upload_bytes=config.max_upload_bytes,
                max_audio_seconds=config.maximum_window_seconds + 0.05,
            )
        )

    def load(self) -> None:
        load = getattr(self.base_engine, "load", None)
        if load is not None:
            load()

    def _run_piece(
        self,
        waveform: np.ndarray,
        *,
        direction: str,
        chunk_ms: int,
        request_dir: Path,
        plan_index: int,
        boundary_rms: float,
        start_sample: int,
        end_sample: int,
        depth: int,
        counter: list[int],
        retry_reason: str | None = None,
    ) -> list[tuple[LongFormWindowRecord, np.ndarray]]:
        index = counter[0]
        counter[0] += 1
        start_seconds = start_sample / SAMPLE_RATE
        end_seconds = end_sample / SAMPLE_RATE
        record = LongFormWindowRecord(
            index=index,
            plan_index=plan_index,
            depth=depth,
            source_start_seconds=start_seconds,
            source_end_seconds=end_seconds,
            status="running",
            boundary_rms=boundary_rms,
            chunk_ms=chunk_ms,
            retry_reason=retry_reason,
        )
        piece_dir = request_dir / "windows" / f"window_{index:04d}_d{depth}"
        piece_dir.mkdir(parents=True, exist_ok=False)
        piece_path = piece_dir / "source.wav"
        sf.write(
            piece_path,
            np.asarray(waveform[start_sample:end_sample], dtype=np.float32),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        try:
            final = None
            for update in self.base_engine.stream(
                piece_path, direction=direction, chunk_ms=chunk_ms
            ):
                if update.result is not None:
                    final = update.result
            if final is None:
                raise RuntimeError("window engine returned no final result")
            target, target_rate = sf.read(
                final.translation_path, dtype="float32", always_2d=False
            )
            target = np.asarray(target, dtype=np.float32).reshape(-1)
            if target_rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected target sample rate: {target_rate}")
            if target.size == 0 or not np.isfinite(target).all():
                raise RuntimeError("window target audio is empty or invalid")
            record.status = "completed"
            record.first_write_local_ms = final.first_write_source_ms
            record.first_audio_local_ms = final.first_audio_source_ms
            if final.first_audio_source_ms is not None:
                record.first_audio_global_ms = (
                    start_seconds * 1000.0 + final.first_audio_source_ms
                )
            record.processing_seconds = final.processing_seconds
            record.rtf = final.rtf
            record.wait_events = final.wait_events
            record.write_events = final.write_events
            record.committed_text_tokens = final.committed_text_tokens
            record.semantic_tokens = final.semantic_tokens
            record.semantic_per_text_token = final.semantic_tokens / max(
                final.committed_text_tokens, 1
            )
            record.semantic_rejections = sorted(
                {
                    str(event.semantic_rejected_reason)
                    for event in getattr(final, "events", [])
                    if getattr(event, "semantic_rejected_reason", None)
                }
            )
            record.translation = final.translation.strip()
            record.translation_audio_path = final.translation_path
            record.window_result_path = final.result_path
            maximum_text_tokens = int(
                getattr(getattr(self.base_engine, "config", None), "max_text_tokens", 160)
            )
            violations: list[str] = []
            if final.committed_text_tokens >= maximum_text_tokens:
                violations.append(
                    f"text_token_saturation:{final.committed_text_tokens}>="
                    f"{maximum_text_tokens}"
                )
            if (
                final.committed_text_tokens >= 16
                and record.semantic_per_text_token < 1.5
            ):
                violations.append(
                    "semantic_coverage:"
                    f"{record.semantic_per_text_token:.4f}<1.5"
                )
            if violations:
                raise RuntimeError("window_quality_gate:" + ",".join(violations))
            return [(record, target)]
        except Exception as exc:
            duration = (end_sample - start_sample) / SAMPLE_RATE
            half = (end_sample - start_sample) // 2
            can_retry = (
                duration / 2.0 >= self.config.minimum_retry_seconds and half > 0
            )
            if can_retry:
                midpoint = start_sample + half
                reason = f"{type(exc).__name__}: {exc}"
                return [
                    *self._run_piece(
                        waveform,
                        direction=direction,
                        chunk_ms=chunk_ms,
                        request_dir=request_dir,
                        plan_index=plan_index,
                        boundary_rms=boundary_rms,
                        start_sample=start_sample,
                        end_sample=midpoint,
                        depth=depth + 1,
                        counter=counter,
                        retry_reason=reason,
                    ),
                    *self._run_piece(
                        waveform,
                        direction=direction,
                        chunk_ms=chunk_ms,
                        request_dir=request_dir,
                        plan_index=plan_index,
                        boundary_rms=boundary_rms,
                        start_sample=midpoint,
                        end_sample=end_sample,
                        depth=depth + 1,
                        counter=counter,
                        retry_reason=reason,
                    ),
                ]
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            return [(record, np.zeros(0, dtype=np.float32))]

    def run(
        self, input_audio: str | Path, *, direction: str, chunk_ms: int
    ) -> Iterator[LongFormUpdate]:
        if int(chunk_ms) not in {320, 480, 640}:
            raise ValueError("chunk_ms must be one of 320, 480 or 640")
        cleanup_expired(self.config.output_root, self.config.output_ttl_hours)
        request_dir = create_request_directory(self.config.output_root / f"chunk_{chunk_ms}ms")
        source_path = request_dir / "source_16k.wav"
        metadata = normalize_uploaded_audio(
            input_audio,
            source_path,
            max_upload_bytes=self.config.max_upload_bytes,
            min_audio_seconds=0.5,
            max_audio_seconds=self.config.max_audio_seconds,
        )
        source, source_rate = sf.read(source_path, dtype="float32", always_2d=False)
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        if source_rate != SAMPLE_RATE:
            raise RuntimeError(f"unexpected normalized sample rate: {source_rate}")
        spans = plan_bounded_windows(
            source,
            SAMPLE_RATE,
            target_seconds=self.config.target_window_seconds,
            minimum_seconds=self.config.minimum_window_seconds,
            maximum_seconds=self.config.maximum_window_seconds,
            search_seconds=self.config.boundary_search_seconds,
        )
        yield LongFormUpdate(
            status=(
                f"已规范化 {metadata['duration_seconds']:.1f}s；规划 {len(spans)} 个"
                f" {self.config.minimum_window_seconds:.0f}–"
                f"{self.config.maximum_window_seconds:.0f}s 有界窗口。"
            ),
            progress=0.0,
        )

        started = time.perf_counter()
        counter = [0]
        completed_text: list[str] = []
        records_and_audio: list[tuple[LongFormWindowRecord, np.ndarray]] = []
        for plan_position, span in enumerate(spans):
            pieces = self._run_piece(
                source,
                direction=direction,
                chunk_ms=int(chunk_ms),
                request_dir=request_dir,
                plan_index=span.index,
                boundary_rms=span.boundary_rms,
                start_sample=span.start_sample,
                end_sample=span.end_sample,
                depth=0,
                counter=counter,
                retry_reason=None,
            )
            records_and_audio.extend(pieces)
            completed_text.extend(
                record.translation
                for record, _ in pieces
                if record.status == "completed" and record.translation
            )
            yield LongFormUpdate(
                status=(
                    f"窗口 {plan_position + 1}/{len(spans)} 完成 · "
                    f"源时间 {span.start_sample / SAMPLE_RATE:.1f}–"
                    f"{span.end_sample / SAMPLE_RATE:.1f}s"
                ),
                progress=(plan_position + 1) / len(spans),
                translation="\n\n".join(completed_text),
            )

        placements: list[tuple[int, np.ndarray]] = []
        continuous_parts: list[np.ndarray] = []
        target_cursor = 0
        translations: list[str] = []
        records: list[LongFormWindowRecord] = []
        for record, target in records_and_audio:
            records.append(record)
            if record.status != "completed" or target.size == 0:
                continue
            local_available_ms = record.first_audio_local_ms
            if local_available_ms is None:
                local_available_ms = record.source_duration_seconds * 1000.0
            available = int(
                round(
                    (
                        record.source_start_seconds
                        + float(local_available_ms) / 1000.0
                    )
                    * SAMPLE_RATE
                )
            )
            target_start, target_end = place_target_without_overlap(
                placements,
                target,
                available_sample=available,
                cursor=target_cursor,
            )
            target_cursor = target_end
            record.target_start_seconds = target_start / SAMPLE_RATE
            record.target_end_seconds = target_end / SAMPLE_RATE
            record.first_audio_global_ms = record.target_start_seconds * 1000.0
            continuous_parts.append(target)
            if record.translation:
                translations.append(record.translation)

        failed = sum(record.status == "failed" for record in records)
        if failed:
            errors = "; ".join(
                f"window {record.index}: {record.error}"
                for record in records
                if record.status == "failed"
            )
            raise RuntimeError(f"{failed} long-form windows failed: {errors}")
        if not continuous_parts:
            raise RuntimeError("long-form inference produced no target audio")

        continuous = np.concatenate(continuous_parts).astype(np.float32, copy=False)
        timeline = render_target_timeline(placements, len(source))
        stereo = stereo_waveform(source, timeline)
        translation_path = request_dir / "translation_continuous.wav"
        timeline_path = request_dir / "translation_global_timeline.wav"
        stereo_path = request_dir / "stereo_left_source_right_translation.wav"
        result_path = request_dir / "result.json"
        sf.write(translation_path, continuous, SAMPLE_RATE, subtype="PCM_16")
        sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
        sf.write(stereo_path, stereo, SAMPLE_RATE, subtype="PCM_16")

        processing = time.perf_counter() - started
        selected = -1
        manifest = getattr(self.base_engine, "adapter_manifest", None)
        if manifest:
            selected = int(manifest.get("selected_iteration", -1))
        completed = sum(record.status == "completed" for record in records)
        retries = sum(record.depth > 0 for record in records)
        first_audio = min(
            (
                record.first_audio_global_ms
                for record in records
                if record.first_audio_global_ms is not None
            ),
            default=None,
        )
        result = LongFormResult(
            request_dir=str(request_dir.resolve()),
            source_path=str(source_path.resolve()),
            translation_path=str(translation_path.resolve()),
            timeline_path=str(timeline_path.resolve()),
            stereo_path=str(stereo_path.resolve()),
            result_path=str(result_path.resolve()),
            direction=direction,
            chunk_ms=int(chunk_ms),
            selected_iteration=selected,
            source_duration_seconds=len(source) / SAMPLE_RATE,
            translation_duration_seconds=len(continuous) / SAMPLE_RATE,
            timeline_duration_seconds=len(timeline) / SAMPLE_RATE,
            processing_seconds=processing,
            rtf=processing / max(len(source) / SAMPLE_RATE, 1e-9),
            first_audio_global_ms=first_audio,
            translation="\n\n".join(translations),
            planned_windows=len(spans),
            completed_windows=completed,
            failed_windows=0,
            retry_windows=retries,
            maximum_observed_window_seconds=max(
                record.source_duration_seconds for record in records
            ),
            records=records,
        )
        payload = result.to_dict()
        payload["mode"] = "adaptive bounded-window five-minute pseudo-streaming"
        payload["window_plan"] = [span.to_dict(SAMPLE_RATE) for span in spans]
        payload["model_state_policy"] = {
            "phase3_adapter": "iter_0008000",
            "window_translation_state": "reset per bounded source window",
            "global_state": "translation text, target timeline, and non-overlap cursor",
            "speaker_state": "extracted independently per source window",
        }
        payload["claim_boundary"] = (
            "Online-visibility simulation with bounded recomputation; not a causal cached encoder."
        )
        write_json(result_path, payload)
        yield LongFormUpdate(
            status=(
                f"完成：{completed} 个窗口全部成功 · "
                f"RTF={result.rtf:.3f} · 总耗时={processing:.1f}s"
            ),
            progress=1.0,
            translation=result.translation,
            result=result,
        )
