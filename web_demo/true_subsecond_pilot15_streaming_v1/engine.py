"""End-to-end PCM -> causal WhisperVQ -> Qwen KV -> BiCodec streaming engine."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.inference.scheduler import (
    DeadlineScheduler,
)
from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
    StreamingWhisperVQTeacher,
)
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)
from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    cleanup_expired,
    concatenate_audio,
    create_request_directory,
    normalize_uploaded_audio,
    write_json,
)

from .causal_frontend import BoundedCausalWhisperVQFrontend
from .config import DemoConfig
from .model_loader import load_runtime_models
from .qwen_runtime import IncrementalQwenRuntime


DIRECTIONS = {
    "中文 → 英文": ("cmn", "eng"),
    "英文 → 中文": ("eng", "cmn"),
    "zh-en": ("cmn", "eng"),
    "en-zh": ("eng", "cmn"),
}


@dataclass
class StreamEvent:
    index: int
    source_end_ms: int
    is_final: bool
    speech_active: bool
    frontend_window_start_ms: int
    frontend_stable_end_ms: int
    frontend_seconds: float
    new_source_codes: int
    total_source_codes: int
    write_probability: float
    support_bucket: int
    action: str
    action_reason: str
    deadline_forced: bool
    new_text: str = ""
    new_text_tokens: int = 0
    safe_probabilities: list[float] = field(default_factory=list)
    semantic_tokens: int = 0
    quality_rejected_reason: str | None = None
    emitted_audio_samples: int = 0
    compute_seconds: float = 0.0


@dataclass
class StreamResult:
    request_dir: str
    source_path: str
    translation_path: str
    timeline_path: str
    stereo_path: str
    result_path: str
    direction: str
    selected_iteration: int
    decision_chunk_ms: int
    acoustic_chunk_ms: int
    acoustic_right_context_ms: int
    frontend_window_ms: int
    source_duration_seconds: float
    translation_duration_seconds: float
    processing_seconds: float
    rtf: float
    first_write_source_ms: int | None
    first_useful_audio_source_ms: int | None
    first_useful_audio_wall_ms: float | None
    first_write_compute_ms: float | None
    write_to_pcm_ms: float | None
    committed_translation: str
    committed_text_tokens: int
    semantic_tokens: int
    natural_writes: int
    forced_writes: int
    wait_events: int
    empty_after_write: int
    committed_revision_violations: int
    maximum_frontend_buffer_ms: float
    speaker_observed_ms: int
    speaker_reference_seconds: float
    semantic_unique_ratio: float
    semantic_max_identical_run: int
    translation_coverage_ratio: float
    quality_passed: bool
    quality_failures: list[str] = field(default_factory=list)
    true_input_streaming: bool = True
    bounded_frontend_memory: bool = True
    qwen_kv_cache: bool = True
    uploaded_file_realtime_replay: bool = True
    browser_webrtc_live: bool = False
    events: list[StreamEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StreamUpdate:
    status: str
    translation: str = ""
    progress: float = 0.0
    audio_chunk: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    event: StreamEvent | None = None
    result: StreamResult | None = None


def timeline_audio(chunks: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    placements: list[tuple[int, np.ndarray]] = []
    cursor = 0
    for source_end_ms, raw in chunks:
        chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not len(chunk):
            continue
        start = max(cursor, int(round(source_end_ms * SAMPLE_RATE / 1000)))
        placements.append((start, chunk))
        cursor = start + len(chunk)
    result = np.zeros(cursor, dtype=np.float32)
    for start, chunk in placements:
        result[start : start + len(chunk)] = chunk
    return result


def stereo_waveform(source: np.ndarray, target_timeline: np.ndarray) -> np.ndarray:
    left = np.asarray(source, dtype=np.float32).reshape(-1)
    right = np.asarray(target_timeline, dtype=np.float32).reshape(-1)
    total = max(len(left), len(right))
    result = np.zeros((total, 2), dtype=np.float32)
    result[: len(left), 0] = left
    result[: len(right), 1] = right
    return result


def speech_active(chunk: np.ndarray, *, threshold: float = 0.003) -> bool:
    values = np.asarray(chunk, dtype=np.float32).reshape(-1)
    return bool(len(values) and float(np.sqrt(np.mean(values * values))) >= threshold)


def active_speech_ms(
    chunk: np.ndarray, *, frame_ms: int = 20, threshold: float = 0.006
) -> int:
    values = np.asarray(chunk, dtype=np.float32).reshape(-1)
    frame_samples = max(1, SAMPLE_RATE * frame_ms // 1000)
    active = 0
    for start in range(0, len(values), frame_samples):
        frame = values[start : start + frame_samples]
        if len(frame) and float(np.sqrt(np.mean(frame * frame))) >= threshold:
            active += len(frame)
    return int(round(active * 1000 / SAMPLE_RATE))


def vad_speaker_reference(
    source: np.ndarray,
    *,
    warmup_ms: int,
    frame_ms: int,
    threshold: float,
    minimum_reference_ms: int = 400,
) -> tuple[np.ndarray, int]:
    """Select ordered voiced frames from an observed-only speaker warm-up."""

    values = np.asarray(source, dtype=np.float32).reshape(-1)
    observed_samples = min(len(values), SAMPLE_RATE * warmup_ms // 1000)
    observed = values[:observed_samples]
    frame_samples = max(1, SAMPLE_RATE * frame_ms // 1000)
    frames: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(observed), frame_samples):
        frame = observed[start : start + frame_samples]
        if not len(frame):
            continue
        rms = float(np.sqrt(np.mean(frame * frame)))
        frames.append((rms, frame))
    selected = [frame for rms, frame in frames if rms >= threshold]
    minimum_samples = min(
        len(observed), SAMPLE_RATE * minimum_reference_ms // 1000
    )
    if sum(len(frame) for frame in selected) < minimum_samples:
        # Very quiet recordings still need a non-empty reference. Select the
        # highest-energy observed frames, then restore chronological order.
        ranked = sorted(enumerate(frames), key=lambda item: item[1][0], reverse=True)
        chosen: set[int] = set()
        total = 0
        for index, (_, frame) in ranked:
            chosen.add(index)
            total += len(frame)
            if total >= minimum_samples:
                break
        selected = [frame for index, (_, frame) in enumerate(frames) if index in chosen]
    reference = concatenate_audio(selected)
    if not len(reference):
        raise RuntimeError("speaker warm-up contains no usable audio")
    return reference, observed_samples


def maximum_identical_run(values: Sequence[int]) -> int:
    best = current = 0
    previous: int | None = None
    for raw in values:
        value = int(raw)
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


class TrueSubsecondStreamingEngine:
    def __init__(self, config: DemoConfig) -> None:
        config.validate_assets(require_export=True)
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.whisper = None
        self.model = None
        self.tokenizer = None
        self.objective = None
        self.manifest = None
        self.bicodec: BiCodecTokenizer | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.whisper = StreamingWhisperVQTeacher(
            self.config.whispervq_dir,
            device=str(self.device),
            chunk_ms=self.config.acoustic_chunk_ms,
            right_context_ms=self.config.acoustic_right_context_ms,
        )
        (
            self.model,
            self.tokenizer,
            self.objective,
            self.manifest,
            _,
        ) = load_runtime_models(
            self.config.exported_runtime,
            codebook_weight=self.whisper.model.codebook.weight,
            device=self.device,
        )
        self.bicodec = BiCodecTokenizer(
            model_dir=self.config.speech_tokenizer_dir / "bicodec",
            device=self.device,
        )

    def _speaker_tokens(self, prefix: np.ndarray, path: Path) -> list[int]:
        assert self.bicodec is not None
        sf.write(path, np.asarray(prefix, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
        tokens = self.bicodec.encode_wav_to_tokens(str(path)).detach().reshape(-1).cpu()
        values = [int(value) for value in tokens[:32].tolist()]
        if len(values) != 32:
            raise RuntimeError(f"BiCodec returned {len(values)} speaker tokens")
        return values

    def _codec(self, speaker: Sequence[int]) -> StreamingBiCodecDecoder:
        assert self.bicodec is not None
        codec = StreamingBiCodecDecoder(
            bicodec_decode_function(self.bicodec),
            sample_rate=SAMPLE_RATE,
            semantic_rate=50.0,
            left_context_tokens=50,
            holdback_tokens=5,
            overlap_ms=80.0,
        )
        codec.set_speaker_tokens(speaker)
        return codec

    def stream(
        self,
        input_audio: str | Path,
        *,
        direction: str,
        decision_chunk_ms: int | None = None,
    ) -> Iterator[StreamUpdate]:
        if direction not in DIRECTIONS:
            raise ValueError(f"unsupported direction: {direction}")
        active_chunk_ms = int(decision_chunk_ms or self.config.decision_chunk_ms)
        if active_chunk_ms not in {320, 480, 640}:
            raise ValueError("decision chunk must be 320, 480 or 640ms")
        source_lang, target_lang = DIRECTIONS[direction]
        del source_lang
        self.load()
        assert self.whisper is not None
        assert self.model is not None and self.tokenizer is not None
        assert self.objective is not None and self.manifest is not None

        cleanup_expired(self.config.output_root, 48.0)
        request = create_request_directory(
            self.config.output_root / f"chunk_{active_chunk_ms}ms"
        )
        source_path = request / "source_16k.wav"
        metadata = normalize_uploaded_audio(
            input_audio,
            source_path,
            max_upload_bytes=self.config.max_upload_bytes,
            min_audio_seconds=0.32,
            max_audio_seconds=self.config.max_audio_seconds,
        )
        source, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"normalized source has unexpected sample rate {sample_rate}")
        step_samples = active_chunk_ms * SAMPLE_RATE // 1000
        speaker_reference, speaker_end = vad_speaker_reference(
            source,
            warmup_ms=self.config.speaker_warmup_ms,
            frame_ms=self.config.speaker_vad_frame_ms,
            threshold=self.config.speaker_vad_min_rms,
        )
        sf.write(
            request / "speaker_observed_prefix.wav",
            source[:speaker_end],
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        speaker = self._speaker_tokens(
            speaker_reference, request / "speaker_vad_reference.wav"
        )
        boundaries = list(range(step_samples, len(source), step_samples)) + [len(source)]
        boundaries = sorted(set(end for end in boundaries if end >= speaker_end))
        if not boundaries:
            boundaries = [len(source)]
        frontend = BoundedCausalWhisperVQFrontend(
            self.whisper,
            chunk_ms=self.config.acoustic_chunk_ms,
            right_context_ms=self.config.acoustic_right_context_ms,
            window_ms=self.config.frontend_window_ms,
        )
        qwen = IncrementalQwenRuntime(
            self.model,
            self.tokenizer,
            self.objective,
            target_lang=target_lang,
            speaker_global=speaker,
            device=self.device,
            semantic_history_tokens=self.config.semantic_history_tokens,
            seed=self.config.seed,
        )
        codec = self._codec(speaker)
        scheduler = DeadlineScheduler(
            soft_deadline_ms=self.config.soft_deadline_ms,
            hard_deadline_ms=self.config.hard_deadline_ms,
            write_threshold=0.5,
            minimum_commit_tokens=1,
            maximum_commit_tokens=self.config.max_text_tokens_per_write,
        )

        events: list[StreamEvent] = []
        audio_chunks: list[tuple[int, np.ndarray]] = []
        semantic_count = 0
        elapsed_since_write = 0
        previous = 0
        first_write: int | None = None
        first_audio: int | None = None
        first_audio_wall: float | None = None
        first_write_compute: float | None = None
        empty_after_write = 0
        started = time.perf_counter()
        yield StreamUpdate(
            status=(
                f"iter_{int(self.manifest['selected_iteration']):07d} 已载入；"
                f"先用前 {speaker_end * 1000 // SAMPLE_RATE}ms 已到达音频完成 VAD speaker warm-up，"
                f"随后严格按 {active_chunk_ms}ms 到达顺序回放。"
            )
        )

        for index, end in enumerate(boundaries):
            event_started = time.perf_counter()
            final = end == len(source)
            pcm = source[previous:end]
            previous = end
            active_ms = active_speech_ms(
                pcm,
                frame_ms=self.config.speaker_vad_frame_ms,
                threshold=self.config.speaker_vad_min_rms,
            )
            active = active_ms > 0
            elapsed_since_write += active_ms
            front = frontend.push(pcm, is_final=final)
            qwen.append_source_codes(front.new_tokens)
            observation = qwen.observe_policy()
            supported = observation.support_bucket
            decision = scheduler.decide(
                elapsed_speech_ms=elapsed_since_write,
                write_probability=observation.write_probability,
                supported_tokens=supported,
                speech_active=active,
                final=final,
            )
            event = StreamEvent(
                index=index,
                source_end_ms=front.source_end_ms,
                is_final=final,
                speech_active=active,
                frontend_window_start_ms=front.window_start_ms,
                frontend_stable_end_ms=front.stable_end_ms,
                frontend_seconds=front.encode_seconds,
                new_source_codes=len(front.new_tokens),
                total_source_codes=front.committed_tokens,
                write_probability=observation.write_probability,
                support_bucket=observation.support_bucket,
                action=decision.action,
                action_reason=decision.reason,
                deadline_forced=decision.deadline_forced,
            )
            emitted = np.zeros(0, dtype=np.float32)
            if decision.action == "WRITE":
                if first_write is None:
                    first_write = front.source_end_ms
                    first_write_compute = (time.perf_counter() - started) * 1000.0
                budget = max(1, decision.commit_tokens or supported)
                write = qwen.micro_write(
                    observation,
                    maximum_text_tokens=min(
                        budget, self.config.max_text_tokens_per_write
                    ),
                    semantic_block_tokens=self.config.semantic_block_tokens,
                    forced=decision.deadline_forced,
                )
                event.new_text = write.text
                event.new_text_tokens = len(write.text_ids)
                event.safe_probabilities = list(write.safe_probabilities)
                event.semantic_tokens = len(write.semantic_ids)
                event.quality_rejected_reason = write.quality_rejected_reason
                if write.semantic_ids:
                    emitted = codec.push(write.semantic_ids, is_final=False)
                    semantic_count += len(write.semantic_ids)
                if not write.text_ids and not write.semantic_ids:
                    empty_after_write += 1
                if len(emitted):
                    audio_chunks.append((front.source_end_ms, emitted))
                    if first_audio is None:
                        first_audio = front.source_end_ms
                        first_audio_wall = (time.perf_counter() - started) * 1000.0
                elapsed_since_write = 0
            if final and semantic_count:
                tail = codec.push([], is_final=True)
                if len(tail):
                    emitted = concatenate_audio((emitted, tail))
                    audio_chunks.append((front.source_end_ms, tail))
                    if first_audio is None:
                        first_audio = front.source_end_ms
                        first_audio_wall = (time.perf_counter() - started) * 1000.0
            event.emitted_audio_samples = len(emitted)
            event.compute_seconds = time.perf_counter() - event_started
            events.append(event)
            yield StreamUpdate(
                status=(
                    f"{index + 1}/{len(boundaries)} · source={front.source_end_ms}ms · "
                    f"{decision.action} ({decision.reason}) · "
                    f"p(write)={observation.write_probability:.3f} · "
                    f"support={observation.support_bucket} · +audio={len(emitted) / SAMPLE_RATE:.2f}s"
                ),
                translation=qwen.committed_text,
                progress=(index + 1) / len(boundaries),
                audio_chunk=emitted,
                event=event,
            )

        processing = time.perf_counter() - started
        translation = concatenate_audio([chunk for _, chunk in audio_chunks])
        timeline = timeline_audio(audio_chunks)
        stereo = stereo_waveform(source, timeline)
        translation_path = request / "translation_continuous.wav"
        timeline_path = request / "translation_timeline.wav"
        stereo_path = request / "stereo_left_source_right_translation.wav"
        result_path = request / "result.json"
        sf.write(translation_path, translation, SAMPLE_RATE, subtype="PCM_16")
        sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
        sf.write(stereo_path, stereo, SAMPLE_RATE, subtype="PCM_16")
        semantic_values = qwen.committed_semantic_ids
        semantic_unique_ratio = (
            len(set(semantic_values)) / len(semantic_values) if semantic_values else 0.0
        )
        semantic_max_run = maximum_identical_run(semantic_values)
        source_seconds = float(metadata["duration_seconds"])
        coverage_ratio = (len(translation) / SAMPLE_RATE) / max(source_seconds, 1e-9)
        natural_writes = sum(
            event.action == "WRITE" and not event.deadline_forced for event in events
        )
        forced_writes = sum(event.deadline_forced for event in events)
        quality_failures: list[str] = []
        if not qwen.committed_text_ids or not semantic_values:
            quality_failures.append("no_safe_streaming_translation")
        if natural_writes == 0:
            quality_failures.append("no_natural_write")
        if forced_writes > natural_writes:
            quality_failures.append("forced_write_dominant")
        if semantic_values and semantic_max_run >= 16:
            quality_failures.append(f"semantic_identical_run:{semantic_max_run}")
        if len(semantic_values) >= 64 and semantic_unique_ratio < 0.10:
            quality_failures.append(
                f"semantic_unique_ratio:{semantic_unique_ratio:.4f}"
            )
        if coverage_ratio < 0.35:
            quality_failures.append(f"audio_coverage:{coverage_ratio:.4f}")
        result = StreamResult(
            request_dir=str(request.resolve()),
            source_path=str(source_path.resolve()),
            translation_path=str(translation_path.resolve()),
            timeline_path=str(timeline_path.resolve()),
            stereo_path=str(stereo_path.resolve()),
            result_path=str(result_path.resolve()),
            direction=direction,
            selected_iteration=int(self.manifest["selected_iteration"]),
            decision_chunk_ms=active_chunk_ms,
            acoustic_chunk_ms=self.config.acoustic_chunk_ms,
            acoustic_right_context_ms=self.config.acoustic_right_context_ms,
            frontend_window_ms=self.config.frontend_window_ms,
            source_duration_seconds=source_seconds,
            translation_duration_seconds=len(translation) / SAMPLE_RATE,
            processing_seconds=processing,
            rtf=processing / max(float(metadata["duration_seconds"]), 1e-9),
            first_write_source_ms=first_write,
            first_useful_audio_source_ms=first_audio,
            first_useful_audio_wall_ms=first_audio_wall,
            first_write_compute_ms=first_write_compute,
            write_to_pcm_ms=(
                float(first_audio - first_write)
                if first_audio is not None and first_write is not None
                else None
            ),
            committed_translation=qwen.committed_text,
            committed_text_tokens=len(qwen.committed_text_ids),
            semantic_tokens=semantic_count,
            natural_writes=natural_writes,
            forced_writes=forced_writes,
            wait_events=sum(event.action == "READ" for event in events),
            empty_after_write=empty_after_write,
            committed_revision_violations=frontend.committed_revision_violations,
            maximum_frontend_buffer_ms=(
                frontend.maximum_buffer_samples * 1000.0 / SAMPLE_RATE
            ),
            speaker_observed_ms=int(round(speaker_end * 1000 / SAMPLE_RATE)),
            speaker_reference_seconds=len(speaker_reference) / SAMPLE_RATE,
            semantic_unique_ratio=semantic_unique_ratio,
            semantic_max_identical_run=semantic_max_run,
            translation_coverage_ratio=coverage_ratio,
            quality_passed=not quality_failures,
            quality_failures=quality_failures,
            events=events,
        )
        payload = result.to_dict()
        payload["runtime_contract"] = {
            "future_pcm_visible_to_model": False,
            "frontend": "bounded-causal WhisperVQ 160ms + 80ms right context",
            "frontend_recompute_window_ms": self.config.frontend_window_ms,
            "qwen": "append-only KV cache with per-WRITE branch copy",
            "codec": "append-only semantic history with bounded left context",
            "five_minute_supported": self.config.max_audio_seconds >= 300,
        }
        write_json(result_path, payload)
        quality_status = "质量门通过" if result.quality_passed else (
            "质量门失败：" + ", ".join(result.quality_failures)
        )
        yield StreamUpdate(
            status=(
                f"完成 · First WRITE={first_write if first_write is not None else 'N/A'}ms · "
                f"First audio={first_audio if first_audio is not None else 'N/A'}ms · "
                f"RTF={result.rtf:.3f} · {quality_status}"
            ),
            translation=result.committed_translation,
            progress=1.0,
            result=result,
        )
