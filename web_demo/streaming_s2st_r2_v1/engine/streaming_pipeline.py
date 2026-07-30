"""End-to-end upload replay and microphone prefix pseudo-streaming pipeline."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf
import torch

from training.simul_uniss.schema import chunk_spans, tokens_per_chunk
from uniss import UniSSTokenizer
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)
from uniss.streaming.policy import PolicyDecision

from ..audio_io import (
    SAMPLE_RATE,
    cleanup_expired,
    concatenate_audio,
    create_request_directory,
    normalize_uploaded_audio,
    write_json,
)
from ..config import StreamingDemoConfig
from ..session_manager import BrowserSession
from .prefix_frontend import CumulativePrefixFrontend
from .qwen_live_adapter import QwenLiveAdapter


DIRECTION_TO_LANGUAGES = {
    "中文 → 英文": ("cmn", "eng"),
    "英文 → 中文": ("eng", "cmn"),
}


@dataclass
class StreamingEvent:
    event_index: int
    source_end_ms: float
    source_is_final: bool
    action: str
    raw_action_token_id: int
    forced_reason: str | None
    action_seconds: float
    generated_text: str = ""
    generated_text_ids: list[int] = field(default_factory=list)
    generated_semantic_values: list[int] = field(default_factory=list)
    write_seconds: float = 0.0
    codec_seconds: float = 0.0
    audio_samples: int = 0
    write_structurally_valid: bool | None = None


@dataclass
class StreamingResult:
    request_dir: str
    mode: str
    direction: str
    model_label: str
    source_audio_path: str
    translation_audio_path: str
    timeline_audio_path: str
    aligned_stereo_path: str
    result_json_path: str
    translation: str
    source_duration_seconds: float
    translation_duration_seconds: float
    total_seconds: float
    first_write_ms: float | None
    first_audio_ms: float | None
    forced_actions: int
    structural_recoveries: int
    max_prompt_tokens: int
    training_context_exceeded: bool
    events: list[StreamingEvent]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["events"] = [asdict(event) for event in self.events]
        return value


@dataclass
class StreamingUpdate:
    status: str
    translation: str
    event: StreamingEvent | None = None
    audio_chunk: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    result: StreamingResult | None = None
    frontend: dict[str, object] | None = None


@dataclass
class LiveEngineState:
    direction: str
    request_dir: Path
    frontend: CumulativePrefixFrontend
    adapter: QwenLiveAdapter | None = None
    codec: StreamingBiCodecDecoder | None = None
    speaker_tokens: list[int] | None = None
    next_boundary_samples: int = 0
    appended_committed_tokens: int = 0
    events: list[StreamingEvent] = field(default_factory=list)
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    total_frontend_seconds: float = 0.0
    first_write_ms: float | None = None
    first_audio_ms: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    finalized: bool = False


class StreamingDemoEngine:
    """One frozen model and one serialized GPU runtime for the public demo."""

    def __init__(self, config: StreamingDemoConfig):
        config.validate()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.speech_tokenizer: UniSSTokenizer | None = None
        self.lock = threading.Lock()
        self.loaded_at: float | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.speech_tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.speech_tokenizer = UniSSTokenizer.from_pretrained(
            self.config.speech_tokenizer_path,
            device=self.device,
        )
        self.loaded_at = time.time()

    def _target_language(self, direction: str) -> str:
        try:
            return DIRECTION_TO_LANGUAGES[direction][1]
        except KeyError as exc:
            raise ValueError(f"Unsupported translation direction: {direction!r}") from exc

    def _adapter(self, direction: str, speaker_tokens: Sequence[int]) -> QwenLiveAdapter:
        assert self.model is not None and self.tokenizer is not None
        return QwenLiveAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            target_language=self._target_language(direction),
            speaker_tokens=speaker_tokens,
            max_write_tokens=self.config.max_write_tokens,
            max_model_len=self.config.max_model_len,
            training_context_limit=self.config.training_context_limit,
            repetition_penalty=self.config.repetition_penalty,
        )

    def _codec(self) -> StreamingBiCodecDecoder:
        assert self.speech_tokenizer is not None
        return StreamingBiCodecDecoder(
            bicodec_decode_function(self.speech_tokenizer.bicodec),
            sample_rate=SAMPLE_RATE,
            semantic_rate=50.0,
            left_context_tokens=self.config.codec_left_context_tokens,
            holdback_tokens=self.config.codec_holdback_tokens,
            overlap_ms=self.config.codec_overlap_ms,
        )

    def _run_event(
        self,
        adapter: QwenLiveAdapter,
        codec: StreamingBiCodecDecoder,
        *,
        event_index: int,
        source_end_ms: float,
        source_is_final: bool,
        speaker_tokens: Sequence[int],
    ) -> tuple[StreamingEvent, np.ndarray]:
        action = adapter.choose_action(is_final=source_is_final)
        assert adapter.last_action is not None
        event = StreamingEvent(
            event_index=event_index,
            source_end_ms=source_end_ms,
            source_is_final=source_is_final,
            action=action.value,
            raw_action_token_id=adapter.last_action.raw_token_id,
            forced_reason=adapter.last_action.forced_reason,
            action_seconds=adapter.last_action.seconds,
        )
        if action == PolicyDecision.WAIT:
            adapter.commit_wait()
            return event, np.zeros(0, dtype=np.float32)
        write = adapter.generate_write(is_final=source_is_final)
        assert adapter.last_write is not None
        event.generated_text = adapter.last_write.text
        event.generated_text_ids = list(write.target_text_ids)
        event.generated_semantic_values = list(write.semantic_tokens)
        event.write_seconds = adapter.last_write.seconds
        event.write_structurally_valid = adapter.last_write.structurally_valid
        started = time.perf_counter()
        waveform = codec.push(
            write.semantic_tokens,
            speaker_tokens=speaker_tokens,
            is_final=source_is_final,
        )
        if source_is_final and waveform.size == 0 and codec.semantic_history:
            waveform = codec.push([], is_final=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        event.codec_seconds = time.perf_counter() - started
        event.audio_samples = int(waveform.size)
        return event, waveform

    @staticmethod
    def _timeline_audio(
        event_chunks: Sequence[tuple[float, np.ndarray]], sample_rate: int = SAMPLE_RATE
    ) -> np.ndarray:
        pieces: list[tuple[int, np.ndarray]] = []
        cursor = 0
        for source_end_ms, raw in event_chunks:
            chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue
            start = max(cursor, int(round(source_end_ms * sample_rate / 1000.0)))
            pieces.append((start, chunk))
            cursor = start + len(chunk)
        timeline = np.zeros(cursor, dtype=np.float32)
        for start, chunk in pieces:
            timeline[start : start + len(chunk)] = chunk
        return timeline

    @staticmethod
    def _write_stereo(source: np.ndarray, target_timeline: np.ndarray, path: Path) -> None:
        total = max(len(source), len(target_timeline))
        stereo = np.zeros((total, 2), dtype=np.float32)
        stereo[: len(source), 0] = source
        stereo[: len(target_timeline), 1] = target_timeline
        sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_16")

    def stream_upload(self, input_audio: str | Path, *, direction: str) -> Iterator[StreamingUpdate]:
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
        with self.lock:
            self.load()
            assert self.speech_tokenizer is not None
            started = time.perf_counter()
            yield StreamingUpdate("正在将完整上传音频编码为评估兼容 source token…", "")
            linguistic_tokens, bicodec_tokens = self.speech_tokenizer.tokenize(source_path)
            source_glm = [int(value) for value in linguistic_tokens]
            source_bicodec = [int(value) for value in bicodec_tokens]
            if not source_glm or len(source_bicodec) < 33:
                raise RuntimeError("speech tokenizer returned incomplete GLM/BiCodec tokens")
            speaker_tokens = source_bicodec[:32]
            adapter = self._adapter(direction, speaker_tokens)
            codec = self._codec()
            events: list[StreamingEvent] = []
            event_chunks: list[tuple[float, np.ndarray]] = []
            first_write_ms: float | None = None
            first_audio_ms: float | None = None
            spans = chunk_spans(len(source_glm), tokens_per_chunk(self.config.chunk_ms))
            source_duration_ms = float(metadata["duration_seconds"]) * 1000.0
            for index, span in enumerate(spans):
                is_final = index == len(spans) - 1
                adapter.append_source(source_glm[span.start : span.end])
                source_end_ms = (
                    source_duration_ms
                    if is_final
                    else min(source_duration_ms, (index + 1) * self.config.chunk_ms)
                )
                event, waveform = self._run_event(
                    adapter,
                    codec,
                    event_index=index,
                    source_end_ms=source_end_ms,
                    source_is_final=is_final,
                    speaker_tokens=speaker_tokens,
                )
                events.append(event)
                if event.action == "write" and first_write_ms is None:
                    first_write_ms = source_end_ms
                if waveform.size:
                    event_chunks.append((source_end_ms, waveform))
                    if first_audio_ms is None:
                        first_audio_ms = source_end_ms
                yield StreamingUpdate(
                    status=(
                        f"事件 {index + 1}/{len(spans)} · {event.action.upper()} · "
                        f"source={source_end_ms:.0f} ms"
                    ),
                    translation=adapter.translation,
                    event=event,
                    audio_chunk=waveform,
                )
            translation_audio = concatenate_audio([chunk for _, chunk in event_chunks])
            if translation_audio.size == 0:
                raise RuntimeError("R2/R3 completed without a decodable translation waveform")
            source_waveform, _ = sf.read(source_path, dtype="float32", always_2d=False)
            target_timeline = self._timeline_audio(event_chunks)
            translation_path = request_dir / "translation.wav"
            timeline_path = request_dir / "translation_timeline.wav"
            stereo_path = request_dir / "aligned_stereo.wav"
            sf.write(translation_path, translation_audio, SAMPLE_RATE, subtype="PCM_16")
            sf.write(timeline_path, target_timeline, SAMPLE_RATE, subtype="PCM_16")
            self._write_stereo(source_waveform, target_timeline, stereo_path)
            result_path = request_dir / "session_summary.json"
            total_seconds = time.perf_counter() - started
            result = StreamingResult(
                request_dir=str(request_dir.resolve()),
                mode="evaluation-compatible replay (pseudo-streaming)",
                direction=direction,
                model_label=self.config.model_label,
                source_audio_path=str(source_path.resolve()),
                translation_audio_path=str(translation_path.resolve()),
                timeline_audio_path=str(timeline_path.resolve()),
                aligned_stereo_path=str(stereo_path.resolve()),
                result_json_path=str(result_path.resolve()),
                translation=adapter.translation,
                source_duration_seconds=float(metadata["duration_seconds"]),
                translation_duration_seconds=len(translation_audio) / SAMPLE_RATE,
                total_seconds=total_seconds,
                first_write_ms=first_write_ms,
                first_audio_ms=first_audio_ms,
                forced_actions=adapter.forced_actions,
                structural_recoveries=adapter.structural_recoveries,
                max_prompt_tokens=adapter.max_prompt_tokens,
                training_context_exceeded=adapter.training_context_exceeded,
                events=events,
            )
            payload = result.to_dict()
            manifest = self.config.export_manifest()
            payload["model_manifest"] = manifest
            payload["model_sha256"] = manifest["weight_files"]["model.safetensors"][
                "sha256"
            ]
            payload["config"] = {
                "chunk_ms": self.config.chunk_ms,
                "repetition_penalty": self.config.repetition_penalty,
                "codec_left_context_tokens": self.config.codec_left_context_tokens,
                "codec_holdback_tokens": self.config.codec_holdback_tokens,
                "codec_overlap_ms": self.config.codec_overlap_ms,
            }
            write_json(result_path, payload)
            yield StreamingUpdate(
                "完成：右侧时间线音频保留模型实际 WAIT/WRITE 起点。",
                adapter.translation,
                result=result,
            )

    def translate_upload(self, input_audio: str | Path, *, direction: str) -> StreamingResult:
        result = None
        for update in self.stream_upload(input_audio, direction=direction):
            result = update.result or result
        if result is None:
            raise RuntimeError("upload streaming pipeline did not produce a final result")
        return result

    def _new_live_state(self, session: BrowserSession, direction: str) -> LiveEngineState:
        assert self.speech_tokenizer is not None
        request_dir = session.ensure_request_dir()
        state = LiveEngineState(
            direction=direction,
            request_dir=request_dir,
            frontend=CumulativePrefixFrontend(
                self.speech_tokenizer,
                holdback_tokens=self.config.stable_prefix_holdback_tokens,
            ),
            next_boundary_samples=int(round(self.config.chunk_ms * SAMPLE_RATE / 1000.0)),
        )
        session.engine_state = state
        return state

    def process_microphone(
        self,
        session: BrowserSession,
        audio_chunk: tuple[int, np.ndarray] | None,
        *,
        direction: str,
        is_final: bool = False,
    ) -> list[StreamingUpdate]:
        """Process one Gradio microphone chunk; state remains server-side by session id."""

        with self.lock:
            self.load()
            appended = session.ingress.append(audio_chunk)
            del appended
            if session.cancelled:
                raise RuntimeError("microphone session was cancelled")
            state = session.engine_state
            if state is None:
                state = self._new_live_state(session, direction)
            if not isinstance(state, LiveEngineState):
                raise TypeError("browser session contains incompatible engine state")
            if state.direction != direction:
                raise ValueError("translation direction cannot change during a live session")
            if state.finalized:
                return [StreamingUpdate("麦克风会话已经完成。", state.adapter.translation if state.adapter else "")]
            waveform = session.ingress.waveform
            if waveform.size == 0:
                return [StreamingUpdate("等待麦克风音频。", "")]
            boundaries: list[tuple[int, bool]] = []
            while state.next_boundary_samples <= len(waveform):
                boundaries.append((state.next_boundary_samples, False))
                state.next_boundary_samples += int(
                    round(self.config.chunk_ms * SAMPLE_RATE / 1000.0)
                )
            if is_final and (not boundaries or boundaries[-1][0] != len(waveform)):
                boundaries.append((len(waveform), True))
            elif is_final and boundaries:
                boundaries[-1] = (boundaries[-1][0], True)
            updates: list[StreamingUpdate] = []
            for end_sample, final_step in boundaries:
                prefix = waveform[:end_sample]
                frontend = state.frontend.encode(prefix, is_final=final_step)
                state.total_frontend_seconds += frontend.encode_seconds
                speaker_capture_samples = int(3.2 * SAMPLE_RATE)
                if state.speaker_tokens is None and (
                    end_sample >= speaker_capture_samples or final_step
                ):
                    state.speaker_tokens = state.frontend.extract_speaker_tokens(
                        prefix,
                        state.request_dir / "speaker_prefix.wav",
                    )
                    state.adapter = self._adapter(direction, state.speaker_tokens)
                    state.codec = self._codec()
                if state.adapter is not None:
                    new_committed = frontend.committed_tokens[state.appended_committed_tokens :]
                    state.adapter.append_source(new_committed)
                    state.appended_committed_tokens = len(frontend.committed_tokens)
                    event, target_audio = self._run_event(
                        state.adapter,
                        state.codec,
                        event_index=len(state.events),
                        source_end_ms=end_sample * 1000.0 / SAMPLE_RATE,
                        source_is_final=final_step,
                        speaker_tokens=state.speaker_tokens,
                    )
                    state.events.append(event)
                    if event.action == "write" and state.first_write_ms is None:
                        state.first_write_ms = event.source_end_ms
                    if target_audio.size:
                        state.audio_chunks.append(target_audio)
                        if state.first_audio_ms is None:
                            state.first_audio_ms = event.source_end_ms
                    updates.append(
                        StreamingUpdate(
                            status=(
                                f"Online pseudo-streaming · {event.action.upper()} · "
                                f"source={event.source_end_ms:.0f} ms"
                            ),
                            translation=state.adapter.translation,
                            event=event,
                            audio_chunk=target_audio,
                            frontend={
                                "candidate_tokens": len(frontend.candidate_tokens),
                                "committed_tokens": len(frontend.committed_tokens),
                                "revision_events": frontend.revision_events,
                                "frontend_seconds": frontend.encode_seconds,
                                "frontend_rtf": state.total_frontend_seconds
                                / max(end_sample / SAMPLE_RATE, 1e-6),
                            },
                        )
                    )
                else:
                    updates.append(
                        StreamingUpdate(
                            "正在收集目标音色并确认稳定 WhisperVQ 前缀…",
                            "",
                            frontend={
                                "candidate_tokens": len(frontend.candidate_tokens),
                                "committed_tokens": len(frontend.committed_tokens),
                                "revision_events": frontend.revision_events,
                            },
                        )
                    )
            if is_final:
                state.finalized = True
                if state.adapter is None or state.codec is None:
                    raise RuntimeError("microphone audio ended before runtime initialization")
                source_path = state.request_dir / "source_16k.wav"
                sf.write(source_path, waveform, SAMPLE_RATE, subtype="PCM_16")
                translation_audio = concatenate_audio(state.audio_chunks)
                if translation_audio.size == 0:
                    raise RuntimeError("live session produced no decodable translation audio")
                translation_path = state.request_dir / "translation.wav"
                timeline_path = state.request_dir / "translation_timeline.wav"
                stereo_path = state.request_dir / "aligned_stereo.wav"
                sf.write(translation_path, translation_audio, SAMPLE_RATE, subtype="PCM_16")
                event_chunks = [
                    (event.source_end_ms, chunk)
                    for event, chunk in zip(
                        [event for event in state.events if event.audio_samples > 0],
                        state.audio_chunks,
                    )
                ]
                target_timeline = self._timeline_audio(event_chunks)
                sf.write(timeline_path, target_timeline, SAMPLE_RATE, subtype="PCM_16")
                self._write_stereo(waveform, target_timeline, stereo_path)
                result_path = state.request_dir / "session_summary.json"
                result = StreamingResult(
                    request_dir=str(state.request_dir.resolve()),
                    mode="online WhisperVQ cumulative-prefix pseudo-streaming",
                    direction=direction,
                    model_label=self.config.model_label,
                    source_audio_path=str(source_path.resolve()),
                    translation_audio_path=str(translation_path.resolve()),
                    timeline_audio_path=str(timeline_path.resolve()),
                    aligned_stereo_path=str(stereo_path.resolve()),
                    result_json_path=str(result_path.resolve()),
                    translation=state.adapter.translation,
                    source_duration_seconds=len(waveform) / SAMPLE_RATE,
                    translation_duration_seconds=len(translation_audio) / SAMPLE_RATE,
                    total_seconds=time.perf_counter() - state.started_at,
                    first_write_ms=state.first_write_ms,
                    first_audio_ms=state.first_audio_ms,
                    forced_actions=state.adapter.forced_actions,
                    structural_recoveries=state.adapter.structural_recoveries,
                    max_prompt_tokens=state.adapter.max_prompt_tokens,
                    training_context_exceeded=state.adapter.training_context_exceeded,
                    events=state.events,
                )
                write_json(
                    result_path,
                    {
                        **result.to_dict(),
                        "source_frontend": "WhisperVQ cumulative-prefix re-encoding",
                        "frontend_total_seconds": state.total_frontend_seconds,
                        "prefix_revision_events": state.frontend.committer.revision_events,
                    },
                )
                updates.append(
                    StreamingUpdate(
                        "麦克风同传完成并已 final flush。",
                        state.adapter.translation,
                        result=result,
                    )
                )
            return updates or [
                StreamingUpdate(
                    f"已接收 {session.ingress.duration_seconds:.2f}s；等待下一个 640 ms 边界。",
                    state.adapter.translation if state.adapter else "",
                )
            ]
