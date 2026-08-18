"""End-to-end Stage09 -> Stage10 -> Streaming BiCodec session."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

from evaluation.uniss_outputs import parse_with_tokenizer
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge import (
    replace_embedding_span,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import Stage09Config
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.model_loader import (
    Stage09Bundle,
    load_stage09_bundle,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.runtime import (
    Stage09OnlineRuntime,
)
from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write.adapter import (
    CachedMicroWriteAdapter,
    CachedWrite,
    apply_repetition_penalty,
    block_collapsed_semantic,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.sample_builders import build_performance_sample
from uniss import UniSSTokenizer
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)
from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    concatenate_audio,
    write_aligned_stereo,
    write_json,
)

from .config import Stage11Config


@dataclass
class Stage11Event:
    index: int
    source_end_ms: float
    final: bool
    policy_action: str
    ctc_text_delta: str
    qwen_text_delta: str
    qwen_structurally_valid: bool | None
    semantic_tokens: int
    semantic_unique_ratio: float
    semantic_max_run: int
    semantic_rejected_reason: str | None
    first_qwen_token_seconds: float
    qwen_seconds: float
    codec_seconds: float
    audio_samples: int
    cache_tokens: int
    wall_elapsed_ms: float


@dataclass
class Stage11Result:
    direction: str
    source_audio_path: str
    translation_audio_path: str
    timeline_audio_path: str
    stereo_audio_path: str
    result_json_path: str
    translation: str
    transcription: str
    ctc_translation: str
    source_seconds: float
    target_seconds: float
    wall_seconds: float
    first_write_ms: float | None
    first_audio_nca_ms: float | None
    first_audio_ca_ms: float | None
    valid_audio_writes: int
    rejected_writes: int
    fallback_used: bool
    fallback_reason: str | None
    events: list[Stage11Event]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["events"] = [asdict(event) for event in self.events]
        return value


@dataclass
class Stage11Update:
    status: str
    translation: str
    audio_chunk: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    event: Stage11Event | None = None
    result: Stage11Result | None = None


class Stage11Session:
    def __init__(
        self,
        engine: "Stage11Engine",
        *,
        direction: str,
        speaker_tokens: Sequence[int],
        request_dir: Path,
    ) -> None:
        self.engine = engine
        self.direction = direction
        self.request_dir = request_dir
        self.runtime = Stage09OnlineRuntime(engine.bundle, direction=direction)
        target_language = "cmn" if direction == "eng->cmn" else "eng"
        self.adapter = CachedMicroWriteAdapter(
            engine.bundle.qwen,
            engine.bundle.tokenizer,
            engine.bundle.device,
            target_language,
            speaker_tokens,
            max_write_tokens=engine.config.max_write_tokens,
        )
        self.codec = StreamingBiCodecDecoder(
            bicodec_decode_function(engine.speech_tokenizer.bicodec),
            sample_rate=SAMPLE_RATE,
            semantic_rate=50.0,
            left_context_tokens=engine.config.codec_left_context_tokens,
            holdback_tokens=engine.config.codec_holdback_tokens,
            overlap_ms=engine.config.codec_overlap_ms,
        )
        self.speaker_tokens = [int(value) for value in speaker_tokens]
        self.source_chunks: list[np.ndarray] = []
        self.audio_chunks: list[np.ndarray] = []
        self.timeline_chunks: list[tuple[int, np.ndarray]] = []
        self.events: list[Stage11Event] = []
        self.source_embeddings: list[torch.Tensor] = []
        self.started = time.perf_counter()
        self.first_write_ms: float | None = None
        self.first_audio_nca_ms: float | None = None
        self.first_audio_ca_ms: float | None = None
        self.finalized = False
        self.fallback_used = False
        self.fallback_reason: str | None = None

    def _rejection(self, write: CachedWrite) -> str | None:
        if not write.structurally_valid:
            return "invalid_structure"
        if not write.semantic_values:
            return "empty_semantic"
        if write.semantic_max_identical_run >= self.engine.config.semantic_max_run:
            return f"semantic_run:{write.semantic_max_identical_run}"
        if write.semantic_unique_ratio < self.engine.config.semantic_unique_ratio_min:
            return f"semantic_unique_ratio:{write.semantic_unique_ratio:.4f}"
        return None

    def push(self, samples: Sequence[float] | np.ndarray, *, final: bool = False):
        if self.finalized:
            raise RuntimeError("Stage11 session already finalized")
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size:
            self.source_chunks.append(values.copy())
        stage09_events = self.runtime.push_audio(values, final=final)
        for source_event in stage09_events:
            self.source_embeddings.append(source_event.qwen_speech_embeddings.detach())
            self.adapter.append_source(source_event.qwen_speech_embeddings)
            write = None
            rejection = None
            audio = np.zeros(0, dtype=np.float32)
            codec_seconds = 0.0
            if source_event.action == "WRITE":
                if self.first_write_ms is None:
                    self.first_write_ms = source_event.source_end_ms
                write = self.adapter.generate_write()
                rejection = self._rejection(write)
                if rejection is None:
                    codec_started = time.perf_counter()
                    audio = self.codec.push(
                        write.semantic_values,
                        speaker_tokens=self.speaker_tokens,
                        is_final=source_event.final,
                    )
                    if self.engine.bundle.device.type == "cuda":
                        torch.cuda.synchronize(self.engine.bundle.device)
                    codec_seconds = time.perf_counter() - codec_started
            else:
                self.adapter.commit_wait()
            wall_ms = (time.perf_counter() - self.started) * 1000.0
            if audio.size:
                if self.first_audio_nca_ms is None:
                    self.first_audio_nca_ms = source_event.source_end_ms
                    self.first_audio_ca_ms = wall_ms
                self.audio_chunks.append(audio)
                offset = int(round(source_event.source_end_ms * SAMPLE_RATE / 1000.0))
                self.timeline_chunks.append((offset, audio.copy()))
            event = Stage11Event(
                index=len(self.events),
                source_end_ms=source_event.source_end_ms,
                final=source_event.final,
                policy_action=source_event.action,
                ctc_text_delta=source_event.new_target_text,
                qwen_text_delta=write.text if write else "",
                qwen_structurally_valid=write.structurally_valid if write else None,
                semantic_tokens=len(write.semantic_values) if write else 0,
                semantic_unique_ratio=write.semantic_unique_ratio if write else 0.0,
                semantic_max_run=write.semantic_max_identical_run if write else 0,
                semantic_rejected_reason=rejection,
                first_qwen_token_seconds=write.first_token_seconds if write else 0.0,
                qwen_seconds=write.total_seconds if write else 0.0,
                codec_seconds=codec_seconds,
                audio_samples=int(audio.size),
                cache_tokens=self.adapter._cache_length(),
                wall_elapsed_ms=wall_ms,
            )
            self.events.append(event)
            yield Stage11Update(
                status=(
                    f"{source_event.source_end_ms:.0f} ms {source_event.action}; "
                    f"audio={audio.size / SAMPLE_RATE:.2f}s"
                ),
                translation=self.adapter.translation,
                audio_chunk=audio,
                event=event,
            )
        if final:
            tail = np.zeros(0, dtype=np.float32)
            if self.codec.semantic_history:
                started = time.perf_counter()
                tail = self.codec.push([], is_final=True)
                if self.engine.bundle.device.type == "cuda":
                    torch.cuda.synchronize(self.engine.bundle.device)
                if tail.size:
                    self.audio_chunks.append(tail)
                    offset = int(round(len(self.runtime.audio) / SAMPLE_RATE * SAMPLE_RATE))
                    self.timeline_chunks.append((offset, tail.copy()))
            if not self.audio_chunks:
                fallback_text, fallback_semantic = self.engine.performance_fallback(
                    self.source_embeddings,
                    self.speaker_tokens,
                    "cmn" if self.direction == "eng->cmn" else "eng",
                )
                fallback_audio = self.codec.push(
                    fallback_semantic,
                    speaker_tokens=self.speaker_tokens,
                    is_final=True,
                )
                if self.engine.bundle.device.type == "cuda":
                    torch.cuda.synchronize(self.engine.bundle.device)
                if fallback_audio.size == 0:
                    raise RuntimeError("Stage11 offline safety fallback produced no audio")
                self.audio_chunks.append(fallback_audio)
                offset = len(self.runtime.audio)
                self.timeline_chunks.append((offset, fallback_audio.copy()))
                self.adapter.generated_text_ids = self.engine.bundle.tokenizer.encode(
                    fallback_text, add_special_tokens=False
                )
                self.fallback_used = True
                self.fallback_reason = "no_accepted_online_semantic"
                if self.first_audio_nca_ms is None:
                    self.first_audio_nca_ms = len(self.runtime.audio) / 16.0
                    self.first_audio_ca_ms = (time.perf_counter() - self.started) * 1000.0
            self.finalized = True
            result = self._write_result()
            yield Stage11Update(
                status="Stage11 streaming audio complete",
                translation=result.translation,
                audio_chunk=tail,
                result=result,
            )

    def _write_result(self) -> Stage11Result:
        source = concatenate_audio(self.source_chunks)
        target = concatenate_audio(self.audio_chunks)
        if target.size == 0:
            raise RuntimeError("Stage11 produced no accepted translated audio")
        timeline_length = max(
            len(source),
            max(offset + len(chunk) for offset, chunk in self.timeline_chunks),
        )
        timeline = np.zeros(timeline_length, dtype=np.float32)
        for offset, chunk in self.timeline_chunks:
            end = offset + len(chunk)
            timeline[offset:end] += chunk
        source_path = self.request_dir / "source.wav"
        target_path = self.request_dir / "translation.wav"
        timeline_path = self.request_dir / "translation_timeline.wav"
        stereo_path = self.request_dir / "aligned_stereo.wav"
        result_path = self.request_dir / "result.json"
        sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")
        sf.write(target_path, target, SAMPLE_RATE, subtype="PCM_16")
        sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
        first_offset = self.first_audio_nca_ms or 0.0
        write_aligned_stereo(
            source,
            target,
            stereo_path,
            translation_offset_ms=first_offset,
        )
        result = Stage11Result(
            direction=self.direction,
            source_audio_path=str(source_path.resolve()),
            translation_audio_path=str(target_path.resolve()),
            timeline_audio_path=str(timeline_path.resolve()),
            stereo_audio_path=str(stereo_path.resolve()),
            result_json_path=str(result_path.resolve()),
            translation=self.adapter.translation,
            transcription=self.runtime.source_transcription,
            ctc_translation=self.runtime.committed_translation,
            source_seconds=len(source) / SAMPLE_RATE,
            target_seconds=len(target) / SAMPLE_RATE,
            wall_seconds=time.perf_counter() - self.started,
            first_write_ms=self.first_write_ms,
            first_audio_nca_ms=self.first_audio_nca_ms,
            first_audio_ca_ms=self.first_audio_ca_ms,
            valid_audio_writes=sum(event.audio_samples > 0 for event in self.events),
            rejected_writes=sum(event.semantic_rejected_reason is not None for event in self.events),
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            events=self.events,
        )
        write_json(result_path, result.to_dict())
        return result


class Stage11Engine:
    def __init__(
        self,
        stage09_config: Stage09Config | None = None,
        config: Stage11Config | None = None,
    ) -> None:
        self.stage09_config = stage09_config or Stage09Config()
        self.config = config or Stage11Config()
        self.bundle: Stage09Bundle | None = None
        self.speech_tokenizer = None
        self.text_encoder = None

    @property
    def loaded(self) -> bool:
        return self.bundle is not None and self.speech_tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        self.config.validate()
        self.bundle = load_stage09_bundle(self.stage09_config)
        self.speech_tokenizer = UniSSTokenizer.from_pretrained(
            self.config.speech_tokenizer_path,
            device=self.bundle.device,
        )
        self.text_encoder = load_hf_text_encoder(self.bundle.tokenizer)

    @torch.inference_mode()
    def performance_fallback(
        self,
        source_embeddings: Sequence[torch.Tensor],
        speaker_tokens: Sequence[int],
        target_language: str,
    ) -> tuple[str, list[int]]:
        """Final-only safety path using all accumulated Stage09 B1 embeddings."""

        if not source_embeddings:
            raise RuntimeError("offline fallback has no accumulated B1 embeddings")
        assert self.bundle is not None and self.text_encoder is not None
        speech = torch.cat([value.to(self.bundle.device) for value in source_embeddings], dim=0)
        sample = build_performance_sample(
            source_glm=[0] * len(speech),
            bicodec_global=speaker_tokens,
            tgt_lang=target_language,
            translation="fallback",
            target_bicodec=[0],
            text_encoder=self.text_encoder,
        )
        ids = torch.tensor(sample.prompt_ids, dtype=torch.long, device=self.bundle.device)
        embeddings = self.bundle.qwen.get_input_embeddings()(ids)
        span_start = sample.prompt_length - 5 - len(speech)
        embeddings = replace_embedding_span(
            embeddings,
            speech,
            span_start=span_start,
            speech_length=len(speech),
        )
        with torch.autocast(
            device_type=self.bundle.device.type,
            dtype=torch.bfloat16,
            enabled=self.bundle.device.type == "cuda",
        ):
            output = self.bundle.qwen(inputs_embeds=embeddings.unsqueeze(0), use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float()
        generated: list[int] = []
        for _ in range(1500):
            logical = apply_repetition_penalty(
                logits[:, : c.VOCAB_SIZE], generated, 1.1
            )
            logical = block_collapsed_semantic(logical, generated)
            token = int(logical.argmax(dim=-1)[0])
            generated.append(token)
            if token in {c.TOKEN_END_SEMANTIC, c.TOKEN_EOS}:
                break
            next_id = torch.tensor([[token]], device=self.bundle.device)
            with torch.autocast(
                device_type=self.bundle.device.type,
                dtype=torch.bfloat16,
                enabled=self.bundle.device.type == "cuda",
            ):
                output = self.bundle.qwen(
                    input_ids=next_id,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = output.past_key_values
            logits = output.logits[:, -1].float()
        parsed = parse_with_tokenizer(
            generated,
            mode="performance",
            tokenizer=self.bundle.tokenizer,
        )
        semantic = [int(value) for value in parsed.get("semantic_values") or []]
        if not semantic:
            raise RuntimeError("offline fallback generated no semantic tokens")
        return str(parsed.get("generated_translation") or "").strip(), semantic

    def new_session(
        self,
        *,
        direction: str,
        speaker_tokens: Sequence[int],
        request_dir: Path,
    ) -> Stage11Session:
        self.load()
        request_dir.mkdir(parents=True, exist_ok=False)
        assert self.bundle is not None and self.speech_tokenizer is not None
        return Stage11Session(
            self,
            direction=direction,
            speaker_tokens=speaker_tokens,
            request_dir=request_dir,
        )
