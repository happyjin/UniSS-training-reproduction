"""Three-head inference matching the full198 prefix-streaming v3 objectives.

Runtime order:
  1. cumulative WhisperVQ/GLM prefix encoding;
  2. the trained WAIT/WRITE action pair;
  3. streaming S2TT full-prefix hypothesis with irreversible stable commit;
  4. streaming TTS semantic-block generation and incremental BiCodec decode.

This is deliberately source-side pseudo-streaming.  The acoustic prefix is
re-encoded cumulatively because the frozen WhisperVQ frontend is not causal.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_prefix_streaming_full198_v1 import builders
from training import constants_uniss as c
from uniss import UniSSTokenizer
from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder, bicodec_decode_function
from uniss.streaming.stable_prefix import StablePrefixCommitter
from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    create_request_directory,
    normalize_uploaded_audio,
    write_json,
)
from web_demo.streaming_s2st_r2_v1.engine.prefix_frontend import CumulativePrefixFrontend

from .metrics import latency_metrics
from .model_loader import load_model_and_tokenizer


DIRECTIONS = {
    "zh-en": ("cmn", "eng"),
    "en-zh": ("eng", "cmn"),
    "中文 → 英文": ("cmn", "eng"),
    "英文 → 中文": ("eng", "cmn"),
}


@dataclass(frozen=True)
class EngineConfig:
    adapter_dir: Path
    speech_tokenizer_dir: Path
    output_root: Path
    device: str = "cuda:0"
    chunk_ms: int = 480
    frontend_bootstrap_ms: int = 3200
    stable_glm_holdback: int = 2
    stable_text_holdback: int = 2
    semantic_block_tokens: int = 64
    semantic_history_tokens: int = 200
    final_semantic_tokens_per_text_token: int = 12
    max_final_tts_blocks: int = 8
    codec_left_context_tokens: int = 50
    codec_holdback_tokens: int = 5
    codec_overlap_ms: float = 80.0
    max_text_tokens: int = 160
    max_upload_bytes: int = 100 * 1024 * 1024
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 60.0
    seed: int = 20260809

    def validate(self) -> None:
        if self.chunk_ms not in {320, 480, 640}:
            raise ValueError("chunk_ms must be one of 320, 480 or 640")
        if self.frontend_bootstrap_ms < self.chunk_ms:
            raise ValueError("frontend bootstrap cannot be shorter than one chunk")
        for path in (self.adapter_dir, self.speech_tokenizer_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)


@dataclass
class StreamEvent:
    index: int
    source_end_ms: float
    is_final: bool
    candidate_glm_tokens: int
    committed_glm_tokens: int
    new_glm_tokens: int
    frontend_seconds: float
    action: str
    wait_logit: float
    write_logit: float
    forced_write: bool
    action_seconds: float
    candidate_text: str = ""
    candidate_text_tokens: int = 0
    committed_text: str = ""
    new_text_tokens: int = 0
    text_seconds: float = 0.0
    semantic_tokens: int = 0
    semantic_invalid_tokens: int = 0
    semantic_unique_ratio: float = 0.0
    semantic_max_run: int = 0
    semantic_rejected_reason: str | None = None
    tts_seconds: float = 0.0
    codec_seconds: float = 0.0
    emitted_audio_samples: int = 0


@dataclass
class StreamResult:
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
    processing_seconds: float
    rtf: float
    first_write_source_ms: float | None
    first_audio_source_ms: float | None
    first_audio_wall_ms: float | None
    finalization_lag_ms: float | None
    translation: str
    committed_text_tokens: int
    semantic_tokens: int
    wait_events: int
    write_events: int
    frontend_revision_events: int
    al_ms: float | None
    laal_ms: float | None
    ap: float | None
    pseudo_streaming: bool
    events: list[StreamEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StreamUpdate:
    status: str
    translation: str
    event: StreamEvent | None = None
    audio_chunk: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    result: StreamResult | None = None


def _maximum_run(values: Sequence[int]) -> int:
    best = current = 0
    previous: int | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def _semantic_rejection(values: Sequence[int], invalid: int) -> str | None:
    if invalid:
        return f"invalid_semantic_tokens:{invalid}"
    if not values:
        return "empty_semantic"
    ratio = len(set(values)) / len(values)
    run = _maximum_run(values)
    if len(values) >= 16 and run >= 8:
        return f"semantic_identical_run:{run}"
    if len(values) >= 16 and ratio < 0.10:
        return f"semantic_unique_ratio:{ratio:.4f}"
    return None


def write_stereo(source: np.ndarray, target_timeline: np.ndarray, path: Path) -> None:
    left = np.asarray(source, dtype=np.float32).reshape(-1)
    right = np.asarray(target_timeline, dtype=np.float32).reshape(-1)
    total = max(len(left), len(right))
    stereo = np.zeros((total, 2), dtype=np.float32)
    stereo[: len(left), 0] = left
    stereo[: len(right), 1] = right
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_16")


def timeline_audio(chunks: Sequence[tuple[float, np.ndarray]]) -> np.ndarray:
    placements: list[tuple[int, np.ndarray]] = []
    cursor = 0
    for source_end_ms, raw in chunks:
        chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not len(chunk):
            continue
        start = max(cursor, int(round(source_end_ms * SAMPLE_RATE / 1000.0)))
        placements.append((start, chunk))
        cursor = start + len(chunk)
    output = np.zeros(cursor, dtype=np.float32)
    for start, chunk in placements:
        output[start : start + len(chunk)] = chunk
    return output


class PrefixStreamingEngine:
    def __init__(self, config: EngineConfig):
        config.validate()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.adapter_manifest: dict[str, object] | None = None
        self.speech_tokenizer: UniSSTokenizer | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        model, tokenizer, manifest, _ = load_model_and_tokenizer(
            self.config.adapter_dir, device=self.device
        )
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_manifest = manifest
        self.speech_tokenizer = UniSSTokenizer.from_pretrained(
            self.config.speech_tokenizer_dir, device=self.device
        )

    @staticmethod
    def _record(source_glm: Sequence[int], globals_: Sequence[int], target: str) -> dict[str, object]:
        return {
            "source_glm": [int(value) for value in source_glm],
            "bicodec_global": [int(value) for value in globals_],
            "tgt_lang": target,
        }

    def _forward_last(self, ids: Sequence[int]) -> torch.Tensor:
        assert self.model is not None
        inputs = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            logits = self.model(input_ids=inputs, use_cache=False).logits[0, -1]
        return logits.float()

    def _action(
        self, source_glm: Sequence[int], globals_: Sequence[int], target: str, is_final: bool
    ) -> tuple[str, float, float, bool, float]:
        prompt = builders.build_action_prompt(self._record(source_glm, globals_, target), 1.0)
        started = time.perf_counter()
        logits = self._forward_last(prompt)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        wait = float(logits[c.TOKEN_WAIT_READ].item())
        write = float(logits[c.TOKEN_WRITE_GENERATE].item())
        forced = bool(is_final and write < wait)
        return ("write" if is_final or write >= wait else "wait", wait, write, forced, seconds)

    def _generate_text(
        self, source_glm: Sequence[int], globals_: Sequence[int], target: str
    ) -> tuple[list[int], float]:
        assert self.model is not None
        record = self._record(source_glm, globals_, target)
        record["translation_ids"] = [c.TOKEN_PAD]
        sample = builders.build_streaming_s2tt(record, 1.0)
        prompt = sample.prompt_ids
        inputs = torch.tensor([prompt], dtype=torch.long, device=self.device)
        suppressed = list(range(c.VOCAB_SIZE, int(self.model.config.vocab_size)))
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=self.config.max_text_tokens,
                repetition_penalty=1.05,
                pad_token_id=c.TOKEN_PAD,
                eos_token_id=[c.TOKEN_END_CONTENT, c.TOKEN_EOS],
                suppress_tokens=suppressed,
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tail = [int(value) for value in output[0, len(prompt) :].tolist()]
        text: list[int] = []
        for token in tail:
            if token in {c.TOKEN_END_CONTENT, c.TOKEN_EOS}:
                break
            if token <= c.QWEN_BASE_VOCAB_END:
                text.append(token)
        return text, time.perf_counter() - started

    def _generate_semantic(
        self,
        committed_text: Sequence[int],
        semantic_history: Sequence[int],
        globals_: Sequence[int],
        target: str,
    ) -> tuple[list[int], int, float]:
        assert self.model is not None
        history = [int(value) for value in semantic_history[-self.config.semantic_history_tokens :]]
        prompt = [
            c.TOKEN_TASK_STREAMING_TTS,
            c.TOKEN_STREAMING_MODE,
            c.language_token_id(target),
            *c.wrap_global_tokens(globals_),
            c.TOKEN_START_CONTENT,
            *[int(value) for value in committed_text],
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            *c.encode_bicodec_semantic(history),
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_START_SEMANTIC,
        ]
        inputs = torch.tensor([prompt], dtype=torch.long, device=self.device)
        torch.manual_seed(self.config.seed + len(semantic_history))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed + len(semantic_history))
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                max_new_tokens=self.config.semantic_block_tokens + 1,
                repetition_penalty=1.05,
                pad_token_id=c.TOKEN_PAD,
                eos_token_id=[c.TOKEN_END_SEMANTIC, c.TOKEN_EOS],
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        values: list[int] = []
        invalid = 0
        for token in [int(value) for value in output[0, len(prompt) :].tolist()]:
            if token in {c.TOKEN_END_SEMANTIC, c.TOKEN_EOS}:
                break
            if c.BICODEC_SEMANTIC_OFFSET <= token <= c.BICODEC_SEMANTIC_SPAN.last_id:
                values.append(c.BICODEC_SEMANTIC_SPAN.value_for(token))
            else:
                invalid += 1
        return values, invalid, time.perf_counter() - started

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

    def stream(self, input_audio: Path | str, *, direction: str) -> Iterator[StreamUpdate]:
        if direction not in DIRECTIONS:
            raise ValueError(f"unsupported direction: {direction}")
        source_lang, target_lang = DIRECTIONS[direction]
        del source_lang
        self.load()
        assert self.speech_tokenizer is not None and self.tokenizer is not None
        request_dir = create_request_directory(self.config.output_root / f"chunk_{self.config.chunk_ms}ms")
        source_path = request_dir / "source_16k.wav"
        metadata = normalize_uploaded_audio(
            input_audio,
            source_path,
            max_upload_bytes=self.config.max_upload_bytes,
            min_audio_seconds=self.config.min_audio_seconds,
            max_audio_seconds=self.config.max_audio_seconds,
        )
        source, _ = sf.read(source_path, dtype="float32", always_2d=False)
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        duration_ms = len(source) * 1000.0 / SAMPLE_RATE
        step_samples = int(round(self.config.chunk_ms * SAMPLE_RATE / 1000.0))
        bootstrap_samples = min(
            len(source), int(round(self.config.frontend_bootstrap_ms * SAMPLE_RATE / 1000.0))
        )
        boundaries: list[int] = []
        current = bootstrap_samples
        while current < len(source):
            boundaries.append(current)
            current += step_samples
        boundaries.append(len(source))
        boundaries = sorted(set(value for value in boundaries if value > 0))

        frontend = CumulativePrefixFrontend(
            self.speech_tokenizer, holdback_tokens=self.config.stable_glm_holdback
        )
        text_committer = StablePrefixCommitter(holdback_tokens=self.config.stable_text_holdback)
        codec = self._codec()
        globals_: list[int] | None = None
        events: list[StreamEvent] = []
        semantic_history: list[int] = []
        audio_chunks: list[tuple[float, np.ndarray]] = []
        text_emission_ms: list[float] = []
        first_write: float | None = None
        first_audio: float | None = None
        first_audio_wall: float | None = None
        started = time.perf_counter()
        yield StreamUpdate(
            f"载入 iter_0008000；开始 {self.config.chunk_ms} ms 累计前缀重编码…", ""
        )
        for index, end in enumerate(boundaries):
            is_final = end == len(source)
            source_end_ms = end * 1000.0 / SAMPLE_RATE
            prefix = source[:end]
            front = frontend.encode(prefix, is_final=is_final)
            if globals_ is None:
                globals_ = frontend.extract_speaker_tokens(
                    prefix, request_dir / "speaker_prefix.wav"
                )
                codec.set_speaker_tokens(globals_)
            committed_glm = list(front.committed_tokens)
            if committed_glm:
                action, wait_logit, write_logit, forced, action_seconds = self._action(
                    committed_glm, globals_, target_lang, is_final
                )
            else:
                # The first non-final cumulative encode intentionally commits
                # nothing: stability needs two observations.  This is a real
                # frontend WAIT, not a model decision.
                action, wait_logit, write_logit, forced, action_seconds = (
                    "wait",
                    0.0,
                    0.0,
                    False,
                    0.0,
                )
            event = StreamEvent(
                index=index,
                source_end_ms=source_end_ms,
                is_final=is_final,
                candidate_glm_tokens=len(front.candidate_tokens),
                committed_glm_tokens=len(committed_glm),
                new_glm_tokens=len(front.new_committed_tokens),
                frontend_seconds=front.encode_seconds,
                action=action,
                wait_logit=wait_logit,
                write_logit=write_logit,
                forced_write=forced,
                action_seconds=action_seconds,
            )
            emitted = np.zeros(0, dtype=np.float32)
            if action == "write" and committed_glm:
                if first_write is None:
                    first_write = source_end_ms
                candidate, event.text_seconds = self._generate_text(
                    committed_glm, globals_, target_lang
                )
                new_text = text_committer.update(candidate, is_final=is_final)
                event.candidate_text_tokens = len(candidate)
                event.candidate_text = self.tokenizer.decode(candidate, skip_special_tokens=False).strip()
                event.new_text_tokens = len(new_text)
                event.committed_text = self.tokenizer.decode(
                    text_committer.committed, skip_special_tokens=False
                ).strip()
                text_emission_ms.extend([source_end_ms] * len(new_text))
                if new_text or (is_final and text_committer.committed):
                    event_semantic: list[int] = []
                    emitted_parts: list[np.ndarray] = []
                    block_limit = self.config.max_final_tts_blocks if is_final else 1
                    target_total = (
                        max(
                            50,
                            len(text_committer.committed)
                            * self.config.final_semantic_tokens_per_text_token,
                        )
                        if is_final
                        else math.inf
                    )
                    for _ in range(block_limit):
                        semantic, invalid, tts_seconds = self._generate_semantic(
                            text_committer.committed,
                            semantic_history,
                            globals_,
                            target_lang,
                        )
                        event.tts_seconds += tts_seconds
                        event.semantic_invalid_tokens += invalid
                        rejection = _semantic_rejection(semantic, invalid)
                        if rejection is not None:
                            event.semantic_rejected_reason = rejection
                            break
                        event_semantic.extend(semantic)
                        semantic_history.extend(semantic)
                        codec_started = time.perf_counter()
                        part = codec.push(semantic, is_final=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        event.codec_seconds += time.perf_counter() - codec_started
                        if len(part):
                            emitted_parts.append(part)
                        if not is_final or len(semantic_history) >= target_total:
                            break
                    if is_final and semantic_history:
                        codec_started = time.perf_counter()
                        tail = codec.push([], is_final=True)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        event.codec_seconds += time.perf_counter() - codec_started
                        if len(tail):
                            emitted_parts.append(tail)
                    event.semantic_tokens = len(event_semantic)
                    event.semantic_unique_ratio = len(set(event_semantic)) / max(
                        1, len(event_semantic)
                    )
                    event.semantic_max_run = _maximum_run(event_semantic)
                    emitted = (
                        np.concatenate(emitted_parts)
                        if emitted_parts
                        else np.zeros(0, dtype=np.float32)
                    )
                    event.emitted_audio_samples = len(emitted)
                    if len(emitted):
                        audio_chunks.append((source_end_ms, emitted))
                        if first_audio is None:
                            first_audio = source_end_ms
                            first_audio_wall = (time.perf_counter() - started) * 1000.0
            events.append(event)
            yield StreamUpdate(
                status=(
                    f"{index + 1}/{len(boundaries)} · source={source_end_ms:.0f} ms · "
                    f"{action.upper()} · +text={event.new_text_tokens} · "
                    f"+semantic={event.semantic_tokens}"
                ),
                translation=event.committed_text or self.tokenizer.decode(
                    text_committer.committed, skip_special_tokens=False
                ).strip(),
                event=event,
                audio_chunk=emitted,
            )

        processing = time.perf_counter() - started
        target_timeline = timeline_audio(audio_chunks)
        translation = (
            np.concatenate([chunk for _, chunk in audio_chunks])
            if audio_chunks
            else np.zeros(0, dtype=np.float32)
        )
        translation_path = request_dir / "translation.wav"
        timeline_path = request_dir / "translation_timeline.wav"
        stereo_path = request_dir / "stereo_left_source_right_translation.wav"
        sf.write(translation_path, translation, SAMPLE_RATE, subtype="PCM_16")
        sf.write(timeline_path, target_timeline, SAMPLE_RATE, subtype="PCM_16")
        write_stereo(source, target_timeline, stereo_path)
        measured = latency_metrics(text_emission_ms, duration_ms)
        result_path = request_dir / "result.json"
        selected = int(self.adapter_manifest["selected_iteration"]) if self.adapter_manifest else -1
        result = StreamResult(
            request_dir=str(request_dir.resolve()),
            source_path=str(source_path.resolve()),
            translation_path=str(translation_path.resolve()),
            timeline_path=str(timeline_path.resolve()),
            stereo_path=str(stereo_path.resolve()),
            result_path=str(result_path.resolve()),
            direction=direction,
            chunk_ms=self.config.chunk_ms,
            selected_iteration=selected,
            source_duration_seconds=duration_ms / 1000.0,
            translation_duration_seconds=len(translation) / SAMPLE_RATE,
            processing_seconds=processing,
            rtf=processing / max(duration_ms / 1000.0, 1e-9),
            first_write_source_ms=first_write,
            first_audio_source_ms=first_audio,
            first_audio_wall_ms=first_audio_wall,
            finalization_lag_ms=(max(0.0, len(target_timeline) * 1000.0 / SAMPLE_RATE - duration_ms) if len(target_timeline) else None),
            translation=self.tokenizer.decode(
                text_committer.committed, skip_special_tokens=False
            ).strip(),
            committed_text_tokens=len(text_committer.committed),
            semantic_tokens=len(semantic_history),
            wait_events=sum(event.action == "wait" for event in events),
            write_events=sum(event.action == "write" for event in events),
            frontend_revision_events=frontend.committer.revision_events,
            al_ms=measured["al_ms"],
            laal_ms=measured["laal_ms"],
            ap=measured["ap"],
            pseudo_streaming=True,
            events=events,
        )
        payload = result.to_dict()
        payload["adapter_manifest"] = self.adapter_manifest
        payload["frontend"] = {
            "type": "cumulative WhisperVQ/GLM re-encoding",
            "causal": False,
            "bootstrap_ms": self.config.frontend_bootstrap_ms,
        }
        write_json(result_path, payload)
        yield StreamUpdate("完成", result.translation, result=result)
