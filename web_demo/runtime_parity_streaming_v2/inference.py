"""Natural-action real-PCM inference for runtime-parity streaming checkpoints."""

from __future__ import annotations

import difflib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.audio_cached_frontend import (
    BLOCK_SAMPLES,
    SAMPLE_RATE,
    StreamingCachedWhisperVQFrontend,
)
from training import constants_uniss as c
from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder
from web_demo.runtime_parity_streaming_v2.hf_backend import HuggingFaceKVBackend
from web_demo.runtime_parity_streaming_v2.session import PersistentPromptSession


@dataclass(frozen=True)
class GeneratedWrite:
    text_ids: tuple[int, ...]
    semantic_codes: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeEvent:
    event_index: int
    source_end_ms: int
    source_finished: bool
    new_source_codes: int
    write_probability: float
    action: str
    text_ids: tuple[int, ...]
    semantic_codes: tuple[int, ...]
    emitted_audio_samples: int
    compute_ms: float
    continuation_choice: str | None
    eos_probability: float | None


@dataclass(frozen=True)
class RuntimeResult:
    sample_id: str
    source_duration_ms: int
    target_text: str
    generated_text: str
    text_similarity: float
    events: tuple[RuntimeEvent, ...]
    natural_writes: int
    semantic_tokens: int
    first_write_source_ms: int | None
    first_audio_source_ms: int | None
    source_finished_before_first_write: bool
    forced_writes: int
    committed_revision_violations: int
    natural_eos: bool
    drain_ticks: int
    processing_seconds: float
    rtf: float
    translation_audio: np.ndarray
    timeline_audio: np.ndarray
    quality_passed: bool
    quality_failures: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("translation_audio")
        value.pop("timeline_audio")
        return value


def _decode_text_choice(logits: torch.Tensor) -> int:
    values = logits.reshape(-1).float()
    text_value, text_index = values[: c.QWEN_BASE_VOCAB_END + 1].max(dim=0)
    if values[c.TOKEN_END_CONTENT] >= text_value:
        return c.TOKEN_END_CONTENT
    return int(text_index)


def _decode_semantic_choice(logits: torch.Tensor, *, allow_end: bool) -> int:
    values = logits.reshape(-1).float()
    start = c.BICODEC_SEMANTIC_OFFSET
    semantic_value, semantic_index = values[
        start : start + c.BICODEC_SEMANTIC_SIZE
    ].max(dim=0)
    if allow_end and values[c.TOKEN_END_SEMANTIC] >= semantic_value:
        return c.TOKEN_END_SEMANTIC
    return int(semantic_index)


def _decode_continuation_choice(logits: torch.Tensor) -> tuple[str, float]:
    """Choose the model's natural ready-state continuation.

    Dense training has exactly two legal tokens after a complete non-final or
    final tick: ``START_GLM`` for another observation, or ``EOS`` to close the
    session.  Restricting decoding to that grammar is ordinary constrained
    decoding; it does not override the model's WAIT/WRITE policy.
    """

    values = logits.reshape(-1).float()
    candidates = torch.stack(
        (values[c.TOKEN_START_GLM], values[c.TOKEN_EOS])
    )
    probabilities = F.softmax(candidates, dim=0)
    eos_probability = float(probabilities[1])
    choice = "EOS" if candidates[1] >= candidates[0] else "START_GLM"
    return choice, eos_probability


class NaturalRuntimeParityGenerator:
    """Generate only model-selected natural WRITEs; no deadline override."""

    def __init__(
        self,
        model,
        tokenizer,
        objective,
        *,
        target_lang: str,
        speaker_global: Sequence[int],
        device: str | torch.device,
        maximum_text_tokens: int = 16,
        maximum_semantic_tokens: int = 80,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.objective = objective
        self.backend = HuggingFaceKVBackend(model, objective, device=device)
        self.session = PersistentPromptSession(
            self.backend,
            target_lang=target_lang,
            speaker_global=speaker_global,
        )
        self.maximum_text_tokens = int(maximum_text_tokens)
        self.maximum_semantic_tokens = int(maximum_semantic_tokens)
        self.text_ids: list[int] = []
        self.semantic_codes: list[int] = []

    def action_probability(self, last_hidden: torch.Tensor) -> float:
        logits = self.objective.action_head(last_hidden)
        return float(F.softmax(logits.float(), dim=-1)[0, 1])

    def generate_write(self) -> GeneratedWrite:
        result = self.session.begin_write()
        text_ids: list[int] = []
        logits = result.logits
        for _ in range(self.maximum_text_tokens):
            token = _decode_text_choice(logits)
            if token == c.TOKEN_END_CONTENT:
                break
            text_ids.append(token)
            logits = self.session.append_text_ids((token,)).logits
        result = self.session.end_text()
        semantic: list[int] = []
        logits = result.logits
        for _ in range(self.maximum_semantic_tokens):
            token = _decode_semantic_choice(logits, allow_end=bool(semantic))
            if token == c.TOKEN_END_SEMANTIC:
                break
            semantic.append(token)
            logits = self.session.append_semantic_codes((token,)).logits
        if not semantic:
            raise RuntimeError("natural WRITE produced no semantic token")
        self.session.finish_write()
        self.text_ids.extend(text_ids)
        self.semantic_codes.extend(semantic)
        return GeneratedWrite(tuple(text_ids), tuple(semantic))

    @property
    def text(self) -> str:
        return self.tokenizer.decode(
            self.text_ids, skip_special_tokens=True
        ).strip()


def _timeline(chunks: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    placements: list[tuple[int, np.ndarray]] = []
    cursor = 0
    for source_ms, raw in chunks:
        chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not len(chunk):
            continue
        start = max(cursor, int(round(source_ms * SAMPLE_RATE / 1000)))
        placements.append((start, chunk))
        cursor = start + len(chunk)
    output = np.zeros(cursor, dtype=np.float32)
    for start, chunk in placements:
        output[start : start + len(chunk)] = chunk
    return output


def evaluate_waveform(
    *,
    sample_id: str,
    waveform: np.ndarray,
    target_text: str,
    target_lang: str,
    speaker_global: Sequence[int],
    frontend: StreamingCachedWhisperVQFrontend,
    generator: NaturalRuntimeParityGenerator,
    codec: StreamingBiCodecDecoder,
    maximum_drain_ticks: int = 32,
) -> RuntimeResult:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    source_duration_ms = int(round(len(values) * 1000 / SAMPLE_RATE))
    state = None
    events: list[RuntimeEvent] = []
    audio_chunks: list[tuple[int, np.ndarray]] = []
    first_write: int | None = None
    first_audio: int | None = None
    natural_writes = 0
    natural_eos = False
    drain_ticks = 0
    started = time.perf_counter()

    def tick(new_codes: Sequence[int], source_end_ms: int, source_finished: bool) -> None:
        nonlocal first_write, first_audio, natural_writes, natural_eos
        tick_started = time.perf_counter()
        observation = generator.session.begin_tick(new_codes)
        write_probability = generator.action_probability(observation.last_hidden)
        action = "WRITE" if write_probability >= 0.5 else "WAIT"
        text_ids: tuple[int, ...] = ()
        semantic: tuple[int, ...] = ()
        emitted = np.zeros(0, dtype=np.float32)
        if action == "WRITE":
            generated = generator.generate_write()
            text_ids = generated.text_ids
            semantic = generated.semantic_codes
            emitted = codec.push(semantic, is_final=False)
            natural_writes += 1
            if first_write is None:
                first_write = source_end_ms
            if len(emitted):
                audio_chunks.append((source_end_ms, emitted))
                if first_audio is None:
                    first_audio = source_end_ms
        else:
            generator.session.commit_wait()
        committed = generator.session.committed_ticks[-1]
        continuation_choice: str | None = None
        eos_probability: float | None = None
        if source_finished:
            continuation_choice, eos_probability = _decode_continuation_choice(
                committed.continuation_logits
            )
            if continuation_choice == "EOS":
                generator.session.finish_session()
                natural_eos = True
        events.append(
            RuntimeEvent(
                event_index=len(events),
                source_end_ms=source_end_ms,
                source_finished=source_finished,
                new_source_codes=len(new_codes),
                write_probability=write_probability,
                action=action,
                text_ids=text_ids,
                semantic_codes=semantic,
                emitted_audio_samples=len(emitted),
                compute_ms=(time.perf_counter() - tick_started) * 1000,
                continuation_choice=continuation_choice,
                eos_probability=eos_probability,
            )
        )

    for start in range(0, len(values), BLOCK_SAMPLES):
        end = min(len(values), start + BLOCK_SAMPLES)
        final = end == len(values)
        step = frontend.push(values[start:end], state, is_final=final)
        state = step.state
        tick(step.new_tokens, step.source_end_ms, final)

    for _ in range(maximum_drain_ticks):
        if natural_eos:
            break
        drain_ticks += 1
        tick((), source_duration_ms, True)

    if generator.semantic_codes:
        tail = codec.push((), is_final=True)
        if len(tail):
            audio_chunks.append((source_duration_ms, tail))
            if first_audio is None:
                first_audio = source_duration_ms
    translation_audio = (
        np.concatenate([chunk for _, chunk in audio_chunks])
        if audio_chunks
        else np.zeros(0, dtype=np.float32)
    )
    elapsed = time.perf_counter() - started
    generated_text = generator.text
    text_similarity = difflib.SequenceMatcher(
        None, generated_text.casefold(), target_text.casefold(), autojunk=False
    ).ratio()
    failures: list[str] = []
    if natural_writes <= 0:
        failures.append("no_natural_write")
    if not generator.semantic_codes:
        failures.append("no_semantic_audio")
    if first_write is None or first_write >= source_duration_ms:
        failures.append("first_write_not_before_source_eos")
    if first_write is None or first_write >= 1_000:
        failures.append("first_write_not_subsecond")
    if not len(translation_audio):
        failures.append("no_playable_audio")
    if text_similarity < 0.50:
        failures.append("translation_text_similarity_below_0.50")
    if not natural_eos:
        failures.append("no_natural_eos")
    return RuntimeResult(
        sample_id=sample_id,
        source_duration_ms=source_duration_ms,
        target_text=target_text,
        generated_text=generated_text,
        text_similarity=float(text_similarity),
        events=tuple(events),
        natural_writes=natural_writes,
        semantic_tokens=len(generator.semantic_codes),
        first_write_source_ms=first_write,
        first_audio_source_ms=first_audio,
        source_finished_before_first_write=(
            first_write is None or first_write >= source_duration_ms
        ),
        forced_writes=0,
        committed_revision_violations=0,
        natural_eos=natural_eos,
        drain_ticks=drain_ticks,
        processing_seconds=elapsed,
        rtf=elapsed / max(source_duration_ms / 1000, 1e-9),
        translation_audio=translation_audio,
        timeline_audio=_timeline(audio_chunks),
        quality_passed=not failures,
        quality_failures=tuple(failures),
    )


__all__ = [
    "GeneratedWrite",
    "NaturalRuntimeParityGenerator",
    "RuntimeEvent",
    "RuntimeResult",
    "_decode_continuation_choice",
    "evaluate_waveform",
]
