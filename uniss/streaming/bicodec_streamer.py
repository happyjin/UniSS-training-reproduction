"""Overlap/holdback wrapper for append-only BiCodec waveform streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


DecodeFunction = Callable[[Sequence[int], Sequence[int]], np.ndarray]


def bicodec_decode_function(tokenizer) -> DecodeFunction:
    """Adapt the existing BiCodecTokenizer(global, semantic) API lazily."""

    def decode(speaker_tokens: Sequence[int], semantic_tokens: Sequence[int]) -> np.ndarray:
        import torch

        global_tensor = torch.tensor([list(speaker_tokens)], dtype=torch.long, device=tokenizer.device)
        semantic_tensor = torch.tensor([list(semantic_tokens)], dtype=torch.long, device=tokenizer.device)
        return np.asarray(tokenizer.detokenize(global_tensor, semantic_tensor), dtype=np.float32)

    return decode


def equal_power_crossfade(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    if old.shape != new.shape:
        raise ValueError("crossfade arrays must have the same shape")
    if old.size == 0:
        return new.copy()
    phase = np.linspace(0.0, np.pi / 2.0, old.size, endpoint=True, dtype=np.float32)
    return old * np.cos(phase) + new * np.sin(phase)


@dataclass
class StreamingBiCodecDecoder:
    decode: DecodeFunction
    sample_rate: int = 16000
    semantic_rate: float = 50.0
    left_context_tokens: int = 50
    holdback_tokens: int = 5
    overlap_ms: float = 80.0
    semantic_history: list[int] = field(default_factory=list)
    speaker_tokens: list[int] | None = None
    emitted_samples: int = 0
    pending_tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def samples_per_token(self) -> float:
        return self.sample_rate / self.semantic_rate

    @property
    def overlap_samples(self) -> int:
        return max(0, int(round(self.overlap_ms * self.sample_rate / 1000.0)))

    def set_speaker_tokens(self, tokens: Sequence[int]) -> None:
        values = [int(token) for token in tokens]
        if len(values) != 32:
            raise ValueError("speaker tokens must contain exactly 32 values")
        if self.speaker_tokens is not None and values != self.speaker_tokens:
            raise ValueError("speaker tokens are frozen for the lifetime of a session")
        self.speaker_tokens = values

    def push(
        self,
        semantic_tokens: Sequence[int],
        *,
        speaker_tokens: Sequence[int] | None = None,
        is_final: bool = False,
    ) -> np.ndarray:
        if speaker_tokens is not None:
            self.set_speaker_tokens(speaker_tokens)
        if self.speaker_tokens is None:
            raise ValueError("speaker tokens must be set before decoding")
        self.semantic_history.extend(int(token) for token in semantic_tokens)
        if not self.semantic_history:
            return np.zeros(0, dtype=np.float32)

        window_start_token = max(0, len(self.semantic_history) - len(semantic_tokens) - self.left_context_tokens)
        window_tokens = self.semantic_history[window_start_token:]
        waveform = np.asarray(self.decode(self.speaker_tokens, window_tokens), dtype=np.float32).reshape(-1)
        window_start_sample = int(round(window_start_token * self.samples_per_token))
        total_sample = int(round(len(self.semantic_history) * self.samples_per_token))
        if is_final:
            stable_end_sample = total_sample
        else:
            stable_end_sample = max(0, total_sample - int(round(self.holdback_tokens * self.samples_per_token)))
        if stable_end_sample <= self.emitted_samples:
            return np.zeros(0, dtype=np.float32)
        relative_start = self.emitted_samples - window_start_sample
        relative_end = stable_end_sample - window_start_sample
        if relative_start < 0 or relative_end > len(waveform):
            raise ValueError(
                "decode window does not cover the required stable region; increase left_context_tokens"
            )
        revised = waveform[relative_start:relative_end]

        if self.pending_tail.size:
            overlap = min(len(self.pending_tail), len(revised), self.overlap_samples)
            if overlap:
                blended = equal_power_crossfade(self.pending_tail[:overlap], revised[:overlap])
                revised = np.concatenate([blended, revised[overlap:]])
            elif len(revised) == 0:
                revised = self.pending_tail.copy()

        if is_final:
            output = revised
            self.pending_tail = np.zeros(0, dtype=np.float32)
        else:
            tail_length = min(self.overlap_samples, len(revised))
            if tail_length:
                output = revised[:-tail_length]
                self.pending_tail = revised[-tail_length:].copy()
            else:
                output = revised
                self.pending_tail = np.zeros(0, dtype=np.float32)
        self.emitted_samples += len(output)
        if is_final:
            self.emitted_samples = stable_end_sample
        return output.astype(np.float32, copy=False)

    def reset(self) -> None:
        self.semantic_history.clear()
        self.speaker_tokens = None
        self.emitted_samples = 0
        self.pending_tail = np.zeros(0, dtype=np.float32)
