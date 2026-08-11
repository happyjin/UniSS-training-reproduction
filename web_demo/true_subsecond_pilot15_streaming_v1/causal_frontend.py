"""Bounded-memory PCM to append-only WhisperVQ code streaming.

The encoder is invoked only with samples that have already arrived.  A fixed
recent waveform window is re-encoded under the same 160ms/80ms bounded-causal
attention used to build the training cache.  Tokens are committed only after
their right-context clock has elapsed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


SAMPLE_RATE = 16_000
TOKEN_HOP_MS = 80
TOKEN_HOP_SAMPLES = SAMPLE_RATE * TOKEN_HOP_MS // 1000


class WhisperVQEncoder(Protocol):
    def encode(self, audio: Sequence[tuple[object, int]]): ...


@dataclass(frozen=True)
class FrontendStep:
    source_end_ms: int
    window_start_ms: int
    stable_end_ms: int
    candidate_tokens: tuple[int, ...]
    new_tokens: tuple[int, ...]
    committed_tokens: int
    encode_seconds: float
    committed_revision_violations: int


class BoundedCausalWhisperVQFrontend:
    def __init__(
        self,
        encoder: WhisperVQEncoder,
        *,
        right_context_ms: int = 80,
        window_ms: int = 4_800,
    ) -> None:
        if right_context_ms < 0 or right_context_ms % TOKEN_HOP_MS:
            raise ValueError("right context must be a non-negative 80ms multiple")
        if window_ms <= right_context_ms or window_ms % TOKEN_HOP_MS:
            raise ValueError("window must be an 80ms multiple larger than right context")
        self.encoder = encoder
        self.right_context_ms = int(right_context_ms)
        self.window_samples = window_ms * SAMPLE_RATE // 1000
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_sample = 0
        self.total_samples = 0
        self.committed: list[int] = []
        self._observed: dict[int, int] = {}
        self.committed_revision_violations = 0
        self.maximum_buffer_samples = 0

    @property
    def source_end_ms(self) -> int:
        return int(round(self.total_samples * 1000 / SAMPLE_RATE))

    @property
    def buffer_start_ms(self) -> int:
        return int(round(self.buffer_start_sample * 1000 / SAMPLE_RATE))

    def _trim(self) -> None:
        excess = len(self.buffer) - self.window_samples
        if excess <= 0:
            return
        drop = math.ceil(excess / TOKEN_HOP_SAMPLES) * TOKEN_HOP_SAMPLES
        if drop >= len(self.buffer):
            raise RuntimeError("frontend trim would discard the complete observation")
        self.buffer = self.buffer[drop:].copy()
        self.buffer_start_sample += drop
        start_token = self.buffer_start_sample // TOKEN_HOP_SAMPLES
        self._observed = {
            index: token for index, token in self._observed.items() if index >= start_token
        }

    def push(self, pcm: np.ndarray, *, is_final: bool = False) -> FrontendStep:
        chunk = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not len(chunk):
            raise ValueError("cannot push an empty PCM chunk")
        if not np.isfinite(chunk).all():
            raise ValueError("PCM chunk contains a non-finite sample")
        self.buffer = np.concatenate((self.buffer, chunk))
        self.total_samples += len(chunk)
        self._trim()
        self.maximum_buffer_samples = max(self.maximum_buffer_samples, len(self.buffer))

        import torch

        waveform = torch.from_numpy(self.buffer.copy()).unsqueeze(0)
        started = time.perf_counter()
        encoded = self.encoder.encode([(waveform, SAMPLE_RATE)])
        seconds = time.perf_counter() - started
        if len(encoded) != 1:
            raise RuntimeError(f"WhisperVQ returned {len(encoded)} rows for one PCM window")
        candidate = tuple(int(value) for value in encoded[0].tokens.reshape(-1).tolist())
        if not candidate:
            raise RuntimeError("WhisperVQ returned an empty causal token window")

        start_index = self.buffer_start_sample // TOKEN_HOP_SAMPLES
        for local_index, token in enumerate(candidate):
            global_index = start_index + local_index
            previous = self._observed.get(global_index)
            if previous is not None and previous != token and global_index < len(self.committed):
                self.committed_revision_violations += 1
            self._observed[global_index] = token

        if is_final:
            stable_count = math.ceil(self.total_samples / TOKEN_HOP_SAMPLES)
            stable_end_ms = self.source_end_ms
        else:
            stable_samples = max(
                0,
                self.total_samples - self.right_context_ms * SAMPLE_RATE // 1000,
            )
            stable_count = stable_samples // TOKEN_HOP_SAMPLES
            stable_end_ms = int(round(stable_samples * 1000 / SAMPLE_RATE))
        new_tokens: list[int] = []
        for global_index in range(len(self.committed), stable_count):
            local_index = global_index - start_index
            if not 0 <= local_index < len(candidate):
                raise RuntimeError(
                    "bounded frontend lost an uncommitted token; increase window_ms "
                    f"(global={global_index}, start={start_index}, candidates={len(candidate)})"
                )
            token = candidate[local_index]
            self.committed.append(token)
            new_tokens.append(token)
        return FrontendStep(
            source_end_ms=self.source_end_ms,
            window_start_ms=self.buffer_start_ms,
            stable_end_ms=stable_end_ms,
            candidate_tokens=candidate,
            new_tokens=tuple(new_tokens),
            committed_tokens=len(self.committed),
            encode_seconds=seconds,
            committed_revision_violations=self.committed_revision_violations,
        )
