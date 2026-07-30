"""Cumulative WhisperVQ prefix encoding and stable-token commit helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from uniss.streaming.stable_prefix import StablePrefixCommitter

from ..audio_io import SAMPLE_RATE


@dataclass
class PrefixStep:
    candidate_tokens: list[int]
    new_committed_tokens: list[int]
    committed_tokens: list[int]
    revision_events: int
    encode_seconds: float


class CumulativePrefixFrontend:
    def __init__(self, speech_tokenizer, holdback_tokens: int = 2):
        self.speech_tokenizer = speech_tokenizer
        self.committer = StablePrefixCommitter(holdback_tokens=holdback_tokens)

    def encode(self, waveform: np.ndarray, *, is_final: bool = False) -> PrefixStep:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("cannot encode an empty waveform prefix")
        tensor = torch.from_numpy(values).unsqueeze(0)
        started = time.perf_counter()
        tokens = self.speech_tokenizer.glm4.tokenize(speech=tensor, sr=SAMPLE_RATE)
        encode_seconds = time.perf_counter() - started
        candidate = [int(value) for value in tokens.squeeze(0).detach().cpu().tolist()]
        new_tokens = self.committer.update(candidate, is_final=is_final)
        return PrefixStep(
            candidate_tokens=candidate,
            new_committed_tokens=new_tokens,
            committed_tokens=list(self.committer.committed),
            revision_events=self.committer.revision_events,
            encode_seconds=encode_seconds,
        )

    def extract_speaker_tokens(self, waveform: np.ndarray, temporary_path: Path) -> list[int]:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temporary_path, values, SAMPLE_RATE, subtype="PCM_16")
        tokens = self.speech_tokenizer.bicodec.encode_wav_to_tokens(str(temporary_path))
        flattened = tokens.detach().reshape(-1).cpu()
        speaker = [int(value) for value in flattened[:32].tolist()]
        if len(speaker) != 32:
            raise ValueError(f"BiCodec returned {len(speaker)} speaker tokens, expected 32")
        return speaker
