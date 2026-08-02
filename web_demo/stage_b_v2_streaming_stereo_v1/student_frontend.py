"""Cached PCM frontend for the quantization-aware Stage-B-v2 student."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F

from web_demo.streaming_s2st_r2_v1.audio_io import SAMPLE_RATE
from web_demo.streaming_s2st_r2_v1.engine.prefix_frontend import PrefixStep


@dataclass(frozen=True)
class StudentChunkEvent:
    source_end_ms: float
    compute_ms: float
    output_frames: int
    emitted_tokens: int


class LatentStudentStreamingSession:
    """Incremental PCM, causal STFT and Emformer-cache inference for Student-v2."""

    def __init__(self, model, *, synchronize_cuda: bool = True) -> None:
        if model.training:
            raise ValueError("streaming session requires model.eval()")
        self.model = model
        self.device = next(model.parameters()).device
        self.synchronize_cuda = synchronize_cuda and self.device.type == "cuda"
        self.sample_rate = int(model.config.sample_rate)
        self.segment_frames = int(model.config.segment_frames)
        self.right_context_frames = int(model.config.right_context_frames)
        self.stack_factor = int(model.config.stack_factor)
        self._pcm = torch.empty(0, dtype=torch.float32, device=self.device)
        self._mel = torch.empty(
            0, int(model.config.n_mels), dtype=torch.float32, device=self.device
        )
        self._projected = torch.empty(
            1, 0, int(model.config.hidden_size), dtype=torch.float32, device=self.device
        )
        self._states: list[list[torch.Tensor]] | None = None
        self._received_samples = 0
        self._output_frames = 0
        self._active_compute_seconds = 0.0
        self._finalized = False
        self.tokens: list[int] = []
        self.stability_probabilities: list[float] = []
        self.events: list[StudentChunkEvent] = []

    @property
    def active_rtf(self) -> float:
        audio_seconds = self._received_samples / self.sample_rate
        return self._active_compute_seconds / max(audio_seconds, 1e-9)

    def _sync(self) -> None:
        if self.synchronize_cuda:
            torch.cuda.synchronize(self.device)

    def _extract_complete_mel_frames(self) -> None:
        if self._pcm.numel() < 400:
            return
        frame_count = (self._pcm.numel() - 400) // 160 + 1
        mel = torch.log(self.model.mel(self._pcm.unsqueeze(0)).clamp_min(1e-5))
        mel = mel.transpose(1, 2).squeeze(0)[:frame_count]
        self._mel = torch.cat((self._mel, mel), dim=0)
        self._pcm = self._pcm[frame_count * 160 :]

    def _project_complete_stacks(self, *, final: bool) -> None:
        complete = self._mel.shape[0] // self.stack_factor
        if final and self._mel.shape[0] % self.stack_factor:
            complete += 1
        if complete == 0:
            return
        consumed = min(self._mel.shape[0], complete * self.stack_factor)
        mel = self._mel[:consumed]
        if mel.shape[0] < complete * self.stack_factor:
            mel = F.pad(mel, (0, 0, 0, complete * self.stack_factor - mel.shape[0]))
        stacked = mel.reshape(complete, -1)
        projected = self.model.input_projection(stacked).unsqueeze(0)
        self._projected = torch.cat((self._projected, projected), dim=1)
        self._mel = self._mel[consumed:]

    def _infer_available(self, *, final: bool) -> tuple[int, int]:
        output_count = 0
        token_count = 0
        required = self.segment_frames + self.right_context_frames
        while self._projected.shape[1] >= required or (
            final and self._projected.shape[1] > 0
        ):
            available = int(self._projected.shape[1])
            actual = min(self.segment_frames, available)
            encoder_input = self._projected[:, :required]
            if encoder_input.shape[1] < required:
                encoder_input = F.pad(
                    encoder_input, (0, 0, 0, required - encoder_input.shape[1])
                )
            lengths = torch.tensor([required], dtype=torch.long, device=self.device)
            hidden, _, self._states = self.model.encoder.infer(
                encoder_input, lengths, self._states
            )
            hidden = self.model.output_norm(hidden[:, :actual])
            heads = self.model._heads(
                hidden,
                torch.tensor([actual], dtype=torch.long, device=self.device),
            )
            length = int(heads["token_lengths"][0])
            latent = heads["glm_latent"][:, :length]
            values = self.model.quantize(latent).reshape(-1).tolist()
            stability = torch.sigmoid(
                heads["stability_logits"][:, :length].float()
            ).reshape(-1).tolist()
            self.tokens.extend(int(value) for value in values)
            self.stability_probabilities.extend(float(value) for value in stability)
            output_count += actual
            token_count += len(values)
            self._output_frames += actual
            self._projected = self._projected[:, actual:]
            if not final and self._projected.shape[1] < required:
                break
        return output_count, token_count

    @torch.inference_mode()
    def feed(self, pcm: torch.Tensor | np.ndarray, *, final: bool = False) -> StudentChunkEvent:
        if self._finalized:
            raise RuntimeError("streaming session is already finalized")
        value = torch.as_tensor(pcm, dtype=torch.float32, device=self.device).reshape(-1)
        self._received_samples += value.numel()
        self._pcm = torch.cat((self._pcm, value), dim=0)
        self._sync()
        started = time.perf_counter()
        self._extract_complete_mel_frames()
        self._project_complete_stacks(final=final)
        output_frames, emitted = self._infer_available(final=final)
        self._sync()
        elapsed = time.perf_counter() - started
        self._active_compute_seconds += elapsed
        event = StudentChunkEvent(
            source_end_ms=self._received_samples * 1000.0 / self.sample_rate,
            compute_ms=elapsed * 1000.0,
            output_frames=output_frames,
            emitted_tokens=emitted,
        )
        self.events.append(event)
        self._finalized = final
        return event


class StudentV2GlmTokenizer:
    def __init__(self, model) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def tokenize(self, speech: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
        waveform = torch.as_tensor(speech, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:1]
        if sr != self.model.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, int(sr), int(self.model.config.sample_rate)
            )
        waveform = waveform.to(self.device)
        output = self.model.infer_waveform(waveform)
        length = int(output["token_lengths"][0])
        tokens = self.model.quantize(output["glm_latent"][:, :length])
        return tokens.detach().cpu()


class StudentV2SpeechTokenizerAdapter:
    """Retain BiCodec while replacing source GLM tokenization with Student-v2."""

    def __init__(self, base_tokenizer, model) -> None:
        self.base_tokenizer = base_tokenizer
        self.model = model
        self.glm4 = StudentV2GlmTokenizer(model)
        self.bicodec = base_tokenizer.bicodec

    def tokenize(self, audio_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        glm = self.glm4.tokenize(waveform[:1], int(sample_rate)).reshape(-1)
        bicodec = self.bicodec.encode_wav_to_tokens(str(audio_path)).detach().reshape(-1).cpu()
        return glm, bicodec

    def decode(self, tokens: torch.Tensor) -> Any:
        return self.base_tokenizer.decode(tokens)


class StudentV2StreamingFrontend:
    """PrefixFrontend-compatible wrapper backed by an actual cached session."""

    def __init__(self, speech_tokenizer: StudentV2SpeechTokenizerAdapter, feed_ms: int = 160):
        self.speech_tokenizer = speech_tokenizer
        self.feed_samples = int(round(feed_ms * SAMPLE_RATE / 1000.0))
        self.session = LatentStudentStreamingSession(speech_tokenizer.model)
        self.consumed_samples = 0
        # Compatibility with the audited legacy finalizer, which records
        # ``frontend.committer.revision_events`` for cumulative WhisperVQ.
        # Student-v2 emits append-only cached tokens, so revisions are always 0.
        self.committer = self
        self.revision_events = 0

    def encode(self, waveform: np.ndarray, *, is_final: bool = False) -> PrefixStep:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if len(values) < self.consumed_samples:
            raise ValueError("streaming waveform prefix cannot shrink")
        delta = values[self.consumed_samples :]
        started = time.perf_counter()
        cursor = 0
        while cursor < len(delta):
            end = min(len(delta), cursor + self.feed_samples)
            final_piece = is_final and end == len(delta)
            self.session.feed(delta[cursor:end], final=final_piece)
            cursor = end
        if is_final and len(delta) == 0:
            self.session.feed(np.zeros(0, dtype=np.float32), final=True)
        self.consumed_samples = len(values)
        elapsed = time.perf_counter() - started
        tokens = list(self.session.tokens)
        return PrefixStep(
            candidate_tokens=tokens,
            new_committed_tokens=tokens,
            committed_tokens=tokens,
            revision_events=0,
            encode_seconds=elapsed,
        )

    def extract_speaker_tokens(self, waveform: np.ndarray, temporary_path: Path) -> list[int]:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temporary_path, values, SAMPLE_RATE, subtype="PCM_16")
        tokens = self.speech_tokenizer.bicodec.encode_wav_to_tokens(str(temporary_path))
        speaker = [int(value) for value in tokens.detach().reshape(-1).cpu()[:32].tolist()]
        if len(speaker) != 32:
            raise ValueError(f"BiCodec returned {len(speaker)} speaker tokens, expected 32")
        return speaker
