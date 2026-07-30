"""Stateful PCM-to-token inference for the Stage-B causal audio student.

The session consumes real PCM chunks.  It keeps the causal STFT overlap,
40-ms stacked-Mel remainder, and Emformer cache instead of re-encoding an
ever-growing waveform prefix.  Reported computation-aware timestamps model a
single real-time worker: computation starts when both audio and the previous
chunk's computation are available.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2


@dataclass(frozen=True)
class TokenEmission:
    head: str
    token_id: int
    output_frame: int
    nca_ms: float
    ca_ms: float
    stability_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkEvent:
    chunk_index: int
    source_end_ms: float
    compute_ms: float
    computation_end_ms: float
    output_frames: int
    glm_emissions: int
    source_emissions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalStudentStreamingSession:
    """Incremental waveform frontend and cached Emformer inference session."""

    def __init__(self, model: CausalAudioStudentV2, *, synchronize_cuda: bool = True) -> None:
        if model.training:
            raise ValueError("streaming session requires model.eval()")
        self.model = model
        self.device = next(model.parameters()).device
        self.synchronize_cuda = synchronize_cuda and self.device.type == "cuda"
        self.sample_rate = model.config.sample_rate
        self.segment_frames = model.config.segment_frames
        self.right_context_frames = model.config.right_context_frames
        self.stack_factor = model.config.stack_factor

        self._pcm = torch.empty(0, dtype=torch.float32, device=self.device)
        self._mel = torch.empty(0, model.config.n_mels, dtype=torch.float32, device=self.device)
        self._projected = torch.empty(
            1, 0, model.config.hidden_size, dtype=torch.float32, device=self.device
        )
        self._states: list[list[torch.Tensor]] | None = None
        self._previous_glm = 0
        self._previous_source = 0
        self._received_samples = 0
        self._output_frames = 0
        self._chunk_index = 0
        self._compute_finish_seconds = 0.0
        self._active_compute_seconds = 0.0
        self._finalized = False
        self.glm_emissions: list[TokenEmission] = []
        self.source_emissions: list[TokenEmission] = []
        self.chunk_events: list[ChunkEvent] = []

    @property
    def audio_seconds(self) -> float:
        return self._received_samples / self.sample_rate

    @property
    def active_compute_seconds(self) -> float:
        return self._active_compute_seconds

    @property
    def active_rtf(self) -> float:
        return self._active_compute_seconds / max(self.audio_seconds, 1e-9)

    @property
    def final_backlog_ms(self) -> float:
        return max(0.0, self._compute_finish_seconds - self.audio_seconds) * 1000.0

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

    @staticmethod
    def _collapse(
        values: torch.Tensor,
        stability: torch.Tensor,
        *,
        previous: int,
        head: str,
        first_frame: int,
        nca_ms: float,
        ca_ms: float,
    ) -> tuple[list[TokenEmission], int]:
        emissions: list[TokenEmission] = []
        ids = values.argmax(dim=-1).reshape(-1).tolist()
        probabilities = torch.sigmoid(stability.float()).reshape(-1).tolist()
        for offset, (token, probability) in enumerate(zip(ids, probabilities)):
            token = int(token)
            if token != 0 and token != previous:
                emissions.append(
                    TokenEmission(
                        head=head,
                        token_id=token - 1,
                        output_frame=first_frame + offset,
                        nca_ms=nca_ms,
                        ca_ms=ca_ms,
                        stability_probability=float(probability),
                    )
                )
            previous = token
        return emissions, previous

    def _infer_available(self, *, final: bool, nca_ms: float, ca_ms: float) -> tuple[int, int, int]:
        segment = self.segment_frames
        right = self.right_context_frames
        output_count = 0
        glm_count = 0
        source_count = 0
        while self._projected.shape[1] >= segment + right or (
            final and self._projected.shape[1] > 0
        ):
            available = self._projected.shape[1]
            actual = min(segment, available)
            required = segment + right
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
            stability = self.model.stability_head(hidden).squeeze(-1)
            glm, self._previous_glm = self._collapse(
                self.model.teacher_glm_head(hidden),
                stability,
                previous=self._previous_glm,
                head="teacher_glm",
                first_frame=self._output_frames,
                nca_ms=nca_ms,
                ca_ms=ca_ms,
            )
            source, self._previous_source = self._collapse(
                self.model.source_ctc_head(hidden),
                stability,
                previous=self._previous_source,
                head="source_ctc",
                first_frame=self._output_frames,
                nca_ms=nca_ms,
                ca_ms=ca_ms,
            )
            self.glm_emissions.extend(glm)
            self.source_emissions.extend(source)
            glm_count += len(glm)
            source_count += len(source)
            output_count += actual
            self._output_frames += actual
            self._projected = self._projected[:, actual:]
            if not final and self._projected.shape[1] < segment + right:
                break
        return output_count, glm_count, source_count

    @torch.inference_mode()
    def feed(self, pcm: torch.Tensor, *, final: bool = False) -> ChunkEvent:
        if self._finalized:
            raise RuntimeError("streaming session is already finalized")
        value = torch.as_tensor(pcm, dtype=torch.float32, device=self.device).reshape(-1)
        self._received_samples += value.numel()
        source_end_seconds = self.audio_seconds
        self._pcm = torch.cat((self._pcm, value), dim=0)

        self._sync()
        started = time.perf_counter()
        self._extract_complete_mel_frames()
        if final and self._pcm.numel() >= 400:
            self._extract_complete_mel_frames()
        self._project_complete_stacks(final=final)

        # The CA timestamp is finalized after synchronization.  Emissions are
        # temporarily stamped with NCA and then replaced below.
        output_count, glm_count, source_count = self._infer_available(
            final=final,
            nca_ms=source_end_seconds * 1000.0,
            ca_ms=source_end_seconds * 1000.0,
        )
        self._sync()
        elapsed = time.perf_counter() - started
        compute_start = max(source_end_seconds, self._compute_finish_seconds)
        self._compute_finish_seconds = compute_start + elapsed
        self._active_compute_seconds += elapsed
        ca_ms = self._compute_finish_seconds * 1000.0

        if glm_count:
            start = len(self.glm_emissions) - glm_count
            self.glm_emissions[start:] = [
                TokenEmission(**{**item.to_dict(), "ca_ms": ca_ms})
                for item in self.glm_emissions[start:]
            ]
        if source_count:
            start = len(self.source_emissions) - source_count
            self.source_emissions[start:] = [
                TokenEmission(**{**item.to_dict(), "ca_ms": ca_ms})
                for item in self.source_emissions[start:]
            ]

        event = ChunkEvent(
            chunk_index=self._chunk_index,
            source_end_ms=source_end_seconds * 1000.0,
            compute_ms=elapsed * 1000.0,
            computation_end_ms=ca_ms,
            output_frames=output_count,
            glm_emissions=glm_count,
            source_emissions=source_count,
        )
        self.chunk_events.append(event)
        self._chunk_index += 1
        self._finalized = final
        return event

    def summary(self) -> dict[str, Any]:
        acts = [event.compute_ms for event in self.chunk_events]
        return {
            "audio_seconds": self.audio_seconds,
            "active_compute_seconds": self.active_compute_seconds,
            "active_rtf": self.active_rtf,
            "final_backlog_ms": self.final_backlog_ms,
            "chunks": len(self.chunk_events),
            "output_frames": self._output_frames,
            "glm_tokens": len(self.glm_emissions),
            "source_tokens": len(self.source_emissions),
            "chunk_act_ms": acts,
        }
