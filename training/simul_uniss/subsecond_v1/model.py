"""Emformer-based causal audio student for true streaming Stage B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Emformer

from training import constants_uniss as c


@dataclass(frozen=True)
class StageBModelConfig:
    policy_vocab_size: int
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    sample_rate: int = 16000
    n_mels: int = 128
    stack_factor: int = 4
    segment_frames: int = 4
    right_context_frames: int = 2
    left_context_frames: int = 50

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageBModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


class CausalAudioStudentV2(nn.Module):
    """40 ms frame-stacked log-Mel frontend plus stateful Emformer encoder."""

    def __init__(self, config: StageBModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.segment_frames <= 0 or config.right_context_frames < 0:
            raise ValueError("invalid streaming frame configuration")
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=config.n_mels,
            center=False,
            power=2.0,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(config.n_mels * config.stack_factor, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.encoder = Emformer(
            input_dim=config.hidden_size,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            segment_length=config.segment_frames,
            dropout=config.dropout,
            activation="gelu",
            left_context_length=config.left_context_frames,
            right_context_length=config.right_context_frames,
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.teacher_glm_head = nn.Linear(config.hidden_size, c.GLM_SEMANTIC_SIZE + 1)
        self.source_ctc_head = nn.Linear(config.hidden_size, config.policy_vocab_size)
        self.target_capacity_head = nn.Linear(config.hidden_size, 1)
        self.stability_head = nn.Linear(config.hidden_size, 1)

    @staticmethod
    def mel_lengths(sample_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div((sample_lengths - 400).clamp_min(0), 160, rounding_mode="floor") + 1

    def stacked_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        mel_lengths = self.mel_lengths(sample_lengths)
        return torch.div(
            mel_lengths + self.config.stack_factor - 1,
            self.config.stack_factor,
            rounding_mode="floor",
        ).clamp_min(1)

    def extract_projected(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = torch.log(self.mel(waveform).clamp_min(1e-5)).transpose(1, 2)
        remainder = mel.shape[1] % self.config.stack_factor
        if remainder:
            mel = F.pad(mel, (0, 0, 0, self.config.stack_factor - remainder))
        stacked = mel.reshape(
            mel.shape[0],
            mel.shape[1] // self.config.stack_factor,
            self.config.n_mels * self.config.stack_factor,
        )
        return self.input_projection(stacked)

    def _emformer_batch(
        self,
        projected: torch.Tensor,
        total_lengths: torch.Tensor,
        utterance_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = projected.shape[0]
        max_utterance = int(utterance_lengths.max())
        right = self.config.right_context_frames
        encoder_input = projected.new_zeros(batch, max_utterance + right, projected.shape[-1])
        for row in range(batch):
            utterance = int(utterance_lengths[row])
            total = int(total_lengths[row])
            encoder_input[row, :utterance] = projected[row, :utterance]
            available_right = min(right, max(0, total - utterance))
            if available_right:
                encoder_input[row, max_utterance : max_utterance + available_right] = projected[
                    row, utterance : utterance + available_right
                ]
        encoded, output_lengths = self.encoder(encoder_input, utterance_lengths)
        return self.output_norm(encoded), output_lengths

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        utterance_sample_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        projected = self.extract_projected(waveform)
        total_lengths = self.stacked_lengths(waveform_lengths)
        utterance_lengths = self.stacked_lengths(utterance_sample_lengths)
        hidden, output_lengths = self._emformer_batch(projected, total_lengths, utterance_lengths)
        return {
            "hidden": hidden,
            "output_lengths": output_lengths,
            "teacher_glm_logits": self.teacher_glm_head(hidden),
            "source_ctc_logits": self.source_ctc_head(hidden),
            "target_capacity_logits": self.target_capacity_head(hidden).squeeze(-1),
            "stability_logits": self.stability_head(hidden).squeeze(-1),
        }

    def forward_projected(
        self, projected: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        right = self.config.right_context_frames
        padded = F.pad(projected, (0, 0, 0, right))
        output, output_lengths = self.encoder(padded, lengths)
        return self.output_norm(output), output_lengths

    @torch.inference_mode()
    def infer_projected(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.shape[0] != 1:
            raise ValueError("streaming inference currently requires batch size one")
        segment = self.config.segment_frames
        right = self.config.right_context_frames
        states = None
        outputs: list[torch.Tensor] = []
        total = projected.shape[1]
        for start in range(0, total, segment):
            actual = min(segment, total - start)
            chunk = projected[:, start : start + segment + right]
            if chunk.shape[1] < segment + right:
                chunk = F.pad(chunk, (0, 0, 0, segment + right - chunk.shape[1]))
            lengths = torch.tensor([segment + right], dtype=torch.long, device=projected.device)
            output, _, states = self.encoder.infer(chunk, lengths, states)
            outputs.append(output[:, :actual])
        if not outputs:
            return projected.new_zeros(1, 0, self.config.hidden_size)
        return self.output_norm(torch.cat(outputs, dim=1))

    @torch.inference_mode()
    def infer_waveform(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        projected = self.extract_projected(waveform)
        hidden = self.infer_projected(projected)
        return {
            "hidden": hidden,
            "teacher_glm_logits": self.teacher_glm_head(hidden),
            "source_ctc_logits": self.source_ctc_head(hidden),
            "target_capacity_logits": self.target_capacity_head(hidden).squeeze(-1),
            "stability_logits": self.stability_head(hidden).squeeze(-1),
        }


def greedy_ctc_tokens(logits: torch.Tensor, blank: int = 0) -> list[int]:
    values = logits.argmax(dim=-1).reshape(-1).tolist()
    result: list[int] = []
    previous = blank
    for value in values:
        value = int(value)
        if value != blank and value != previous:
            result.append(value)
        previous = value
    return result
