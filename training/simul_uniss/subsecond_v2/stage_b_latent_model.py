"""Causal Stage-B student with frozen WhisperVQ codebook supervision.

This module is intentionally independent from the historical CTC Stage-B
implementation.  The old model remains importable for exact reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Emformer


DEFAULT_CODEBOOK_KEY = "codebook.weight"


def load_whispervq_codebook(
    model_path: str | Path,
    *,
    key: str = DEFAULT_CODEBOOK_KEY,
) -> torch.Tensor:
    """Load only the frozen VQ codebook instead of the complete teacher."""

    path = Path(model_path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"WhisperVQ codebook key {key!r} is missing from {path}")
        value = handle.get_tensor(key).float().contiguous()
    if value.ndim != 2 or value.shape[0] != 16_384:
        raise ValueError(f"unexpected WhisperVQ codebook shape: {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class LatentStageBModelConfig:
    policy_vocab_size: int
    codebook_size: int = 16_384
    codebook_dim: int = 1_280
    hidden_size: int = 768
    num_layers: int = 16
    num_heads: int = 12
    ffn_dim: int = 3_072
    dropout: float = 0.1
    sample_rate: int = 16_000
    n_mels: int = 128
    mel_scale: str = "htk"
    mel_norm: str | None = None
    stack_factor: int = 4
    student_frames_per_glm: int = 2
    segment_frames: int = 4
    right_context_frames: int = 2
    left_context_frames: int = 50

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LatentStageBModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


def pool_student_frames(
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool fixed-rate 40 ms student frames onto 80 ms WhisperVQ steps."""

    if factor <= 0:
        raise ValueError("pooling factor must be positive")
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, time, channel]")
    padded_time = ((hidden.shape[1] + factor - 1) // factor) * factor
    if padded_time != hidden.shape[1]:
        hidden = F.pad(hidden, (0, 0, 0, padded_time - hidden.shape[1]))
    batch, _, channel = hidden.shape
    grouped = hidden.reshape(batch, padded_time // factor, factor, channel)
    positions = torch.arange(padded_time, device=lengths.device).reshape(1, -1)
    valid = positions < lengths.reshape(-1, 1)
    valid = valid.reshape(batch, padded_time // factor, factor)
    denominator = valid.sum(dim=2, keepdim=True).clamp_min(1).to(hidden.dtype)
    pooled = (grouped * valid.unsqueeze(-1).to(hidden.dtype)).sum(dim=2) / denominator
    pooled_lengths = torch.div(lengths + factor - 1, factor, rounding_mode="floor")
    return pooled, pooled_lengths.clamp_min(1)


def nearest_codebook_tokens(
    latent: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Quantize latent vectors with the teacher's Euclidean VQ geometry."""

    if latent.shape[-1] != codebook.shape[-1]:
        raise ValueError("latent and codebook dimensions differ")
    flat = latent.float().reshape(-1, latent.shape[-1])
    codebook_float = codebook.float()
    codebook_norm = codebook_float.square().sum(dim=1).reshape(1, -1)
    values: list[torch.Tensor] = []
    for start in range(0, len(flat), chunk_size):
        current = flat[start : start + chunk_size]
        distance = (
            current.square().sum(dim=1, keepdim=True)
            + codebook_norm
            - 2.0 * current @ codebook_float.T
        )
        values.append(distance.argmin(dim=1))
    return torch.cat(values).reshape(latent.shape[:-1])


class LatentCausalAudioStudent(nn.Module):
    """40 ms causal Emformer that predicts 80 ms WhisperVQ latent vectors."""

    def __init__(
        self,
        config: LatentStageBModelConfig,
        codebook: torch.Tensor,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if tuple(codebook.shape) != (config.codebook_size, config.codebook_dim):
            raise ValueError(
                f"codebook shape {tuple(codebook.shape)} does not match "
                f"{(config.codebook_size, config.codebook_dim)}"
            )
        self.config = config
        self.register_buffer("codebook", codebook.float().contiguous(), persistent=False)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=config.n_mels,
            mel_scale=config.mel_scale,
            norm=config.mel_norm,
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
        self.glm_latent_head = nn.Linear(config.hidden_size, config.codebook_dim)
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

    @staticmethod
    def _pack_emformer_input(
        projected: torch.Tensor,
        total_lengths: torch.Tensor,
        utterance_lengths: torch.Tensor,
        right: int,
    ) -> torch.Tensor:
        max_utterance = int(utterance_lengths.max())
        time = torch.arange(max_utterance, device=projected.device).unsqueeze(0)
        left_mask = time < utterance_lengths.unsqueeze(1)
        left = projected[:, :max_utterance].masked_fill(~left_mask.unsqueeze(-1), 0)
        if not right:
            return left
        offsets = torch.arange(right, device=projected.device).unsqueeze(0)
        right_indices = utterance_lengths.unsqueeze(1) + offsets
        right_indices = right_indices.clamp_max(projected.shape[1] - 1)
        right_values = projected.gather(
            1, right_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        )
        available_right = (total_lengths - utterance_lengths).clamp(min=0, max=right)
        right_mask = offsets < available_right.unsqueeze(1)
        right_values = right_values.masked_fill(~right_mask.unsqueeze(-1), 0)
        return torch.cat((left, right_values), dim=1)

    def _heads(self, hidden: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        token_hidden, token_lengths = pool_student_frames(
            hidden, lengths, self.config.student_frames_per_glm
        )
        return {
            "hidden": hidden,
            "output_lengths": lengths,
            "token_hidden": token_hidden,
            "token_lengths": token_lengths,
            "glm_latent": self.glm_latent_head(token_hidden),
            "source_ctc_logits": self.source_ctc_head(hidden),
            "target_capacity_logits": self.target_capacity_head(token_hidden).squeeze(-1),
            "stability_logits": self.stability_head(token_hidden).squeeze(-1),
        }

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        utterance_sample_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        projected = self.extract_projected(waveform)
        total_lengths = self.stacked_lengths(waveform_lengths)
        utterance_lengths = self.stacked_lengths(utterance_sample_lengths)
        encoder_input = self._pack_emformer_input(
            projected,
            total_lengths,
            utterance_lengths,
            self.config.right_context_frames,
        )
        encoded, output_lengths = self.encoder(encoder_input, utterance_lengths)
        return self._heads(self.output_norm(encoded), output_lengths)

    def forward_projected(
        self,
        projected: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        right = self.config.right_context_frames
        padded = F.pad(projected, (0, 0, 0, right))
        output, output_lengths = self.encoder(padded, lengths)
        return self.output_norm(output), output_lengths

    def infer_projected(
        self,
        projected: torch.Tensor,
        *,
        output_frames: int | None = None,
    ) -> torch.Tensor:
        if projected.shape[0] != 1:
            raise ValueError("streaming inference currently requires batch size one")
        segment = self.config.segment_frames
        right = self.config.right_context_frames
        states = None
        outputs: list[torch.Tensor] = []
        total = projected.shape[1] if output_frames is None else int(output_frames)
        if total < 0 or total > projected.shape[1]:
            raise ValueError(
                f"output_frames must be in [0, {projected.shape[1]}], got {total}"
            )
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
    def infer_waveform(
        self,
        waveform: torch.Tensor,
        *,
        utterance_sample_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Infer committed frames while optionally consuming lookahead samples.

        ``waveform`` may include right-context-only audio.  When
        ``utterance_sample_length`` is provided, only frames supported by that
        committed prefix are returned; later samples are available to Emformer
        attention but can never create output tokens of their own.
        """

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        projected = self.extract_projected(waveform)
        output_frames = None
        if utterance_sample_length is not None:
            if utterance_sample_length < 400 or utterance_sample_length > waveform.shape[-1]:
                raise ValueError(
                    "utterance_sample_length must cover at least one mel frame and "
                    "cannot exceed the supplied waveform"
                )
            sample_lengths = torch.tensor(
                [utterance_sample_length], dtype=torch.long, device=projected.device
            )
            output_frames = min(
                int(self.stacked_lengths(sample_lengths)[0]), projected.shape[1]
            )
        hidden = self.infer_projected(projected, output_frames=output_frames)
        lengths = torch.tensor([hidden.shape[1]], device=hidden.device)
        return self._heads(hidden, lengths)

    @torch.inference_mode()
    def quantize(self, latent: torch.Tensor, *, chunk_size: int = 512) -> torch.Tensor:
        return nearest_codebook_tokens(latent, self.codebook, chunk_size=chunk_size)
