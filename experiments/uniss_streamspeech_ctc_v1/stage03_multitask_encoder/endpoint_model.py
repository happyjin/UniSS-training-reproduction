"""Endpoint-supervised causal Emformer initialized from Stage-B-v3."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Emformer

from training.simul_uniss.subsecond_v2.stage_b_latent_model import LatentStageBModelConfig


class EndpointCTCStudent(nn.Module):
    def __init__(
        self,
        config: LatentStageBModelConfig,
        *,
        eng_vocab_size: int,
        cmn_vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
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
        self.heads = nn.ModuleDict(
            {
                "asr_eng": nn.Linear(config.hidden_size, eng_vocab_size + 1),
                "asr_cmn": nn.Linear(config.hidden_size, cmn_vocab_size + 1),
                "nar_s2tt_eng": nn.Linear(config.hidden_size, eng_vocab_size + 1),
                "nar_s2tt_cmn": nn.Linear(config.hidden_size, cmn_vocab_size + 1),
            }
        )

    @classmethod
    def from_stage_b_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        eng_vocab_size: int,
        cmn_vocab_size: int,
    ) -> "EndpointCTCStudent":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = LatentStageBModelConfig.from_dict(checkpoint["model_config"])
        model = cls(
            config,
            eng_vocab_size=eng_vocab_size,
            cmn_vocab_size=cmn_vocab_size,
        )
        destination = model.state_dict()
        source = checkpoint["model"]
        transferable = {
            name: value
            for name, value in source.items()
            if name in destination and not name.startswith("heads.")
        }
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        allowed_missing = {name for name in destination if name.startswith("heads.")}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                f"Stage-B initialization mismatch: missing={missing}, unexpected={unexpected}"
            )
        return model

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

    def encode(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.extract_projected(waveform)
        lengths = self.stacked_lengths(waveform_lengths)
        right = self.config.right_context_frames
        padded = F.pad(projected, (0, 0, 0, right))
        encoded, output_lengths = self.encoder(padded, lengths)
        hidden = self.output_norm(encoded)
        return hidden, output_lengths

    def forward(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        hidden, output_lengths = self.encode(waveform, waveform_lengths)
        return {
            "hidden": hidden,
            "output_lengths": output_lengths,
            "logits": {name: layer(hidden) for name, layer in self.heads.items()},
        }

    def encoder_parameters(self):
        for module in (self.input_projection, self.encoder, self.output_norm):
            yield from module.parameters()

    def metadata(self) -> dict[str, object]:
        return asdict(self.config)
