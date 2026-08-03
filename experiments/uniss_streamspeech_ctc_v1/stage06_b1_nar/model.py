"""Continuous residual interface initialized exactly from the B2 bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from bridge import BridgeOutput, pool_frames
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.model import (
    FrozenEncoderB2Bridge,
)


@dataclass(frozen=True)
class B1Output:
    qwen_speech_embeddings: torch.Tensor
    token_lengths: torch.Tensor
    hard_code_ids: torch.Tensor
    residual_mse: torch.Tensor
    residual_rms: torch.Tensor


class FrozenB2ResidualBridge(nn.Module):
    def __init__(self, base: FrozenEncoderB2Bridge, residual_scale: float = 0.05) -> None:
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        encoder_dim = int(base.encoder.config.hidden_size)
        qwen_dim = int(base.bridge.qwen_glm_embeddings.shape[-1])
        self.residual = nn.Linear(encoder_dim, qwen_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

    @classmethod
    def from_checkpoints(
        cls,
        *,
        endpoint_checkpoint: str | Path,
        historical_stage_b_checkpoint: str | Path,
        stage04_b2_checkpoint: str | Path,
        codebook_model: str | Path,
        qwen_glm_embeddings: torch.Tensor,
        eng_vocab_size: int,
        cmn_vocab_size: int,
    ) -> "FrozenB2ResidualBridge":
        base = FrozenEncoderB2Bridge.from_checkpoints(
            endpoint_checkpoint=endpoint_checkpoint,
            historical_stage_b_checkpoint=historical_stage_b_checkpoint,
            codebook_model=codebook_model,
            qwen_glm_embeddings=qwen_glm_embeddings,
            eng_vocab_size=eng_vocab_size,
            cmn_vocab_size=cmn_vocab_size,
        )
        checkpoint = torch.load(stage04_b2_checkpoint, map_location="cpu", weights_only=False)
        base.load_state_dict(checkpoint["model"])
        return cls(base)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor) -> B1Output:
        with torch.no_grad():
            hidden, lengths = self.base.encoder.encode(waveform, waveform_lengths)
            b2: BridgeOutput = self.base.bridge(hidden, lengths)
            pooled, _ = pool_frames(hidden, lengths, factor=2)
        # A bounded correction protects the frozen BF16 Qwen input-gradient
        # path from the first-step overflow observed with an unconstrained
        # additive residual, while retaining exact B2 equivalence at zero.
        residual = self.residual_scale * torch.tanh(self.residual(pooled))
        corrected = b2.qwen_speech_embeddings + residual
        residual_mse = residual.float().square().mean()
        return B1Output(
            qwen_speech_embeddings=corrected,
            token_lengths=b2.token_lengths,
            hard_code_ids=b2.hard_code_ids,
            # Optimize MSE directly. Backpropagating through sqrt(MSE)^2 at
            # the exact zero initialization creates 0 * inf and NaN grads.
            residual_mse=residual_mse,
            residual_rms=residual_mse.detach().sqrt(),
        )
