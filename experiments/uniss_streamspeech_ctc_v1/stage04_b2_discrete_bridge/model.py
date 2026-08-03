"""Frozen causal encoder plus trainable B2 GLM projection."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from bridge import BridgeOutput, StraightThroughCodebookBridge
from endpoint_model import EndpointCTCStudent
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    LatentStageBModelConfig,
    load_whispervq_codebook,
)


class FrozenEncoderB2Bridge(nn.Module):
    def __init__(self, encoder: EndpointCTCStudent, bridge: StraightThroughCodebookBridge) -> None:
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @classmethod
    def from_checkpoints(
        cls,
        *,
        endpoint_checkpoint: str | Path,
        historical_stage_b_checkpoint: str | Path,
        codebook_model: str | Path,
        qwen_glm_embeddings: torch.Tensor,
        eng_vocab_size: int,
        cmn_vocab_size: int,
        top_k: int = 32,
        temperature: float = 0.1,
    ) -> "FrozenEncoderB2Bridge":
        endpoint = torch.load(endpoint_checkpoint, map_location="cpu", weights_only=False)
        config = LatentStageBModelConfig.from_dict(endpoint["model_config"])
        encoder = EndpointCTCStudent(
            config, eng_vocab_size=eng_vocab_size, cmn_vocab_size=cmn_vocab_size
        )
        state = endpoint["model"]
        if any(name.startswith("base.") for name in state):
            state = {
                name[len("base.") :]: value
                for name, value in state.items()
                if name.startswith("base.") and name[len("base.") :] in encoder.state_dict()
            }
        encoder.load_state_dict(state)
        codebook = load_whispervq_codebook(codebook_model)
        bridge = StraightThroughCodebookBridge(
            encoder_dim=config.hidden_size,
            codebook=codebook,
            qwen_glm_embeddings=qwen_glm_embeddings,
            top_k=top_k,
            temperature=temperature,
        )
        historical = torch.load(
            historical_stage_b_checkpoint, map_location="cpu", weights_only=False
        )["model"]
        bridge.initialize_projection(
            historical["glm_latent_head.weight"], historical["glm_latent_head.bias"]
        )
        return cls(encoder, bridge)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor) -> BridgeOutput:
        with torch.no_grad():
            hidden, lengths = self.encoder.encode(waveform, waveform_lengths)
        return self.bridge(hidden, lengths)

