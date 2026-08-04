"""Shared-encoder Stage03b + B1 model for frozen-Phase3 joint supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from bridge import BridgeOutput, pool_frames
from experiments.uniss_streamspeech_ctc_v1.stage03_multitask_encoder.ar_s2tt_v1.model import (
    EndpointCTCARStudent,
)
from experiments.uniss_streamspeech_ctc_v1.stage03_multitask_encoder.endpoint_model import (
    EndpointCTCStudent,
)
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.model import (
    FrozenEncoderB2Bridge,
)
from experiments.uniss_streamspeech_ctc_v1.stage06_b1_nar.model import B1Output
from experiments.uniss_streamspeech_ctc_v1.stage07_end_to_end_eval.checkpoint_io import (
    load_residual_into_model,
)
from training.simul_uniss.subsecond_v2.stage_b_latent_model import LatentStageBModelConfig


@dataclass(frozen=True)
class JointInitialization:
    stage03b_checkpoint: str
    stage04_checkpoint: str
    stage06_checkpoint: str
    stage06_iteration: int


class JointEmformerB1(nn.Module):
    def __init__(
        self,
        endpoint: EndpointCTCARStudent,
        bridge: nn.Module,
        residual: nn.Linear,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.bridge = bridge
        self.bridge.requires_grad_(False)
        self.bridge.eval()
        self.residual = residual
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

    @classmethod
    def from_checkpoints(
        cls,
        *,
        stage03b_checkpoint: str | Path,
        historical_stage_b_checkpoint: str | Path,
        stage04_checkpoint: str | Path,
        stage06_checkpoint: str | Path,
        codebook_model: str | Path,
        qwen_glm_embeddings: torch.Tensor,
        eng_vocab_size: int,
        cmn_vocab_size: int,
        unfreeze_encoder_layers: int = 4,
    ) -> tuple["JointEmformerB1", JointInitialization]:
        stage03b = torch.load(stage03b_checkpoint, map_location="cpu", weights_only=False)
        config = LatentStageBModelConfig.from_dict(stage03b["model_config"])
        endpoint = EndpointCTCARStudent(
            EndpointCTCStudent(
                config,
                eng_vocab_size=eng_vocab_size,
                cmn_vocab_size=cmn_vocab_size,
            ),
            eng_vocab_size=eng_vocab_size,
            cmn_vocab_size=cmn_vocab_size,
        )
        endpoint.load_state_dict(stage03b["model"])
        frozen_b2 = FrozenEncoderB2Bridge.from_checkpoints(
            endpoint_checkpoint=stage03b_checkpoint,
            historical_stage_b_checkpoint=historical_stage_b_checkpoint,
            codebook_model=codebook_model,
            qwen_glm_embeddings=qwen_glm_embeddings,
            eng_vocab_size=eng_vocab_size,
            cmn_vocab_size=cmn_vocab_size,
        )
        stage04 = torch.load(stage04_checkpoint, map_location="cpu", weights_only=False)
        frozen_b2.load_state_dict(stage04["model"])
        residual = nn.Linear(config.hidden_size, qwen_glm_embeddings.shape[-1])
        model = cls(endpoint, frozen_b2.bridge, residual, residual_scale=0.05)
        provenance = load_residual_into_model(model, stage06_checkpoint)
        model.configure_trainable(unfreeze_encoder_layers)
        return model, JointInitialization(
            stage03b_checkpoint=str(Path(stage03b_checkpoint).resolve()),
            stage04_checkpoint=str(Path(stage04_checkpoint).resolve()),
            stage06_checkpoint=str(Path(stage06_checkpoint).resolve()),
            stage06_iteration=int(provenance["iteration"]),
        )

    def configure_trainable(self, unfreeze_encoder_layers: int) -> None:
        layers = self.endpoint.base.encoder.emformer_layers
        if not 1 <= unfreeze_encoder_layers <= len(layers):
            raise ValueError(
                f"unfreeze_encoder_layers must be in [1,{len(layers)}], got "
                f"{unfreeze_encoder_layers}"
            )
        self.endpoint.requires_grad_(False)
        for layer in layers[-unfreeze_encoder_layers:]:
            layer.requires_grad_(True)
        self.endpoint.base.output_norm.requires_grad_(True)
        self.endpoint.base.heads.requires_grad_(True)
        self.endpoint.target_embeddings.requires_grad_(True)
        self.endpoint.target_positions.requires_grad_(True)
        self.endpoint.decoder.requires_grad_(True)
        self.endpoint.target_outputs.requires_grad_(True)
        self.residual.requires_grad_(True)
        self.bridge.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bridge.eval()
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        target_padded: torch.Tensor,
        target_lengths: torch.Tensor,
        direction_ids: torch.Tensor,
    ) -> tuple[dict[str, object], B1Output]:
        endpoint = self.endpoint(
            waveform,
            waveform_lengths,
            target_padded,
            target_lengths,
            direction_ids,
        )
        hidden = endpoint["hidden"]
        lengths = endpoint["output_lengths"]
        return endpoint, self.b1_from_hidden(hidden, lengths)

    def b1_from_hidden(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> B1Output:
        """Map one shared Emformer hidden sequence into Phase3 embeddings."""

        b2: BridgeOutput = self.bridge(hidden, lengths)
        pooled, _ = pool_frames(hidden, lengths, factor=2)
        residual = self.residual_scale * torch.tanh(self.residual(pooled))
        residual_mse = residual.float().square().mean()
        b1 = B1Output(
            qwen_speech_embeddings=b2.qwen_speech_embeddings + residual,
            token_lengths=b2.token_lengths,
            hard_code_ids=b2.hard_code_ids,
            residual_mse=residual_mse,
            residual_rms=residual_mse.detach().sqrt(),
        )
        return b1

    def encode_to_b1(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> B1Output:
        """Inference path without constructing unused CTC or AR logits."""

        hidden, lengths = self.endpoint.base.encode(waveform, waveform_lengths)
        return self.b1_from_hidden(hidden, lengths)

    def trainable_parameter_counts(self) -> dict[str, int]:
        groups = {
            "encoder": self.endpoint.base.encoder,
            "ctc_heads": self.endpoint.base.heads,
            "ar_decoder": nn.ModuleList(
                [
                    self.endpoint.target_embeddings,
                    self.endpoint.target_positions,
                    self.endpoint.decoder,
                    self.endpoint.target_outputs,
                ]
            ),
            "b1_residual": self.residual,
        }
        return {
            name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            for name, module in groups.items()
        }
