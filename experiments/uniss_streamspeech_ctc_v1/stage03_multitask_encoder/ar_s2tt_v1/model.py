"""Causal endpoint CTC model with shared AR translation supervision."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from endpoint_model import EndpointCTCStudent
from training.simul_uniss.subsecond_v2.stage_b_latent_model import LatentStageBModelConfig


class EndpointCTCARStudent(nn.Module):
    def __init__(
        self,
        base: EndpointCTCStudent,
        *,
        eng_vocab_size: int,
        cmn_vocab_size: int,
        decoder_layers: int = 4,
        max_target_positions: int = 512,
    ) -> None:
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.vocab = {"eng": eng_vocab_size, "cmn": cmn_vocab_size}
        self.target_embeddings = nn.ModuleDict(
            {
                language: nn.Embedding(size + 1, hidden)
                for language, size in self.vocab.items()
            }
        )
        self.target_positions = nn.Embedding(max_target_positions, hidden)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=base.config.num_heads,
            dim_feedforward=base.config.ffn_dim,
            dropout=base.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=decoder_layers, norm=nn.LayerNorm(hidden)
        )
        self.target_outputs = nn.ModuleDict(
            {
                language: nn.Linear(hidden, size)
                for language, size in self.vocab.items()
            }
        )

    @classmethod
    def from_stage03_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        eng_vocab_size: int,
        cmn_vocab_size: int,
    ) -> "EndpointCTCARStudent":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = LatentStageBModelConfig.from_dict(checkpoint["model_config"])
        base = EndpointCTCStudent(
            config, eng_vocab_size=eng_vocab_size, cmn_vocab_size=cmn_vocab_size
        )
        base.load_state_dict(checkpoint["model"])
        return cls(
            base,
            eng_vocab_size=eng_vocab_size,
            cmn_vocab_size=cmn_vocab_size,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        target_padded: torch.Tensor,
        target_lengths: torch.Tensor,
        direction_ids: torch.Tensor,
    ) -> dict[str, object]:
        output = self.base(waveform, waveform_lengths)
        memory = output["hidden"]
        memory_lengths = output["output_lengths"]
        memory_positions = torch.arange(memory.shape[1], device=memory.device)
        memory_padding = memory_positions.unsqueeze(0) >= memory_lengths.unsqueeze(1)
        ar_logits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for direction_id, language in ((0, "cmn"), (1, "eng")):
            rows = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
            if not len(rows):
                continue
            references = target_padded[rows]
            lengths = target_lengths[rows]
            maximum = references.shape[1]
            if maximum > self.target_positions.num_embeddings:
                raise ValueError(f"target length {maximum} exceeds configured AR positions")
            bos = self.vocab[language]
            decoder_inputs = references.new_zeros(references.shape)
            decoder_inputs[:, 0] = bos
            if maximum > 1:
                decoder_inputs[:, 1:] = references[:, :-1].clamp_min(0)
            positions = torch.arange(maximum, device=memory.device)
            embedded = self.target_embeddings[language](decoder_inputs)
            embedded = embedded + self.target_positions(positions).unsqueeze(0)
            causal_mask = torch.triu(
                torch.ones(maximum, maximum, dtype=torch.bool, device=memory.device),
                diagonal=1,
            )
            target_padding = positions.unsqueeze(0) >= lengths.unsqueeze(1)
            decoded = self.decoder(
                embedded,
                memory[rows],
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding,
                memory_key_padding_mask=memory_padding[rows],
            )
            ar_logits[language] = (self.target_outputs[language](decoded), rows)
        anchor = sum(
            parameter.sum() * 0.0
            for module in (self.target_embeddings, self.target_outputs)
            for parameter in module.parameters()
        )
        return {**output, "ar_logits": ar_logits, "ar_anchor": anchor}

