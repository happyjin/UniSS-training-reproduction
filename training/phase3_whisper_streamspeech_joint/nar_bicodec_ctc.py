"""StreamSpeech-style non-autoregressive text-to-BiCodec CTC head."""

from __future__ import annotations

import torch
from torch import nn


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class NARBiCodecCTC(nn.Module):
    def __init__(
        self,
        *,
        qwen_hidden_size: int = 896,
        model_size: int = 512,
        semantic_vocab_size: int = 8192,
        upsample_ratio: int = 48,
        num_heads: int = 8,
        t2u_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if upsample_ratio <= 0:
            raise ValueError("upsample_ratio must be positive")
        self.upsample_ratio = int(upsample_ratio)
        self.blank_id = int(semantic_vocab_size)
        self.input_projection = nn.Linear(qwen_hidden_size, model_size)
        layer = nn.TransformerEncoderLayer(
            model_size,
            num_heads,
            dim_feedforward=4 * model_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.t2u_encoder = nn.TransformerEncoder(layer, num_layers=t2u_layers)
        decoder_layer = nn.TransformerEncoderLayer(
            model_size,
            num_heads,
            dim_feedforward=4 * model_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.unit_decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(model_size, semantic_vocab_size + 1)

    def forward(
        self, text_hidden: torch.Tensor, text_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_hidden.ndim != 3 or text_lengths.shape != text_hidden.shape[:1]:
            raise ValueError("invalid text hidden/length geometry")
        encoded = self.t2u_encoder(self.input_projection(text_hidden))
        expanded = encoded.repeat_interleave(self.upsample_ratio, dim=1)
        output_lengths = text_lengths * self.upsample_ratio
        positions = torch.arange(expanded.shape[1], device=expanded.device)
        padding = positions[None, :] >= output_lengths[:, None]
        decoded = self.unit_decoder(
            expanded,
            mask=causal_mask(expanded.shape[1], expanded.device),
            src_key_padding_mask=padding,
        )
        return self.output(decoded), output_lengths
