"""Differentiable hard-forward B2 bridge into frozen Phase3 embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BridgeOutput:
    latent: torch.Tensor
    hard_code_ids: torch.Tensor
    qwen_speech_embeddings: torch.Tensor
    token_lengths: torch.Tensor
    posterior_entropy: torch.Tensor


def pool_frames(
    hidden: torch.Tensor, lengths: torch.Tensor, factor: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or factor <= 0:
        raise ValueError("hidden must be [batch,time,channel] and factor positive")
    padded = ((hidden.shape[1] + factor - 1) // factor) * factor
    if padded != hidden.shape[1]:
        hidden = F.pad(hidden, (0, 0, 0, padded - hidden.shape[1]))
    batch, _, channel = hidden.shape
    grouped = hidden.reshape(batch, padded // factor, factor, channel)
    positions = torch.arange(padded, device=lengths.device).reshape(1, -1)
    mask = (positions < lengths.reshape(-1, 1)).reshape(batch, padded // factor, factor)
    denominator = mask.sum(-1, keepdim=True).clamp_min(1).to(hidden.dtype)
    pooled = (grouped * mask.unsqueeze(-1).to(hidden.dtype)).sum(2) / denominator
    pooled_lengths = torch.div(lengths + factor - 1, factor, rounding_mode="floor")
    return pooled, pooled_lengths.clamp_min(1)


class StraightThroughCodebookBridge(nn.Module):
    def __init__(
        self,
        *,
        encoder_dim: int,
        codebook: torch.Tensor,
        qwen_glm_embeddings: torch.Tensor,
        top_k: int = 32,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if codebook.ndim != 2 or qwen_glm_embeddings.ndim != 2:
            raise ValueError("codebook and Qwen GLM embeddings must be matrices")
        if codebook.shape[0] != qwen_glm_embeddings.shape[0]:
            raise ValueError("codebook and Qwen GLM vocabulary sizes differ")
        if not 1 <= top_k <= codebook.shape[0]:
            raise ValueError("invalid top_k")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.projection = nn.Linear(encoder_dim, codebook.shape[1])
        self.register_buffer("codebook", codebook.float().contiguous())
        self.register_buffer(
            "qwen_glm_embeddings", qwen_glm_embeddings.float().contiguous()
        )
        self.top_k = top_k
        self.temperature = temperature

    def initialize_projection(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        if self.projection.weight.shape != weight.shape or self.projection.bias.shape != bias.shape:
            raise ValueError("historical GLM latent head does not match bridge projection")
        self.projection.weight.data.copy_(weight)
        self.projection.bias.data.copy_(bias)

    def forward(self, hidden_40ms: torch.Tensor, lengths_40ms: torch.Tensor) -> BridgeOutput:
        token_hidden, token_lengths = pool_frames(hidden_40ms, lengths_40ms, factor=2)
        latent = self.projection(token_hidden)
        flat = latent.float().reshape(-1, latent.shape[-1])
        codebook = self.codebook
        distance = (
            flat.square().sum(-1, keepdim=True)
            + codebook.square().sum(-1).reshape(1, -1)
            - 2.0 * flat @ codebook.T
        )
        nearest_distance, nearest_ids = torch.topk(
            distance, self.top_k, dim=-1, largest=False, sorted=True
        )
        probabilities = torch.softmax(-nearest_distance / self.temperature, dim=-1)
        hard = torch.zeros_like(probabilities)
        hard[:, 0] = 1.0
        straight_through = hard - probabilities.detach() + probabilities
        candidate_embeddings = self.qwen_glm_embeddings[nearest_ids]
        qwen_embeddings = (straight_through.unsqueeze(-1) * candidate_embeddings).sum(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)
        shape = latent.shape[:2]
        return BridgeOutput(
            latent=latent,
            hard_code_ids=nearest_ids[:, 0].reshape(shape),
            qwen_speech_embeddings=qwen_embeddings.reshape(
                *shape, self.qwen_glm_embeddings.shape[-1]
            ),
            token_lengths=token_lengths,
            posterior_entropy=entropy.reshape(shape),
        )


def replace_embedding_span(
    token_embeddings: torch.Tensor,
    speech_embeddings: torch.Tensor,
    *,
    span_start: int,
    speech_length: int,
) -> torch.Tensor:
    if token_embeddings.ndim != 2 or speech_embeddings.ndim != 2:
        raise ValueError("embeddings must be [time,channel]")
    if token_embeddings.shape[-1] != speech_embeddings.shape[-1]:
        raise ValueError("embedding dimensions differ")
    if speech_length < 0 or speech_length > len(speech_embeddings):
        raise ValueError("invalid speech_length")
    span_end = span_start + speech_length
    if span_start < 0 or span_end > len(token_embeddings):
        raise ValueError("replacement span is outside token sequence")
    output = token_embeddings.clone()
    output[span_start:span_end] = speech_embeddings[:speech_length]
    return output

