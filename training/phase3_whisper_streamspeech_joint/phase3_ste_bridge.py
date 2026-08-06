"""Hard-forward, straight-through bridge into Phase3 GLM embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class STEBridgeOutput:
    embeddings: torch.Tensor
    hard_embeddings: torch.Tensor
    hard_code_ids: torch.Tensor
    commitment_loss: torch.Tensor


class Phase3STEBridge(nn.Module):
    """Quantize Whisper hidden while preserving gradients to the frontend."""

    def __init__(
        self,
        whisper_hidden_size: int,
        qwen_hidden_size: int,
        codebook: torch.Tensor,
        qwen_glm_embeddings: torch.Tensor,
        *,
        surrogate: str = "projection",
        topk: int = 8,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if codebook.ndim != 2 or qwen_glm_embeddings.ndim != 2:
            raise ValueError("codebook and Qwen embeddings must be matrices")
        if codebook.shape[0] != qwen_glm_embeddings.shape[0]:
            raise ValueError("GLM codebook sizes differ")
        if codebook.shape[1] != whisper_hidden_size:
            raise ValueError("Whisper hidden size does not match codebook")
        if qwen_glm_embeddings.shape[1] != qwen_hidden_size:
            raise ValueError("Qwen hidden size does not match embeddings")
        if surrogate not in {"projection", "topk_soft"}:
            raise ValueError(f"unsupported STE surrogate: {surrogate}")
        if topk <= 0 or (surrogate == "topk_soft" and topk > codebook.shape[0]):
            raise ValueError("topk must be in [1, codebook size]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.surrogate = surrogate
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.continuous_projection = (
            nn.Linear(whisper_hidden_size, qwen_hidden_size)
            if surrogate == "projection"
            else None
        )
        self.register_buffer("codebook", codebook.detach().float().clone())
        self.register_buffer(
            "qwen_glm_embeddings", qwen_glm_embeddings.detach().float().clone()
        )

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> STEBridgeOutput:
        if hidden.ndim != 3 or hidden.shape[-1] != self.codebook.shape[-1]:
            raise ValueError("hidden has incompatible geometry")
        flat = hidden.float().reshape(-1, hidden.shape[-1])
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.square().sum(dim=1).unsqueeze(0)
        )
        ids = distances.argmin(dim=-1).reshape(hidden.shape[:-1])
        hard = F.embedding(ids, self.qwen_glm_embeddings).to(hidden.dtype)
        if self.surrogate == "projection":
            if self.continuous_projection is None:
                raise RuntimeError("projection surrogate is not initialized")
            continuous = self.continuous_projection(hidden)
        else:
            nearest_distances, nearest_ids = distances.topk(
                self.topk, dim=-1, largest=False, sorted=False
            )
            logits = -nearest_distances / (
                hidden.shape[-1] * self.temperature
            )
            weights = logits.softmax(dim=-1)
            nearest_embeddings = F.embedding(
                nearest_ids, self.qwen_glm_embeddings
            )
            continuous = (
                weights.unsqueeze(-1) * nearest_embeddings
            ).sum(dim=-2).reshape(*hidden.shape[:-1], -1).to(hidden.dtype)
        embeddings = continuous + (hard - continuous).detach()
        chosen_code = F.embedding(ids.reshape(-1), self.codebook).reshape_as(hidden)
        squared_error = (hidden.float() - chosen_code.detach()).square()
        if lengths is None:
            commitment = squared_error.mean()
        else:
            if lengths.ndim != 1 or lengths.shape[0] != hidden.shape[0]:
                raise ValueError("lengths must have shape [batch]")
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            denominator = valid.sum().clamp_min(1) * hidden.shape[-1]
            commitment = (
                squared_error * valid.unsqueeze(-1)
            ).sum() / denominator
        return STEBridgeOutput(embeddings, hard, ids, commitment)
