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
        self.continuous_projection = nn.Linear(whisper_hidden_size, qwen_hidden_size)
        self.register_buffer("codebook", codebook.detach().float().clone())
        self.register_buffer(
            "qwen_glm_embeddings", qwen_glm_embeddings.detach().float().clone()
        )

    def forward(self, hidden: torch.Tensor) -> STEBridgeOutput:
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
        continuous = self.continuous_projection(hidden)
        embeddings = continuous + (hard - continuous).detach()
        chosen_code = F.embedding(ids.reshape(-1), self.codebook).reshape_as(hidden)
        commitment = F.mse_loss(hidden.float(), chosen_code.detach())
        return STEBridgeOutput(embeddings, hard, ids, commitment)
