"""Per-target-token irreversible commit classifier."""

from __future__ import annotations

import torch
from torch import nn


class SafeCommitHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.source_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.target_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self, source_summary: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        if source_summary.ndim != 2 or target_hidden.ndim != 3:
            raise ValueError("source_summary must be [B,H] and target_hidden [B,T,H]")
        fused = self.source_projection(source_summary).unsqueeze(1) + self.target_projection(
            target_hidden
        )
        return self.classifier(fused).squeeze(-1)
