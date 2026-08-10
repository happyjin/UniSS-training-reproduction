"""Support-count and READ/WRITE heads."""

from __future__ import annotations

import torch
from torch import nn


class SupportOrdinalHead(nn.Module):
    def __init__(self, hidden_size: int, buckets: int = 5) -> None:
        super().__init__()
        if buckets != 5:
            raise ValueError("the frozen support schema requires buckets {0,1,2,3,4+}")
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, buckets),
        )

    def forward(self, source_summary: torch.Tensor) -> torch.Tensor:
        return self.network(source_summary)


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, source_summary: torch.Tensor) -> torch.Tensor:
        return self.network(source_summary)
