"""Language/task-conditioned CTC projections."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class TaskCTCHeads(nn.Module):
    """Four independent heads matching StreamSpeech's ASR/NAR supervision."""

    REQUIRED = ("asr_eng", "asr_cmn", "nar_s2tt_eng", "nar_s2tt_cmn")

    def __init__(self, hidden_size: int, output_sizes: Mapping[str, int]) -> None:
        super().__init__()
        missing = set(self.REQUIRED) - set(output_sizes)
        if missing:
            raise ValueError(f"missing CTC output sizes: {sorted(missing)}")
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, int(output_sizes[name])) for name in self.REQUIRED}
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B,T,H]")
        return {name: head(hidden) for name, head in self.heads.items()}
