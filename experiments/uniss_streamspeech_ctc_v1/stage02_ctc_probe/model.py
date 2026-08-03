"""Four language/task-conditioned linear CTC heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CTCProbeConfig:
    hidden_size: int = 1280
    eng_vocab_size: int = 8000
    cmn_vocab_size: int = 8000
    input_dropout: float = 0.05


class LanguageConditionalCTCProbe(nn.Module):
    def __init__(self, config: CTCProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.normalization = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.dropout = nn.Dropout(config.input_dropout)
        self.heads = nn.ModuleDict(
            {
                "asr_eng": nn.Linear(config.hidden_size, config.eng_vocab_size + 1),
                "asr_cmn": nn.Linear(config.hidden_size, config.cmn_vocab_size + 1),
                "nar_s2tt_eng": nn.Linear(
                    config.hidden_size, config.eng_vocab_size + 1
                ),
                "nar_s2tt_cmn": nn.Linear(
                    config.hidden_size, config.cmn_vocab_size + 1
                ),
            }
        )

    def forward(self, hidden: torch.Tensor, head: str) -> torch.Tensor:
        return self.heads[head](self.dropout(self.normalization(hidden)))

    def metadata(self) -> dict[str, object]:
        return asdict(self.config)

