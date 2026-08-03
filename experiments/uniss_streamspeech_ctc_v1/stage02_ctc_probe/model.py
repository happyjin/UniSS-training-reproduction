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

    def forward(
        self, hidden: torch.Tensor, head: str | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        value = self.dropout(self.normalization(hidden))
        if head is not None:
            return self.heads[head](value)
        # DDP must observe one forward graph per backward pass.  Returning all
        # heads in one call also keeps every parameter active on direction-pure
        # batches and raises useful H200 GEMM occupancy for this linear probe.
        return {name: layer(value) for name, layer in self.heads.items()}

    def metadata(self) -> dict[str, object]:
        return asdict(self.config)
