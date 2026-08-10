"""Minimal isolated LoRA injection for the frozen Phase3 Qwen backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid LoRA rank/alpha/dropout")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b)
        return self.base(value) + update * self.scale


@dataclass(frozen=True)
class LoRAInjectionSummary:
    attention_modules: int
    mlp_modules: int
    trainable_parameters: int
    total_parameters: int


def inject_phase3_qwen_lora(
    model: nn.Module,
    *,
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
    mlp_last_layers: int = 12,
) -> LoRAInjectionSummary:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not isinstance(layers, nn.ModuleList):
        raise TypeError("expected a Qwen-style model.model.layers ModuleList")
    if not 0 <= mlp_last_layers <= len(layers):
        raise ValueError("mlp_last_layers is outside the Qwen layer range")
    attention_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_names = ("gate_proj", "up_proj", "down_proj")
    attention_count = 0
    mlp_count = 0
    mlp_start = len(layers) - mlp_last_layers
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn")
        for name in attention_names:
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Qwen attention {layer_index}.{name} is not Linear")
            setattr(
                attention,
                name,
                LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
            )
            attention_count += 1
        if layer_index >= mlp_start:
            mlp = getattr(layer, "mlp")
            for name in mlp_names:
                base = getattr(mlp, name)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Qwen MLP {layer_index}.{name} is not Linear")
                setattr(
                    mlp,
                    name,
                    LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
                )
                mlp_count += 1
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise AssertionError("LoRA injection left no trainable parameters")
    return LoRAInjectionSummary(attention_count, mlp_count, trainable, total)
