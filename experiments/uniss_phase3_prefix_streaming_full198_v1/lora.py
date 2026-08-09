"""Isolated LoRA implementation with an exact base-model teacher switch."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAInjection:
    module_names: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float
    trainable_parameters: int


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0.0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if not self.enabled:
            return base
        update = self.lora_B(self.lora_A(self.dropout(value)))
        return base + update.to(dtype=base.dtype) * self.scaling


def _parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def inject_lora(
    model: nn.Module,
    *,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> LoRAInjection:
    targets = frozenset(str(value) for value in target_modules)
    model.requires_grad_(False)
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)
    ]
    if not selected:
        raise ValueError(f"no modules matched LoRA targets {sorted(targets)}")
    for name, module in selected:
        parent, child = _parent_and_child(model, name)
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    return LoRAInjection(tuple(name for name, _ in selected), rank, alpha, dropout, trainable)


@contextlib.contextmanager
def lora_enabled(model: nn.Module, enabled: bool) -> Iterator[None]:
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = bool(enabled)
        yield
    finally:
        for module, value in zip(modules, previous):
            module.enabled = value


def set_lora_training(model: nn.Module, mode: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.train(mode)
            module.base.eval()


def lora_update_rms(model: nn.Module) -> torch.Tensor:
    values = [
        value.float().square().mean()
        for name, value in model.named_parameters()
        if ".lora_B." in name
    ]
    if not values:
        raise ValueError("no LoRA B tensors found")
    return torch.stack(values).mean().sqrt()

