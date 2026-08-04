"""Small dependency-free LoRA modules for the Stage08 research validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

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
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False, **factory)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = self.lora_B(self.lora_A(self.dropout(value)))
        return base + update * self.scaling


def _parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child


def inject_lora(
    model: nn.Module,
    *,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> LoRAInjection:
    """Freeze ``model`` and replace every requested leaf linear with LoRA."""

    targets = frozenset(str(value) for value in target_modules)
    if not targets:
        raise ValueError("at least one LoRA target module is required")
    model.requires_grad_(False)
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)
    ]
    if not selected:
        raise ValueError(f"no linear modules matched LoRA targets: {sorted(targets)}")
    for name, module in selected:
        parent, child = _parent_and_child(model, name)
        setattr(
            parent,
            child,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
    count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (".lora_A." in name or ".lora_B." in name)
    )
    if count <= 0:
        raise ValueError("LoRA injection produced no trainable parameters")
    return LoRAInjection(
        module_names=tuple(name for name, _ in selected),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        trainable_parameters=count,
    )


def set_lora_training(model: nn.Module, mode: bool) -> None:
    """Keep the frozen Qwen base in eval mode while toggling LoRA dropout."""

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.train(mode)
            module.base.eval()


def lora_tensor_names(model: nn.Module) -> list[str]:
    names = sorted(
        name
        for name in model.state_dict()
        if ".lora_A." in name or ".lora_B." in name
    )
    if not names:
        raise ValueError("model has no LoRA tensors")
    return names


def lora_update_rms(model: nn.Module) -> torch.Tensor:
    values = [
        parameter.float().square().mean()
        for name, parameter in model.named_parameters()
        if ".lora_B." in name
    ]
    if not values:
        raise ValueError("model has no LoRA B tensors")
    return torch.stack(values).mean().sqrt()
