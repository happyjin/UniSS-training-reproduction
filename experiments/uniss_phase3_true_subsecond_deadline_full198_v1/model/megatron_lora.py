"""Additive LoRA hooks for a native Megatron Qwen model.

The Phase3 distributed checkpoint expects the original Megatron parameter
names. Replacing its fused linear modules would rename those parameters and
make the handoff fragile. These adapters leave every base module untouched and
add a registered low-rank branch through forward hooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class AdditiveLoRABranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid LoRA rank/alpha/dropout")
        self.rank = int(rank)
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(value).to(dtype=self.lora_a.dtype)
        update = F.linear(
            F.linear(value, self.lora_a), self.lora_b
        )
        return update * self.scale


class MegatronLoRAController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleDict()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.module_names: list[str] = []

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(".", "__")

    def add(
        self,
        name: str,
        module: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"LoRA target {name} has no matrix weight")
        key = self._safe_name(name)
        if key in self.branches:
            raise ValueError(f"duplicate LoRA target: {name}")
        branch = AdditiveLoRABranch(
            int(weight.shape[1]),
            int(weight.shape[0]),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=weight.device)
        self.branches[key] = branch
        self.module_names.append(name)

        def add_update(_module, inputs, output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"LoRA target {name} received malformed input")
            update = branch(inputs[0])
            if isinstance(output, tuple):
                if not output or not isinstance(output[0], torch.Tensor):
                    raise TypeError(f"LoRA target {name} returned malformed tuple")
                return (output[0] + update.to(output[0].dtype), *output[1:])
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"LoRA target {name} returned {type(output).__name__}")
            return output + update.to(output.dtype)

        self._handles.append(module.register_forward_hook(add_update))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass(frozen=True)
class MegatronLoRASummary:
    module_names: tuple[str, ...]
    attention_modules: int
    mlp_modules: int
    trainable_parameters: int


def _layer_index(name: str) -> int | None:
    values = name.split(".")
    try:
        position = values.index("layers")
        return int(values[position + 1])
    except (ValueError, IndexError):
        return None


def inject_native_megatron_lora(
    model: nn.Module,
    *,
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
    mlp_last_layers: int = 12,
    attention_suffixes: Iterable[str] = ("linear_qkv", "linear_proj"),
    mlp_suffixes: Iterable[str] = ("linear_fc1", "linear_fc2"),
) -> MegatronLoRASummary:
    """Freeze a native GPT and attach fused-Qwen equivalent LoRA branches."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = [
        index
        for name, _ in model.named_modules()
        if (index := _layer_index(name)) is not None
    ]
    if not layers:
        raise ValueError("could not discover Megatron transformer layers")
    layer_count = max(layers) + 1
    if not 0 <= mlp_last_layers <= layer_count:
        raise ValueError("mlp_last_layers is outside the transformer depth")
    mlp_start = layer_count - mlp_last_layers
    attention_suffixes = tuple(attention_suffixes)
    mlp_suffixes = tuple(mlp_suffixes)
    selected: list[tuple[str, nn.Module, str]] = []
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        layer = _layer_index(name)
        if layer is None:
            continue
        if suffix in attention_suffixes:
            selected.append((name, module, "attention"))
        elif layer >= mlp_start and suffix in mlp_suffixes:
            selected.append((name, module, "mlp"))
    expected_attention = layer_count * len(attention_suffixes)
    expected_mlp = mlp_last_layers * len(mlp_suffixes)
    if sum(kind == "attention" for _, _, kind in selected) != expected_attention:
        raise ValueError("native attention LoRA target discovery is incomplete")
    if sum(kind == "mlp" for _, _, kind in selected) != expected_mlp:
        raise ValueError("native MLP LoRA target discovery is incomplete")

    controller = MegatronLoRAController()
    model.add_module("true_subsecond_lora", controller)
    for name, module, _ in selected:
        controller.add(
            name,
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    for parameter in controller.parameters():
        parameter.requires_grad_(True)
        parameter.uniss_lr_qwen_lora = True
    return MegatronLoRASummary(
        tuple(controller.module_names),
        sum(kind == "attention" for _, _, kind in selected),
        sum(kind == "mlp" for _, _, kind in selected),
        sum(parameter.numel() for parameter in controller.parameters()),
    )


__all__ = [
    "AdditiveLoRABranch",
    "MegatronLoRAController",
    "MegatronLoRASummary",
    "inject_native_megatron_lora",
]
