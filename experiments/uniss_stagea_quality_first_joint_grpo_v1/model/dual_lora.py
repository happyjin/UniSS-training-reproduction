"""Dual policy/reference LoRA hooks for a native Megatron Qwen model.

The historical Stage-A tensors keep their exact names and values.  Additive
branches are registered under one new experiment-only module, so a fresh DCP
handoff can load Stage A non-strictly while later checkpoints resume strictly.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


class LowRankBranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid LoRA geometry")
        self.scale = float(alpha) / int(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(value).to(dtype=self.lora_a.dtype)
        return F.linear(F.linear(value, self.lora_a), self.lora_b) * self.scale


class DualLoRAController(nn.Module):
    """Policy branches plus a frozen bootstrap reference snapshot."""

    MODES = {"policy", "reference", "disabled"}

    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.ModuleDict()
        self.reference = nn.ModuleDict()
        self.module_names: list[str] = []
        self._mode = "policy"
        self._active_mask: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.register_buffer("reference_ready", torch.tensor(False), persistent=True)

    @staticmethod
    def _key(name: str) -> str:
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
        key = self._key(name)
        policy = LowRankBranch(
            int(weight.shape[1]),
            int(weight.shape[0]),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=weight.device)
        reference = LowRankBranch(
            int(weight.shape[1]),
            int(weight.shape[0]),
            rank=rank,
            alpha=alpha,
            dropout=0.0,
        ).to(device=weight.device)
        reference.load_state_dict(policy.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.policy[key] = policy
        self.reference[key] = reference
        self.module_names.append(name)

        def add_update(_module, inputs, output):
            if self._mode == "disabled":
                return output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"LoRA target {name} received malformed input")
            branch = self.policy[key] if self._mode == "policy" else self.reference[key]
            update = branch(inputs[0])
            if self._active_mask is not None:
                mask = self._active_mask.to(device=update.device, dtype=update.dtype)
                if update.ndim == 3 and mask.numel() == update.shape[0] * update.shape[1]:
                    mask = mask.reshape(update.shape[0], update.shape[1], 1)
                elif update.ndim == 3 and mask.numel() == update.shape[0]:
                    mask = mask.reshape(update.shape[0], 1, 1)
                elif update.ndim == 2 and mask.numel() == update.shape[0]:
                    mask = mask.reshape(update.shape[0], 1)
                else:
                    raise ValueError(
                        f"LoRA route mask {tuple(mask.shape)} cannot cover "
                        f"activation {tuple(update.shape)}"
                    )
                update = update * mask
            if isinstance(output, tuple):
                return (output[0] + update.to(output[0].dtype), *output[1:])
            return output + update.to(output.dtype)

        self._handles.append(module.register_forward_hook(add_update))

    @contextmanager
    def use(self, mode: str) -> Iterator[None]:
        if mode not in self.MODES:
            raise ValueError(f"unknown LoRA mode: {mode}")
        previous = self._mode
        self._mode = mode
        try:
            yield
        finally:
            self._mode = previous

    @torch.no_grad()
    def snapshot_reference(self) -> None:
        if bool(self.reference_ready.item()):
            return
        for key in self.policy:
            self.reference[key].load_state_dict(self.policy[key].state_dict())
        self.reference_ready.fill_(True)

    def set_active_mask(self, mask: torch.Tensor | None) -> None:
        if mask is not None and mask.dtype != torch.bool:
            raise TypeError("LoRA route mask must be boolean")
        self._active_mask = mask

    def reference_anchor(self) -> torch.Tensor:
        values = []
        for key, policy in self.policy.items():
            reference = self.reference[key]
            values.extend(
                (left.float() - right.float()).square().mean()
                for left, right in zip(policy.parameters(), reference.parameters())
            )
        if not values:
            raise RuntimeError("no policy/reference LoRA branches")
        return torch.stack(values).mean()

    def policy_update_rms(self) -> torch.Tensor:
        values = [branch.lora_b.float().square().mean() for branch in self.policy.values()]
        if not values:
            raise RuntimeError("no policy LoRA branches")
        return torch.stack(values).mean().sqrt()


@dataclass(frozen=True)
class DualLoRASummary:
    module_names: tuple[str, ...]
    first_layer: int
    last_layer: int
    trainable_parameters: int
    reference_parameters: int


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None


def inject_top_layer_dual_lora(
    model: nn.Module,
    *,
    top_layers: int = 8,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> DualLoRASummary:
    """Freeze Stage A and attach QKV/proj/MLP LoRA to its top layers."""

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
    if not 1 <= int(top_layers) <= layer_count:
        raise ValueError("top-layer count is outside the transformer depth")
    first_layer = layer_count - int(top_layers)
    suffixes = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
    selected: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        layer = _layer_index(name)
        if layer is not None and layer >= first_layer and name.rsplit(".", 1)[-1] in suffixes:
            selected.append((name, module))
    expected = int(top_layers) * len(suffixes)
    if len(selected) != expected:
        raise ValueError(f"expected {expected} top-layer LoRA targets, found {len(selected)}")
    controller = DualLoRAController()
    model.add_module("quality_grpo_lora", controller)
    for name, module in selected:
        controller.add(
            name,
            module,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
        )
    for parameter in controller.policy.parameters():
        parameter.requires_grad_(True)
        parameter.uniss_grpo_adapter = True
    return DualLoRASummary(
        tuple(controller.module_names),
        first_layer,
        layer_count - 1,
        sum(parameter.numel() for parameter in controller.policy.parameters()),
        sum(parameter.numel() for parameter in controller.reference.parameters()),
    )


__all__ = [
    "DualLoRAController",
    "DualLoRASummary",
    "inject_top_layer_dual_lora",
]
