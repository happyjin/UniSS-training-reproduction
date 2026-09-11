"""A minimal LoRA, because ``peft`` is not in this environment.

NaturalFlow adapts its base model with LoRA at rank 128 rather than tuning
every weight, and that is the right shape for us for a second reason: the two
earlier attempts to bake a reranker into these weights both damaged the speech
stream, once catastrophically.  A low-rank adapter bounds how far the policy
can move and -- because ``b`` starts at zero -- lets the *same* module serve as
its own frozen reference simply by switching the adapters off.  That removes
the second copy of the model from memory and makes "policy equals reference at
step 0" exact rather than approximate.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn

# Attention and MLP projections.  The embedding and the output head are left
# alone on purpose: this is a 16k-code extended vocabulary shared with the
# speech stream, and moving it is how the earlier attempts broke the audio.
DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class LoRALinear(nn.Module):
    """``base(x) + scaling * b @ a @ x``, with ``b`` zeroed at init."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        device = base.weight.device
        # The adapter is held in fp32 even when the base is bf16: these are the
        # only weights an optimiser touches, and bf16 has too few mantissa bits
        # to accumulate the small updates a 5e-6 learning rate makes.
        self.a = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.a.dtype), self.a), self.b
        )
        return out + self.scaling * delta.to(out.dtype)


def inject(
    model: nn.Module,
    *,
    rank: int = 128,
    alpha: float = 256.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> list[LoRALinear]:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``."""
    wrapped: list[LoRALinear] = []
    for parent_name, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                layer = LoRALinear(child, rank, alpha)
                # Carry the qualified name so a saved adapter can be matched
                # back to its module by name rather than by traversal order.
                layer.lora_name = f"{parent_name}.{name}" if parent_name else name
                setattr(module, name, layer)
                wrapped.append(layer)
    if not wrapped:
        raise SystemExit(f"no module matched {targets}; the model layout changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in wrapped:
        layer.a.requires_grad_(True)
        layer.b.requires_grad_(True)
    return wrapped


@contextmanager
def adapters_disabled(layers: list[LoRALinear]):
    """Run the frozen reference policy: the base weights, untouched."""
    for layer in layers:
        layer.enabled = False
    try:
        yield
    finally:
        for layer in layers:
            layer.enabled = True


def trainable_parameters(layers: list[LoRALinear]) -> list[nn.Parameter]:
    return [p for layer in layers for p in (layer.a, layer.b)]


def state_dict(layers: list[LoRALinear]) -> dict:
    return {
        f"{layer.lora_name}.{which}": getattr(layer, which).detach().cpu()
        for layer in layers
        for which in ("a", "b")
    }


def load_state_dict(layers: list[LoRALinear], state: dict) -> None:
    by_name = {layer.lora_name: layer for layer in layers}
    missing = set(by_name) - {k.rsplit(".", 1)[0] for k in state}
    if missing:
        raise SystemExit(f"adapter is missing {len(missing)} layers, e.g. {sorted(missing)[:3]}")
    for key, value in state.items():
        name, which = key.rsplit(".", 1)
        with torch.no_grad():
            getattr(by_name[name], which).copy_(value.to(by_name[name].a.device))


def merge(layers: list[LoRALinear]) -> None:
    """Fold each adapter into its base weight, so export needs no new code."""
    for layer in layers:
        with torch.no_grad():
            delta = (layer.b @ layer.a) * layer.scaling
            layer.base.weight.add_(delta.to(layer.base.weight.dtype))
            layer.b.zero_()
