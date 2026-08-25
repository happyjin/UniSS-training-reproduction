"""Load a Megatron policy LoRA checkpoint into the Stage-A Hugging Face model.

Training attaches adapters to Megatron's fused QKV and gated-FC1 projections
and masks their output by token family.  A normal PEFT merge is therefore not
equivalent at inference time.  This module splits fused tensors into the exact
HF projections and retains an explicit per-forward route mask.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.nn import functional as F
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer


POLICY_PREFIX = "quality_grpo_lora.policy."
KEY_RE = re.compile(
    r"^quality_grpo_lora\.policy\.decoder__layers__(\d+)__"
    r"(self_attention__linear_qkv|self_attention__linear_proj|"
    r"mlp__linear_fc1|mlp__linear_fc2)\.(lora_a|lora_b)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def split_megatron_qkv_b(
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo Megatron's per-query-group ``[Q heads, K, V]`` interleave."""
    if num_attention_heads <= 0 or num_query_groups <= 0 or head_dim <= 0:
        raise ValueError("invalid QKV geometry")
    if num_attention_heads % num_query_groups:
        raise ValueError("query heads must be divisible by query groups")
    rank = int(value.shape[-1])
    total_heads = num_attention_heads + 2 * num_query_groups
    if tuple(value.shape) != (total_heads * head_dim, rank):
        raise ValueError(f"unexpected fused QKV LoRA-B shape: {tuple(value.shape)}")
    grouped = value.reshape(total_heads, head_dim, rank)
    heads_per_group = num_attention_heads // num_query_groups
    width = heads_per_group + 2
    q_index = torch.cat(
        [
            torch.arange(group * width, group * width + heads_per_group)
            for group in range(num_query_groups)
        ]
    )
    k_index = torch.arange(width - 2, total_heads, width)
    v_index = torch.arange(width - 1, total_heads, width)
    q = grouped[q_index].reshape(num_attention_heads * head_dim, rank)
    k = grouped[k_index].reshape(num_query_groups * head_dim, rank)
    v = grouped[v_index].reshape(num_query_groups * head_dim, rank)
    if q.numel() + k.numel() + v.numel() != value.numel():
        raise AssertionError("QKV split lost adapter values")
    return q, k, v


def split_megatron_fc1_b(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron SwiGLU ``[gate; up]`` LoRA-B rows."""
    if value.ndim != 2 or value.shape[0] % 2:
        raise ValueError(f"unexpected fused FC1 LoRA-B shape: {tuple(value.shape)}")
    return value[: value.shape[0] // 2], value[value.shape[0] // 2 :]


def load_policy_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    checkpoint = checkpoint.resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"missing DCP metadata: {checkpoint}")
    reader = FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata().state_dict_metadata
    keys = sorted(key for key in metadata if key.startswith(POLICY_PREFIX))
    if len(keys) != 64 or any(KEY_RE.fullmatch(key) is None for key in keys):
        raise ValueError(f"expected 64 policy LoRA tensors, found {len(keys)}")
    state = {
        key: torch.empty(
            tuple(metadata[key].size), dtype=metadata[key].properties.dtype
        )
        for key in keys
    }
    dcp.load(state_dict=state, storage_reader=reader)
    if any(not torch.isfinite(value.float()).all() for value in state.values()):
        raise FloatingPointError("policy LoRA checkpoint contains NaN/Inf")
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def policy_state_to_hf(
    state: Mapping[str, torch.Tensor],
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``HF module name -> (A, B)`` without globally merging adapters."""
    parsed: dict[tuple[int, str, str], torch.Tensor] = {}
    for key, value in state.items():
        match = KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected policy key: {key}")
        parsed[(int(match.group(1)), match.group(2), match.group(3))] = value
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(16, 24):
        qkv_a = parsed[(layer, "self_attention__linear_qkv", "lora_a")]
        qkv_b = parsed[(layer, "self_attention__linear_qkv", "lora_b")]
        q_b, k_b, v_b = split_megatron_qkv_b(
            qkv_b,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            head_dim=head_dim,
        )
        prefix = f"model.layers.{layer}"
        output[f"{prefix}.self_attn.q_proj"] = (qkv_a, q_b)
        output[f"{prefix}.self_attn.k_proj"] = (qkv_a, k_b)
        output[f"{prefix}.self_attn.v_proj"] = (qkv_a, v_b)
        output[f"{prefix}.self_attn.o_proj"] = (
            parsed[(layer, "self_attention__linear_proj", "lora_a")],
            parsed[(layer, "self_attention__linear_proj", "lora_b")],
        )
        fc1_a = parsed[(layer, "mlp__linear_fc1", "lora_a")]
        gate_b, up_b = split_megatron_fc1_b(
            parsed[(layer, "mlp__linear_fc1", "lora_b")]
        )
        output[f"{prefix}.mlp.gate_proj"] = (fc1_a, gate_b)
        output[f"{prefix}.mlp.up_proj"] = (fc1_a, up_b)
        output[f"{prefix}.mlp.down_proj"] = (
            parsed[(layer, "mlp__linear_fc2", "lora_a")],
            parsed[(layer, "mlp__linear_fc2", "lora_b")],
        )
    if len(output) != 56:
        raise AssertionError(f"expected 56 HF LoRA targets, found {len(output)}")
    return output


@dataclass
class _Branch:
    a: torch.Tensor
    b: torch.Tensor


class RoutedHFLoRA:
    """Additive HF hooks with an explicit token-position route."""

    def __init__(
        self,
        model: nn.Module,
        branches: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        scale: float,
    ) -> None:
        if scale <= 0:
            raise ValueError("LoRA scale must be positive")
        modules = dict(model.named_modules())
        self.scale = float(scale)
        self.enabled = False
        self.active_mask: torch.Tensor | None = None
        self.branches: dict[str, _Branch] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, (a, b) in branches.items():
            module = modules.get(name)
            weight = getattr(module, "weight", None)
            if not isinstance(module, nn.Module) or not isinstance(weight, torch.Tensor):
                raise KeyError(f"HF LoRA target is missing: {name}")
            if tuple(a.shape) != (a.shape[0], weight.shape[1]) or tuple(b.shape) != (
                weight.shape[0],
                a.shape[0],
            ):
                raise ValueError(
                    f"LoRA shape mismatch for {name}: A={tuple(a.shape)} "
                    f"B={tuple(b.shape)} weight={tuple(weight.shape)}"
                )
            branch = _Branch(
                a=a.to(device=weight.device, dtype=weight.dtype),
                b=b.to(device=weight.device, dtype=weight.dtype),
            )
            self.branches[name] = branch

            def hook(_module, inputs, output, *, target=name):
                if not self.enabled:
                    return output
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(f"LoRA target {target} received malformed input")
                pair = self.branches[target]
                update = F.linear(F.linear(inputs[0], pair.a), pair.b) * self.scale
                if self.active_mask is not None:
                    mask = self.active_mask.to(device=update.device, dtype=update.dtype)
                    if tuple(mask.shape) != tuple(update.shape[:-1]):
                        raise ValueError(
                            f"route mask {tuple(mask.shape)} cannot cover "
                            f"activation {tuple(update.shape)}"
                        )
                    update = update * mask.unsqueeze(-1)
                return output + update.to(output.dtype)

            self.handles.append(module.register_forward_hook(hook))

    def set_route(self, enabled: bool, mask: torch.Tensor | None = None) -> None:
        if mask is not None and mask.dtype != torch.bool:
            raise TypeError("LoRA route mask must be boolean")
        self.enabled = bool(enabled)
        self.active_mask = mask

    @contextmanager
    def route(
        self, enabled: bool, mask: torch.Tensor | None = None
    ) -> Iterator[None]:
        previous = (self.enabled, self.active_mask)
        self.set_route(enabled, mask)
        try:
            yield
        finally:
            self.enabled, self.active_mask = previous

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_model_and_adapter(
    base_hf: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    alpha: float = 32.0,
    rank: int = 16,
):
    tokenizer = AutoTokenizer.from_pretrained(base_hf, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_hf,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    state = load_policy_state(checkpoint)
    config = model.config
    num_heads = int(config.num_attention_heads)
    num_groups = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // num_heads
    branches = policy_state_to_hf(
        state,
        num_attention_heads=num_heads,
        num_query_groups=num_groups,
        head_dim=head_dim,
    )
    controller = RoutedHFLoRA(model, branches, scale=float(alpha) / int(rank))
    manifest = {
        "base_hf": str(base_hf.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
        "megatron_policy_tensors": len(state),
        "hf_route_targets": len(branches),
        "layers": [16, 23],
        "rank": int(rank),
        "alpha": float(alpha),
        "scale": float(alpha) / int(rank),
        "route_semantics": "disabled_for_asr_enabled_for_mt_semantic_boundary_eos",
    }
    return model, tokenizer, controller, manifest


__all__ = [
    "RoutedHFLoRA",
    "load_model_and_adapter",
    "load_policy_state",
    "policy_state_to_hf",
    "split_megatron_fc1_b",
    "split_megatron_qkv_b",
]

