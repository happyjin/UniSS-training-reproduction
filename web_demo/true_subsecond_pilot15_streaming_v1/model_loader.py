"""Load the frozen Phase3 HF base, mapped native LoRA and streaming sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_prefix_streaming_full198_v1.lora import LoRALinear
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    TrueSubsecondObjective,
)


def _parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


class _PreNormInputCapture:
    """Keep the input seen by Megatron's fused LayerNormLinear module.

    Native Megatron attaches the LoRA forward hook to ``linear_qkv`` and
    ``linear_fc1``.  Both are fused LayerNormLinear modules, so the hook's
    positional input is the hidden state *before* layer normalization.  In
    Hugging Face Qwen the normalization is a separate module and q/k/v or
    gate/up receive the normalized value.  A normal ``LoRALinear`` therefore
    changes the trained function even when every exported tensor is exact.

    The pre-hook below records the raw input to the matching HF RMSNorm.  Base
    projections still consume their normal post-norm value; only the additive
    LoRA branch consumes this captured pre-norm value, matching Megatron.
    """

    def __init__(self, norm: nn.Module) -> None:
        self.value: torch.Tensor | None = None
        self._handle = norm.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: tuple[object, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Qwen RMSNorm received malformed hidden states")
        self.value = inputs[0]

    def current(self, normalized: torch.Tensor) -> torch.Tensor:
        value = self.value
        if value is None:
            raise RuntimeError("pre-norm LoRA input was not captured")
        if value.shape != normalized.shape:
            raise RuntimeError(
                "captured pre-norm LoRA input shape differs from projection input"
            )
        return value


class _CapturedInputLoRALinear(LoRALinear):
    """HF base projection plus a LoRA branch evaluated on captured input."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        capture: _PreNormInputCapture,
    ) -> None:
        super().__init__(base, rank=rank, alpha=alpha, dropout=0.0)
        self._capture = capture

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if not self.enabled:
            return base
        lora_input = self._capture.current(value)
        update = self.lora_B(self.lora_A(lora_input))
        return base + update.to(dtype=base.dtype) * self.scaling


def inject_exact_runtime_lora(
    model: nn.Module, *, rank: int, alpha: float
) -> tuple[str, ...]:
    selected: list[str] = []
    for layer in range(24):
        selected.extend(
            f"model.layers.{layer}.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
    for layer in range(12, 24):
        selected.extend(
            f"model.layers.{layer}.mlp.{name}"
            for name in ("gate_proj", "up_proj", "down_proj")
        )
    captures: list[_PreNormInputCapture] = []
    for layer_index in range(24):
        layer = model.get_submodule(f"model.layers.{layer_index}")
        attention_capture = _PreNormInputCapture(layer.input_layernorm)
        mlp_capture = _PreNormInputCapture(layer.post_attention_layernorm)
        captures.extend((attention_capture, mlp_capture))
        for target in ("q_proj", "k_proj", "v_proj"):
            name = f"model.layers.{layer_index}.self_attn.{target}"
            parent, child = _parent_and_child(model, name)
            base = getattr(parent, child)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"runtime LoRA target is not Linear: {name}")
            setattr(
                parent,
                child,
                _CapturedInputLoRALinear(
                    base,
                    rank=rank,
                    alpha=alpha,
                    capture=attention_capture,
                ),
            )
        if layer_index >= 12:
            for target in ("gate_proj", "up_proj"):
                name = f"model.layers.{layer_index}.mlp.{target}"
                parent, child = _parent_and_child(model, name)
                base = getattr(parent, child)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"runtime LoRA target is not Linear: {name}")
                setattr(
                    parent,
                    child,
                    _CapturedInputLoRALinear(
                        base,
                        rank=rank,
                        alpha=alpha,
                        capture=mlp_capture,
                    ),
                )
    # Output projections do not fuse a preceding normalization in Megatron;
    # their ordinary HF input is already the exact native LoRA-hook input.
    captured_targets = {
        f"model.layers.{layer}.self_attn.{target}"
        for layer in range(24)
        for target in ("q_proj", "k_proj", "v_proj")
    }
    captured_targets.update(
        f"model.layers.{layer}.mlp.{target}"
        for layer in range(12, 24)
        for target in ("gate_proj", "up_proj")
    )
    for name in selected:
        if name in captured_targets:
            continue
        parent, child = _parent_and_child(model, name)
        base = getattr(parent, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"runtime LoRA target is not Linear: {name}")
        setattr(parent, child, LoRALinear(base, rank=rank, alpha=alpha, dropout=0.0))
    # Keep hook owners alive for the lifetime of the model.  RemovableHandle
    # itself is not an nn.Module and must not enter the checkpoint state dict.
    model._uniss_prenorm_lora_captures = captures
    return tuple(selected)


def load_runtime_models(
    export_dir: Path,
    *,
    codebook_weight: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    base = Path(manifest["base_model"])
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    # The native Phase3/streaming Megatron launches use the MCore default
    # ``--layernorm-epsilon 1e-5``.  The historical HF export retained the
    # Qwen reference config's 1e-6, which is close enough for casual offline
    # generation but not for exact layer-by-layer LoRA/runtime parity.
    config.rms_norm_eps = float(manifest.get("layernorm_epsilon", 1.0e-5))
    kwargs = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "config": config,
    }
    if device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    selected = inject_exact_runtime_lora(
        model, rank=int(manifest["rank"]), alpha=float(manifest["alpha"])
    )
    adapter = load_file(export_dir / "adapter_model.safetensors")
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    missing_adapter = [
        name
        for name, _ in model.named_parameters()
        if (".lora_A." in name or ".lora_B." in name) and name not in adapter
    ]
    unexpected_adapter = [name for name in unexpected if name in adapter]
    if missing_adapter or unexpected_adapter:
        raise ValueError(
            "runtime LoRA mismatch: "
            f"missing={missing_adapter[:8]}, unexpected={unexpected_adapter[:8]}"
        )
    del missing
    model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)

    objective = TrueSubsecondObjective(
        hidden_size=int(model.config.hidden_size),
        codebook_weight=codebook_weight.detach().float().cpu(),
        adapter_layers=4,
        adapter_kernel_size=5,
        adapter_expansion=2,
        adapter_dropout=0.0,
        kd_temperature=1.5,
        action_write_weight=1.0,
        safe_positive_alpha=0.5,
    )
    objective_weights = load_file(export_dir / "objective_model.safetensors")
    missing, unexpected = objective.load_state_dict(objective_weights, strict=False)
    if missing != ["codebook.weight"] or unexpected:
        raise ValueError(
            f"runtime objective mismatch: missing={missing}, unexpected={unexpected}"
        )
    objective.to(device=device, dtype=dtype).eval()
    objective.requires_grad_(False)
    return model, tokenizer, objective, manifest, selected
