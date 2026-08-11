"""Load the frozen Phase3 HF base, mapped native LoRA and streaming sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_prefix_streaming_full198_v1.lora import LoRALinear
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    TrueSubsecondObjective,
)


def _parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


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
    for name in selected:
        parent, child = _parent_and_child(model, name)
        base = getattr(parent, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"runtime LoRA target is not Linear: {name}")
        setattr(parent, child, LoRALinear(base, rank=rank, alpha=alpha, dropout=0.0))
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
    kwargs = {
        "local_files_only": True,
        "torch_dtype": dtype,
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
