"""Load the v5 parallel-semantic objective beside the exact Phase3 HF runtime."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.pretrain_overfit5 import (
    RuntimeParityOverfit5Objective,
)
from web_demo.true_subsecond_pilot15_streaming_v1.model_loader import (
    inject_exact_runtime_lora,
)


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
    _, unexpected = model.load_state_dict(adapter, strict=False)
    missing_adapter = [
        name
        for name, _ in model.named_parameters()
        if (".lora_A." in name or ".lora_B." in name) and name not in adapter
    ]
    if missing_adapter or [name for name in unexpected if name in adapter]:
        raise ValueError("runtime v5 LoRA state does not match the exact HF graph")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    objective = RuntimeParityOverfit5Objective(
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
            f"runtime v5 objective mismatch: missing={missing}, unexpected={unexpected}"
        )
    objective.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, tokenizer, objective, manifest, selected


__all__ = ["load_runtime_models"]

