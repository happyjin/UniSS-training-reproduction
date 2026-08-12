"""Load fixed15 event-rollout heads beside the exact Phase3 HF runtime."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
)
from web_demo.true_subsecond_pilot15_streaming_v1.model_loader import (
    inject_exact_runtime_lora,
)


def validate_objective_state(
    objective: EventRolloutJointObjective,
    state: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Load every learned sidecar tensor and reject structural drift."""

    missing, unexpected = objective.load_state_dict(state, strict=False)
    if missing != ["codebook.weight"] or unexpected:
        raise ValueError(
            "event-rollout objective state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    required = {
        "action_head.network.3.weight",
        "semantic_microblock_head.continue_head.3.weight",
        "continuation_head.weight",
    }
    absent = sorted(required - set(state))
    if absent:
        raise ValueError(f"event-rollout export lacks runtime heads: {absent}")
    return {
        "objective_tensor_count": len(state),
        "continuation_head_shape": list(state["continuation_head.weight"].shape),
        "microblock_size": int(objective.semantic_microblock_head.block_size),
    }


def load_runtime_models(export_dir: Path, *, codebook_weight, device, dtype=None):
    """Reconstruct the exact HF graph used by the deployed causal runtime."""

    export_dir = Path(export_dir).resolve()
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    base = Path(manifest["base_model"]).resolve()
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    config.rms_norm_eps = float(manifest.get("layernorm_epsilon", 1.0e-5))
    kwargs = {"local_files_only": True, "torch_dtype": dtype, "config": config}
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
        raise ValueError("event-rollout LoRA state does not match the exact HF graph")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    objective = EventRolloutJointObjective(
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
    state = dict(load_file(export_dir / "objective_model.safetensors"))
    objective_audit = validate_objective_state(objective, state)
    objective.to(device=device, dtype=dtype).eval().requires_grad_(False)
    manifest = dict(manifest)
    manifest["event_rollout_objective_audit"] = objective_audit
    return model, tokenizer, objective, manifest, selected


__all__ = ["load_runtime_models", "validate_objective_state"]
