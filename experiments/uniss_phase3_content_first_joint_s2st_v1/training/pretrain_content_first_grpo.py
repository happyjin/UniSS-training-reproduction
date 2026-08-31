#!/usr/bin/env python3
"""Coverage GRPO on top of the exact frozen content-first SFT graph."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader

import experiments.uniss_phasea_coverage_constrained_grpo_v3.training.pretrain_event_grpo as grpo
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model.megatron_lora import (
    inject_native_megatron_lora,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    load_whispervq_codebook,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.model.dual_lora import (
    inject_top_layer_dual_lora,
)


CONTENT_PREFIXES = (
    "embedding.",
    "decoder.",
    "output_layer.",
    "true_subsecond_lora.",
    "true_subsecond_objective.",
)
POLICY_PREFIX = "quality_grpo_lora."


def _base_key(value: str) -> str:
    return str(value).split("/shard_", 1)[0]


def audit_content_first_handoff(model, load_root: str | Path) -> dict[str, object]:
    """Require exact SFT state coverage while allowing only the GRPO overlay."""

    from megatron.core import parallel_state

    checkpoint = grpo.e2e._resolve_checkpoint(load_root)
    metadata = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    checkpoint_keys = {
        _base_key(key)
        for key in metadata
        if _base_key(key).startswith((*CONTENT_PREFIXES, POLICY_PREFIX))
    }
    current_raw = model.sharded_state_dict(
        metadata={
            "dp_cp_group": parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        }
    )
    current = {
        _base_key(getattr(value, "key", key)) for key, value in current_raw.items()
    }
    missing = sorted(checkpoint_keys - current)
    illegal = sorted(
        key
        for key in current - checkpoint_keys
        if key.startswith(CONTENT_PREFIXES)
    )
    content_checkpoint = sorted(
        key for key in checkpoint_keys if key.startswith(CONTENT_PREFIXES)
    )
    policy_current = sorted(key for key in current if key.startswith(POLICY_PREFIX))
    policy_checkpoint = sorted(
        key for key in checkpoint_keys if key.startswith(POLICY_PREFIX)
    )
    if (
        missing
        or illegal
        or not content_checkpoint
        or not policy_current
        or (policy_checkpoint and policy_checkpoint != policy_current)
    ):
        raise RuntimeError(
            "content-first GRPO handoff failed: "
            f"missing={missing[:8]} illegal={illegal[:8]} "
            f"content={len(content_checkpoint)} "
            f"policy_checkpoint={len(policy_checkpoint)} "
            f"policy_current={len(policy_current)}"
        )
    return {
        "content_checkpoint_keys": len(content_checkpoint),
        "policy_checkpoint_keys": len(policy_checkpoint),
        "policy_current_keys": len(policy_current),
        "missing_checkpoint_keys": 0,
        "illegal_content_keys": 0,
    }


def augment_content_first_model(model, args) -> dict[str, object]:
    """Rebuild frozen SFT modules, then add only the trainable GRPO delta."""

    codebook_path = Path(args.event_whispervq_model) / "model.safetensors"
    content_lora = inject_native_megatron_lora(
        model,
        rank=32,
        alpha=64.0,
        dropout=0.0,
        mlp_last_layers=12,
    )
    objective = EventRolloutJointObjective(
        hidden_size=int(args.hidden_size),
        codebook_weight=load_whispervq_codebook(codebook_path),
        adapter_layers=4,
        adapter_kernel_size=5,
        adapter_expansion=2,
        adapter_dropout=0.0,
        kd_temperature=1.5,
        action_write_weight=1.0,
        safe_positive_alpha=0.5,
    )
    model.add_module("true_subsecond_objective", objective)
    policy = inject_top_layer_dual_lora(
        model,
        top_layers=int(args.event_top_layers),
        rank=int(args.event_lora_rank),
        alpha=float(args.event_lora_alpha),
        dropout=float(args.event_lora_dropout),
    )
    grpo.attach_forward(model)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != policy.trainable_parameters:
        raise RuntimeError(
            "content-first GRPO trainable scope escaped policy LoRA: "
            f"actual={trainable} expected={policy.trainable_parameters}"
        )
    return {
        "frozen_content_lora_targets": len(content_lora.module_names),
        "frozen_content_lora_parameters": content_lora.trainable_parameters,
        "frozen_objective_parameters": sum(
            parameter.numel() for parameter in objective.parameters()
        ),
        "policy_layers": [policy.first_layer, policy.last_layer],
        "policy_targets": len(policy.module_names),
        "trainable_policy_parameters": policy.trainable_parameters,
    }


def main() -> None:
    grpo.augment_model = augment_content_first_model
    grpo.audit_stage_a_handoff = audit_content_first_handoff
    grpo.main()


if __name__ == "__main__":
    main()

