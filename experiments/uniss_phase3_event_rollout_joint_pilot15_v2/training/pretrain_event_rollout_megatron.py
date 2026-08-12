#!/usr/bin/env python3
"""Megatron fixed15 event rollout with a trainable causal source frontend."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_event_rollout_joint_full198_v1.training.pretrain_event_rollout_megatron as event
import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
    distributed_event_rollout_objective,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.pretrain_event_rollout_megatron import (
    add_experiment_args,
    train_valid_test_datasets_provider,
    validate_experiment_args,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


_V1_EVENT_TRAINABLE = event._is_event_rollout_trainable
FRONTEND_PREFIXES = (
    "true_subsecond_objective.frontend_adapter.",
    "true_subsecond_objective.frontend_projection.",
)


def is_event_rollout_v2_trainable_parameter(name: str) -> bool:
    """Retain the v1 scope and repair its omitted causal frontend."""

    return _V1_EVENT_TRAINABLE(name) or any(prefix in name for prefix in FRONTEND_PREFIXES)


def install_event_rollout_v2_model() -> None:
    event._is_event_rollout_trainable = is_event_rollout_v2_trainable_parameter
    event.install_event_rollout_model()


def main() -> None:
    base.TrueSubsecondObjective = EventRolloutJointObjective
    base.METRIC_NAMES = event.METRIC_NAMES
    dense.METRIC_NAMES = event.METRIC_NAMES
    dense._distributed_dense_objective = distributed_event_rollout_objective
    dense.JointValidationDataset = SynchronizedValidationDataset
    install_event_rollout_v2_model()

    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    load_root = Path(args.load).resolve()
    phase3_root = Path(
        "checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4"
    ).resolve()
    save_root = Path(args.save).resolve()
    fresh = load_root == phase3_root
    resume = load_root == save_root
    if not (fresh or resume):
        raise ValueError("pilot15 v2 may load only Phase3 v4 or its own checkpoint root")
    if fresh:
        latest = (load_root / "latest_checkpointed_iteration.txt").read_text().strip()
        if latest != "9075":
            raise ValueError("pilot15 v2 fresh run must start at Phase3 iter_0009075")
        if str(args.dist_ckpt_strictness) not in {"log_all", "StrictHandling.LOG_ALL"}:
            raise ValueError("fresh Phase3 handoff requires log_all key audit")
    elif str(args.dist_ckpt_strictness) not in {"raise_all", "StrictHandling.RAISE_ALL"}:
        raise ValueError("pilot15 v2 self-resume requires raise_all")

    dense.install_dense_lr_overrides(args)
    dense.install_coverage_sampler()
    base.install_joint_collate()
    base.install_rerun_checkpoint_compatibility()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "experiment": "uniss_phase3_event_rollout_joint_pilot15_v2",
                    "repair": "trainable_causal_frontend",
                    "scope": "fixed_shards_00000_00014",
                    "fresh_phase3": fresh,
                    "load": str(load_root),
                    "event_rollout": "exact_model_induced_variable_grammar",
                    "shuffle": "global_randperm_over_multifile_pack_namespace",
                    "metric_count": len(event.METRIC_NAMES),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        event.forward_step,
        model_provider=base.model_provider,
    )


if __name__ == "__main__":
    main()
