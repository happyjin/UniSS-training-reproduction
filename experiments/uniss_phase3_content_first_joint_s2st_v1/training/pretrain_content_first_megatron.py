#!/usr/bin/env python3
"""Phase-3-rooted Megatron SFT with phrase-gated streaming supervision.

This isolated entry point reuses the audited fixed15 trajectory namespace and
the native event-rollout implementation.  It installs only process-local
overrides: short WRITE labels are coalesced into phrases and the objective
emphasizes content/replay over action classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset as event_dataset
import experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective as event_objective
import experiments.uniss_phase3_event_rollout_joint_pilot15_v2.training.pretrain_event_rollout_megatron as prior
import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
    ROLLOUT_METRIC_NAMES,
    distributed_event_rollout_objective,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.pretrain_event_rollout_megatron import (
    add_experiment_args,
    train_valid_test_datasets_provider,
    validate_experiment_args,
)
from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_phase3_content_first_joint_s2st_v1.training.phrase_oracle import (
    phrase_oracle_sessions,
)


MINIMUM_COMMIT_TOKENS = 4
CONTENT_FIRST_WEIGHTS = {
    "phase3_replay": 1.50,
    "interleaved_trajectory": 0.50,
    "real_prefix_kd": 0.50,
    "support_ordinal": 0.10,
    "token_safe_commit": 0.25,
    "deadline_survival": 0.50,
    "prefix_stability": 0.50,
    "ar_semantic_microblock": 0.25,
    "speaker_consistency": 0.25,
    "boundary_continuity": 0.50,
    "microblock_semantic_content": 2.00,
    "microblock_final_length": 1.00,
    "microblock_continue": 1.00,
    "runtime_text_content": 5.00,
    "runtime_critical_boundary": 1.50,
    "runtime_action": 0.35,
    "runtime_continuation": 1.25,
}


def install_content_first_overrides() -> None:
    """Install process-local pack view and content-first objective weights."""

    def phrase_view(value):
        return phrase_oracle_sessions(value, minimum_tokens=MINIMUM_COMMIT_TOKENS)

    # Both dataset functions resolve this imported global at call time.
    event_dataset.oracle_sessions_from_pack = phrase_view
    for name, weight in CONTENT_FIRST_WEIGHTS.items():
        if name not in event_objective.ROLLOUT_WEIGHTS:
            raise KeyError(f"unknown event-rollout loss {name}")
        event_objective.ROLLOUT_WEIGHTS[name] = float(weight)


def main() -> None:
    install_content_first_overrides()
    base.TrueSubsecondObjective = EventRolloutJointObjective
    base.METRIC_NAMES = ROLLOUT_METRIC_NAMES
    dense.METRIC_NAMES = ROLLOUT_METRIC_NAMES
    dense._distributed_dense_objective = distributed_event_rollout_objective
    prior.install_event_rollout_v2_model()

    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    load_root = Path(args.load).resolve()
    phase3_root = Path("checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4").resolve()
    save_root = Path(args.save).resolve()
    fresh = load_root == phase3_root
    resume = load_root == save_root
    if not (fresh or resume):
        raise ValueError("content-first training may load only Phase3 v4 or its own checkpoint root")
    if fresh:
        if (load_root / "latest_checkpointed_iteration.txt").read_text().strip() != "9075":
            raise ValueError("fresh content-first run must start at Phase3 iter_0009075")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "experiment": "uniss_phase3_content_first_joint_s2st_v1",
                    "load": str(load_root),
                    "fresh_phase3": fresh,
                    "scope": "fixed_shards_00000_00014",
                    "phrase_minimum_tokens": MINIMUM_COMMIT_TOKENS,
                    "objective_weights": CONTENT_FIRST_WEIGHTS,
                    "shuffle": "global_randperm_over_multifile_pack_namespace",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    prior.dense.install_dense_lr_overrides(args)
    prior.dense.install_coverage_sampler()
    base.install_joint_collate()
    base.install_rerun_checkpoint_compatibility()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        prior.event.forward_step,
        model_provider=base.model_provider,
    )


if __name__ == "__main__":
    main()
