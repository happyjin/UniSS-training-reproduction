#!/usr/bin/env python3
"""Megatron entrypoint for the fixed15 multi-file exact event rollout."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_event_rollout_joint_full198_v1.training.pretrain_event_rollout_megatron as event
import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    collate_event_rollout,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
    distributed_event_rollout_objective,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_training_manifest import (
    MANIFEST_SCHEMA,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MultiFileIndexedEventRolloutTrajectoryDataset,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


def add_experiment_args(parser):
    parser = dense.add_experiment_args(parser)
    group = parser.add_argument_group(title="UniSS event rollout fixed15 multi-file")
    group.add_argument("--pilot15-trajectory-manifest", required=True)
    group.add_argument("--pilot15-valid-trajectory-manifest", required=True)
    return parser


def _manifest(args) -> dict[str, object]:
    value = json.loads(Path(args.dense_training_manifest).read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected/incomplete pilot15 training manifest")
    checks = {
        "coverage_epochs": int(args.dense_coverage_epochs),
        "train_iters": int(args.train_iters),
        "micro_batch_size": int(args.micro_batch_size),
        "global_batch_size": int(args.global_batch_size),
        "seq_length": int(args.seq_length),
    }
    for key, expected in checks.items():
        if int(value[key]) != expected:
            raise ValueError(f"pilot15 manifest {key} differs from Megatron args")
    if Path(value["trajectory_manifest"]).resolve() != Path(
        args.pilot15_trajectory_manifest
    ).resolve():
        raise ValueError("pilot15 train trajectory manifest changed")
    if Path(value["valid_trajectory_manifest"]).resolve() != Path(
        args.pilot15_valid_trajectory_manifest
    ).resolve():
        raise ValueError("pilot15 validation trajectory manifest changed")
    if not bool(value.get("strict_global_shuffle")):
        raise ValueError("pilot15 manifest does not require strict global shuffle")
    return value


def validate_experiment_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("pilot15 event rollout requires --sft")
    if int(args.seq_length) != 18_000:
        raise ValueError("pilot15 event rollout requires seq_length=18000")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("pilot15 event rollout requires TP=PP=1")
    if int(args.micro_batch_size) != 2 or int(args.global_batch_size) != 128:
        raise ValueError("pilot15 event rollout requires MBS=2, GBS=128")
    for path in (
        args.dense_training_manifest,
        args.pilot15_trajectory_manifest,
        args.pilot15_valid_trajectory_manifest,
        args.true_replay_packed,
        args.true_replay_offsets,
        args.true_whispervq_codebook,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    _manifest(args)


def _target_count(values, index: int) -> int | None:
    if values is None or index >= len(values) or values[index] is None:
        return None
    return int(values[index])


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    manifest = _manifest(args)
    trajectory = MultiFileIndexedEventRolloutTrajectoryDataset(
        args.pilot15_trajectory_manifest,
        seq_length=int(args.seq_length),
        expected_split="train",
    )
    replay = dense.IndexedPhase3ReplayDataset(
        args.true_replay_packed,
        args.true_replay_offsets,
        seq_length=int(args.seq_length),
        require_complete=True,
    )
    target_train = _target_count(train_val_test_num_samples, 0)
    if target_train is None:
        raise ValueError("Megatron did not provide a pilot15 train sample count")
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    train = dense.ThreeEpochGlobalShuffleSchedule(
        trajectory,
        replay,
        coverage_epochs=int(args.dense_coverage_epochs),
        data_parallel_group_size=dp_group,
        global_batch_size=int(args.global_batch_size),
        shuffle_seed=int(args.seed),
        target_replay_fraction=float(args.dense_replay_fraction),
    )
    train.collate_fn = collate_event_rollout
    if len(train) != target_train or len(train) != int(manifest["total_samples"]):
        raise ValueError("pilot15 schedule differs from the frozen training manifest")
    valid_trajectory = MultiFileIndexedEventRolloutTrajectoryDataset(
        args.pilot15_valid_trajectory_manifest,
        seq_length=int(args.seq_length),
        expected_split="valid",
    )
    valid = SynchronizedValidationDataset([valid_trajectory])
    valid.collate_fn = collate_event_rollout
    runtime.print_rank_0(
        "> pilot15 multi-file event-rollout datasets: "
        f"trajectory={len(trajectory)} replay={len(replay)} "
        f"epochs={train.coverage_epochs} epoch_samples={train.epoch_samples} "
        f"total={len(train)} valid={len(valid_trajectory)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    base.TrueSubsecondObjective = EventRolloutJointObjective
    base.METRIC_NAMES = event.METRIC_NAMES
    dense.METRIC_NAMES = event.METRIC_NAMES
    dense._distributed_dense_objective = distributed_event_rollout_objective
    dense.JointValidationDataset = SynchronizedValidationDataset
    event.install_event_rollout_model()

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
        raise ValueError("pilot15 may load only Phase3 v4 or its own checkpoint root")
    if fresh:
        latest = (load_root / "latest_checkpointed_iteration.txt").read_text().strip()
        if latest != "9075":
            raise ValueError("pilot15 fresh run must start at Phase3 iter_0009075")
        if str(args.dist_ckpt_strictness) not in {"log_all", "StrictHandling.LOG_ALL"}:
            raise ValueError("fresh Phase3 handoff requires log_all key audit")
    elif str(args.dist_ckpt_strictness) not in {"raise_all", "StrictHandling.RAISE_ALL"}:
        raise ValueError("pilot15 self-resume requires raise_all")

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
                    "experiment": "uniss_phase3_event_rollout_joint_pilot15_v1",
                    "scope": "fixed_shards_00000_00014",
                    "fresh_phase3": fresh,
                    "load": str(load_root),
                    "trajectory_manifest": str(Path(args.pilot15_trajectory_manifest).resolve()),
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

