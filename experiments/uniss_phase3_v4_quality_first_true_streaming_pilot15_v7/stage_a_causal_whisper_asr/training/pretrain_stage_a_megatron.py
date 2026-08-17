#!/usr/bin/env python3
"""Megatron Stage A v7 with independent curriculum and optimizer horizons."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    load_megatron_runtime,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    curriculum_group_multiplier as v5_curriculum_group_multiplier,
)


_CURRICULUM_SCALE = 1.0
_native = v2_entrypoint.implementation
_original_add_experiment_args = _native.add_experiment_args
_original_validate_experiment_args = _native.validate_experiment_args
_original_train_valid_test_datasets_provider = (
    _native.train_valid_test_datasets_provider
)


class PrefixStageASchedule(Dataset):
    """Bounded diagnostic prefix of an already globally shuffled schedule."""

    def __init__(self, schedule: Dataset, total_samples: int) -> None:
        if total_samples <= 0 or total_samples > len(schedule):
            raise ValueError("invalid v7 prefix schedule length")
        group_size = int(schedule.data_parallel_group_size)
        if total_samples % group_size:
            raise ValueError("v7 prefix must end on a data-parallel group")
        global_batch_size = int(schedule.global_batch_size)
        if total_samples % global_batch_size:
            raise ValueError("v7 prefix must end on a global update")
        self.schedule = schedule
        self.total_samples = int(total_samples)
        self.data_parallel_group_size = group_size
        self.global_batch_size = global_batch_size
        self.coverage_epochs = int(schedule.coverage_epochs)
        self.epoch_samples = int(schedule.epoch_samples)
        self.shuffle_seed = int(schedule.shuffle_seed)
        self.synchronize_sample_kind = True
        self.split = "train"
        self.collate_fn = schedule.collate_fn

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.schedule[index]

    def source_index(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.schedule.source_index(index)


def effective_curriculum_progress(
    consumed_samples: int,
    global_batch_size: int,
    curriculum_iters: int,
) -> float:
    """Return clamped curriculum progress using its explicit horizon."""

    if consumed_samples < 0 or global_batch_size <= 0 or curriculum_iters <= 0:
        raise ValueError("invalid v7 curriculum position")
    denominator = global_batch_size * curriculum_iters
    return min(1.0, max(0.0, consumed_samples / denominator))


def configure_training_horizons(args: SimpleNamespace) -> float:
    """Validate and install independent optimizer/curriculum clocks."""

    global _CURRICULUM_SCALE

    train_iters = int(args.train_iters)
    curriculum_iters = int(args.stage_a_curriculum_iters)
    optimizer_iters = int(args.stage_a_optimizer_iters)
    warmup_iters = int(args.stage_a_optimizer_warmup_iters)
    if train_iters <= 0:
        raise ValueError("v7 train iters must be positive")
    if not 1 <= curriculum_iters <= optimizer_iters <= train_iters:
        raise ValueError(
            "v7 horizons must satisfy 1 <= curriculum <= optimizer <= train"
        )
    if not 0 <= warmup_iters < optimizer_iters:
        raise ValueError("v7 optimizer warmup must be in [0, optimizer_iters)")

    args.lr_decay_iters = optimizer_iters
    args.lr_warmup_iters = warmup_iters
    _CURRICULUM_SCALE = optimizer_iters / curriculum_iters
    return _CURRICULUM_SCALE


def add_experiment_args(parser):
    parser = _original_add_experiment_args(parser)
    group = parser.add_argument_group(title="UniSS quality-first Stage A v7")
    group.add_argument("--stage-a-curriculum-iters", type=int, required=True)
    group.add_argument("--stage-a-optimizer-iters", type=int, required=True)
    group.add_argument(
        "--stage-a-optimizer-warmup-iters", type=int, required=True
    )
    group.add_argument("--stage-a-prefix-schedule", action="store_true")
    return parser


def validate_experiment_args(args) -> None:
    _original_validate_experiment_args(args)
    configure_training_horizons(args)


def curriculum_group_multiplier(group, progress: float) -> float:
    """Map optimizer-clock progress onto the independent curriculum clock."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A v7 LR progress")
    effective = min(1.0, progress * _CURRICULUM_SCALE)
    return v5_curriculum_group_multiplier(group, effective)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Use an exact shuffled prefix only for the explicit diagnostic canary."""

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not bool(args.stage_a_prefix_schedule):
        return _original_train_valid_test_datasets_provider(
            train_val_test_num_samples, vp_stage=vp_stage
        )
    del vp_stage
    source = _native._dataset(args, args.stage_a_train_packs)
    target_train = int(train_val_test_num_samples[0])
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    complete = _native.ThreeEpochStageASchedule(
        source,
        coverage_epochs=int(args.stage_a_coverage_epochs),
        data_parallel_group_size=dp_group,
        global_batch_size=int(args.global_batch_size),
        shuffle_seed=int(args.seed),
    )
    train = PrefixStageASchedule(complete, target_train)
    valid = None
    valid_source_length = 0
    if args.stage_a_valid_packs:
        valid_source = _native._dataset(args, args.stage_a_valid_packs)
        valid_source_length = len(valid_source)
        eval_micro_batch = int(
            getattr(args, "eval_micro_batch_size", None) or args.micro_batch_size
        )
        valid = _native.PaddedStageAValidationDataset(
            valid_source,
            minimum_samples=int(train_val_test_num_samples[1]),
            data_parallel_group_size=int(args.data_parallel_size)
            * eval_micro_batch,
        )
    runtime.print_rank_0(
        "> Stage A v7 prefix datasets: "
        f"source_packs={len(source)} coverage_epochs={complete.coverage_epochs} "
        f"complete_samples={len(complete)} prefix_samples={len(train)} "
        f"global_shuffle_seed={complete.shuffle_seed} "
        f"valid_source={valid_source_length} "
        f"valid_effective={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = base.prepare_packed_batch(next(data_iterator), int(args.seq_length))
    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    global_batch_size = max(1, int(args.global_batch_size))
    progress = effective_curriculum_progress(
        consumed,
        global_batch_size,
        int(args.stage_a_curriculum_iters),
    )
    batch["training_progress"] = torch.tensor(
        progress,
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["training_update"] = torch.tensor(
        consumed // global_batch_size,
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    packed_seq_params = base.build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        stage_a_batch=batch,
    )
    return output, _native.loss_func


def install_v7_overrides() -> None:
    _native.add_experiment_args = add_experiment_args
    _native.validate_experiment_args = validate_experiment_args
    _native.forward_step = forward_step
    _native.train_valid_test_datasets_provider = train_valid_test_datasets_provider
    _native.StageAObjective = StageAObjective
    _native.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    _native.TERM_NAMES = TERM_NAMES
    _native.chunk_pair_for_progress = chunk_pair_for_progress
    _native.distributed_stage_a_objective = distributed_stage_a_objective
    _native.curriculum_group_multiplier = curriculum_group_multiplier
    _native.METRIC_NAMES = (
        *TERM_NAMES,
        *DIAGNOSTIC_NAMES,
        *_native.CURRICULUM_METRICS,
    )


def main() -> None:
    install_v7_overrides()
    v2_entrypoint.main()


if __name__ == "__main__":
    main()


__all__ = [
    "add_experiment_args",
    "configure_training_horizons",
    "curriculum_group_multiplier",
    "effective_curriculum_progress",
    "forward_step",
    "install_v7_overrides",
    "main",
    "PrefixStageASchedule",
    "train_valid_test_datasets_provider",
    "validate_experiment_args",
]
