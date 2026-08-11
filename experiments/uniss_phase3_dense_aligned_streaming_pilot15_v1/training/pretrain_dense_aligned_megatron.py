#!/usr/bin/env python3
"""Native Megatron entrypoint for three dense-aligned fixed15 coverage epochs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    CoverageEpochSampler,
    IndexedDenseTrajectoryDataset,
    IndexedPhase3ReplayDataset,
    ThreeEpochGlobalShuffleSchedule,
    collate_replay_or_dense,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


METRIC_NAMES = base.METRIC_NAMES


class JointValidationDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = tuple(value for value in datasets if value is not None)
        if not self.datasets or any(len(value) <= 0 for value in self.datasets):
            raise ValueError("validation datasets must be non-empty")
        self.boundaries: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.boundaries.append(total)
        self.split = "valid"
        self.collate_fn = collate_replay_or_dense

    def __len__(self) -> int:
        return self.boundaries[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        previous = 0
        for dataset, boundary in zip(self.datasets, self.boundaries):
            if index < boundary:
                return dataset[index - previous]
            previous = boundary
        raise IndexError(index)


def _curriculum(progress: float) -> tuple[OrderedDict[str, float], float, float]:
    """Keep core Phase3/aligned CE active while adding policy losses progressively."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.10:
        policy = 0.0
        deadline = 0.0
        frontend = 0.25 + 0.75 * progress / 0.10
    elif progress < 0.50:
        policy = min(1.0, (progress - 0.10) / 0.10)
        deadline = 0.0
        frontend = 1.0
    else:
        policy = 1.0
        deadline = 0.20 * (progress - 0.50) / 0.50
        frontend = 1.0 if progress < 0.85 else 0.5
    weights = OrderedDict(
        (
            ("phase3_replay", 1.0),
            ("interleaved_trajectory", 1.0),
            # No synthetic teacher distribution is created.  Phase3 replay
            # and the frozen Phase3 root provide the quality anchor in v1.
            ("real_prefix_kd", 0.0),
            ("support_ordinal", 0.5 * policy),
            ("token_safe_commit", 0.5 * policy),
            ("deadline_survival", deadline),
            ("prefix_stability", 0.0),
            ("ar_semantic_microblock", 1.0),
            ("speaker_consistency", 0.0),
            ("boundary_continuity", 0.1 * policy),
        )
    )
    return weights, deadline, frontend


def _distributed_dense_objective(output, *, progress: float):
    if tuple(output.terms) != TERM_NAMES:
        raise ValueError("dense objective term order changed")
    weights, deadline_weight, frontend_multiplier = _curriculum(progress)
    numerators = torch.stack(
        [output.terms[name].numerator for name in TERM_NAMES]
    )
    denominators = torch.stack(
        [
            output.terms[name].denominator.to(numerators.dtype)
            for name in TERM_NAMES
        ]
    )
    global_numerators = numerators.detach().clone()
    global_denominators = denominators.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_numerators)
        dist.all_reduce(global_denominators)
        world_size = dist.get_world_size()
    active = global_denominators > 0
    local_means = torch.where(
        active,
        world_size * numerators / global_denominators.clamp_min(1),
        numerators * 0.0,
    )
    scales = numerators.new_tensor(list(weights.values()))
    total = (local_means * scales).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(TERM_NAMES)
    )
    diagnostics = torch.stack(
        [value.detach().float() for value in output.diagnostics.values()]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index])
        for index, name in enumerate(DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_tensor(
        deadline_weight
    )
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(0.35)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_tensor(
        frontend_multiplier
    )
    return total, metrics


def install_dense_lr_overrides(args) -> None:
    """Install exact parameter groups and this experiment's LR curriculum."""

    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    original_config = megatron_training.get_megatron_optimizer_config
    if not getattr(original_config, "_uniss_dense_aligned_groups", False):

        def with_dense_aligned_groups(parsed_args):
            config, overrides = original_config(parsed_args)
            overrides = dict(overrides or {})
            for attribute, values in base.lr_group_values(parsed_args).items():
                overrides[ParamKey(attr=attribute)] = values
            return config, overrides

        with_dense_aligned_groups._uniss_dense_aligned_groups = True
        megatron_training.get_megatron_optimizer_config = with_dense_aligned_groups

    original_get_lr = OptimizerParamScheduler.get_lr
    if not getattr(original_get_lr, "_uniss_dense_aligned_frontend", False):

        def get_lr_with_dense_curriculum(self, param_group):
            value = original_get_lr(self, param_group)
            if not param_group.get("uniss_dynamic_frontend_lr", False):
                return value
            denominator = max(1.0, float(self.lr_decay_steps))
            progress = min(1.0, max(0.0, float(self.num_steps) / denominator))
            return value * _curriculum(progress)[2]

        get_lr_with_dense_curriculum._uniss_dense_aligned_frontend = True
        OptimizerParamScheduler.get_lr = get_lr_with_dense_curriculum


def _dense_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    objective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("dense TP=PP=1 expects flattened [tokens,1,*]")
    hidden = hidden[:, 0]
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
    batch = context["batch"]
    if context["sample_kind"] == "replay":
        output = objective.replay(logits, labels, loss_mask)
    elif context["sample_kind"] == "trajectory":
        output = objective.trajectory(
            hidden,
            logits,
            labels,
            loss_mask,
            batch["token_roles"].reshape(-1),
            context["word_embedding_weight"],
            batch,
            frontend_residual_rms=context["frontend_residual_rms"],
        )
    else:
        raise ValueError(f"unknown dense sample kind: {context['sample_kind']}")
    total, metrics = _distributed_dense_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("dense metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite dense-aligned loss component")
    return torch.stack(values)


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base.add_experiment_args(parser)
    group = parser.add_argument_group(title="UniSS dense-aligned fixed15")
    group.add_argument("--dense-training-manifest", required=True)
    group.add_argument("--dense-coverage-epochs", type=int, default=3)
    group.add_argument("--dense-replay-fraction", type=float, default=0.35)
    return parser


def _manifest(args) -> dict[str, object]:
    value = json.loads(Path(args.dense_training_manifest).read_text(encoding="utf-8"))
    if value.get("schema_version") != "uniss_dense_aligned_streaming_training_manifest_v1":
        raise ValueError("unexpected dense training manifest schema")
    if int(value["coverage_epochs"]) != int(args.dense_coverage_epochs):
        raise ValueError("coverage epoch count differs from frozen manifest")
    if int(value["train_iters"]) != int(args.train_iters):
        raise ValueError("Megatron train-iters differs from frozen manifest")
    if int(value["micro_batch_size"]) != int(args.micro_batch_size):
        raise ValueError("MBS differs from frozen manifest")
    if int(value["global_batch_size"]) != int(args.global_batch_size):
        raise ValueError("GBS differs from frozen manifest")
    return value


def validate_experiment_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("dense-aligned training requires --sft")
    if int(args.seq_length) != 18_000:
        raise ValueError("dense-aligned formal packs require seq-length 18000")
    if int(args.tensor_model_parallel_size) != 1:
        raise ValueError("dense-aligned v1 requires TP=1")
    if int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("dense-aligned v1 requires PP=1")
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("validated dense micro batch sizes are 1 and 2")
    if int(args.global_batch_size) != 128:
        raise ValueError("dense-aligned training requires GBS=128")
    for path in (
        args.true_trajectory_packed,
        args.true_trajectory_offsets,
        args.true_replay_packed,
        args.true_replay_offsets,
        args.true_whispervq_codebook,
        args.dense_training_manifest,
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
    trajectory = IndexedDenseTrajectoryDataset(
        args.true_trajectory_packed, seq_length=int(args.seq_length)
    )
    replay = IndexedPhase3ReplayDataset(
        args.true_replay_packed,
        args.true_replay_offsets,
        seq_length=int(args.seq_length),
        require_complete=True,
    )
    target_train = _target_count(train_val_test_num_samples, 0)
    if target_train is None:
        raise ValueError("Megatron did not provide a dense train sample count")
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    train = ThreeEpochGlobalShuffleSchedule(
        trajectory,
        replay,
        coverage_epochs=int(args.dense_coverage_epochs),
        data_parallel_group_size=dp_group,
        global_batch_size=int(args.global_batch_size),
        shuffle_seed=int(args.seed),
        target_replay_fraction=float(args.dense_replay_fraction),
    )
    if len(train) != target_train or len(train) != int(manifest["total_samples"]):
        raise ValueError(
            f"dense schedule length {len(train)} differs from target {target_train}"
        )
    valid_sources: list[Dataset] = []
    if args.true_valid_trajectory_packed:
        valid_sources.append(
            IndexedDenseTrajectoryDataset(
                args.true_valid_trajectory_packed,
                seq_length=int(args.seq_length),
            )
        )
    if args.true_valid_replay_packed:
        valid_sources.append(
            IndexedPhase3ReplayDataset(
                args.true_valid_replay_packed,
                args.true_valid_replay_offsets,
                seq_length=int(args.seq_length),
                require_complete=True,
            )
        )
    valid = JointValidationDataset(valid_sources) if valid_sources else None
    runtime.print_rank_0(
        "> dense-aligned datasets: "
        f"trajectory={len(trajectory)} replay_subset={len(replay)} "
        f"coverage_epochs={train.coverage_epochs} epoch_samples={train.epoch_samples} "
        f"total_samples={len(train)} seed={train.shuffle_seed} "
        f"valid={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def install_coverage_sampler() -> None:
    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_dense_coverage_sampler", False):
        return

    def coverage_or_default(dataset, *args, **kwargs):
        if isinstance(dataset, ThreeEpochGlobalShuffleSchedule):
            return CoverageEpochSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    coverage_or_default._uniss_dense_coverage_sampler = True
    data_samplers.MegatronPretrainingRandomSampler = coverage_or_default


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    # The historical model wrapper resolves this module-global processor at
    # runtime; replace it only inside this isolated entrypoint.
    base._joint_output_processor = _dense_output_processor
    install_dense_lr_overrides(args)
    install_coverage_sampler()
    base.install_joint_collate()
    base.install_rerun_checkpoint_compatibility()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        base.forward_step,
        model_provider=base.model_provider,
    )


if __name__ == "__main__":
    main()
