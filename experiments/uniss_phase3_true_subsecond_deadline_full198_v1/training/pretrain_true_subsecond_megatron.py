#!/usr/bin/env python3
"""Native Megatron entrypoint for the full198 true-subsecond joint epoch.

The Phase3 GPT remains the checkpoint root module: its historical parameter
names are unchanged, while LoRA branches and the causal streaming objective are
registered as new children.  This lets a fresh run non-strictly load Phase3 v4
and lets every later checkpoint resume strictly with optimizer, scheduler,
sampler, and curriculum state intact.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model.megatron_lora import (
    MegatronLoRASummary,
    inject_native_megatron_lora,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.curriculum import (
    point_for_progress,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.dataset import (
    CurriculumKindRandomSampler,
    DeterministicReplayTrajectorySchedule,
    IndexedPhase3ReplayDataset,
    IndexedTrajectoryDataset,
    collate_replay_or_trajectory,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    TrueSubsecondObjective,
    distributed_weighted_objective,
    load_whispervq_codebook,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


METRIC_NAMES = (
    *TERM_NAMES,
    *DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)


class JointValidationDataset(Dataset):
    """Immutable validation concatenation without curriculum resampling."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = tuple(dataset for dataset in datasets if dataset is not None)
        if not self.datasets or any(len(dataset) <= 0 for dataset in self.datasets):
            raise ValueError("validation datasets must be non-empty")
        self.boundaries: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.boundaries.append(total)
        self.split = "valid"
        self.collate_fn = collate_replay_or_trajectory

    def __len__(self) -> int:
        return self.boundaries[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        previous = 0
        for dataset, boundary in zip(self.datasets, self.boundaries):
            if index < boundary:
                return dataset[index - previous]
            previous = boundary
        raise AssertionError("validation boundary lookup failed")


def lr_group_values(args) -> dict[str, dict[str, float | bool]]:
    return {
        "uniss_lr_qwen_lora": {
            "lr_mult": float(args.true_lr_qwen_lora) / float(args.lr),
            "max_lr": float(args.true_lr_qwen_lora),
            "min_lr": float(args.true_min_lr),
        },
        "uniss_lr_frontend": {
            "lr_mult": float(args.true_lr_frontend) / float(args.lr),
            "max_lr": float(args.true_lr_frontend),
            "min_lr": float(args.true_min_lr),
            "uniss_dynamic_frontend_lr": True,
        },
        "uniss_lr_new_heads": {
            "lr_mult": float(args.true_lr_new_heads) / float(args.lr),
            "max_lr": float(args.true_lr_new_heads),
            "min_lr": float(args.true_min_lr),
        },
    }


def install_megatron_lr_overrides(args) -> None:
    """Install isolated exact LR groups and the progress frontend multiplier."""

    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    original_config = megatron_training.get_megatron_optimizer_config
    if not getattr(original_config, "_uniss_true_subsecond_groups", False):

        def with_true_subsecond_groups(parsed_args):
            config, overrides = original_config(parsed_args)
            overrides = dict(overrides or {})
            for attribute, values in lr_group_values(parsed_args).items():
                overrides[ParamKey(attr=attribute)] = values
            return config, overrides

        with_true_subsecond_groups._uniss_true_subsecond_groups = True
        megatron_training.get_megatron_optimizer_config = with_true_subsecond_groups

    original_get_lr = OptimizerParamScheduler.get_lr
    if not getattr(original_get_lr, "_uniss_true_subsecond_frontend", False):

        def get_lr_with_frontend_curriculum(self, param_group):
            value = original_get_lr(self, param_group)
            if not param_group.get("uniss_dynamic_frontend_lr", False):
                return value
            denominator = max(1.0, float(self.lr_decay_steps))
            progress = min(1.0, max(0.0, float(self.num_steps) / denominator))
            return value * point_for_progress(progress).frontend_lr_multiplier

        get_lr_with_frontend_curriculum._uniss_true_subsecond_frontend = True
        OptimizerParamScheduler.get_lr = get_lr_with_frontend_curriculum


def install_curriculum_sampler() -> None:
    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_true_subsecond_sampler", False):
        return

    def curriculum_or_default(dataset, *args, **kwargs):
        if getattr(dataset, "synchronize_sample_kind", False):
            return CurriculumKindRandomSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    curriculum_or_default._uniss_true_subsecond_sampler = True
    data_samplers.MegatronPretrainingRandomSampler = curriculum_or_default


def install_joint_collate() -> None:
    import megatron.training.datasets.data_samplers as data_samplers
    import megatron.training.training as megatron_training

    original = data_samplers.build_pretraining_data_loader
    if getattr(original, "_uniss_true_subsecond_collate", False):
        return

    def build_with_collate(dataset, *args, **kwargs):
        collate = getattr(dataset, "collate_fn", None)
        if not callable(collate):
            return original(dataset, *args, **kwargs)
        original_loader = torch.utils.data.DataLoader

        def data_loader(*loader_args, **loader_kwargs):
            loader_kwargs.setdefault("collate_fn", collate)
            return original_loader(*loader_args, **loader_kwargs)

        torch.utils.data.DataLoader = data_loader
        try:
            return original(dataset, *args, **kwargs)
        finally:
            torch.utils.data.DataLoader = original_loader

    build_with_collate._uniss_true_subsecond_collate = True
    data_samplers.build_pretraining_data_loader = build_with_collate
    megatron_training.build_pretraining_data_loader = build_with_collate


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    runtime = load_megatron_runtime()
    if runtime.megatron_gpt.has_nvidia_modelopt:
        parser = runtime.megatron_gpt.add_modelopt_args(parser)
    group = parser.add_argument_group(title="UniSS true-subsecond full198")
    for name in (
        "trajectory-packed",
        "trajectory-offsets",
        "replay-packed",
        "replay-offsets",
        "whispervq-codebook",
    ):
        group.add_argument(f"--true-{name}", required=True)
    group.add_argument("--true-valid-trajectory-packed")
    group.add_argument("--true-valid-trajectory-offsets")
    group.add_argument("--true-valid-replay-packed")
    group.add_argument("--true-valid-replay-offsets")
    group.add_argument("--true-phase3-fingerprint")
    group.add_argument("--true-lora-rank", type=int, default=32)
    group.add_argument("--true-lora-alpha", type=float, default=64.0)
    group.add_argument("--true-lora-dropout", type=float, default=0.05)
    group.add_argument("--true-lora-mlp-last-layers", type=int, default=12)
    group.add_argument("--true-adapter-layers", type=int, default=4)
    group.add_argument("--true-adapter-kernel-size", type=int, default=5)
    group.add_argument("--true-adapter-expansion", type=int, default=2)
    group.add_argument("--true-adapter-dropout", type=float, default=0.0)
    group.add_argument("--true-kd-temperature", type=float, default=1.5)
    group.add_argument("--true-lr-qwen-lora", type=float, default=1e-5)
    group.add_argument("--true-lr-frontend", type=float, default=5e-6)
    group.add_argument("--true-lr-new-heads", type=float, default=5e-5)
    group.add_argument("--true-min-lr", type=float, default=1e-6)
    group.add_argument("--true-npz-lru-capacity", type=int, default=8)
    group.add_argument("--true-allow-partial-index", action="store_true")
    group.add_argument("--true-smoke", action="store_true")
    return parser


def validate_experiment_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("true-subsecond packed training requires --sft")
    if int(args.seq_length) != 18_000:
        raise ValueError("true-subsecond training requires seq-length 18000")
    if int(args.tensor_model_parallel_size) != 1:
        raise ValueError("native sidecars currently require tensor parallel size 1")
    if int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("native sidecars currently require pipeline parallel size 1")
    if bool(args.create_attention_mask_in_dataloader):
        raise ValueError("packed THD training must not create a dense attention mask")
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("validated micro batch sizes are 1 and 2")
    if not args.true_smoke and int(args.global_batch_size) != 128:
        raise ValueError("formal full198 training requires global batch size 128")
    if float(args.lr) != float(args.true_lr_new_heads):
        raise ValueError("base --lr must equal --true-lr-new-heads")
    if float(args.min_lr) != float(args.true_min_lr):
        raise ValueError("base --min-lr must equal --true-min-lr")
    if int(args.true_npz_lru_capacity) <= 0:
        raise ValueError("NPZ LRU capacity must be positive")
    if not 0.0 <= float(args.true_lora_dropout) < 1.0:
        raise ValueError("invalid LoRA dropout")
    required = (
        args.true_trajectory_packed,
        args.true_trajectory_offsets,
        args.true_replay_packed,
        args.true_replay_offsets,
        args.true_whispervq_codebook,
    )
    for value in required:
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    optional_pairs = (
        (args.true_valid_trajectory_packed, args.true_valid_trajectory_offsets),
        (args.true_valid_replay_packed, args.true_valid_replay_offsets),
    )
    for packed, offsets in optional_pairs:
        if bool(packed) != bool(offsets):
            raise ValueError("validation packed and offsets must be supplied together")
        for value in (packed, offsets):
            if value and not Path(value).is_file():
                raise FileNotFoundError(value)
    has_validation = any(packed for packed, _ in optional_pairs)
    if (bool(args.full_validation) or int(args.eval_iters or 0) > 0) and not has_validation:
        raise ValueError("evaluation is enabled but no true-subsecond validation data was supplied")
    if args.true_phase3_fingerprint and not Path(args.true_phase3_fingerprint).is_file():
        raise FileNotFoundError(args.true_phase3_fingerprint)


def _target_count(values, index: int) -> int | None:
    if values is None or index >= len(values) or values[index] is None:
        return None
    return int(values[index])


def _trajectory_dataset(args, packed: str, offsets: str) -> IndexedTrajectoryDataset:
    return IndexedTrajectoryDataset(
        packed,
        offsets,
        seq_length=int(args.seq_length),
        npz_lru_capacity=int(args.true_npz_lru_capacity),
        require_complete=not bool(args.true_allow_partial_index),
    )


def _replay_dataset(args, packed: str, offsets: str) -> IndexedPhase3ReplayDataset:
    return IndexedPhase3ReplayDataset(
        packed,
        offsets,
        seq_length=int(args.seq_length),
        require_complete=not bool(args.true_allow_partial_index),
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    trajectory = _trajectory_dataset(
        args, args.true_trajectory_packed, args.true_trajectory_offsets
    )
    replay = _replay_dataset(args, args.true_replay_packed, args.true_replay_offsets)
    target_train = _target_count(train_val_test_num_samples, 0)
    if target_train is None:
        raise ValueError("Megatron did not provide the formal train sample count")
    dp_microbatch = int(args.data_parallel_size) * int(args.micro_batch_size)
    train = DeterministicReplayTrajectorySchedule(
        trajectory,
        replay,
        total_samples=target_train,
        data_parallel_group_size=dp_microbatch,
    )
    train.split = "train"
    train.collate_fn = collate_replay_or_trajectory

    valid_sources: list[Dataset] = []
    if args.true_valid_trajectory_packed:
        valid_sources.append(
            _trajectory_dataset(
                args,
                args.true_valid_trajectory_packed,
                args.true_valid_trajectory_offsets,
            )
        )
    if args.true_valid_replay_packed:
        valid_sources.append(
            _replay_dataset(
                args, args.true_valid_replay_packed, args.true_valid_replay_offsets
            )
        )
    valid = JointValidationDataset(valid_sources) if valid_sources else None
    runtime.print_rank_0(
        "> true-subsecond datasets: "
        f"trajectory={len(trajectory)} replay={len(replay)} "
        f"schedule={len(train)} replay_groups={train.replay_groups} "
        f"trajectory_groups={train.trajectory_groups} "
        f"valid={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _normalize_kind(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        kinds = {str(item) for item in value}
        if len(kinds) == 1:
            return kinds.pop()
    raise ValueError(f"malformed sample kind: {value!r}")


def _cuda_batch(batch: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_packed_batch(batch: Mapping[str, object], seq_length: int) -> dict[str, object]:
    """Move sidecars to CUDA and flatten only Megatron's packed token tensors."""

    from megatron.core.utils import flatten_batch_for_packed_sequences

    result = _cuda_batch(batch)
    result["sample_kind"] = _normalize_kind(result["sample_kind"])
    primary = {
        key: result.get(key)
        for key in ("tokens", "labels", "loss_mask", "position_ids", "cu_seqlens", "max_seqlen")
    }
    primary["attention_mask"] = None
    primary["cu_seqlens_padded"] = None
    flattened = flatten_batch_for_packed_sequences(primary)
    result.update(flattened)
    if result["sample_kind"] == "trajectory":
        result["token_roles"] = result["token_roles"].reshape(-1)
    result["original_seq_length"] = torch.tensor(
        int(seq_length), dtype=torch.long, device=result["tokens"].device
    )
    return result


def build_packed_seq_params(batch: Mapping[str, object], seq_length: int):
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.training.training import update_seqlen_stats_from_cu_seqlens

    cu_seqlens = batch["cu_seqlens"].squeeze(0)
    update_seqlen_stats_from_cu_seqlens(cu_seqlens)
    max_seqlen = int(batch["max_seqlen"].reshape(-1).max().item())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        tokens_per_sample=int(seq_length),
    )


def _embedding_weight(model: nn.Module) -> torch.Tensor:
    embedding = getattr(model, "embedding", None)
    word_embeddings = getattr(embedding, "word_embeddings", None)
    weight = getattr(word_embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("native GPT word embedding weight was not found")
    return weight


def verify_phase3_fingerprint(model: nn.Module, path: str | Path | None) -> None:
    if not path or getattr(model, "_true_phase3_fingerprint_verified", False):
        return
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "uniss_phase3_embedding_fingerprint_v1":
        raise ValueError("unexpected Phase3 fingerprint schema")
    weight = _embedding_weight(model)
    rows = torch.tensor(payload["rows"], dtype=torch.long, device=weight.device)
    columns = torch.tensor(payload["columns"], dtype=torch.long, device=weight.device)
    actual = weight.index_select(0, rows).index_select(1, columns).float().cpu()
    expected = torch.tensor(payload["values"], dtype=torch.float32)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        maximum = float((actual - expected).abs().max()) if actual.shape == expected.shape else math.inf
        raise RuntimeError(
            "native Phase3 checkpoint fingerprint mismatch; base weights were not loaded "
            f"correctly (max_abs={maximum})"
        )
    model._true_phase3_fingerprint_verified = True


def _joint_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    objective: TrueSubsecondObjective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("true-subsecond TP=PP=1 expects flattened [tokens,1,*] tensors")
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
        raise ValueError(f"unknown sample kind: {context['sample_kind']}")
    total, metrics = distributed_weighted_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("true-subsecond metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite true-subsecond loss component")
    return torch.stack(values)


def attach_true_subsecond_forward(model: nn.Module, fingerprint: str | None) -> None:
    original_forward = model.forward

    def forward_with_true_subsecond(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        padding_mask=None,
        output_processor=None,
        output_processor_context=None,
        true_subsecond_batch=None,
    ):
        if output_processor is not None or output_processor_context is not None:
            raise ValueError("the training entrypoint owns the GPT output processor")
        if true_subsecond_batch is None:
            raise ValueError("missing true-subsecond sidecar batch")
        verify_phase3_fingerprint(self, fingerprint)
        kind = str(true_subsecond_batch["sample_kind"])
        residual_rms = _embedding_weight(self).sum() * 0.0
        if kind == "trajectory":
            if decoder_input is not None:
                raise ValueError("trajectory frontend cannot replace a pipeline decoder input")
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
            decoder_input, residual_rms = self.true_subsecond_objective.inject_frontend_residual(
                decoder_input,
                true_subsecond_batch,
                original_seq_length=int(true_subsecond_batch["original_seq_length"].item()),
            )
        context = {
            "objective": self.true_subsecond_objective,
            "batch": true_subsecond_batch,
            "sample_kind": kind,
            "frontend_residual_rms": residual_rms,
            "word_embedding_weight": _embedding_weight(self),
            "progress": float(true_subsecond_batch["training_progress"].item()),
        }
        return original_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            output_processor=_joint_output_processor,
            output_processor_context=context,
        )

    model.forward = types.MethodType(forward_with_true_subsecond, model)


def augment_native_gpt(model: nn.Module, args) -> MegatronLoRASummary:
    summary = inject_native_megatron_lora(
        model,
        rank=int(args.true_lora_rank),
        alpha=float(args.true_lora_alpha),
        dropout=float(args.true_lora_dropout),
        mlp_last_layers=int(args.true_lora_mlp_last_layers),
    )
    objective = TrueSubsecondObjective(
        int(args.hidden_size),
        load_whispervq_codebook(args.true_whispervq_codebook),
        adapter_layers=int(args.true_adapter_layers),
        adapter_kernel_size=int(args.true_adapter_kernel_size),
        adapter_expansion=int(args.true_adapter_expansion),
        adapter_dropout=float(args.true_adapter_dropout),
        kd_temperature=float(args.true_kd_temperature),
    )
    model.add_module("true_subsecond_objective", objective)
    attach_true_subsecond_forward(model, args.true_phase3_fingerprint)
    return summary


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    from gpt_builders import gpt_builder

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("true-subsecond v1 is intentionally restricted to TP=PP=1")
    model = gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    summary = augment_native_gpt(model, args)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        grouped: dict[str, int] = {}
        for _, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = next(
                (
                    name
                    for name in lr_group_values(args)
                    if getattr(parameter, name, False)
                ),
                "untagged",
            )
            grouped[group] = grouped.get(group, 0) + parameter.numel()
        print(
            json.dumps(
                {
                    "model": "native_megatron_phase3_true_subsecond_v1",
                    "lora": summary.__dict__,
                    "trainable_parameter_groups": grouped,
                    "learning_rates": lr_group_values(args),
                    "phase3_fingerprint": args.true_phase3_fingerprint,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index + 1]) for index, name in enumerate(METRIC_NAMES)
    )


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = prepare_packed_batch(next(data_iterator), int(args.seq_length))
    denominator = max(1, int(args.train_iters) * int(args.global_batch_size))
    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    batch["training_progress"] = torch.tensor(
        min(1.0, max(0.0, consumed / denominator)),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    packed_seq_params = build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        true_subsecond_batch=batch,
    )
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    install_megatron_lr_overrides(args)
    install_curriculum_sampler()
    install_joint_collate()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )


if __name__ == "__main__":
    main()
