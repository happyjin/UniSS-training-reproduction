#!/usr/bin/env python3
"""Native Megatron Stage A causal WhisperVQ plus source-ASR training."""

from __future__ import annotations

import argparse
import json
import math
import os
import types
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

# torchrun ranks must not atomically publish the same Triton/Inductor bundle.
# Configure isolated caches before importing torch or Transformer Engine.
_cache_root = os.environ.get("UNISS_STAGE_A_COMPILE_CACHE_ROOT")
if _cache_root:
    _rank_cache = Path(_cache_root) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    _rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(_rank_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_rank_cache / "inductor")

import torch
from torch import nn

import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    CoverageEpochSampler,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.dataset import (
    IndexedStageAPackDataset,
    PaddedStageAValidationDataset,
    ThreeEpochStageASchedule,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


CURRICULUM_METRICS = (
    "curriculum_progress",
    "curriculum_chunk_ms",
    "curriculum_consistency_chunk_ms",
)
METRIC_NAMES = (*TERM_NAMES, *DIAGNOSTIC_NAMES, *CURRICULUM_METRICS)
PHASE3_NATIVE_PREFIXES = ("embedding.", "decoder.", "output_layer.")
STAGE_A_PREFIX = "stage_a_objective."


def _metadata_base_key(value: str) -> str:
    return str(value).split("/shard_", 1)[0]


def validate_phase3_handoff_key_sets(
    checkpoint_keys, current_model_keys
) -> dict[str, object]:
    checkpoint_native = {
        _metadata_base_key(key)
        for key in checkpoint_keys
        if _metadata_base_key(key).startswith(PHASE3_NATIVE_PREFIXES)
    }
    current = {_metadata_base_key(key) for key in current_model_keys}
    missing_native = sorted(checkpoint_native - current)
    illegal_new = sorted(
        key
        for key in current - checkpoint_native
        if not key.startswith(STAGE_A_PREFIX)
    )
    new_keys = sorted(key for key in current if key.startswith(STAGE_A_PREFIX))
    if missing_native or illegal_new or not checkpoint_native or not new_keys:
        raise RuntimeError(
            "Stage A Phase3 handoff key audit failed: "
            f"missing_native={missing_native[:20]} "
            f"illegal_new={illegal_new[:20]} new_keys={len(new_keys)}"
        )
    return {
        "native_checkpoint_keys": len(checkpoint_native),
        "current_model_keys": len(current),
        "allowed_new_keys": len(new_keys),
        "allowed_new_prefix": STAGE_A_PREFIX,
    }


def audit_phase3_handoff_structure(model: nn.Module, load_root: str | Path) -> dict[str, object]:
    from megatron.core import parallel_state
    from torch.distributed.checkpoint import FileSystemReader

    root = Path(load_root).resolve()
    latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = root / f"iter_{latest:07d}"
    checkpoint_keys = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    current = model.sharded_state_dict(
        metadata={
            "dp_cp_group": parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        }
    )
    canonical = {getattr(value, "key", key) for key, value in current.items()}
    return validate_phase3_handoff_key_sets(checkpoint_keys, canonical)


def lr_group_values(args) -> dict[str, dict[str, float | bool]]:
    values = {
        "uniss_stage_a_new_head": float(args.stage_a_lr_new_head),
        "uniss_stage_a_bridge": float(args.stage_a_lr_bridge),
        "uniss_stage_a_whisper_top": float(args.stage_a_lr_whisper_top),
        "uniss_stage_a_whisper_bottom": float(args.stage_a_lr_whisper_bottom),
        "uniss_stage_a_whisper_conv": float(args.stage_a_lr_whisper_conv),
        "uniss_stage_a_qwen": float(args.stage_a_lr_qwen),
        "uniss_stage_a_qwen_io": float(args.stage_a_lr_qwen_io),
    }
    return {
        name: {
            "lr_mult": maximum / float(args.lr),
            "max_lr": maximum,
            "min_lr": maximum * 0.1,
            "uniss_stage_a_curriculum_group": True,
            name: True,
        }
        for name, maximum in values.items()
    }


def curriculum_group_multiplier(group: Mapping[str, object], progress: float) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A LR progress")
    if group.get("uniss_stage_a_qwen") or group.get("uniss_stage_a_qwen_io"):
        return 1.0 if progress >= 0.05 else 0.0
    if group.get("uniss_stage_a_whisper_top"):
        return 1.0 if progress >= 0.05 else 0.0
    if group.get("uniss_stage_a_whisper_bottom") or group.get(
        "uniss_stage_a_whisper_conv"
    ):
        return 1.0 if progress >= 0.30 else 0.0
    return 1.0


def stage_a_curriculum_position(args, *, training: bool) -> tuple[float, int]:
    """Return a rank-stable curriculum position for the current optimizer update.

    Megatron stores the checkpoint iteration in ``args.iteration`` but advances
    the live training-loop position through ``args.curr_iteration``.  In
    contrast, ``consumed_train_samples`` can transiently differ inside a rerun
    of a resumed training step.  Training choices must therefore be keyed by
    ``curr_iteration``.  At the first strict-resume boundary, Megatron can
    switch replayed microbatches from the checkpoint iteration to the live
    iteration one call before it advances the other counter.  This experiment
    uses a fixed global batch size, so the monotonic maximum of the live update
    and the completed-sample update is the unambiguous optimizer position.
    Evaluation runs after an update and uses the completed global sample count.
    """

    train_iters = max(1, int(args.train_iters))
    sample_updates = int(getattr(args, "consumed_train_samples", 0) or 0) // max(
        1, int(args.global_batch_size)
    )
    if training:
        live_update = int(
            getattr(args, "curr_iteration", getattr(args, "iteration", 0)) or 0
        )
        if live_update < 0:
            raise ValueError("Stage A iteration cannot be negative")
        completed_updates = max(live_update, sample_updates)
    else:
        completed_updates = sample_updates
    if completed_updates < 0:
        raise ValueError("Stage A iteration cannot be negative")
    progress = min(1.0, completed_updates / train_iters)
    return progress, completed_updates


def synchronized_stage_a_curriculum_position(
    args, *, training: bool, device: torch.device
) -> tuple[float, int]:
    """Make world rank zero authoritative at strict-resume microbatch boundaries."""

    _, local_update = stage_a_curriculum_position(args, training=training)
    update_tensor = torch.tensor(local_update, dtype=torch.long, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(update_tensor, src=0)
    update = int(update_tensor.item())
    if update < 0:
        raise ValueError("Stage A synchronized iteration cannot be negative")
    progress = min(1.0, update / max(1, int(args.train_iters)))
    return progress, update


def install_stage_a_lr_overrides(args) -> None:
    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    original_config = megatron_training.get_megatron_optimizer_config
    if not getattr(original_config, "_uniss_stage_a_groups", False):

        def with_stage_a_groups(parsed_args):
            config, overrides = original_config(parsed_args)
            overrides = dict(overrides or {})
            for attribute, values in lr_group_values(parsed_args).items():
                overrides[ParamKey(attr=attribute)] = values
            return config, overrides

        with_stage_a_groups._uniss_stage_a_groups = True
        megatron_training.get_megatron_optimizer_config = with_stage_a_groups

    original_get_lr = OptimizerParamScheduler.get_lr
    if not getattr(original_get_lr, "_uniss_stage_a_curriculum", False):

        def get_lr_with_stage_a_curriculum(self, param_group):
            value = original_get_lr(self, param_group)
            if not param_group.get("uniss_stage_a_curriculum_group", False):
                return value
            denominator = max(1.0, float(self.lr_decay_steps))
            progress = min(1.0, max(0.0, float(self.num_steps) / denominator))
            return value * curriculum_group_multiplier(param_group, progress)

        get_lr_with_stage_a_curriculum._uniss_stage_a_curriculum = True
        OptimizerParamScheduler.get_lr = get_lr_with_stage_a_curriculum


def install_coverage_sampler() -> None:
    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_stage_a_coverage", False):
        return

    def coverage_or_default(dataset, *args, **kwargs):
        if isinstance(dataset, ThreeEpochStageASchedule):
            return CoverageEpochSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    coverage_or_default._uniss_stage_a_coverage = True
    data_samplers.MegatronPretrainingRandomSampler = coverage_or_default


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    runtime = load_megatron_runtime()
    if runtime.megatron_gpt.has_nvidia_modelopt:
        parser = runtime.megatron_gpt.add_modelopt_args(parser)
    group = parser.add_argument_group(title="UniSS quality-first Stage A")
    group.add_argument("--stage-a-train-packs", required=True)
    group.add_argument("--stage-a-valid-packs")
    group.add_argument("--stage-a-whispervq-model", required=True)
    group.add_argument("--stage-a-frontend-gate", required=True)
    group.add_argument("--stage-a-phase3-fingerprint", required=True)
    group.add_argument("--stage-a-coverage-epochs", type=int, default=3)
    group.add_argument("--stage-a-max-acoustics-per-pack", type=int, default=2)
    group.add_argument("--stage-a-lr-new-head", type=float, default=1e-4)
    group.add_argument("--stage-a-lr-bridge", type=float, default=5e-5)
    group.add_argument("--stage-a-lr-whisper-top", type=float, default=1e-6)
    group.add_argument("--stage-a-lr-whisper-bottom", type=float, default=2e-7)
    group.add_argument("--stage-a-lr-whisper-conv", type=float, default=1e-7)
    group.add_argument("--stage-a-lr-qwen", type=float, default=2e-6)
    group.add_argument("--stage-a-lr-qwen-io", type=float, default=5e-7)
    group.add_argument("--stage-a-smoke", action="store_true")
    group.add_argument("--stage-a-audit-gradients", action="store_true")
    return parser


def _passed_gate(path: str | Path) -> bool:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(value.get("passed")) and value.get("schema_version") == (
        "uniss_stage_a_trainable_frontend_gate_v1"
    )


def validate_experiment_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("Stage A packed training requires --sft")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("Stage A v1 is restricted to TP=PP=1")
    if int(args.seq_length) != 18_000 and not (
        bool(args.stage_a_smoke) and int(args.seq_length) == 4_096
    ):
        raise ValueError("formal Stage A requires seq=18000; smoke may use 4096")
    if not args.stage_a_smoke and int(args.global_batch_size) != 128:
        raise ValueError("formal Stage A requires global batch 128")
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("validated Stage A micro batch sizes are 1 and 2")
    if bool(args.create_attention_mask_in_dataloader):
        raise ValueError("Stage A packed THD training must not create a dense mask")
    if int(args.stage_a_coverage_epochs) != 3 and not args.stage_a_smoke:
        raise ValueError("formal Stage A requires three strict coverage epochs")
    if int(args.stage_a_max_acoustics_per_pack) <= 0:
        raise ValueError("Stage A must materialize at least one acoustic sample")
    required = (
        args.stage_a_train_packs,
        args.stage_a_whispervq_model,
        args.stage_a_frontend_gate,
        args.stage_a_phase3_fingerprint,
    )
    for value in required:
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if args.stage_a_valid_packs and not Path(args.stage_a_valid_packs).is_file():
        raise FileNotFoundError(args.stage_a_valid_packs)
    if not _passed_gate(args.stage_a_frontend_gate):
        raise RuntimeError("Stage A trainable frontend gate has not passed")
    fingerprint = json.loads(
        Path(args.stage_a_phase3_fingerprint).read_text(encoding="utf-8")
    )
    if fingerprint.get("schema_version") != "uniss_phase3_embedding_fingerprint_v1":
        raise ValueError("unexpected Phase3 embedding fingerprint")
    if float(args.lr) != float(args.stage_a_lr_new_head):
        raise ValueError("base --lr must equal Stage A new-head LR")


def _dataset(args, path: str, *, load_audio: bool = True) -> IndexedStageAPackDataset:
    return IndexedStageAPackDataset(
        path,
        seq_length=int(args.seq_length),
        max_acoustics_per_pack=int(args.stage_a_max_acoustics_per_pack),
        load_audio=load_audio,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    source = _dataset(args, args.stage_a_train_packs)
    target_train = int(train_val_test_num_samples[0])
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    train = ThreeEpochStageASchedule(
        source,
        coverage_epochs=int(args.stage_a_coverage_epochs),
        data_parallel_group_size=dp_group,
        global_batch_size=int(args.global_batch_size),
        shuffle_seed=int(args.seed),
    )
    if len(train) != target_train:
        raise ValueError(
            f"Stage A strict schedule length {len(train)} differs from Megatron target {target_train}"
        )
    valid = None
    valid_source_length = 0
    if args.stage_a_valid_packs:
        valid_source = _dataset(args, args.stage_a_valid_packs)
        valid_source_length = len(valid_source)
        eval_micro_batch = int(
            getattr(args, "eval_micro_batch_size", None) or args.micro_batch_size
        )
        valid = PaddedStageAValidationDataset(
            valid_source,
            minimum_samples=int(train_val_test_num_samples[1]),
            data_parallel_group_size=int(args.data_parallel_size) * eval_micro_batch,
        )
    runtime.print_rank_0(
        "> Stage A datasets: "
        f"source_packs={len(source)} coverage_epochs={train.coverage_epochs} "
        f"epoch_samples={train.epoch_samples} total_samples={len(train)} "
        f"global_shuffle_seed={train.shuffle_seed} "
        f"valid_source={valid_source_length} valid_effective={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _tag_native_qwen(model: nn.Module) -> dict[str, int]:
    counts = {"qwen": 0, "qwen_io": 0}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        if name.startswith(("embedding.", "output_layer.")):
            parameter.uniss_stage_a_qwen_io = True
            counts["qwen_io"] += parameter.numel()
        else:
            parameter.uniss_stage_a_qwen = True
            counts["qwen"] += parameter.numel()
    return counts


def augment_native_gpt(model: nn.Module, args) -> dict[str, int]:
    counts = _tag_native_qwen(model)
    embedding = base._embedding_weight(model)
    frontend = TrainableSharedCausalWhisperVQ(
        args.stage_a_whispervq_model,
        gradient_checkpointing=True,
    ).to(device=embedding.device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    objective = StageAObjective(frontend, qwen_hidden_size=int(args.hidden_size))
    model.add_module("stage_a_objective", objective)
    attach_stage_a_forward(model, args.stage_a_phase3_fingerprint)
    counts["stage_a_objective"] = sum(
        parameter.numel() for parameter in objective.parameters() if parameter.requires_grad
    )
    return counts


def _stage_a_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("Stage A TP=PP=1 expects flattened [tokens,1,*]")
    batch = context["batch"]
    output = context["objective"].compute(
        context["prepared"],
        logits[:, 0],
        kwargs["labels"].reshape(-1),
        kwargs["loss_mask"].reshape(-1),
        batch["loss_kinds"].reshape(-1),
        batch,
        original_seq_length=context["original_seq_length"],
    )
    total, metrics = distributed_stage_a_objective(output)
    metrics["curriculum_progress"] = total.detach().new_tensor(context["progress"])
    metrics["curriculum_chunk_ms"] = total.detach().new_tensor(context["chunk_ms"])
    metrics["curriculum_consistency_chunk_ms"] = total.detach().new_tensor(
        context["consistency_chunk_ms"]
    )
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("Stage A metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite Stage A loss or diagnostic")
    return torch.stack(values)


def attach_stage_a_forward(model: nn.Module, fingerprint: str) -> None:
    raw_forward = model.forward

    def forward_with_stage_a(
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
        stage_a_batch=None,
    ):
        if output_processor is not None or output_processor_context is not None:
            raise ValueError("Stage A entrypoint owns the native output processor")
        if stage_a_batch is None:
            raise ValueError("missing Stage A sidecar batch")
        base.verify_phase3_fingerprint(self, fingerprint)
        if decoder_input is not None:
            raise ValueError("Stage A cannot replace a pipeline decoder input")
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        progress = float(stage_a_batch["training_progress"].item())
        update = int(stage_a_batch["training_update"].item())
        chunk_ms, consistency_chunk_ms = chunk_pair_for_progress(progress, update)
        original_seq_length = int(stage_a_batch["original_seq_length"].item())
        prepared = self.stage_a_objective.prepare(
            decoder_input,
            base._embedding_weight(self),
            stage_a_batch,
            original_seq_length=original_seq_length,
            chunk_ms=chunk_ms,
            consistency_chunk_ms=consistency_chunk_ms,
        )
        context = {
            "objective": self.stage_a_objective,
            "prepared": prepared,
            "batch": stage_a_batch,
            "original_seq_length": original_seq_length,
            "progress": progress,
            "chunk_ms": chunk_ms,
            "consistency_chunk_ms": consistency_chunk_ms,
        }
        return raw_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=prepared.decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            output_processor=_stage_a_output_processor,
            output_processor_context=context,
        )

    model.forward = types.MethodType(forward_with_stage_a, model)


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
        raise ValueError("Stage A is intentionally restricted to TP=PP=1")
    model = gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    counts = augment_native_gpt(model, args)
    audit = audit_phase3_handoff_structure(model, args.load)
    if args.stage_a_audit_gradients:
        base.install_gradient_audit(model)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "model": "native_megatron_phase3_stage_a_causal_asr_v1",
                    "trainable_parameters": counts,
                    "learning_rate_groups": lr_group_values(args),
                    "phase3_handoff": audit,
                    "phase3_fingerprint": args.stage_a_phase3_fingerprint,
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
    batch = base.prepare_packed_batch(next(data_iterator), int(args.seq_length))
    progress, update = synchronized_stage_a_curriculum_position(
        args,
        training=bool(model.training),
        device=batch["tokens"].device,
    )
    batch["training_progress"] = torch.tensor(
        progress,
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["training_update"] = torch.tensor(
        update,
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
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    install_stage_a_lr_overrides(args)
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
        forward_step,
        model_provider=model_provider,
    )


if __name__ == "__main__":
    main()
