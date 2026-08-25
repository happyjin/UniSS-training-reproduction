#!/usr/bin/env python3
"""Stage-A-rooted matched SFT and quality-first joint GRPO in Megatron."""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as e2e
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TopKTeacherCacheReader,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
    distributed_e2e_objective,
    flattened_e2e_objective,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.runtime_dataset import (
    E2EPackedFamilyDataset,
    collate_e2e_family,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INTERLEAVED,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_NONE,
    LOSS_SEMANTIC,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.model.dual_lora import (
    DualLoRAController,
    inject_top_layer_dual_lora,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.reward import (
    GRPO_METRIC_NAMES,
    candidate_topk,
    group_relative_objective,
    zero_grpo_metrics,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.schedule import (
    OneFamilyCoverageSchedule,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


BASE_METRIC_NAMES = tuple(e2e.METRIC_NAMES)
EXTRA_METRIC_NAMES = (
    *GRPO_METRIC_NAMES,
    "grpo/active",
    "grpo/bootstrap_active",
    "grpo/reference_ready",
    "grpo/reference_anchor",
    "grpo/policy_update_rms",
    "grpo/group_size",
    "grpo/quality_phase",
    "grpo/latency_phase",
)
METRIC_NAMES = (*BASE_METRIC_NAMES, *EXTRA_METRIC_NAMES)
NEW_PREFIX = "quality_grpo_lora."


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = e2e.add_experiment_args(parser)
    group = parser.add_argument_group(title="Stage-A quality-first joint GRPO")
    # ModelOpt/Megatron already owns several --grpo-* option names. Keep this
    # experiment's grouped trajectory controls in a fully isolated namespace.
    group.add_argument("--joint-mode", choices=("sft", "grpo"), required=True)
    group.add_argument("--joint-group-size", type=int, default=4)
    group.add_argument("--joint-bootstrap-updates", type=int, default=256)
    group.add_argument("--joint-candidate-width", type=int, default=16)
    group.add_argument("--joint-clip-epsilon", type=float, default=0.20)
    group.add_argument("--joint-kl-beta", type=float, default=0.02)
    group.add_argument("--joint-sft-replay-weight", type=float, default=0.20)
    group.add_argument("--joint-reference-anchor-weight", type=float, default=0.01)
    group.add_argument("--joint-lora-rank", type=int, default=16)
    group.add_argument("--joint-lora-alpha", type=float, default=32.0)
    group.add_argument("--joint-lora-dropout", type=float, default=0.05)
    group.add_argument("--joint-top-layers", type=int, default=8)
    group.add_argument("--joint-adapter-lr", type=float, default=3e-5)
    group.add_argument("--joint-expected-train-records", type=int, default=40_150)
    group.add_argument("--joint-smoke", action="store_true")
    return parser


def _read_report(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "passed" or value.get("schema_version") != "uniss_phase3_v4_e2e_task_pools_v1":
        raise ValueError("task-pool report is not a passed formal build")
    return value


def validate_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("joint GRPO uses Megatron's SFT forward plumbing")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("joint GRPO is restricted to TP=PP=1")
    if int(args.seq_length) != 18_000 or int(args.micro_batch_size) != 1:
        raise ValueError("validated joint GRPO geometry is seq=18000, MBS=1")
    if int(args.global_batch_size) != 16:
        raise ValueError("four-way comparison requires matched GBS=16")
    if not bool(args.finetune) or not bool(args.no_load_optim) or not bool(args.no_load_rng):
        raise ValueError("fresh Stage-A handoff requires finetune/no-load-optim/no-load-rng")
    strictness = str(args.dist_ckpt_strictness)
    if strictness not in {"log_all", "StrictHandling.LOG_ALL"}:
        raise ValueError("fresh Stage-A plus new LoRA modules requires log_all DCP audit")
    for value in (
        args.e2e_train_build_report,
        args.e2e_valid_build_report,
        args.e2e_phase3_train_cache_audit,
        args.e2e_phase3_valid_cache_audit,
        args.e2e_checkpoint_fingerprints,
    ):
        if not value or not Path(value).is_file():
            raise FileNotFoundError(value)
    if not Path(args.e2e_whispervq_model).exists():
        raise FileNotFoundError(args.e2e_whispervq_model)
    train_report = _read_report(args.e2e_train_build_report)
    records = int(train_report["families"][FAMILY_INTERLEAVED]["records"])
    if records != int(args.joint_expected_train_records):
        raise ValueError(f"expected {args.joint_expected_train_records} interleaved packs, found {records}")
    expected_iters = math.ceil(records / int(args.global_batch_size))
    if not bool(args.joint_smoke) and int(args.train_iters) != expected_iters:
        raise ValueError(f"one exact packed coverage requires train-iters={expected_iters}")
    if bool(args.joint_smoke) and not 1 <= int(args.train_iters) <= 2:
        raise ValueError("joint GRPO smoke is restricted to one or two updates")
    if args.joint_mode == "grpo" and int(args.joint_group_size) < 2:
        raise ValueError("GRPO requires group size >= 2")
    if not 0 < int(args.joint_bootstrap_updates) < int(args.train_iters):
        raise ValueError("bootstrap updates must be inside the formal run")
    if int(args.joint_candidate_width) < int(args.joint_group_size):
        raise ValueError("candidate width must cover the rollout group")
    if not 0.0 < float(args.joint_clip_epsilon) < 1.0:
        raise ValueError("invalid GRPO clip epsilon")
    if any(
        float(value) < 0.0
        for value in (
            args.joint_kl_beta,
            args.joint_sft_replay_weight,
            args.joint_reference_anchor_weight,
        )
    ):
        raise ValueError("GRPO regularization weights must be non-negative")
    e2e.validate_v1_fingerprint_manifest(args.load, args.e2e_checkpoint_fingerprints)


def _interleaved_dataset(args, report: str, split: str) -> E2EPackedFamilyDataset:
    reader = TopKTeacherCacheReader(
        getattr(args, f"e2e_phase3_{split}_cache_audit"),
        cache_kind="phase3",
        verify_manifest_sha256=bool(args.e2e_verify_cache_sha256),
        verify_bundle_sha256=bool(args.e2e_verify_cache_sha256),
    )
    return E2EPackedFamilyDataset.from_build_report(
        report,
        FAMILY_INTERLEAVED,
        verify_sha256=bool(args.e2e_verify_dataset_sha256),
        load_audio=True,
        teacher_readers={"phase3": reader},
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    train_source = _interleaved_dataset(args, args.e2e_train_build_report, "train")
    target_train = int(train_val_test_num_samples[0])
    train = OneFamilyCoverageSchedule(
        train_source,
        total_samples=target_train,
        global_batch_size=int(args.global_batch_size),
        data_parallel_group_size=int(args.data_parallel_size) * int(args.micro_batch_size),
        shuffle_seed=int(args.seed),
        split="train",
        require_full_coverage=not bool(args.joint_smoke),
    )
    train.collate_fn = collate_e2e_family
    valid = None
    target_valid = int(train_val_test_num_samples[1])
    if target_valid > 0:
        valid_source = _interleaved_dataset(args, args.e2e_valid_build_report, "valid")
        eval_global = int(getattr(args, "eval_global_batch_size", 0) or args.global_batch_size)
        valid = OneFamilyCoverageSchedule(
            valid_source,
            total_samples=target_valid,
            global_batch_size=eval_global,
            data_parallel_group_size=int(args.data_parallel_size)
            * int(getattr(args, "eval_micro_batch_size", 0) or args.micro_batch_size),
            shuffle_seed=int(args.seed) + 1,
            split="valid",
            require_full_coverage=False,
        )
        valid.collate_fn = collate_e2e_family
    runtime.print_rank_0(
        f"> Stage-A GRPO datasets: train={len(train_source)} scheduled={len(train)} "
        f"valid={0 if valid is None else len(valid)} strict_global_shuffle=true"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def lr_group_values(args) -> dict[str, dict[str, float]]:
    return {
        "uniss_grpo_adapter": {
            "lr_mult": float(args.joint_adapter_lr) / float(args.lr),
            "max_lr": float(args.joint_adapter_lr),
            "min_lr": float(args.joint_adapter_lr) * 0.1,
        }
    }


def install_lr_overrides(args) -> None:
    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey

    original = megatron_training.get_megatron_optimizer_config
    if getattr(original, "_uniss_stagea_grpo", False):
        return

    def with_grpo(parsed_args):
        config, overrides = original(parsed_args)
        overrides = dict(overrides or {})
        overrides[ParamKey(attr="uniss_grpo_adapter")] = lr_group_values(parsed_args)[
            "uniss_grpo_adapter"
        ]
        return config, overrides

    with_grpo._uniss_stagea_grpo = True
    megatron_training.get_megatron_optimizer_config = with_grpo


def install_family_sampler() -> None:
    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_stagea_grpo", False):
        return

    def scheduled_or_default(dataset, *args, **kwargs):
        if getattr(dataset, "synchronize_task_family", False):
            return e2e.FiveFamilyCoverageSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    scheduled_or_default._uniss_stagea_grpo = True
    data_samplers.MegatronPretrainingRandomSampler = scheduled_or_default


def _metadata_base_key(value: str) -> str:
    return str(value).split("/shard_", 1)[0]


def audit_stage_a_handoff(model, load_root: str | Path) -> dict[str, object]:
    from megatron.core import parallel_state
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = e2e._resolve_checkpoint(load_root)
    checkpoint_keys = {
        _metadata_base_key(key)
        for key in FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
        if _metadata_base_key(key).startswith(e2e.V1_MODEL_PREFIXES)
    }
    current_raw = model.sharded_state_dict(
        metadata={
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
    )
    current = {_metadata_base_key(getattr(value, "key", key)) for key, value in current_raw.items()}
    missing = sorted(checkpoint_keys - current)
    illegal = sorted(
        key
        for key in current - checkpoint_keys
        if key.startswith(("embedding.", "decoder.", "output_layer.", "stage_a_objective."))
        and not key.startswith(NEW_PREFIX)
    )
    new = sorted(key for key in current if key.startswith(NEW_PREFIX))
    if missing or illegal or not checkpoint_keys or not new:
        raise RuntimeError(
            f"Stage-A GRPO handoff failed: missing={missing[:8]} illegal={illegal[:8]} new={len(new)}"
        )
    return {
        "stage_a_checkpoint_keys": len(checkpoint_keys),
        "new_grpo_keys": len(new),
        "missing_stage_a_keys": 0,
    }


def _reference_candidate_processor(**kwargs):
    logits, _ = kwargs["output_layer"](
        kwargs["hidden_states"],
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError("reference GRPO pass expects flattened TP=PP=1 logits")
    batch = kwargs["context"]["batch"]
    positions, indices, values = candidate_topk(
        logits[:, 0],
        kwargs["labels"].reshape(-1),
        batch["loss_kinds"].reshape(-1),
        width=int(batch["grpo_candidate_width"].item()),
    )
    return positions, indices, values


def _reduce_extra_metrics(metrics: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    names = tuple(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return OrderedDict((name, values[index]) for index, name in enumerate(names))


def _joint_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("joint GRPO expects flattened TP=PP=1 tensors")
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
    batch = context["batch"]
    loss_kinds = batch["loss_kinds"].reshape(-1)
    if not torch.equal(loss_mask > 0, loss_kinds != LOSS_NONE):
        raise ValueError("joint GRPO loss mask differs from loss kinds")
    terms = flattened_e2e_objective(
        logits=logits,
        labels=labels,
        loss_kinds=loss_kinds,
        batch=batch,
        original_seq_length=int(context["original_seq_length"]),
        semantic_end_logit_margin=float(context["semantic_end_logit_margin"]),
        semantic_boundary_rollin_mask=context["semantic_boundary_rollin_mask"],
        semantic_rollin_continue_decision_mask=context[
            "semantic_rollin_continue_decision_mask"
        ],
        semantic_rollin_continue_mask=context["semantic_rollin_continue_mask"],
        semantic_rollin_continue_decision_logit_margin=float(
            context["semantic_rollin_continue_decision_logit_margin"]
        ),
        semantic_rollin_continue_logit_margin=float(
            context["semantic_rollin_continue_logit_margin"]
        ),
        semantic_continue_tail=int(context["semantic_continue_tail"]),
        semantic_continue_logit_margin=float(context["semantic_continue_logit_margin"]),
        semantic_boundary_binary_logit_margin=float(
            context["semantic_boundary_binary_logit_margin"]
        ),
    )
    sft_total, base_metrics = distributed_e2e_objective(terms, weights=context["weights"])
    e2e.validate_family_denominators(str(batch["family"]), base_metrics)
    base_metrics.update(
        e2e._distributed_diagnostics(context, sft_total.detach(), logits=logits, labels=labels)
    )
    if tuple(base_metrics) != BASE_METRIC_NAMES:
        raise AssertionError("base E2E metric order changed")
    update = int(batch["training_update"].item())
    progress = float(batch["training_progress"].item())
    mode = str(batch["grpo_mode"])
    bootstrap = int(batch["grpo_bootstrap_updates"].item())
    active = bool(batch.get("grpo_training", False)) and mode == "grpo" and update >= bootstrap
    extra = zero_grpo_metrics(sft_total)
    total = sft_total
    if active:
        objective = group_relative_objective(
            logits,
            labels,
            loss_kinds,
            batch["sample_boundaries"],
            batch["grpo_reference_indices"],
            batch["grpo_reference_logits"],
            sequence_length=int(context["original_seq_length"]),
            group_size=int(batch["grpo_group_size"].item()),
            progress=progress,
            clip_epsilon=float(batch["grpo_clip_epsilon"].item()),
        )
        extra = objective.metrics
        reference_kl = extra["grpo/reference_kl"]
        total = (
            objective.loss
            + float(batch["grpo_kl_beta"].item()) * reference_kl
            + float(batch["grpo_sft_replay_weight"].item()) * sft_total
            + float(batch["grpo_reference_anchor_weight"].item())
            * batch["grpo_reference_anchor"]
        )
    extra = OrderedDict(extra)
    extra.update(
        (
            ("grpo/active", sft_total.new_tensor(float(active))),
            ("grpo/bootstrap_active", sft_total.new_tensor(float(update < bootstrap))),
            ("grpo/reference_ready", batch["grpo_reference_ready"].float()),
            ("grpo/reference_anchor", batch["grpo_reference_anchor"].detach()),
            ("grpo/policy_update_rms", batch["grpo_policy_update_rms"].detach()),
            ("grpo/group_size", sft_total.new_tensor(float(batch["grpo_group_size"].item()))),
            ("grpo/quality_phase", sft_total.new_tensor(float(progress < 0.60))),
            ("grpo/latency_phase", sft_total.new_tensor(float(progress >= 0.60))),
        )
    )
    extra = _reduce_extra_metrics(extra)
    if tuple(extra) != EXTRA_METRIC_NAMES:
        raise AssertionError("joint GRPO metric order changed")
    values = (total.float(), *[base_metrics[name].float() for name in BASE_METRIC_NAMES], *[extra[name].float() for name in EXTRA_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite Stage-A joint GRPO objective")
    return torch.stack(values)


def attach_joint_forward(model, *, allow_missing_teachers: bool = False) -> None:
    e2e.attach_e2e_forward(model, allow_missing_teachers=allow_missing_teachers)
    wrapped = model.forward
    controller = model.quality_grpo_lora
    if not isinstance(controller, DualLoRAController):
        raise TypeError("missing dual LoRA controller")

    def forward_with_reference(self, *args, e2e_batch=None, **kwargs):
        if e2e_batch is None:
            raise ValueError("missing joint GRPO sidecar batch")
        route = (
            (e2e_batch["loss_kinds"] == LOSS_MT)
            | (e2e_batch["loss_kinds"] == LOSS_SEMANTIC)
            | (e2e_batch["loss_kinds"] == LOSS_BOUNDARY)
            | (e2e_batch["loss_kinds"] == LOSS_EOS)
        ).reshape(-1)
        controller.set_active_mask(route)
        e2e_batch["grpo_training"] = bool(self.training)
        update = int(e2e_batch["training_update"].item())
        bootstrap = int(e2e_batch["grpo_bootstrap_updates"].item())
        active = bool(self.training) and str(e2e_batch["grpo_mode"]) == "grpo" and update >= bootstrap
        if active:
            controller.snapshot_reference()
            original_processor = e2e._e2e_output_processor
            e2e._e2e_output_processor = _reference_candidate_processor
            try:
                with controller.use("reference"), torch.no_grad():
                    reference = wrapped(*args, e2e_batch=e2e_batch, **kwargs)
            finally:
                e2e._e2e_output_processor = original_processor
            if not isinstance(reference, tuple) or len(reference) != 3:
                raise TypeError("reference candidate pass returned malformed output")
            _, e2e_batch["grpo_reference_indices"], e2e_batch["grpo_reference_logits"] = reference
        e2e_batch["grpo_reference_ready"] = controller.reference_ready.to(
            device=e2e_batch["tokens"].device
        )
        e2e_batch["grpo_reference_anchor"] = controller.reference_anchor()
        e2e_batch["grpo_policy_update_rms"] = controller.policy_update_rms()
        # Keep the route mask alive through activation recomputation in backward.
        # The next microbatch overwrites it before its own forward pass.
        with controller.use("policy"):
            return wrapped(*args, e2e_batch=e2e_batch, **kwargs)

    model.forward = types.MethodType(forward_with_reference, model)


def augment_model(model, args) -> dict[str, object]:
    embedding = e2e.base._embedding_weight(model)
    frontend = TrainableSharedCausalWhisperVQ(
        args.e2e_whispervq_model,
        gradient_checkpointing=False,
    ).to(device=embedding.device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    model.add_module(
        "stage_a_objective",
        StageAObjective(frontend, qwen_hidden_size=int(args.hidden_size)),
    )
    summary = inject_top_layer_dual_lora(
        model,
        top_layers=int(args.joint_top_layers),
        rank=int(args.joint_lora_rank),
        alpha=float(args.joint_lora_alpha),
        dropout=float(args.joint_lora_dropout),
    )
    attach_joint_forward(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != summary.trainable_parameters:
        raise RuntimeError("joint GRPO trainable scope escaped policy LoRA")
    return {
        "top_layers": [summary.first_layer, summary.last_layer],
        "targets": len(summary.module_names),
        "trainable_parameters": summary.trainable_parameters,
        "reference_parameters": summary.reference_parameters,
    }


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    from gpt_builders import gpt_builder

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("joint GRPO is restricted to TP=PP=1")
    model = gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    counts = augment_model(model, args)
    structure = audit_stage_a_handoff(model, args.load)
    fingerprint = e2e.validate_v1_fingerprint_manifest(args.load, args.e2e_checkpoint_fingerprints)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "model": "stage_a_iter381_top8_dual_lora_joint_grpo_v1",
                    "mode": args.joint_mode,
                    "parameters": counts,
                    "handoff": structure,
                    "fingerprint": fingerprint,
                    "quality_gate_controls_training": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = e2e.prepare_e2e_batch(next(data_iterator), int(args.seq_length))
    denominator = max(1, int(args.train_iters) * int(args.global_batch_size))
    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    batch["training_progress"] = torch.tensor(
        min(1.0, max(0.0, consumed / denominator)),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["training_update"] = torch.tensor(
        consumed // max(1, int(args.global_batch_size)),
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    batch["loss_weights"] = e2e.e2e_weights(args)
    scalar_fields = {
        "semantic_end_logit_margin": (args.e2e_semantic_end_logit_margin, torch.float32),
        "semantic_continue_tail": (args.e2e_semantic_continue_tail, torch.long),
        "semantic_continue_logit_margin": (args.e2e_semantic_continue_logit_margin, torch.float32),
        "semantic_rollin_continue_logit_margin": (args.e2e_semantic_rollin_continue_logit_margin, torch.float32),
        "semantic_rollin_continue_decision_logit_margin": (args.e2e_semantic_rollin_continue_decision_logit_margin, torch.float32),
        "semantic_boundary_binary_logit_margin": (args.e2e_semantic_boundary_binary_logit_margin, torch.float32),
        "semantic_rollin_continue_tail": (args.e2e_semantic_rollin_continue_tail, torch.long),
        "semantic_rollin_continue_ratio": (args.e2e_semantic_rollin_continue_ratio, torch.float32),
        "semantic_prefix_corruption_rate": (0.0, torch.float32),
        "semantic_prefix_corruption_tail": (args.e2e_semantic_prefix_corruption_tail, torch.long),
        "semantic_prefix_corruption_ramp_updates": (0, torch.long),
        "semantic_boundary_rollin_rate": (0.0, torch.float32),
        "semantic_boundary_rollin_ramp_updates": (0, torch.long),
        "grpo_group_size": (args.joint_group_size, torch.long),
        "grpo_bootstrap_updates": (args.joint_bootstrap_updates, torch.long),
        "grpo_candidate_width": (args.joint_candidate_width, torch.long),
        "grpo_clip_epsilon": (args.joint_clip_epsilon, torch.float32),
        "grpo_kl_beta": (args.joint_kl_beta, torch.float32),
        "grpo_sft_replay_weight": (args.joint_sft_replay_weight, torch.float32),
        "grpo_reference_anchor_weight": (args.joint_reference_anchor_weight, torch.float32),
    }
    for name, (value, dtype) in scalar_fields.items():
        batch[name] = torch.tensor(value, dtype=dtype, device=batch["tokens"].device)
    batch["grpo_mode"] = str(args.joint_mode)
    packed_seq_params = e2e.base.build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        e2e_batch=batch,
    )
    return output, e2e.loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_args(args)
    e2e.METRIC_NAMES = METRIC_NAMES
    e2e._e2e_output_processor = _joint_output_processor
    install_lr_overrides(args)
    install_family_sampler()
    e2e.base.install_joint_collate()
    e2e.base.install_rerun_checkpoint_compatibility()
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
