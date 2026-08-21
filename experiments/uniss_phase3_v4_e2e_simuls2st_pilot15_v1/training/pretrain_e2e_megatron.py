#!/usr/bin/env python3
"""Native Megatron entrypoint for the single-run E2E simultaneous S2ST student."""

from __future__ import annotations

import argparse
import json
import os
import types
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

_cache_root = os.environ.get("UNISS_E2E_COMPILE_CACHE_ROOT")
if _cache_root:
    _rank_cache = Path(_cache_root) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    _rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(_rank_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_rank_cache / "inductor")

import torch
import torch.distributed as dist
from torch import nn

import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TopKTeacherCacheReader,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
    E2E_TERM_NAMES,
    E2E_WEIGHTED_NAMES,
    E2ELossWeights,
    distributed_e2e_objective,
    flattened_e2e_objective,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.runtime_dataset import (
    E2EPackedFamilyDataset,
    collate_e2e_family,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.schedule import (
    FiveFamilyCoverageSampler,
    FiveFamilyGlobalSchedule,
    FiveFamilySchedulePrefix,
    FiveFamilySingleBlock,
    FiveFamilyValidationSchedule,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_STREAMING_ASR,
    LOSS_NONE,
    TASK_FAMILIES,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


V1_MODEL_PREFIXES = ("embedding.", "decoder.", "output_layer.", "stage_a_objective.")
FAMILY_IDS = {name: index for index, name in enumerate(TASK_FAMILIES)}
OBJECTIVE_METRIC_NAMES = (
    *(f"loss/{name}" for name in E2E_TERM_NAMES),
    *(f"denominator/{name}" for name in E2E_TERM_NAMES),
    "loss/boundary_eos",
    *(f"weighted/{name}" for name in E2E_WEIGHTED_NAMES),
)
DIAGNOSTIC_NAMES = (
    "diagnostic/causal_glm_agreement",
    "diagnostic/bridge_residual_rms",
    "diagnostic/causal_glm_terminal_extensions",
    "diagnostic/acoustic_rows",
    "diagnostic/chunk_ms",
    "diagnostic/family_id",
    "diagnostic/speaker_continuity_weight",
)
METRIC_NAMES = (*OBJECTIVE_METRIC_NAMES, *DIAGNOSTIC_NAMES)
REQUIRED_FAMILY_DENOMINATORS = {
    FAMILY_STREAMING_ASR: (
        "asr_ce",
        "boundary_ce",
        "v1_asr_kl",
    ),
    FAMILY_INCREMENTAL_MT: (
        "mt_ce",
        "boundary_ce",
        "phase3_kl",
        "commit_consistency",
    ),
    FAMILY_INTERLEAVED: (
        "asr_ce",
        "mt_ce",
        "semantic_ce",
        "boundary_ce",
        "eos_ce",
        "phase3_kl",
    ),
    FAMILY_PHASE3_QUALITY: ("replay_ce",),
    FAMILY_PHASE3_PERFORMANCE: ("replay_ce",),
}


def _metadata_base_key(value: str) -> str:
    return str(value).split("/shard_", 1)[0]


def _model_keys(values) -> set[str]:
    return {
        _metadata_base_key(value)
        for value in values
        if _metadata_base_key(value).startswith(V1_MODEL_PREFIXES)
    }


def validate_v1_checkpoint_key_sets(
    checkpoint_keys, current_model_keys
) -> dict[str, object]:
    checkpoint = _model_keys(checkpoint_keys)
    current = _model_keys(current_model_keys)
    missing = sorted(checkpoint - current)
    unexpected = sorted(current - checkpoint)
    stage_a = sorted(key for key in current if key.startswith("stage_a_objective."))
    if missing or unexpected or not checkpoint or not stage_a:
        raise RuntimeError(
            "E2E V1 compound checkpoint key audit failed: "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"checkpoint={len(checkpoint)} current={len(current)} stage_a={len(stage_a)}"
        )
    return {
        "checkpoint_model_keys": len(checkpoint),
        "current_model_keys": len(current),
        "stage_a_objective_keys": len(stage_a),
        "exact_key_match": True,
    }


def _resolve_checkpoint(load_root: str | Path) -> Path:
    root = Path(load_root).resolve()
    latest_path = root / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    latest = int(latest_path.read_text(encoding="utf-8").strip())
    checkpoint = root / f"iter_{latest:07d}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def validate_v1_fingerprint_manifest(
    load_root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    checkpoint = _resolve_checkpoint(load_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "uniss_checkpoint_tree_fingerprint_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("unexpected V1 checkpoint fingerprint manifest")
    value = manifest.get("checkpoints", {}).get("v1")
    if not isinstance(value, dict):
        raise ValueError("V1 checkpoint fingerprint is missing")
    if Path(str(value.get("path"))).resolve() != checkpoint:
        raise RuntimeError("V1 fingerprint path differs from --load checkpoint")
    sha256 = str(value.get("sha256", ""))
    if len(sha256) != 64:
        raise ValueError("V1 checkpoint tree SHA256 is malformed")
    return {
        "checkpoint": str(checkpoint),
        "tree_sha256": sha256,
        "bytes": int(value["bytes"]),
        "files": int(value["files"]),
    }


def audit_v1_handoff_structure(
    model: nn.Module, load_root: str | Path
) -> dict[str, object]:
    from megatron.core import parallel_state
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = _resolve_checkpoint(load_root)
    checkpoint_keys = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    current = model.sharded_state_dict(
        metadata={
            "dp_cp_group": parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        }
    )
    canonical = {getattr(value, "key", key) for key, value in current.items()}
    return validate_v1_checkpoint_key_sets(checkpoint_keys, canonical)


def e2e_chunk_ms_for_progress(progress: float, update: int) -> int:
    if not 0.0 <= progress <= 1.0 or update < 0:
        raise ValueError("invalid E2E curriculum position")
    if progress < 0.10:
        choices = (1280, 960)
    elif progress < 0.35:
        choices = (960, 640)
    elif progress < 0.70:
        choices = (640, 320)
    else:
        choices = (320, 160)
    return choices[update % len(choices)]


def lr_group_values(args) -> dict[str, dict[str, float]]:
    return {
        "uniss_e2e_qwen": {
            "lr_mult": float(args.e2e_lr_qwen) / float(args.lr),
            "max_lr": float(args.e2e_lr_qwen),
            "min_lr": float(args.e2e_lr_qwen) * 0.1,
        },
        "uniss_e2e_qwen_io": {
            "lr_mult": float(args.e2e_lr_qwen_io) / float(args.lr),
            "max_lr": float(args.e2e_lr_qwen_io),
            "min_lr": float(args.e2e_lr_qwen_io) * 0.1,
        },
    }


def install_e2e_lr_overrides(args) -> None:
    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey

    original = megatron_training.get_megatron_optimizer_config
    if getattr(original, "_uniss_e2e_groups", False):
        return

    def with_e2e_groups(parsed_args):
        config, overrides = original(parsed_args)
        overrides = dict(overrides or {})
        for attribute, values in lr_group_values(parsed_args).items():
            overrides[ParamKey(attr=attribute)] = values
        return config, overrides

    with_e2e_groups._uniss_e2e_groups = True
    megatron_training.get_megatron_optimizer_config = with_e2e_groups


def install_family_sampler() -> None:
    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_e2e_family_schedule", False):
        return

    def family_or_default(dataset, *args, **kwargs):
        if getattr(dataset, "synchronize_task_family", False):
            return FiveFamilyCoverageSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    family_or_default._uniss_e2e_family_schedule = True
    data_samplers.MegatronPretrainingRandomSampler = family_or_default


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    runtime = load_megatron_runtime()
    if runtime.megatron_gpt.has_nvidia_modelopt:
        parser = runtime.megatron_gpt.add_modelopt_args(parser)
    group = parser.add_argument_group(title="UniSS E2E simultaneous S2ST")
    group.add_argument("--e2e-train-build-report", required=True)
    group.add_argument("--e2e-valid-build-report")
    group.add_argument("--e2e-v1-train-cache-audit")
    group.add_argument("--e2e-phase3-train-cache-audit")
    group.add_argument("--e2e-v1-valid-cache-audit")
    group.add_argument("--e2e-phase3-valid-cache-audit")
    group.add_argument("--e2e-whispervq-model", required=True)
    group.add_argument("--e2e-checkpoint-fingerprints", required=True)
    group.add_argument("--e2e-training-gate")
    group.add_argument("--e2e-coverage-epochs", type=int, default=3)
    group.add_argument("--e2e-lr-qwen", type=float, default=2e-6)
    group.add_argument("--e2e-lr-qwen-io", type=float, default=5e-7)
    group.add_argument("--e2e-asr-weight", type=float, default=1.0)
    group.add_argument("--e2e-mt-weight", type=float, default=1.0)
    group.add_argument("--e2e-semantic-weight", type=float, default=1.0)
    group.add_argument("--e2e-replay-weight", type=float, default=0.50)
    group.add_argument("--e2e-v1-asr-kl-weight", type=float, default=0.30)
    group.add_argument("--e2e-phase3-kl-weight", type=float, default=0.25)
    group.add_argument("--e2e-commit-weight", type=float, default=0.20)
    group.add_argument("--e2e-boundary-eos-weight", type=float, default=0.10)
    group.add_argument("--e2e-speaker-continuity-weight", type=float, default=0.0)
    group.add_argument("--e2e-verify-dataset-sha256", action="store_true")
    group.add_argument("--e2e-verify-cache-sha256", action="store_true")
    group.add_argument("--e2e-smoke", action="store_true")
    group.add_argument("--e2e-smoke-family", choices=TASK_FAMILIES)
    group.add_argument("--e2e-allow-missing-teachers", action="store_true")
    group.add_argument("--e2e-audit-gradients", action="store_true")
    return parser


def _require_path(path: str | Path | None) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(path)


def _require_file(path: str | Path | None) -> None:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(path)


def validate_smoke_scope(
    *,
    smoke: bool,
    allow_missing_teachers: bool,
    train_iters: int,
    smoke_family: str | None = None,
) -> None:
    if allow_missing_teachers and not smoke:
        raise ValueError("missing E2E teachers are allowed only in smoke mode")
    if smoke and not 1 <= int(train_iters) <= 2:
        raise ValueError("E2E smoke runs are restricted to one or two updates")
    if smoke_family is not None and (not smoke or int(train_iters) != 1):
        raise ValueError("one-family E2E canary requires one smoke update")


def validate_v1_checkpoint_load_policy(args) -> None:
    if (
        not bool(args.finetune)
        or not bool(args.no_load_optim)
        or not bool(args.no_load_rng)
    ):
        raise ValueError("E2E V1 initialization requires finetune/no-load-optim/no-load-rng")
    if str(args.dist_ckpt_strictness) != "raise_unexpected":
        raise ValueError(
            "E2E V1 initialization requires "
            "dist-ckpt-strictness=raise_unexpected"
        )


def validate_experiment_args(args) -> None:
    validate_smoke_scope(
        smoke=bool(args.e2e_smoke),
        allow_missing_teachers=bool(args.e2e_allow_missing_teachers),
        train_iters=int(args.train_iters),
        smoke_family=args.e2e_smoke_family,
    )
    if not bool(args.sft):
        raise ValueError("E2E packed training requires --sft")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("E2E v1 is restricted to TP=PP=1")
    if int(args.seq_length) != 18_000:
        raise ValueError("E2E task pools and native training require seq-length 18000")
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("validated E2E micro batch sizes are 1 and 2")
    if not args.e2e_smoke and int(args.global_batch_size) != 128:
        raise ValueError("formal E2E training requires global batch size 128")
    if int(args.e2e_coverage_epochs) != 3 and not args.e2e_smoke:
        raise ValueError("formal E2E training requires three coverage epochs")
    if bool(args.create_attention_mask_in_dataloader):
        raise ValueError("E2E packed THD training must not create a dense mask")
    validate_v1_checkpoint_load_policy(args)
    if float(args.lr) != float(args.e2e_lr_qwen):
        raise ValueError("base --lr must equal the E2E Qwen LR")
    if float(args.min_lr) != float(args.e2e_lr_qwen) * 0.1:
        raise ValueError("base --min-lr must equal 0.1 * E2E Qwen LR")
    if float(args.e2e_speaker_continuity_weight) != 0.0:
        raise ValueError(
            "speaker continuity must remain zero until a genuine training sidecar exists"
        )
    _require_file(args.e2e_train_build_report)
    _require_path(args.e2e_whispervq_model)
    _require_file(args.e2e_checkpoint_fingerprints)
    if args.e2e_valid_build_report:
        _require_file(args.e2e_valid_build_report)
    train_cache_paths = (
        args.e2e_v1_train_cache_audit,
        args.e2e_phase3_train_cache_audit,
    )
    if not args.e2e_allow_missing_teachers:
        for path in train_cache_paths:
            _require_file(path)
    if args.e2e_valid_build_report and not args.e2e_allow_missing_teachers:
        for path in (
            args.e2e_v1_valid_cache_audit,
            args.e2e_phase3_valid_cache_audit,
        ):
            _require_file(path)
    if not args.e2e_smoke:
        _require_file(args.e2e_training_gate)
        gate = json.loads(Path(args.e2e_training_gate).read_text(encoding="utf-8"))
        if not bool(gate.get("formal_training_authorized")):
            raise RuntimeError("formal E2E training gate is not authorized")
    validate_v1_fingerprint_manifest(args.load, args.e2e_checkpoint_fingerprints)


def e2e_weights(args) -> E2ELossWeights:
    return E2ELossWeights(
        asr_ce=float(args.e2e_asr_weight),
        mt_ce=float(args.e2e_mt_weight),
        semantic_ce=float(args.e2e_semantic_weight),
        replay_ce=float(args.e2e_replay_weight),
        v1_asr_kl=float(args.e2e_v1_asr_kl_weight),
        phase3_kl=float(args.e2e_phase3_kl_weight),
        commit_consistency=float(args.e2e_commit_weight),
        boundary_eos=float(args.e2e_boundary_eos_weight),
        speaker_continuity=float(args.e2e_speaker_continuity_weight),
    )


def _teacher_readers(args, split: str) -> dict[str, TopKTeacherCacheReader]:
    values = {
        "v1_asr": getattr(args, f"e2e_v1_{split}_cache_audit"),
        "phase3": getattr(args, f"e2e_phase3_{split}_cache_audit"),
    }
    readers = {}
    for kind, path in values.items():
        if path:
            readers[kind] = TopKTeacherCacheReader(
                path,
                cache_kind=kind,
                verify_manifest_sha256=bool(args.e2e_verify_cache_sha256),
                verify_bundle_sha256=bool(args.e2e_verify_cache_sha256),
            )
    return readers


def _family_datasets(args, report: str, split: str):
    readers = _teacher_readers(args, split)
    return {
        family: E2EPackedFamilyDataset.from_build_report(
            report,
            family,
            verify_sha256=bool(args.e2e_verify_dataset_sha256),
            load_audio=True,
            teacher_readers=readers,
        )
        for family in TASK_FAMILIES
    }


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    dp_microbatch = int(args.data_parallel_size) * int(args.micro_batch_size)
    train_sources = _family_datasets(args, args.e2e_train_build_report, "train")
    train = FiveFamilyGlobalSchedule(
        train_sources,
        coverage_epochs=int(args.e2e_coverage_epochs),
        global_batch_size=int(args.global_batch_size),
        data_parallel_group_size=dp_microbatch,
        shuffle_seed=int(args.seed),
    )
    train.collate_fn = collate_e2e_family
    target_train = int(train_val_test_num_samples[0])
    if args.e2e_smoke and 0 < target_train <= len(train):
        if args.e2e_smoke_family:
            if int(args.train_iters) != 1 or target_train != int(args.global_batch_size):
                raise ValueError("one-family E2E canary requires exactly one update")
            train = FiveFamilySingleBlock(train, args.e2e_smoke_family)
        else:
            train = FiveFamilySchedulePrefix(train, target_train)
        train.collate_fn = collate_e2e_family
    elif len(train) != target_train:
        raise ValueError(
            f"E2E schedule length {len(train)} differs from Megatron target {target_train}"
        )
    valid = None
    if args.e2e_valid_build_report and int(train_val_test_num_samples[1]) > 0:
        valid_sources = _family_datasets(args, args.e2e_valid_build_report, "valid")
        eval_global_batch = int(
            getattr(args, "eval_global_batch_size", None) or args.global_batch_size
        )
        eval_micro_batch = int(
            getattr(args, "eval_micro_batch_size", None) or args.micro_batch_size
        )
        valid = FiveFamilyValidationSchedule(
            valid_sources,
            total_samples=int(train_val_test_num_samples[1]),
            global_batch_size=eval_global_batch,
            data_parallel_group_size=int(args.data_parallel_size) * eval_micro_batch,
            shuffle_seed=int(args.seed) + 1,
        )
        valid.collate_fn = collate_e2e_family
    runtime.print_rank_0(
        "> E2E datasets: "
        f"families={{{', '.join(f'{name}:{len(train_sources[name])}' for name in TASK_FAMILIES)}}} "
        f"blocks={train.total_blocks} samples={len(train)} "
        f"family_blocks={train.family_block_counts} valid={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _tag_trainable_qwen_and_freeze_v1(model: nn.Module) -> dict[str, int]:
    counts = {"qwen": 0, "qwen_io": 0, "frozen_stage_a": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("stage_a_objective."):
            parameter.requires_grad_(False)
            counts["frozen_stage_a"] += parameter.numel()
        elif name.startswith(("embedding.", "output_layer.")):
            parameter.requires_grad_(True)
            parameter.uniss_e2e_qwen_io = True
            counts["qwen_io"] += parameter.numel()
        else:
            parameter.requires_grad_(True)
            parameter.uniss_e2e_qwen = True
            counts["qwen"] += parameter.numel()
    trainable = {id(value) for value in model.parameters() if value.requires_grad}
    frozen = {
        id(value)
        for name, value in model.named_parameters()
        if name.startswith("stage_a_objective.")
    }
    if trainable & frozen or not frozen or counts["qwen"] <= 0 or counts["qwen_io"] <= 0:
        raise RuntimeError("E2E frozen/trainable parameter partition is invalid")
    return counts


def augment_native_gpt(model: nn.Module, args) -> dict[str, int]:
    embedding = base._embedding_weight(model)
    frontend = TrainableSharedCausalWhisperVQ(
        args.e2e_whispervq_model,
        gradient_checkpointing=False,
    ).to(device=embedding.device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    model.add_module(
        "stage_a_objective",
        StageAObjective(frontend, qwen_hidden_size=int(args.hidden_size)),
    )
    counts = _tag_trainable_qwen_and_freeze_v1(model)
    attach_e2e_forward(
        model, allow_missing_teachers=bool(args.e2e_allow_missing_teachers)
    )
    return counts


def _distributed_diagnostics(
    context: Mapping[str, object], reference: torch.Tensor
) -> OrderedDict[str, torch.Tensor]:
    active = float(context["acoustic_active"])
    values = reference.new_tensor(
        [
            float(context["causal_glm_agreement"]) * active,
            float(context["bridge_residual_rms"]) * active,
            active,
            float(context["terminal_extensions"]),
            float(context["acoustic_rows"]),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
    divisor = values[2].clamp_min(1.0)
    return OrderedDict(
        (
            ("diagnostic/causal_glm_agreement", values[0] / divisor),
            ("diagnostic/bridge_residual_rms", values[1] / divisor),
            ("diagnostic/causal_glm_terminal_extensions", values[3]),
            ("diagnostic/acoustic_rows", values[4]),
            (
                "diagnostic/chunk_ms",
                reference.new_tensor(float(context["chunk_ms"])),
            ),
            (
                "diagnostic/family_id",
                reference.new_tensor(float(context["family_id"])),
            ),
            (
                "diagnostic/speaker_continuity_weight",
                reference.new_tensor(float(context["speaker_continuity_weight"])),
            ),
        )
    )


def validate_family_denominators(
    family: str,
    metrics: Mapping[str, torch.Tensor],
    *,
    allow_missing_teachers: bool = False,
) -> None:
    """Fail before backward if an active task family silently lost supervision."""

    required = REQUIRED_FAMILY_DENOMINATORS.get(family)
    if required is None:
        raise ValueError(f"unknown E2E task family: {family}")
    missing = []
    for name in required:
        if allow_missing_teachers and name in {"v1_asr_kl", "phase3_kl"}:
            continue
        value = metrics.get(f"denominator/{name}")
        if value is None or value.numel() != 1 or float(value.detach()) <= 0.0:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"E2E family {family} has zero/missing active denominators: {missing}"
        )


def _e2e_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("E2E TP=PP=1 expects flattened [tokens,1,*] tensors")
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
    batch = context["batch"]
    loss_kinds = batch["loss_kinds"].reshape(-1)
    if not torch.equal(loss_mask > 0, loss_kinds != LOSS_NONE):
        raise ValueError("E2E loss mask differs from loss-kind sidecar")
    terms = flattened_e2e_objective(
        logits=logits,
        labels=labels,
        loss_kinds=loss_kinds,
        batch=batch,
        original_seq_length=int(context["original_seq_length"]),
    )
    total, metrics = distributed_e2e_objective(
        terms, weights=context["weights"]
    )
    if tuple(metrics) != OBJECTIVE_METRIC_NAMES:
        raise AssertionError("E2E objective metric order changed")
    validate_family_denominators(
        str(batch["family"]),
        metrics,
        allow_missing_teachers=bool(context["allow_missing_teachers"]),
    )
    metrics.update(_distributed_diagnostics(context, total.detach()))
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("E2E metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite E2E loss or diagnostic")
    return torch.stack(values)


def attach_e2e_forward(model: nn.Module, *, allow_missing_teachers: bool) -> None:
    raw_forward = model.forward

    def forward_with_e2e(
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
        e2e_batch=None,
    ):
        if output_processor is not None or output_processor_context is not None:
            raise ValueError("E2E entrypoint owns the native output processor")
        if e2e_batch is None:
            raise ValueError("missing E2E sidecar batch")
        if decoder_input is not None:
            raise ValueError("E2E v1 cannot replace a pipeline decoder input")
        original_seq_length = int(e2e_batch["original_seq_length"].item())
        progress = float(e2e_batch["training_progress"].item())
        update = int(e2e_batch["training_update"].item())
        chunk_ms = e2e_chunk_ms_for_progress(progress, update)
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        agreement = 0.0
        residual_rms = 0.0
        terminal_extensions = 0.0
        acoustic_rows = 0
        acoustic_active = 0
        if "waveform" in e2e_batch:
            self.stage_a_objective.eval()
            with torch.no_grad():
                acoustic = self.stage_a_objective.frontend(
                    e2e_batch["waveform"],
                    e2e_batch["waveform_lengths"],
                    chunk_ms=chunk_ms,
                )
            (
                decoder_input,
                _,
                agreement_tensor,
                residual_tensor,
                terminal_tensor,
            ) = self.stage_a_objective._inject_causal_glm(
                decoder_input,
                base._embedding_weight(self),
                acoustic.pooled_hidden.detach(),
                acoustic.pooled_lengths.detach(),
                e2e_batch,
                original_seq_length=original_seq_length,
            )
            agreement = float(agreement_tensor.detach())
            residual_rms = float(residual_tensor.detach())
            terminal_extensions = float(terminal_tensor.detach())
            acoustic_rows = int(e2e_batch["waveform"].shape[0])
            acoustic_active = 1
        family = str(e2e_batch["family"])
        context = {
            "batch": e2e_batch,
            "weights": e2e_batch["loss_weights"],
            "original_seq_length": original_seq_length,
            "chunk_ms": chunk_ms,
            "family_id": FAMILY_IDS[family],
            "causal_glm_agreement": agreement,
            "bridge_residual_rms": residual_rms,
            "terminal_extensions": terminal_extensions,
            "acoustic_rows": acoustic_rows,
            "acoustic_active": acoustic_active,
            "speaker_continuity_weight": e2e_batch["loss_weights"].speaker_continuity,
            "allow_missing_teachers": bool(allow_missing_teachers),
        }
        return raw_forward(
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
            output_processor=_e2e_output_processor,
            output_processor_context=context,
        )

    model.forward = types.MethodType(forward_with_e2e, model)


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
        raise ValueError("E2E v1 is intentionally restricted to TP=PP=1")
    model = gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    counts = augment_native_gpt(model, args)
    structure = audit_v1_handoff_structure(model, args.load)
    fingerprint = validate_v1_fingerprint_manifest(
        args.load, args.e2e_checkpoint_fingerprints
    )
    if args.e2e_audit_gradients:
        base.install_gradient_audit(model)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "model": "native_megatron_v1_compound_e2e_simuls2st_v1",
                    "parameters": counts,
                    "learning_rate_groups": lr_group_values(args),
                    "v1_handoff": structure,
                    "v1_fingerprint": fingerprint,
                    "speaker_continuity": {
                        "weight": float(args.e2e_speaker_continuity_weight),
                        "reason": "no genuine cross-fragment training sidecar",
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def _cuda_batch(batch: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_e2e_batch(batch: Mapping[str, object], seq_length: int) -> dict[str, object]:
    from megatron.core.utils import flatten_batch_for_packed_sequences

    result = _cuda_batch(batch)
    primary = {
        key: result.get(key)
        for key in (
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "cu_seqlens",
            "max_seqlen",
        )
    }
    primary["attention_mask"] = None
    primary["cu_seqlens_padded"] = None
    result.update(flatten_batch_for_packed_sequences(primary))
    result["loss_kinds"] = result["loss_kinds"].reshape(-1)
    result["original_seq_length"] = torch.tensor(
        int(seq_length), dtype=torch.long, device=result["tokens"].device
    )
    return result


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index + 1]) for index, name in enumerate(METRIC_NAMES)
    )


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = prepare_e2e_batch(next(data_iterator), int(args.seq_length))
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
    batch["loss_weights"] = e2e_weights(args)
    packed_seq_params = base.build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        e2e_batch=batch,
    )
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    install_e2e_lr_overrides(args)
    install_family_sampler()
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
