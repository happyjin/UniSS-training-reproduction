#!/usr/bin/env python3
"""Native Megatron entrypoint for the single-run E2E simultaneous S2ST student."""

from __future__ import annotations

import argparse
import json
import os
import types
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, NamedTuple

_cache_root = os.environ.get("UNISS_E2E_COMPILE_CACHE_ROOT")
if _cache_root:
    _rank_cache = Path(_cache_root) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    _rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(_rank_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_rank_cache / "inductor")

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

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
    FiveFamilyPhaseStratifiedCanary,
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
from training import constants_uniss as c
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
    "loss/semantic_boundary_binary",
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
    "diagnostic/semantic_prefix_corruption_rate",
    "diagnostic/semantic_prefix_corruption_target_rate",
    "diagnostic/semantic_prefix_corrupted_tokens",
    "diagnostic/semantic_prefix_eligible_tokens",
    "diagnostic/semantic_boundary_rollin_rate",
    "diagnostic/semantic_boundary_rollin_target_rate",
    "diagnostic/semantic_boundary_rollin_end_ce",
    "diagnostic/semantic_boundary_rollin_end_margin",
    "diagnostic/semantic_rollin_continue_decision_signed_margin",
    "diagnostic/semantic_rollin_continue_signed_margin",
    "diagnostic/semantic_boundary_binary_end_count",
    "diagnostic/semantic_boundary_binary_continue_count",
    "diagnostic/semantic_boundary_binary_end_signed_score",
    "diagnostic/semantic_boundary_binary_continue_signed_score",
    "diagnostic/semantic_boundary_binary_balanced_loss",
    "diagnostic/semantic_boundary_rollin_selected_tokens",
    "diagnostic/semantic_boundary_rollin_eligible_tokens",
    "diagnostic/semantic_boundary_rollin_changed_tokens",
    "diagnostic/semantic_boundary_rollin_selected_samples",
    "diagnostic/semantic_boundary_rollin_eligible_samples",
    "diagnostic/semantic_boundary_rollin_sample_rate",
    "diagnostic/semantic_rollin_end_eligible_samples",
    "diagnostic/semantic_rollin_end_selected_samples",
    "diagnostic/semantic_rollin_end_sample_rate",
    "diagnostic/semantic_rollin_continue_eligible_samples",
    "diagnostic/semantic_rollin_continue_selected_samples",
    "diagnostic/semantic_rollin_continue_sample_rate",
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
        "content_end_ce",
        "semantic_end_ce",
        "semantic_end_margin",
        "phase3_kl",
    ),
    FAMILY_PHASE3_QUALITY: ("replay_ce",),
    FAMILY_PHASE3_PERFORMANCE: ("replay_ce",),
}


class SemanticBoundaryRollinResult(NamedTuple):
    input_ids: torch.Tensor
    selected_mask: torch.Tensor
    selected_tokens: int
    eligible_tokens: int
    changed_tokens: int
    effective_rate: float
    selected_samples: int
    eligible_samples: int
    end_mask: torch.Tensor
    continue_decision_mask: torch.Tensor
    continue_mask: torch.Tensor
    selected_end_samples: int
    eligible_end_samples: int
    selected_continue_samples: int
    eligible_continue_samples: int


def _stable_semantic_rollin_hash(*values: int) -> int:
    """Return a process-independent SplitMix64 hash for roll-in decisions."""

    mask = (1 << 64) - 1
    state = 0x9E3779B97F4A7C15
    for index, raw_value in enumerate(values):
        state = (
            state
            + (int(raw_value) & mask)
            + 0x9E3779B97F4A7C15
            + index * 0xD1B54A32D192ED03
        ) & mask
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        state = mixed ^ (mixed >> 31)
    return state & mask


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
    group.add_argument("--e2e-content-end-weight", type=float, default=0.0)
    group.add_argument("--e2e-semantic-end-weight", type=float, default=0.0)
    group.add_argument("--e2e-semantic-end-margin-weight", type=float, default=0.0)
    group.add_argument("--e2e-semantic-end-logit-margin", type=float, default=0.0)
    group.add_argument("--e2e-semantic-rollin-end-weight", type=float, default=0.0)
    group.add_argument(
        "--e2e-semantic-rollin-end-margin-weight", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-decision-margin-weight",
        type=float,
        default=0.0,
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-decision-logit-margin",
        type=float,
        default=0.0,
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-margin-weight", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-logit-margin", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-tail", type=int, default=12
    )
    group.add_argument(
        "--e2e-semantic-rollin-continue-ratio", type=float, default=0.5
    )
    group.add_argument(
        "--e2e-semantic-continue-margin-weight", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-continue-logit-margin", type=float, default=0.0
    )
    group.add_argument("--e2e-semantic-continue-tail", type=int, default=12)
    group.add_argument(
        "--e2e-semantic-boundary-binary-weight", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-boundary-binary-logit-margin", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-prefix-corruption-rate", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-prefix-corruption-tail", type=int, default=8
    )
    group.add_argument(
        "--e2e-semantic-prefix-corruption-ramp-updates", type=int, default=0
    )
    group.add_argument(
        "--e2e-semantic-boundary-rollin-rate", type=float, default=0.0
    )
    group.add_argument(
        "--e2e-semantic-boundary-rollin-ramp-updates", type=int, default=0
    )
    group.add_argument("--e2e-speaker-continuity-weight", type=float, default=0.0)
    group.add_argument("--e2e-verify-dataset-sha256", action="store_true")
    group.add_argument("--e2e-verify-cache-sha256", action="store_true")
    group.add_argument("--e2e-smoke", action="store_true")
    group.add_argument("--e2e-smoke-family", choices=TASK_FAMILIES)
    group.add_argument("--e2e-learning-canary", action="store_true")
    group.add_argument("--e2e-phase-stratified-canary", action="store_true")
    group.add_argument("--e2e-canary-report")
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
    learning_canary: bool = False,
    allow_missing_teachers: bool,
    train_iters: int,
    smoke_family: str | None = None,
    phase_stratified_canary: bool = False,
) -> None:
    if smoke and learning_canary:
        raise ValueError("E2E smoke and learning canary are mutually exclusive")
    if allow_missing_teachers and not smoke:
        raise ValueError("missing E2E teachers are allowed only in smoke mode")
    if smoke and not 1 <= int(train_iters) <= 2:
        raise ValueError("E2E smoke runs are restricted to one or two updates")
    if learning_canary and not 10 <= int(train_iters) <= 100:
        raise ValueError("E2E learning canary is restricted to 10--100 updates")
    if phase_stratified_canary and not learning_canary:
        raise ValueError(
            "phase-stratified E2E canary requires --e2e-learning-canary"
        )
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
        learning_canary=bool(args.e2e_learning_canary),
        allow_missing_teachers=bool(args.e2e_allow_missing_teachers),
        train_iters=int(args.train_iters),
        smoke_family=args.e2e_smoke_family,
        phase_stratified_canary=bool(args.e2e_phase_stratified_canary),
    )
    if not bool(args.sft):
        raise ValueError("E2E packed training requires --sft")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("E2E v1 is restricted to TP=PP=1")
    if int(args.seq_length) != 18_000:
        raise ValueError("E2E task pools and native training require seq-length 18000")
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("validated E2E micro batch sizes are 1 and 2")
    bounded_canary = bool(args.e2e_smoke or args.e2e_learning_canary)
    if not bounded_canary and int(args.global_batch_size) != 128:
        raise ValueError("formal E2E training requires global batch size 128")
    if int(args.e2e_coverage_epochs) != 3 and not bounded_canary:
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
    if float(args.e2e_semantic_end_logit_margin) < 0.0:
        raise ValueError("semantic end logit margin must be non-negative")
    if float(args.e2e_semantic_continue_logit_margin) < 0.0:
        raise ValueError("semantic continue logit margin must be non-negative")
    if float(args.e2e_semantic_rollin_continue_logit_margin) < 0.0:
        raise ValueError("semantic roll-in continue logit margin must be non-negative")
    if float(args.e2e_semantic_rollin_continue_decision_logit_margin) < 0.0:
        raise ValueError(
            "semantic roll-in continue decision logit margin must be non-negative"
        )
    if float(args.e2e_semantic_boundary_binary_logit_margin) < 0.0:
        raise ValueError("semantic boundary binary logit margin must be non-negative")
    if float(args.e2e_semantic_boundary_binary_weight) > 0.0:
        duplicate_weights = {
            "semantic_end_ce": args.e2e_semantic_end_weight,
            "semantic_end_margin": args.e2e_semantic_end_margin_weight,
            "semantic_rollin_end_ce": args.e2e_semantic_rollin_end_weight,
            "semantic_rollin_end_margin": args.e2e_semantic_rollin_end_margin_weight,
            "semantic_rollin_continue_decision_margin": (
                args.e2e_semantic_rollin_continue_decision_margin_weight
            ),
            "semantic_rollin_continue_margin": (
                args.e2e_semantic_rollin_continue_margin_weight
            ),
            "semantic_continue_margin": args.e2e_semantic_continue_margin_weight,
        }
        active_duplicates = [
            name for name, value in duplicate_weights.items() if float(value) != 0.0
        ]
        if active_duplicates:
            raise ValueError(
                "balanced semantic boundary calibration requires duplicate special "
                f"terms to be zero: {active_duplicates}"
            )
    if int(args.e2e_semantic_continue_tail) < 1:
        raise ValueError("semantic continue tail must be positive")
    if int(args.e2e_semantic_rollin_continue_tail) < 1:
        raise ValueError("semantic roll-in continue tail must be positive")
    if not 0.0 <= float(args.e2e_semantic_rollin_continue_ratio) <= 1.0:
        raise ValueError("semantic roll-in continue ratio must be in [0, 1]")
    if not 0.0 <= float(args.e2e_semantic_prefix_corruption_rate) <= 1.0:
        raise ValueError("semantic prefix corruption rate must be in [0, 1]")
    if int(args.e2e_semantic_prefix_corruption_tail) < 1:
        raise ValueError("semantic prefix corruption tail must be positive")
    if int(args.e2e_semantic_prefix_corruption_ramp_updates) < 0:
        raise ValueError("semantic prefix corruption ramp updates must be non-negative")
    if not 0.0 <= float(args.e2e_semantic_boundary_rollin_rate) <= 1.0:
        raise ValueError("semantic boundary roll-in rate must be in [0, 1]")
    if int(args.e2e_semantic_boundary_rollin_ramp_updates) < 0:
        raise ValueError("semantic boundary roll-in ramp updates must be non-negative")
    if (
        float(args.e2e_semantic_prefix_corruption_rate) > 0.0
        and float(args.e2e_semantic_boundary_rollin_rate) > 0.0
    ):
        raise ValueError(
            "random semantic prefix corruption and model boundary roll-in "
            "cannot be enabled together"
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
    if args.e2e_learning_canary:
        _require_file(args.e2e_canary_report)
        canary = json.loads(Path(args.e2e_canary_report).read_text(encoding="utf-8"))
        if canary.get("status") != "passed" or bool(
            canary.get("formal_training_authorized")
        ):
            raise RuntimeError(
                "learning canary requires the passed, unauthorized structural canary"
            )
    elif not args.e2e_smoke:
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
        content_end_ce=float(args.e2e_content_end_weight),
        semantic_end_ce=float(args.e2e_semantic_end_weight),
        semantic_end_margin=float(args.e2e_semantic_end_margin_weight),
        semantic_rollin_end_ce=float(args.e2e_semantic_rollin_end_weight),
        semantic_rollin_end_margin=float(
            args.e2e_semantic_rollin_end_margin_weight
        ),
        semantic_rollin_continue_decision_margin=float(
            args.e2e_semantic_rollin_continue_decision_margin_weight
        ),
        semantic_rollin_continue_margin=float(
            args.e2e_semantic_rollin_continue_margin_weight
        ),
        semantic_continue_margin=float(args.e2e_semantic_continue_margin_weight),
        semantic_boundary_binary=float(args.e2e_semantic_boundary_binary_weight),
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
    if (args.e2e_smoke or args.e2e_learning_canary) and 0 < target_train <= len(train):
        if args.e2e_smoke_family:
            if int(args.train_iters) != 1 or target_train != int(args.global_batch_size):
                raise ValueError("one-family E2E canary requires exactly one update")
            train = FiveFamilySingleBlock(train, args.e2e_smoke_family)
        elif args.e2e_phase_stratified_canary:
            train = FiveFamilyPhaseStratifiedCanary(train, target_train)
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
        f"family_blocks={train.family_block_counts} "
        f"phase_blocks={getattr(train, 'phase_block_counts', None)} "
        f"valid={0 if valid is None else len(valid)}"
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


def semantic_boundary_rollin_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    selected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return diagnostic END CE and signed restricted-choice margin sums."""

    flat_labels = labels.reshape(-1)
    flat_mask = selected_mask.reshape(-1).to(dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != flat_labels.numel():
        raise ValueError("semantic boundary diagnostic logit geometry differs")
    if flat_labels.shape != flat_mask.shape:
        raise ValueError("semantic boundary diagnostic mask geometry differs")
    selected_count = int(flat_mask.sum().item())
    zero = logits.detach().sum() * 0.0
    if selected_count == 0:
        return zero, zero, 0
    if not bool((flat_labels[flat_mask] == c.TOKEN_END_SEMANTIC).all()):
        raise ValueError("semantic boundary roll-in selected a non-END label")
    rows = logits[flat_mask].detach().float()
    targets = flat_labels[flat_mask].long()
    end_ce_sum = F.cross_entropy(rows, targets, reduction="sum")
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    semantic_max = rows[
        :, c.BICODEC_SEMANTIC_OFFSET : semantic_stop
    ].max(dim=-1).values
    end_margin_sum = (
        rows[:, c.TOKEN_END_SEMANTIC] - semantic_max
    ).sum()
    return end_ce_sum, end_margin_sum, selected_count


def semantic_rollin_continue_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    selected_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return target-semantic minus END signed margin over selected rows."""

    flat_labels = labels.reshape(-1)
    flat_mask = selected_mask.reshape(-1).to(dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != flat_labels.numel():
        raise ValueError("semantic continue diagnostic logit geometry differs")
    if flat_labels.shape != flat_mask.shape:
        raise ValueError("semantic continue diagnostic mask geometry differs")
    selected_count = int(flat_mask.sum().item())
    zero = logits.detach().sum() * 0.0
    if selected_count == 0:
        return zero, 0
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    selected_labels = flat_labels[flat_mask]
    if not bool(
        (
            (selected_labels >= c.BICODEC_SEMANTIC_OFFSET)
            & (selected_labels < semantic_stop)
        ).all()
    ):
        raise ValueError("semantic continue roll-in selected a non-semantic label")
    rows = logits[flat_mask].detach().float()
    targets = selected_labels.long()
    target_logits = rows.gather(1, targets[:, None]).squeeze(1)
    signed_margin_sum = (
        target_logits - rows[:, c.TOKEN_END_SEMANTIC]
    ).sum()
    return signed_margin_sum, selected_count


def semantic_rollin_continue_decision_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    selected_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return best-legal-semantic minus END at premature-END decision rows."""

    flat_labels = labels.reshape(-1)
    flat_mask = selected_mask.reshape(-1).to(dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != flat_labels.numel():
        raise ValueError("semantic continue decision diagnostic logit geometry differs")
    if flat_labels.shape != flat_mask.shape:
        raise ValueError("semantic continue decision diagnostic mask geometry differs")
    selected_count = int(flat_mask.sum().item())
    zero = logits.detach().sum() * 0.0
    if selected_count == 0:
        return zero, 0
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    selected_labels = flat_labels[flat_mask]
    if not bool(
        (
            (selected_labels >= c.BICODEC_SEMANTIC_OFFSET)
            & (selected_labels < semantic_stop)
        ).all()
    ):
        raise ValueError(
            "semantic continue decision selected a non-semantic label"
        )
    rows = logits[flat_mask].detach().float()
    semantic_max = rows[
        :, c.BICODEC_SEMANTIC_OFFSET : semantic_stop
    ].max(dim=-1).values
    return (semantic_max - rows[:, c.TOKEN_END_SEMANTIC]).sum(), selected_count


def semantic_boundary_binary_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    end_mask: torch.Tensor,
    continue_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return detached class losses and the common signed score ``END-sem``."""

    if margin < 0:
        raise ValueError("semantic boundary binary margin must be non-negative")
    flat_labels = labels.reshape(-1)
    end = end_mask.reshape(-1).to(dtype=torch.bool)
    cont = continue_mask.reshape(-1).to(dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != flat_labels.numel():
        raise ValueError("semantic boundary binary diagnostic geometry differs")
    if end.shape != flat_labels.shape or cont.shape != flat_labels.shape:
        raise ValueError("semantic boundary binary diagnostic masks differ")
    if bool((end & cont).any()):
        raise ValueError("semantic boundary binary diagnostic masks overlap")
    semantic_start = c.BICODEC_SEMANTIC_OFFSET
    semantic_stop = semantic_start + c.BICODEC_SEMANTIC_SIZE
    if bool(end.any()) and not bool(
        (flat_labels[end] == c.TOKEN_END_SEMANTIC).all()
    ):
        raise ValueError("semantic boundary binary END diagnostic selected non-END")
    if bool(cont.any()) and not bool(
        (
            (flat_labels[cont] >= semantic_start)
            & (flat_labels[cont] < semantic_stop)
        ).all()
    ):
        raise ValueError(
            "semantic boundary binary CONTINUE diagnostic selected non-semantic"
        )
    zero = logits.detach().sum() * 0.0

    def class_values(mask: torch.Tensor, *, target_end: bool):
        count = int(mask.sum().item())
        if count == 0:
            return zero, zero, 0
        rows = logits[mask].detach().float()
        semantic_max = rows[:, semantic_start:semantic_stop].max(dim=-1).values
        score = rows[:, c.TOKEN_END_SEMANTIC] - semantic_max
        loss = F.softplus(
            float(margin) - score if target_end else float(margin) + score
        )
        return loss.sum(), score.sum(), count

    end_loss, end_score, end_count = class_values(end, target_end=True)
    continue_loss, continue_score, continue_count = class_values(
        cont, target_end=False
    )
    return (
        end_loss,
        continue_loss,
        end_score,
        continue_score,
        end_count,
        continue_count,
    )


def _distributed_diagnostics(
    context: Mapping[str, object],
    reference: torch.Tensor,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    active = float(context["acoustic_active"])
    rollin_end_ce_sum, rollin_end_margin_sum, rollin_mask_count = (
        semantic_boundary_rollin_statistics(
            logits,
            labels,
            context["semantic_boundary_rollin_mask"],
        )
    )
    rollin_continue_margin_sum, rollin_continue_mask_count = (
        semantic_rollin_continue_statistics(
            logits,
            labels,
            context["semantic_rollin_continue_mask"],
        )
    )
    rollin_continue_decision_margin_sum, rollin_continue_decision_mask_count = (
        semantic_rollin_continue_decision_statistics(
            logits,
            labels,
            context["semantic_rollin_continue_decision_mask"],
        )
    )
    (
        binary_end_loss_sum,
        binary_continue_loss_sum,
        binary_end_score_sum,
        binary_continue_score_sum,
        binary_end_count,
        binary_continue_count,
    ) = semantic_boundary_binary_statistics(
        logits,
        labels,
        context["semantic_boundary_rollin_mask"],
        context["semantic_rollin_continue_decision_mask"],
        margin=float(context["semantic_boundary_binary_logit_margin"]),
    )
    if rollin_mask_count != int(context["semantic_rollin_end_selected_samples"]):
        raise ValueError("semantic END roll-in mask/count differs")
    if rollin_continue_mask_count != int(
        context["semantic_rollin_continue_selected_samples"]
    ):
        raise ValueError("semantic CONTINUE roll-in mask/count differs")
    if rollin_continue_decision_mask_count != int(
        context["semantic_rollin_continue_selected_samples"]
    ):
        raise ValueError("semantic CONTINUE decision mask/count differs")
    if binary_end_count != rollin_mask_count:
        raise ValueError("semantic boundary binary END mask/count differs")
    if binary_continue_count != rollin_continue_decision_mask_count:
        raise ValueError("semantic boundary binary CONTINUE mask/count differs")
    if rollin_mask_count + rollin_continue_mask_count != int(
        context["semantic_boundary_rollin_selected_tokens"]
    ):
        raise ValueError("semantic roll-in type masks do not cover selected tokens")
    values = reference.new_tensor(
        [
            float(context["causal_glm_agreement"]) * active,
            float(context["bridge_residual_rms"]) * active,
            active,
            float(context["terminal_extensions"]),
            float(context["acoustic_rows"]),
            float(context["semantic_prefix_corrupted_tokens"]),
            float(context["semantic_prefix_eligible_tokens"]),
            float(context["semantic_prefix_corruption_rate"]),
            float(context["semantic_boundary_rollin_selected_tokens"]),
            float(context["semantic_boundary_rollin_eligible_tokens"]),
            float(context["semantic_boundary_rollin_changed_tokens"]),
            float(context["semantic_boundary_rollin_rate"]),
            float(context["semantic_boundary_rollin_selected_samples"]),
            float(context["semantic_boundary_rollin_eligible_samples"]),
            float(rollin_end_ce_sum),
            float(rollin_end_margin_sum),
            float(rollin_continue_decision_margin_sum),
            float(rollin_continue_margin_sum),
            float(context["semantic_rollin_end_selected_samples"]),
            float(context["semantic_rollin_end_eligible_samples"]),
            float(context["semantic_rollin_continue_selected_samples"]),
            float(context["semantic_rollin_continue_eligible_samples"]),
            float(binary_end_loss_sum),
            float(binary_continue_loss_sum),
            float(binary_end_score_sum),
            float(binary_continue_score_sum),
            float(binary_end_count),
            float(binary_continue_count),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
    divisor = values[2].clamp_min(1.0)
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    corruption_denominator = values[6].clamp_min(1.0)
    rollin_denominator = values[9].clamp_min(1.0)
    rollin_selected_denominator = values[18].clamp_min(1.0)
    rollin_sample_denominator = values[13].clamp_min(1.0)
    rollin_continue_selected_denominator = values[20].clamp_min(1.0)
    rollin_end_sample_denominator = values[19].clamp_min(1.0)
    rollin_continue_sample_denominator = values[21].clamp_min(1.0)
    binary_end_denominator = values[26].clamp_min(1.0)
    binary_continue_denominator = values[27].clamp_min(1.0)
    binary_balanced_loss = 0.5 * (
        values[22] / binary_end_denominator
        + values[23] / binary_continue_denominator
    )
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
            (
                "diagnostic/semantic_prefix_corruption_rate",
                values[5] / corruption_denominator,
            ),
            (
                "diagnostic/semantic_prefix_corruption_target_rate",
                values[7] / world_size,
            ),
            ("diagnostic/semantic_prefix_corrupted_tokens", values[5]),
            ("diagnostic/semantic_prefix_eligible_tokens", values[6]),
            (
                "diagnostic/semantic_boundary_rollin_rate",
                values[8] / rollin_denominator,
            ),
            (
                "diagnostic/semantic_boundary_rollin_target_rate",
                values[11] / world_size,
            ),
            (
                "diagnostic/semantic_boundary_rollin_end_ce",
                values[14] / rollin_selected_denominator,
            ),
            (
                "diagnostic/semantic_boundary_rollin_end_margin",
                values[15] / rollin_selected_denominator,
            ),
            (
                "diagnostic/semantic_rollin_continue_decision_signed_margin",
                values[16] / rollin_continue_selected_denominator,
            ),
            (
                "diagnostic/semantic_rollin_continue_signed_margin",
                values[17] / rollin_continue_selected_denominator,
            ),
            ("diagnostic/semantic_boundary_binary_end_count", values[26]),
            ("diagnostic/semantic_boundary_binary_continue_count", values[27]),
            (
                "diagnostic/semantic_boundary_binary_end_signed_score",
                values[24] / binary_end_denominator,
            ),
            (
                "diagnostic/semantic_boundary_binary_continue_signed_score",
                values[25] / binary_continue_denominator,
            ),
            (
                "diagnostic/semantic_boundary_binary_balanced_loss",
                binary_balanced_loss,
            ),
            (
                "diagnostic/semantic_boundary_rollin_selected_tokens",
                values[8],
            ),
            (
                "diagnostic/semantic_boundary_rollin_eligible_tokens",
                values[9],
            ),
            (
                "diagnostic/semantic_boundary_rollin_changed_tokens",
                values[10],
            ),
            (
                "diagnostic/semantic_boundary_rollin_selected_samples",
                values[12],
            ),
            (
                "diagnostic/semantic_boundary_rollin_eligible_samples",
                values[13],
            ),
            (
                "diagnostic/semantic_boundary_rollin_sample_rate",
                values[12] / rollin_sample_denominator,
            ),
            ("diagnostic/semantic_rollin_end_eligible_samples", values[19]),
            ("diagnostic/semantic_rollin_end_selected_samples", values[18]),
            (
                "diagnostic/semantic_rollin_end_sample_rate",
                values[18] / rollin_end_sample_denominator,
            ),
            ("diagnostic/semantic_rollin_continue_eligible_samples", values[21]),
            ("diagnostic/semantic_rollin_continue_selected_samples", values[20]),
            (
                "diagnostic/semantic_rollin_continue_sample_rate",
                values[20] / rollin_continue_sample_denominator,
            ),
        )
    )


def corrupt_interleaved_semantic_prefixes(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    family: str,
    training: bool,
    rate: float,
    tail: int,
    ramp_updates: int,
    update: int,
) -> tuple[torch.Tensor, int, int, float]:
    """Expose semantic fragment endings to deterministic, valid-token prefix noise.

    The inference grammar predicts ``END_SEMANTIC`` after a model-generated
    semantic prefix, while ordinary teacher forcing only presents the exact
    reference prefix.  Corrupting a bounded suffix before each semantic end
    keeps the target sequence and all immutable task pools unchanged, but
    trains the end decision under a small prefix-distribution shift.
    """

    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("semantic prefix corruption rate must be in [0, 1]")
    if int(tail) < 1:
        raise ValueError("semantic prefix corruption tail must be positive")
    if int(ramp_updates) < 0:
        raise ValueError("semantic prefix corruption ramp updates must be non-negative")
    if input_ids.shape != labels.shape:
        raise ValueError("semantic prefix corruption input/label geometry differs")
    if (
        not training
        or family != FAMILY_INTERLEAVED
        or float(rate) == 0.0
        or input_ids.numel() == 0
    ):
        return input_ids, 0, 0, 0.0

    ramp = 1.0
    if int(ramp_updates) > 0:
        ramp = min(1.0, max(0.0, (int(update) + 1) / int(ramp_updates)))
    effective_rate = float(rate) * ramp
    flat_inputs = input_ids.reshape(-1)
    flat_labels = labels.reshape(-1)
    end_positions = torch.nonzero(
        flat_labels == c.TOKEN_END_SEMANTIC, as_tuple=False
    ).reshape(-1)
    if end_positions.numel() == 0:
        return input_ids, 0, 0, effective_rate

    eligible = torch.zeros_like(flat_inputs, dtype=torch.bool)
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    for offset in range(int(tail)):
        positions = end_positions - offset
        valid = positions >= 0
        if not bool(valid.any()):
            break
        positions = positions[valid]
        values = flat_inputs[positions]
        semantic = (values >= c.BICODEC_SEMANTIC_OFFSET) & (
            values < semantic_stop
        )
        if bool(semantic.any()):
            eligible[positions[semantic]] = True

    candidate_positions = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    eligible_count = int(candidate_positions.numel())
    if eligible_count == 0 or effective_rate <= 0.0:
        return input_ids, 0, eligible_count, effective_rate

    candidate_tokens = flat_inputs[candidate_positions].to(dtype=torch.int64)
    # Stateless hashing keeps resumed runs and all ranks deterministic without
    # consuming Megatron's model/data RNG streams.
    hashed = (
        candidate_positions.to(dtype=torch.int64) * 1_103_515_245
        + candidate_tokens * 12_345
        + int(update) * 2_654_435_761
        + 97
    ) % 1_000_003
    threshold = int(round(effective_rate * 1_000_003))
    selected = hashed < threshold
    selected_positions = candidate_positions[selected]
    corrupted_count = int(selected_positions.numel())
    if corrupted_count == 0:
        return input_ids, 0, eligible_count, effective_rate

    output = input_ids.clone()
    flat_output = output.reshape(-1)
    selected_tokens = flat_output[selected_positions].to(dtype=torch.int64)
    selected_hash = hashed[selected]
    delta = 1 + (selected_hash % 31)
    semantic_ids = selected_tokens - c.BICODEC_SEMANTIC_OFFSET
    flat_output[selected_positions] = (
        (semantic_ids + delta) % c.BICODEC_SEMANTIC_SIZE
        + c.BICODEC_SEMANTIC_OFFSET
    ).to(dtype=flat_output.dtype)
    return output, corrupted_count, eligible_count, effective_rate


def semantic_boundary_rollin_candidates(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return model semantic choices that can replace the final gold token.

    For every gold ``END_SEMANTIC`` position, the preceding model logit row is
    evaluated with the same semantic-vs-end restricted choice used by runtime.
    If runtime would already end, that boundary is left ineligible; otherwise
    the model's own semantic continuation becomes the scheduled-sampling input
    for a second, gradient-carrying forward pass.
    """

    flat_inputs = input_ids.reshape(-1)
    flat_labels = labels.reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != flat_inputs.numel():
        raise ValueError("semantic boundary roll-in logits/input geometry differs")
    if flat_inputs.shape != flat_labels.shape:
        raise ValueError("semantic boundary roll-in input/label geometry differs")
    candidates = torch.full_like(flat_inputs, -1)
    ends = torch.nonzero(
        flat_labels == c.TOKEN_END_SEMANTIC, as_tuple=False
    ).reshape(-1)
    if ends.numel() == 0:
        return candidates.reshape_as(input_ids)
    if bool((ends <= 0).any()):
        raise ValueError("END_SEMANTIC cannot be the first packed token")
    current = flat_inputs.index_select(0, ends)
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    if bool(
        ((current < c.BICODEC_SEMANTIC_OFFSET) | (current >= semantic_stop)).any()
    ):
        raise ValueError("END_SEMANTIC is not preceded by a semantic input token")

    prediction_positions = ends - 1
    rows = logits.index_select(0, prediction_positions).float()
    semantic_rows = rows[
        :, c.BICODEC_SEMANTIC_OFFSET : semantic_stop
    ]
    semantic_values, semantic_ids = semantic_rows.max(dim=-1)
    predicted = semantic_ids + c.BICODEC_SEMANTIC_OFFSET
    first_token = (
        flat_inputs.index_select(0, prediction_positions)
        == c.TOKEN_START_SEMANTIC
    )
    semantic_wins = semantic_values > rows[:, c.TOKEN_END_SEMANTIC]
    eligible = first_token | semantic_wins
    if bool(eligible.any()):
        candidates[ends[eligible]] = predicted[eligible].to(candidates.dtype)
    return candidates.reshape_as(input_ids)


def semantic_rollin_continue_candidates(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_boundaries: list[list[tuple[int, int]]],
    tail: int,
) -> torch.Tensor:
    """Return model semantic inputs for pre-END rows that wrongly prefer END.

    Candidate input position ``p`` is inside the final semantic tail before a
    reference ``END_SEMANTIC``.  The no-gradient logit at ``p - 1`` represents
    the runtime choice that supplies input ``p``.  If that choice prefers END,
    the best legal semantic alternative is rolled in at ``p`` and the gradient
    pass trains the semantic label at the same row to remain ahead of END.
    """

    if int(tail) < 1:
        raise ValueError("semantic roll-in continue tail must be positive")
    flat_inputs = input_ids.reshape(-1)
    flat_labels = labels.reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != flat_inputs.numel():
        raise ValueError("semantic continue roll-in logit/input geometry differs")
    if flat_inputs.shape != flat_labels.shape:
        raise ValueError("semantic continue roll-in input/label geometry differs")
    if not isinstance(sample_boundaries, list) or not sample_boundaries:
        raise ValueError("semantic continue roll-in requires packed sample boundaries")
    row_count = len(sample_boundaries)
    if input_ids.numel() % row_count != 0:
        raise ValueError("packed sample boundaries do not divide continue inputs")
    row_width = input_ids.numel() // row_count
    labels_2d = flat_labels.reshape(row_count, row_width)
    inputs_2d = flat_inputs.reshape(row_count, row_width)
    segment_ids = torch.full_like(labels_2d, -1, dtype=torch.long)
    for row, boundaries in enumerate(sample_boundaries):
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("each packed row must contain sample boundaries")
        previous_stop = 0
        for sample_ordinal, raw_boundary in enumerate(boundaries):
            if len(raw_boundary) != 2:
                raise ValueError("packed sample boundary must be a start/stop pair")
            start, stop = int(raw_boundary[0]), int(raw_boundary[1])
            if start != previous_stop or not start < stop <= row_width:
                raise ValueError("packed sample boundaries are not contiguous and valid")
            previous_stop = stop
            segment_ids[row, start:stop] = sample_ordinal
    end = labels_2d == c.TOKEN_END_SEMANTIC
    tail_mask = torch.zeros_like(end)
    for offset in range(1, int(tail) + 1):
        same_sample = segment_ids[:, :-offset] == segment_ids[:, offset:]
        valid_sample = segment_ids[:, :-offset] >= 0
        tail_mask[:, :-offset] |= end[:, offset:] & same_sample & valid_sample
    semantic_start = c.BICODEC_SEMANTIC_OFFSET
    semantic_stop = semantic_start + c.BICODEC_SEMANTIC_SIZE
    tail_mask &= (
        (labels_2d >= semantic_start)
        & (labels_2d < semantic_stop)
        & (inputs_2d >= semantic_start)
        & (inputs_2d < semantic_stop)
    )
    flat_tail = tail_mask.reshape(-1)
    positions = torch.nonzero(flat_tail, as_tuple=False).reshape(-1)
    candidates = torch.full_like(flat_inputs, -1)
    if positions.numel() == 0:
        return candidates.reshape_as(input_ids)
    non_row_start = positions.remainder(row_width) > 0
    positions = positions[non_row_start]
    if positions.numel() == 0:
        return candidates.reshape_as(input_ids)
    prediction_positions = positions - 1
    flat_segments = segment_ids.reshape(-1)
    same_sample = flat_segments.index_select(0, positions) == flat_segments.index_select(
        0, prediction_positions
    )
    positions = positions[same_sample]
    prediction_positions = prediction_positions[same_sample]
    if positions.numel() == 0:
        return candidates.reshape_as(input_ids)
    not_first_runtime_token = (
        flat_inputs.index_select(0, prediction_positions) != c.TOKEN_START_SEMANTIC
    )
    positions = positions[not_first_runtime_token]
    prediction_positions = prediction_positions[not_first_runtime_token]
    if positions.numel() == 0:
        return candidates.reshape_as(input_ids)
    rows = logits.index_select(0, prediction_positions).float()
    semantic_values, semantic_ids = rows[:, semantic_start:semantic_stop].max(dim=-1)
    end_wins = rows[:, c.TOKEN_END_SEMANTIC] >= semantic_values
    if bool(end_wins.any()):
        candidates[positions[end_wins]] = (
            semantic_ids[end_wins] + semantic_start
        ).to(candidates.dtype)
    return candidates.reshape_as(input_ids)


def apply_model_generated_semantic_boundary_rollin(
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    sample_boundaries: list[list[tuple[int, int]]],
    family: str,
    training: bool,
    rate: float,
    ramp_updates: int,
    update: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, float, int, int]:
    """Roll in at most one model-generated semantic boundary per sample.

    ``rate`` is a sample-level scheduled-sampling probability.  A packed
    trajectory can contain many semantic fragments, so independently replacing
    every eligible boundary compounds clean-history candidates inside the
    gradient forward.  Here each independent sample first chooses one eligible
    boundary deterministically, then applies one sample-level Bernoulli draw.
    """

    if input_ids.shape != candidates.shape:
        raise ValueError("semantic boundary roll-in candidate geometry differs")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("semantic boundary roll-in rate must be in [0, 1]")
    if int(ramp_updates) < 0:
        raise ValueError("semantic boundary roll-in ramp updates must be non-negative")
    if not isinstance(sample_boundaries, list) or not sample_boundaries:
        raise ValueError("semantic boundary roll-in requires packed sample boundaries")
    empty_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if (
        not training
        or family != FAMILY_INTERLEAVED
        or float(rate) == 0.0
        or input_ids.numel() == 0
    ):
        return input_ids, empty_mask, 0, 0, 0, 0.0, 0, 0

    ramp = 1.0
    if int(ramp_updates) > 0:
        ramp = min(1.0, max(0.0, (int(update) + 1) / int(ramp_updates)))
    effective_rate = float(rate) * ramp
    flat_candidates = candidates.reshape(-1)
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    eligible = (flat_candidates >= c.BICODEC_SEMANTIC_OFFSET) & (
        flat_candidates < semantic_stop
    )
    positions = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    eligible_count = int(positions.numel())
    if eligible_count == 0 or effective_rate <= 0.0:
        return input_ids, empty_mask, 0, eligible_count, 0, effective_rate, 0, 0

    row_count = len(sample_boundaries)
    if input_ids.numel() % row_count != 0:
        raise ValueError("packed sample boundaries do not divide flattened inputs")
    row_width = input_ids.numel() // row_count
    eligible_positions = [int(value) for value in positions.detach().cpu().tolist()]
    eligible_values = [
        int(value)
        for value in flat_candidates.index_select(0, positions).detach().cpu().tolist()
    ]
    candidate_values = dict(zip(eligible_positions, eligible_values))
    eligible_set = set(eligible_positions)
    covered_eligible: set[int] = set()
    selected_positions_list: list[int] = []
    eligible_sample_count = 0
    modulus = 1_000_003
    threshold = int(round(effective_rate * modulus))

    for row, boundaries in enumerate(sample_boundaries):
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("each packed row must contain sample boundaries")
        previous_stop = 0
        for sample_ordinal, raw_boundary in enumerate(boundaries):
            if len(raw_boundary) != 2:
                raise ValueError("packed sample boundary must be a start/stop pair")
            start, stop = (int(raw_boundary[0]), int(raw_boundary[1]))
            if start != previous_stop or not start < stop <= row_width:
                raise ValueError("packed sample boundaries are not contiguous and valid")
            previous_stop = stop
            global_start = row * row_width + start
            global_stop = row * row_width + stop
            sample_positions = [
                position
                for position in eligible_positions
                if global_start <= position < global_stop
            ]
            covered_eligible.update(sample_positions)
            if not sample_positions:
                continue
            eligible_sample_count += 1

            def candidate_hash(position: int) -> int:
                local_position = position - row * row_width
                token = candidate_values[position]
                return (
                    (int(update) + 1) * 2_654_435_761
                    + (row + 1) * 1_103_515_245
                    + (sample_ordinal + 1) * 97_531
                    + (local_position + 1) * 12_345
                    + token * 193
                    + 17
                ) % modulus

            chosen = min(sample_positions, key=candidate_hash)
            chosen_local = chosen - row * row_width
            chosen_token = candidate_values[chosen]
            sample_hash = (
                (int(update) + 1) * 1_664_525
                + (row + 1) * 1_013_904_223
                + (sample_ordinal + 1) * 69_069
                + (chosen_local + 1) * 36_457
                + chosen_token * 193
                + 911
            ) % modulus
            if sample_hash < threshold:
                selected_positions_list.append(chosen)

    if covered_eligible != eligible_set:
        raise ValueError("eligible semantic boundaries fall outside packed samples")
    selected_sample_count = len(selected_positions_list)
    selected_positions = torch.tensor(
        selected_positions_list, dtype=torch.long, device=input_ids.device
    )
    selected_count = int(selected_positions.numel())
    if selected_count == 0:
        return (
            input_ids,
            empty_mask,
            0,
            eligible_count,
            0,
            effective_rate,
            0,
            eligible_sample_count,
        )

    output = input_ids.clone()
    flat_output = output.reshape(-1)
    selected_values = flat_candidates.index_select(0, selected_positions)
    previous = flat_output.index_select(0, selected_positions)
    flat_output[selected_positions] = selected_values.to(flat_output.dtype)
    mask = torch.zeros_like(flat_output, dtype=torch.bool)
    mask[selected_positions] = True
    changed_count = int((previous != selected_values).sum().item())
    return (
        output,
        mask.reshape_as(input_ids),
        selected_count,
        eligible_count,
        changed_count,
        effective_rate,
        selected_sample_count,
        eligible_sample_count,
    )


def apply_symmetric_model_generated_semantic_rollin(
    input_ids: torch.Tensor,
    end_candidates: torch.Tensor,
    continue_candidates: torch.Tensor,
    *,
    sample_boundaries: list[list[tuple[int, int]]],
    family: str,
    training: bool,
    rate: float,
    ramp_updates: int,
    continue_ratio: float,
    update: int,
) -> SemanticBoundaryRollinResult:
    """Select at most one END or CONTINUE model-history candidate per sample."""

    if input_ids.shape != end_candidates.shape or input_ids.shape != continue_candidates.shape:
        raise ValueError("symmetric semantic roll-in candidate geometry differs")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("semantic boundary roll-in rate must be in [0, 1]")
    if not 0.0 <= float(continue_ratio) <= 1.0:
        raise ValueError("semantic roll-in continue ratio must be in [0, 1]")
    if int(ramp_updates) < 0:
        raise ValueError("semantic boundary roll-in ramp updates must be non-negative")
    if not isinstance(sample_boundaries, list) or not sample_boundaries:
        raise ValueError("symmetric semantic roll-in requires packed sample boundaries")
    empty_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    disabled = (
        not training
        or family != FAMILY_INTERLEAVED
        or float(rate) == 0.0
        or input_ids.numel() == 0
    )
    if disabled:
        return SemanticBoundaryRollinResult(
            input_ids, empty_mask, 0, 0, 0, 0.0, 0, 0,
            empty_mask, empty_mask, empty_mask, 0, 0, 0, 0,
        )

    ramp = 1.0
    if int(ramp_updates) > 0:
        ramp = min(1.0, max(0.0, (int(update) + 1) / int(ramp_updates)))
    effective_rate = float(rate) * ramp
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    flat_end = end_candidates.reshape(-1)
    flat_continue = continue_candidates.reshape(-1)
    end_eligible = (flat_end >= c.BICODEC_SEMANTIC_OFFSET) & (flat_end < semantic_stop)
    continue_eligible = (flat_continue >= c.BICODEC_SEMANTIC_OFFSET) & (
        flat_continue < semantic_stop
    )
    end_positions = [
        int(value)
        for value in torch.nonzero(end_eligible, as_tuple=False).reshape(-1).cpu().tolist()
    ]
    continue_positions = [
        int(value)
        for value in torch.nonzero(continue_eligible, as_tuple=False).reshape(-1).cpu().tolist()
    ]
    end_values = dict(
        zip(
            end_positions,
            [
                int(value)
                for value in flat_end[end_eligible].detach().cpu().tolist()
            ],
        )
    )
    continue_values = dict(
        zip(
            continue_positions,
            [
                int(value)
                for value in flat_continue[continue_eligible].detach().cpu().tolist()
            ],
        )
    )
    eligible_count = len(end_positions) + len(continue_positions)
    if eligible_count == 0 or effective_rate <= 0.0:
        return SemanticBoundaryRollinResult(
            input_ids, empty_mask, 0, eligible_count, 0, effective_rate, 0, 0,
            empty_mask, empty_mask, empty_mask, 0, 0, 0, 0,
        )

    row_count = len(sample_boundaries)
    if input_ids.numel() % row_count != 0:
        raise ValueError("packed sample boundaries do not divide symmetric inputs")
    row_width = input_ids.numel() // row_count
    end_set = set(end_positions)
    continue_set = set(continue_positions)
    covered_end: set[int] = set()
    covered_continue: set[int] = set()
    selected_end: list[int] = []
    selected_continue: list[int] = []
    eligible_samples = 0
    eligible_end_samples = 0
    eligible_continue_samples = 0
    modulus = 1_000_003
    selection_threshold = int(round(effective_rate * modulus))
    type_threshold = int(round(float(continue_ratio) * modulus))

    def candidate_hash(
        *, row: int, sample_ordinal: int, candidate_type: int, position: int, token: int
    ) -> int:
        local_position = position - row * row_width
        return _stable_semantic_rollin_hash(
            int(update) + 1,
            row + 1,
            sample_ordinal + 1,
            candidate_type + 1,
            local_position + 1,
            token,
        ) % modulus

    for row, boundaries in enumerate(sample_boundaries):
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("each packed row must contain sample boundaries")
        previous_stop = 0
        for sample_ordinal, raw_boundary in enumerate(boundaries):
            if len(raw_boundary) != 2:
                raise ValueError("packed sample boundary must be a start/stop pair")
            start, stop = int(raw_boundary[0]), int(raw_boundary[1])
            if start != previous_stop or not start < stop <= row_width:
                raise ValueError("packed sample boundaries are not contiguous and valid")
            previous_stop = stop
            global_start = row * row_width + start
            global_stop = row * row_width + stop
            sample_end = [p for p in end_positions if global_start <= p < global_stop]
            sample_continue = [
                p for p in continue_positions if global_start <= p < global_stop
            ]
            if any(position <= global_start for position in sample_continue):
                raise ValueError(
                    "CONTINUE candidate has no decision row inside its packed sample"
                )
            covered_end.update(sample_end)
            covered_continue.update(sample_continue)
            if sample_end:
                eligible_end_samples += 1
            if sample_continue:
                eligible_continue_samples += 1
            if not sample_end and not sample_continue:
                continue
            eligible_samples += 1

            chosen_end = None
            chosen_continue = None
            if sample_end:
                chosen_end = min(
                    sample_end,
                    key=lambda position: candidate_hash(
                        row=row,
                        sample_ordinal=sample_ordinal,
                        candidate_type=0,
                        position=position,
                        token=end_values[position],
                    ),
                )
            if sample_continue:
                chosen_continue = min(
                    sample_continue,
                    key=lambda position: candidate_hash(
                        row=row,
                        sample_ordinal=sample_ordinal,
                        candidate_type=1,
                        position=position,
                        token=continue_values[position],
                    ),
                )
            if chosen_end is not None and chosen_continue is not None:
                end_hash = candidate_hash(
                    row=row,
                    sample_ordinal=sample_ordinal,
                    candidate_type=0,
                    position=chosen_end,
                    token=end_values[chosen_end],
                )
                continue_hash = candidate_hash(
                    row=row,
                    sample_ordinal=sample_ordinal,
                    candidate_type=1,
                    position=chosen_continue,
                    token=continue_values[chosen_continue],
                )
                type_hash = _stable_semantic_rollin_hash(
                    int(update) + 1,
                    row + 1,
                    sample_ordinal + 1,
                    1,
                    chosen_end - row * row_width + 1,
                    2,
                    chosen_continue - row * row_width + 1,
                    end_hash,
                    continue_hash,
                ) % modulus
                chosen_type = 1 if type_hash < type_threshold else 0
            else:
                chosen_type = 1 if chosen_continue is not None else 0
            chosen = chosen_continue if chosen_type == 1 else chosen_end
            if chosen is None:
                raise AssertionError("symmetric semantic roll-in selected a missing type")
            chosen_values = continue_values if chosen_type == 1 else end_values
            chosen_token = chosen_values[chosen]
            selection_hash = _stable_semantic_rollin_hash(
                int(update) + 1,
                row + 1,
                sample_ordinal + 1,
                chosen_type + 1,
                chosen - row * row_width + 1,
                chosen_token,
                911,
            ) % modulus
            if selection_hash < selection_threshold:
                (selected_continue if chosen_type == 1 else selected_end).append(chosen)

    if covered_end != end_set or covered_continue != continue_set:
        raise ValueError("eligible symmetric semantic candidates fall outside packed samples")
    all_selected = selected_end + selected_continue
    selected_count = len(all_selected)
    if selected_count == 0:
        return SemanticBoundaryRollinResult(
            input_ids, empty_mask, 0, eligible_count, 0, effective_rate, 0,
            eligible_samples, empty_mask, empty_mask, empty_mask, 0,
            eligible_end_samples, 0, eligible_continue_samples,
        )

    output = input_ids.clone()
    flat_output = output.reshape(-1)
    end_mask = torch.zeros_like(flat_output, dtype=torch.bool)
    continue_decision_mask = torch.zeros_like(flat_output, dtype=torch.bool)
    continue_mask = torch.zeros_like(flat_output, dtype=torch.bool)
    changed = 0
    for positions, values, mask in (
        (selected_end, flat_end, end_mask),
        (selected_continue, flat_continue, continue_mask),
    ):
        if not positions:
            continue
        index = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
        replacement = values.index_select(0, index).to(flat_output.dtype)
        changed += int((flat_output.index_select(0, index) != replacement).sum().item())
        flat_output[index] = replacement
        mask[index] = True
    if selected_continue:
        continue_index = torch.tensor(
            selected_continue, dtype=torch.long, device=input_ids.device
        )
        decision_index = continue_index - 1
        if bool((decision_index < 0).any()) or bool(
            (continue_index.remainder(row_width) == 0).any()
        ):
            raise ValueError("CONTINUE decision row crosses a packed sample boundary")
        continue_decision_mask[decision_index] = True
    selected_mask = end_mask | continue_mask
    return SemanticBoundaryRollinResult(
        output,
        selected_mask.reshape_as(input_ids),
        selected_count,
        eligible_count,
        changed,
        effective_rate,
        selected_count,
        eligible_samples,
        end_mask.reshape_as(input_ids),
        continue_decision_mask.reshape_as(input_ids),
        continue_mask.reshape_as(input_ids),
        len(selected_end),
        eligible_end_samples,
        len(selected_continue),
        eligible_continue_samples,
    )


def _semantic_boundary_rollin_output_processor(**kwargs) -> torch.Tensor:
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("semantic roll-in expects flattened TP=PP=1 logits")
    context = kwargs["context"]
    end_candidates = semantic_boundary_rollin_candidates(
        logits[:, 0],
        context["input_ids"],
        kwargs["labels"],
    )
    continue_candidates = semantic_rollin_continue_candidates(
        logits[:, 0],
        context["input_ids"],
        kwargs["labels"],
        sample_boundaries=context["sample_boundaries"],
        tail=int(context["continue_tail"]),
    )
    return torch.stack((end_candidates, continue_candidates), dim=0)


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
        semantic_continue_logit_margin=float(
            context["semantic_continue_logit_margin"]
        ),
        semantic_boundary_binary_logit_margin=float(
            context["semantic_boundary_binary_logit_margin"]
        ),
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
    metrics.update(
        _distributed_diagnostics(
            context,
            total.detach(),
            logits=logits,
            labels=labels,
        )
    )
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
        family = str(e2e_batch["family"])
        (
            effective_input_ids,
            semantic_prefix_corrupted_tokens,
            semantic_prefix_eligible_tokens,
            semantic_prefix_corruption_rate,
        ) = corrupt_interleaved_semantic_prefixes(
            input_ids,
            labels,
            family=family,
            training=bool(self.training),
            rate=float(e2e_batch["semantic_prefix_corruption_rate"].item()),
            tail=int(e2e_batch["semantic_prefix_corruption_tail"].item()),
            ramp_updates=int(
                e2e_batch["semantic_prefix_corruption_ramp_updates"].item()
            ),
            update=update,
        )
        agreement = 0.0
        residual_rms = 0.0
        terminal_extensions = 0.0
        acoustic_rows = 0
        acoustic_active = 0
        acoustic_hidden = None
        acoustic_lengths = None
        if "waveform" in e2e_batch:
            self.stage_a_objective.eval()
            with torch.no_grad():
                acoustic = self.stage_a_objective.frontend(
                    e2e_batch["waveform"],
                    e2e_batch["waveform_lengths"],
                    chunk_ms=chunk_ms,
                )
            acoustic_hidden = acoustic.pooled_hidden.detach()
            acoustic_lengths = acoustic.pooled_lengths.detach()

        def prepare_decoder(selected_input_ids: torch.Tensor):
            prepared = self.embedding(
                input_ids=selected_input_ids, position_ids=position_ids
            )
            if acoustic_hidden is None or acoustic_lengths is None:
                return prepared, None
            (
                prepared,
                _,
                agreement_tensor,
                residual_tensor,
                terminal_tensor,
            ) = self.stage_a_objective._inject_causal_glm(
                prepared,
                base._embedding_weight(self),
                acoustic_hidden,
                acoustic_lengths,
                e2e_batch,
                original_seq_length=original_seq_length,
            )
            return prepared, (
                agreement_tensor,
                residual_tensor,
                terminal_tensor,
            )

        decoder_input, acoustic_diagnostics = prepare_decoder(effective_input_ids)
        if acoustic_diagnostics is not None:
            agreement_tensor, residual_tensor, terminal_tensor = acoustic_diagnostics
            agreement = float(agreement_tensor.detach())
            residual_rms = float(residual_tensor.detach())
            terminal_extensions = float(terminal_tensor.detach())
            acoustic_rows = int(e2e_batch["waveform"].shape[0])
            acoustic_active = 1

        rollin_mask = torch.zeros_like(effective_input_ids, dtype=torch.bool)
        rollin_selected_tokens = 0
        rollin_eligible_tokens = 0
        rollin_changed_tokens = 0
        rollin_rate = 0.0
        rollin_selected_samples = 0
        rollin_eligible_samples = 0
        rollin_end_mask = torch.zeros_like(effective_input_ids, dtype=torch.bool)
        rollin_continue_decision_mask = torch.zeros_like(
            effective_input_ids, dtype=torch.bool
        )
        rollin_continue_mask = torch.zeros_like(effective_input_ids, dtype=torch.bool)
        rollin_selected_end_samples = 0
        rollin_eligible_end_samples = 0
        rollin_selected_continue_samples = 0
        rollin_eligible_continue_samples = 0
        configured_rollin_rate = float(
            e2e_batch["semantic_boundary_rollin_rate"].item()
        )
        if (
            bool(self.training)
            and family == FAMILY_INTERLEAVED
            and configured_rollin_rate > 0.0
        ):
            with torch.no_grad():
                candidates = raw_forward(
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
                    output_processor=_semantic_boundary_rollin_output_processor,
                    output_processor_context={
                        "input_ids": effective_input_ids,
                        "sample_boundaries": e2e_batch["sample_boundaries"],
                        "continue_tail": int(
                            e2e_batch["semantic_rollin_continue_tail"].item()
                        ),
                    },
                )
            if candidates.ndim != effective_input_ids.ndim + 1 or candidates.shape[0] != 2:
                raise ValueError("symmetric semantic roll-in candidate stack is invalid")
            rollin = apply_symmetric_model_generated_semantic_rollin(
                effective_input_ids,
                candidates[0],
                candidates[1],
                sample_boundaries=e2e_batch["sample_boundaries"],
                family=family,
                training=bool(self.training),
                rate=configured_rollin_rate,
                ramp_updates=int(
                    e2e_batch["semantic_boundary_rollin_ramp_updates"].item()
                ),
                continue_ratio=float(
                    e2e_batch["semantic_rollin_continue_ratio"].item()
                ),
                update=update,
            )
            rolled_input_ids = rollin.input_ids
            rollin_mask = rollin.selected_mask
            rollin_selected_tokens = rollin.selected_tokens
            rollin_eligible_tokens = rollin.eligible_tokens
            rollin_changed_tokens = rollin.changed_tokens
            rollin_rate = rollin.effective_rate
            rollin_selected_samples = rollin.selected_samples
            rollin_eligible_samples = rollin.eligible_samples
            rollin_end_mask = rollin.end_mask
            rollin_continue_decision_mask = rollin.continue_decision_mask
            rollin_continue_mask = rollin.continue_mask
            rollin_selected_end_samples = rollin.selected_end_samples
            rollin_eligible_end_samples = rollin.eligible_end_samples
            rollin_selected_continue_samples = rollin.selected_continue_samples
            rollin_eligible_continue_samples = rollin.eligible_continue_samples
            if rollin_selected_tokens > 0:
                decoder_input, _ = prepare_decoder(rolled_input_ids)
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
            "semantic_end_logit_margin": float(
                e2e_batch["semantic_end_logit_margin"].item()
            ),
            "semantic_continue_tail": int(
                e2e_batch["semantic_continue_tail"].item()
            ),
            "semantic_continue_logit_margin": float(
                e2e_batch["semantic_continue_logit_margin"].item()
            ),
            "semantic_rollin_continue_logit_margin": float(
                e2e_batch["semantic_rollin_continue_logit_margin"].item()
            ),
            "semantic_rollin_continue_decision_logit_margin": float(
                e2e_batch[
                    "semantic_rollin_continue_decision_logit_margin"
                ].item()
            ),
            "semantic_boundary_binary_logit_margin": float(
                e2e_batch["semantic_boundary_binary_logit_margin"].item()
            ),
            "semantic_prefix_corruption_rate": semantic_prefix_corruption_rate,
            "semantic_prefix_corrupted_tokens": semantic_prefix_corrupted_tokens,
            "semantic_prefix_eligible_tokens": semantic_prefix_eligible_tokens,
            "semantic_boundary_rollin_mask": rollin_end_mask,
            "semantic_rollin_continue_decision_mask": (
                rollin_continue_decision_mask
            ),
            "semantic_rollin_continue_mask": rollin_continue_mask,
            "semantic_boundary_rollin_rate": rollin_rate,
            "semantic_boundary_rollin_selected_tokens": rollin_selected_tokens,
            "semantic_boundary_rollin_eligible_tokens": rollin_eligible_tokens,
            "semantic_boundary_rollin_changed_tokens": rollin_changed_tokens,
            "semantic_boundary_rollin_selected_samples": rollin_selected_samples,
            "semantic_boundary_rollin_eligible_samples": rollin_eligible_samples,
            "semantic_rollin_end_selected_samples": rollin_selected_end_samples,
            "semantic_rollin_end_eligible_samples": rollin_eligible_end_samples,
            "semantic_rollin_continue_selected_samples": (
                rollin_selected_continue_samples
            ),
            "semantic_rollin_continue_eligible_samples": (
                rollin_eligible_continue_samples
            ),
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
    batch["semantic_end_logit_margin"] = torch.tensor(
        float(args.e2e_semantic_end_logit_margin),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_continue_tail"] = torch.tensor(
        int(args.e2e_semantic_continue_tail),
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    batch["semantic_continue_logit_margin"] = torch.tensor(
        float(args.e2e_semantic_continue_logit_margin),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_rollin_continue_logit_margin"] = torch.tensor(
        float(args.e2e_semantic_rollin_continue_logit_margin),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_rollin_continue_decision_logit_margin"] = torch.tensor(
        float(args.e2e_semantic_rollin_continue_decision_logit_margin),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_boundary_binary_logit_margin"] = torch.tensor(
        float(args.e2e_semantic_boundary_binary_logit_margin),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_rollin_continue_tail"] = torch.tensor(
        int(args.e2e_semantic_rollin_continue_tail),
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    batch["semantic_rollin_continue_ratio"] = torch.tensor(
        float(args.e2e_semantic_rollin_continue_ratio),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_prefix_corruption_rate"] = torch.tensor(
        float(args.e2e_semantic_prefix_corruption_rate),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_prefix_corruption_tail"] = torch.tensor(
        int(args.e2e_semantic_prefix_corruption_tail),
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    batch["semantic_prefix_corruption_ramp_updates"] = torch.tensor(
        int(args.e2e_semantic_prefix_corruption_ramp_updates),
        dtype=torch.long,
        device=batch["tokens"].device,
    )
    batch["semantic_boundary_rollin_rate"] = torch.tensor(
        float(args.e2e_semantic_boundary_rollin_rate),
        dtype=torch.float32,
        device=batch["tokens"].device,
    )
    batch["semantic_boundary_rollin_ramp_updates"] = torch.tensor(
        int(args.e2e_semantic_boundary_rollin_ramp_updates),
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
