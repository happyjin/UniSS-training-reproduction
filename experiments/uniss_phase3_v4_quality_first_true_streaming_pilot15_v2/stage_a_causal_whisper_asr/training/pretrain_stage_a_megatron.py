#!/usr/bin/env python3
"""Isolated v2 wrapper around the validated native Megatron Stage A entrypoint."""

from __future__ import annotations

from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as implementation,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training.dataset import (
    IndexedStageAPackDataset,
    PaddedStageAValidationDataset,
    ThreeEpochStageASchedule,
    collate_stage_a,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    distributed_stage_a_objective,
)


_original_add_experiment_args = implementation.add_experiment_args
_original_validate_experiment_args = implementation.validate_experiment_args
_original_augment_native_gpt = implementation.augment_native_gpt


def checkpoint_contains_stage_a_state(keys) -> bool:
    """Distinguish a trained Stage A resume from pristine Phase3 initialization."""

    return any(
        str(key).split("/shard_", 1)[0].startswith(implementation.STAGE_A_PREFIX)
        for key in keys
    )


def _is_stage_a_resume(load_root: str | Path) -> bool:
    from torch.distributed.checkpoint import FileSystemReader

    root = Path(load_root).resolve()
    latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = root / f"iter_{latest:07d}"
    keys = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    return checkpoint_contains_stage_a_state(keys)


def add_experiment_args(parser):
    parser = _original_add_experiment_args(parser)
    group = parser.add_argument_group(title="UniSS quality-first Stage A v2")
    group.add_argument("--stage-a-train-teacher-cache", required=True)
    group.add_argument("--stage-a-valid-teacher-cache")
    group.add_argument("--stage-a-teacher-lru-capacity", type=int, default=16)
    return parser


def validate_experiment_args(args) -> None:
    _original_validate_experiment_args(args)
    required = [args.stage_a_train_teacher_cache]
    if args.stage_a_valid_packs:
        required.append(args.stage_a_valid_teacher_cache)
    for value in required:
        if value is None or not Path(value).is_file():
            raise FileNotFoundError(value)
    if int(args.stage_a_teacher_lru_capacity) <= 0:
        raise ValueError("Stage A teacher LRU capacity must be positive")


def _dataset(args, path: str, *, load_audio: bool = True):
    cache = (
        args.stage_a_train_teacher_cache
        if Path(path).resolve() == Path(args.stage_a_train_packs).resolve()
        else args.stage_a_valid_teacher_cache
    )
    if cache is None:
        raise ValueError("Stage A v2 dataset has no matching teacher cache")
    return IndexedStageAPackDataset(
        path,
        seq_length=int(args.seq_length),
        max_acoustics_per_pack=int(args.stage_a_max_acoustics_per_pack),
        teacher_cache_manifest=cache,
        teacher_lru_capacity=int(args.stage_a_teacher_lru_capacity),
        load_audio=load_audio,
    )


def augment_native_gpt(model, args):
    counts = _original_augment_native_gpt(model, args)
    if _is_stage_a_resume(args.load):
        # A strict Stage A checkpoint is expected to differ from the pristine
        # Phase3 fingerprint.  Megatron's raise_all load validates every model,
        # optimizer, and RNG key; the original fingerprint remains mandatory
        # only for the first Phase3 -> Stage A initialization.
        model._true_phase3_fingerprint_verified = True
    return counts


implementation.add_experiment_args = add_experiment_args
implementation.validate_experiment_args = validate_experiment_args
implementation._dataset = _dataset
implementation.augment_native_gpt = augment_native_gpt
implementation.IndexedStageAPackDataset = IndexedStageAPackDataset
implementation.PaddedStageAValidationDataset = PaddedStageAValidationDataset
implementation.ThreeEpochStageASchedule = ThreeEpochStageASchedule
implementation.StageAObjective = StageAObjective
implementation.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
implementation.TERM_NAMES = TERM_NAMES
implementation.distributed_stage_a_objective = distributed_stage_a_objective
implementation.METRIC_NAMES = (
    *TERM_NAMES,
    *DIAGNOSTIC_NAMES,
    *implementation.CURRICULUM_METRICS,
)
def main() -> None:
    implementation.main()


if __name__ == "__main__":
    main()


__all__ = ["checkpoint_contains_stage_a_state", "main"]
