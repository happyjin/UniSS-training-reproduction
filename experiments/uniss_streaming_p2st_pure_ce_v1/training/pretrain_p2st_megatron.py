#!/usr/bin/env python3
"""Megatron entry point for prefix-to-prefix training with no speak decision.

What is reused and why that is the point
----------------------------------------
Everything except the data path.  The model provider, the forward step, the
objective, the argument surface and its validation all come from
``pretrain_e2e_megatron`` unchanged, so this run is comparable to B' on the
same code.  The objective needs no extension because this pool is pure CE:
its teacher-KL and commit-consistency terms simply see zero denominators and
go inactive, which the loss audit already reports as such.

Three process-local overrides make the base trainer accept three families it
has never seen, and nothing is edited in the base experiment, whose objective
and Stage-A tensors are under a bit-for-bit frozen audit:

* ``REQUIRED_FAMILY_DENOMINATORS`` gains an entry per family, so
  ``validate_family_denominators`` still fails *before* backward if a family
  silently loses its supervision.  That check is most of what the smoke is
  for: it is the difference between "the run did not crash" and "every
  family's loss actually fired".
* ``FAMILY_IDS`` gains an id per family, used only for logging.
* the dataset provider is this module's own, because the base one builds five
  families from a build report with teacher readers attached.

``FiveFamilyCoverageSampler`` is reused as it is -- it keys off
``synchronize_task_family`` and the batch geometry, never off the family
names.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as base
from experiments.uniss_streaming_p2st_pure_ce_v1.training.p2st_dataset import (
    P2STPackedFamilyDataset,
    collate_p2st_family,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.p2st_schedule import (
    ThreeFamilyGlobalSchedule,
    ThreeFamilySchedulePrefix,
    ThreeFamilySingleBlock,
    ThreeFamilyValidationSchedule,
    POOL_WEIGHTS,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    POOL_FAMILIES,
    P2ST_FAMILIES,
)
from training.pretrain_uniss_megatron import load_megatron_runtime

EXPERIMENT = "uniss_streaming_p2st_pure_ce_v1"

# Each family supervises its own content plus the one boundary token that ends
# its sequence and the EOS after it.  Nothing else may be required: a teacher
# term listed here would silently reintroduce the cache dependency this pool
# exists to shed.
P2ST_REQUIRED_DENOMINATORS = {
    FAMILY_P2ST_ASR: ("asr_ce", "boundary_ce", "eos_ce"),
    FAMILY_P2ST_MT: ("mt_ce", "boundary_ce", "eos_ce"),
    FAMILY_P2ST_TTS: ("semantic_ce", "boundary_ce", "eos_ce"),
}

# The replay families keep the base trainer's own requirement -- they are the
# base builder's samples, so ``base.REQUIRED_FAMILY_DENOMINATORS`` already
# names ``replay_ce`` for both and ``base.FAMILY_IDS`` already holds their ids.
# Listing them here would collide; this table exists only to report what is
# expected of every family in the pool.
POOL_REQUIRED_DENOMINATORS = {
    **P2ST_REQUIRED_DENOMINATORS,
    FAMILY_PHASE3_QUALITY: ("replay_ce",),
    FAMILY_PHASE3_PERFORMANCE: ("replay_ce",),
}


def install_p2st_overrides() -> None:
    """Teach the base trainer's family tables about the three new families.

    Must run before ``add_experiment_args``: ``--e2e-smoke-family`` is declared
    with ``choices=TASK_FAMILIES``, so argparse would reject a p2st family name
    otherwise.  Widening the module-global here is enough because the base
    provider, which is the only other reader of it, is not the provider this
    entry point passes to Megatron.
    """
    for family, required in P2ST_REQUIRED_DENOMINATORS.items():
        if family in base.REQUIRED_FAMILY_DENOMINATORS:
            raise KeyError(f"family {family!r} already known to the base trainer")
        base.REQUIRED_FAMILY_DENOMINATORS[family] = tuple(required)
    for family in P2ST_FAMILIES:
        if family not in base.FAMILY_IDS:
            base.FAMILY_IDS[family] = len(base.FAMILY_IDS)
    base.TASK_FAMILIES = tuple(base.TASK_FAMILIES) + tuple(
        family for family in P2ST_FAMILIES if family not in base.TASK_FAMILIES
    )


def _family_datasets(manifest: str) -> dict[str, P2STPackedFamilyDataset]:
    return {
        family: P2STPackedFamilyDataset.from_pool_manifest(
            manifest, family=family, load_audio=True
        )
        for family in POOL_FAMILIES
    }


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    dp_microbatch = int(args.data_parallel_size) * int(args.micro_batch_size)
    train_sources = _family_datasets(args.e2e_train_build_report)
    train = ThreeFamilyGlobalSchedule(
        train_sources,
        coverage_epochs=int(args.e2e_coverage_epochs),
        global_batch_size=int(args.global_batch_size),
        data_parallel_group_size=dp_microbatch,
        shuffle_seed=int(args.seed),
        weights=POOL_WEIGHTS,
    )
    train.collate_fn = collate_p2st_family
    target_train = int(train_val_test_num_samples[0])
    if args.e2e_smoke_family:
        # A smoke is capped at two updates and one family owns a whole global
        # batch, so a mixed smoke cannot reach all five families.  Naming the
        # family is both the way around that and the stronger check.
        if int(args.train_iters) != 1 or target_train != int(args.global_batch_size):
            raise ValueError("one-family p2st canary requires exactly one update")
        train = ThreeFamilySingleBlock(train, args.e2e_smoke_family)
        train.collate_fn = collate_p2st_family
    elif target_train != len(train):
        bounded = bool(args.e2e_smoke) or bool(args.e2e_learning_canary) or bool(
            args.e2e_extended_canary
        )
        if not bounded or not 0 < target_train <= len(train):
            raise ValueError(
                f"p2st schedule length {len(train)} differs from Megatron target "
                f"{target_train} and this is not a bounded run"
            )
        train = ThreeFamilySchedulePrefix(train, target_train)
        train.collate_fn = collate_p2st_family

    valid = None
    if args.e2e_valid_build_report and int(train_val_test_num_samples[1]) > 0:
        valid_sources = _family_datasets(args.e2e_valid_build_report)
        eval_global_batch = int(
            getattr(args, "eval_global_batch_size", None) or args.global_batch_size
        )
        eval_micro_batch = int(
            getattr(args, "eval_micro_batch_size", None) or args.micro_batch_size
        )
        valid = ThreeFamilyValidationSchedule(
            valid_sources,
            total_samples=int(train_val_test_num_samples[1]),
            global_batch_size=eval_global_batch,
            data_parallel_group_size=int(args.data_parallel_size) * eval_micro_batch,
            shuffle_seed=int(args.seed) + 1,
            weights=POOL_WEIGHTS,
        )
        valid.collate_fn = collate_p2st_family

    runtime.print_rank_0(
        "> p2st datasets: "
        f"families={{{', '.join(f'{n}:{len(train_sources[n])}' for n in POOL_FAMILIES)}}} "
        f"blocks={train.total_blocks} samples={len(train)} "
        f"family_blocks={train.family_block_counts} "
        f"valid={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    # Must precede add_experiment_args: --e2e-smoke-family's argparse choices
    # are read from TASK_FAMILIES at declaration time.
    install_p2st_overrides()

    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=base.add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    base.validate_experiment_args(args)
    # The same four installers the base entry point runs, in the same order.
    # install_joint_collate is the one that matters most here: Megatron's
    # build_pretraining_data_loader never looks at dataset.collate_fn, so
    # without it the loader falls back to default_collate and dies on the
    # first ragged field.
    base.install_e2e_lr_overrides(args)
    base.install_family_sampler()
    # These two live one level down, on the true-subsecond module the base
    # trainer imports as its own ``base``.
    base.base.install_joint_collate()
    base.base.install_rerun_checkpoint_compatibility()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "load": str(Path(args.load).resolve()),
                    "families": list(POOL_FAMILIES),
                    "required_denominators": {
                        family: list(names)
                        for family, names in POOL_REQUIRED_DENOMINATORS.items()
                    },
                    "speak_decision": "removed; switching is external at inference",
                    "teacher_bindings": 0,
                    "shuffle": "uniform_family_blocks_then_per_family_randperm",
                },
                sort_keys=True,
            ),
            flush=True,
        )
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
