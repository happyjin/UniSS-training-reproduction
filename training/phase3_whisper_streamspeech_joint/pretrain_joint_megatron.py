#!/usr/bin/env python3
"""Megatron-Core entrypoint for single-stage Phase3 StreamSpeech training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"
for import_path in (REPO_ROOT, MEGATRON_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import torch

from training.megatron_uniss_dataset import RepeatToLengthDataset
from training.phase3_whisper_streamspeech_joint.config import (
    JointLossWeights,
    MultiChunkConfig,
)
from training.phase3_whisper_streamspeech_joint.dataset import (
    DeterministicReplaySchedule,
    DirectionBalancedJointDataset,
    IndexedPhase3ReplayDataset,
    JointAudioDataset,
    SynchronizedKindRandomSampler,
    collate_joint_or_replay,
)
from training.phase3_whisper_streamspeech_joint.model import (
    COMPONENTS,
    Phase3WhisperStreamSpeechJointModel,
)
from training.phase3_whisper_streamspeech_joint.whisper_multichunk import (
    choose_chunk_ms,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


METRIC_NAMES = (
    *COMPONENTS,
    "ctc/asr_infeasible",
    "ctc/nar_infeasible",
    "ctc/unit_infeasible",
    "bridge/commitment_mse",
    "whisper/quantize_loss",
    "sampler/joint_fraction",
)


def parse_chunks(value: str) -> tuple[int | None, ...]:
    chunks: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        chunks.append(None if item in {"offline", "full", "none"} else int(item))
    return tuple(chunks)


def lr_group_values(base_lr: float, min_lr: float) -> dict[str, dict[str, float]]:
    multipliers = {
        "uniss_lr_new": 1.0,
        "uniss_lr_bridge": 0.5,
        "uniss_lr_whisper_top": 0.1,
        "uniss_lr_whisper_bottom": 0.05,
        "uniss_lr_qwen": 0.02,
        "uniss_lr_qwen_io": 0.01,
    }
    return {
        name: {
            "lr_mult": multiplier,
            "max_lr": base_lr * multiplier,
            "min_lr": min_lr * multiplier,
        }
        for name, multiplier in multipliers.items()
    }


def install_megatron_lr_overrides() -> None:
    """Add isolated parameter-group LRs without patching Megatron files."""

    import megatron.training.training as megatron_training
    from megatron.core.optimizer.optimizer_config import ParamKey

    original = megatron_training.get_megatron_optimizer_config
    if getattr(original, "_uniss_joint_lr_groups", False):
        return

    def with_joint_groups(args):
        config, overrides = original(args)
        overrides = dict(overrides or {})
        for attribute, values in lr_group_values(float(args.lr), float(args.min_lr)).items():
            overrides[ParamKey(attr=attribute)] = values
        return config, overrides

    with_joint_groups._uniss_joint_lr_groups = True
    megatron_training.get_megatron_optimizer_config = with_joint_groups


def install_synchronized_task_sampler() -> None:
    """Keep joint/replay choice identical across all data-parallel ranks."""

    import megatron.training.datasets.data_samplers as data_samplers

    original = data_samplers.MegatronPretrainingRandomSampler
    if getattr(original, "_uniss_joint_synchronized_kinds", False):
        return

    def synchronized_or_default(dataset, *args, **kwargs):
        if getattr(dataset, "synchronize_sample_kind", False):
            return SynchronizedKindRandomSampler(dataset, *args, **kwargs)
        return original(dataset, *args, **kwargs)

    synchronized_or_default._uniss_joint_synchronized_kinds = True
    data_samplers.MegatronPretrainingRandomSampler = synchronized_or_default


def install_joint_collate() -> None:
    """Attach the experiment's variable-length collate without patching Megatron."""

    import megatron.training.datasets.data_samplers as data_samplers
    import megatron.training.training as megatron_training

    original = data_samplers.build_pretraining_data_loader
    if getattr(original, "_uniss_joint_collate", False):
        return

    def build_with_joint_collate(dataset, *args, **kwargs):
        collate = getattr(dataset, "collate_fn", None)
        if not callable(collate):
            return original(dataset, *args, **kwargs)

        original_loader = torch.utils.data.DataLoader

        def data_loader(*loader_args, **loader_kwargs):
            loader_kwargs.setdefault("collate_fn", collate)
            return original_loader(*loader_args, **loader_kwargs)

        # The upstream builder resolves DataLoader through torch.utils.data at
        # call time.  Scope the substitution to this synchronous construction
        # so no shared Megatron source file or unrelated experiment changes.
        torch.utils.data.DataLoader = data_loader
        try:
            return original(dataset, *args, **kwargs)
        finally:
            torch.utils.data.DataLoader = original_loader

    build_with_joint_collate._uniss_joint_collate = True
    data_samplers.build_pretraining_data_loader = build_with_joint_collate
    # training.py imports the function symbol at module import time.
    megatron_training.build_pretraining_data_loader = build_with_joint_collate


def add_joint_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Phase3 Whisper StreamSpeech joint")
    for name in (
        "train-manifest",
        "valid-manifest",
        "tokenizer-map-dir",
        "phase3-replay-packed",
        "phase3-replay-offsets",
        "whisper-model",
        "phase3-model",
    ):
        group.add_argument(f"--joint-{name}", required=True)
    group.add_argument("--joint-direction-index-dir", required=True)
    group.add_argument("--joint-balance-validation", action="store_true")
    group.add_argument("--joint-allow-partial-replay-index", action="store_true")
    group.add_argument("--joint-chunks", default="320,640,960,1280,offline")
    group.add_argument("--joint-right-context-ms", type=int, default=80)
    group.add_argument("--joint-replay-probability", type=float, default=0.20)
    group.add_argument("--joint-bicodec-ctc-weight", type=float, default=1.0)
    group.add_argument("--joint-ar-s2tt-weight", type=float, default=8.0)
    group.add_argument("--joint-asr-ctc-weight", type=float, default=4.0)
    group.add_argument("--joint-nar-s2tt-ctc-weight", type=float, default=4.0)
    group.add_argument("--joint-phase3-replay-weight", type=float, default=0.5)
    group.add_argument("--joint-unit-upsample-ratio", type=int, default=48)
    group.add_argument("--joint-disable-gradient-checkpointing", action="store_true")
    return parser


def validate_joint_args(args) -> None:
    if int(args.micro_batch_size) not in (1, 2):
        raise ValueError("formal joint training supports micro-batch-size 1 or 2")
    if int(args.global_batch_size) != 128:
        raise ValueError("the Phase3-preserving run requires global-batch-size 128")
    if int(args.seq_length) != 18_000:
        raise ValueError("exact Phase3 replay requires seq-length 18000")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("the compound model currently requires TP=PP=1")
    for name in (
        "joint_train_manifest",
        "joint_valid_manifest",
        "joint_tokenizer_map_dir",
        "joint_phase3_replay_packed",
        "joint_phase3_replay_offsets",
        "joint_whisper_model",
        "joint_phase3_model",
        "joint_direction_index_dir",
    ):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {getattr(args, name)}")
    fraction = Fraction(str(args.joint_replay_probability)).limit_denominator(100)
    if not 0 < fraction < 1:
        raise ValueError("joint replay probability must be in (0,1)")
    if fraction != Fraction(1, 5):
        raise ValueError("formal Phase3 preservation currently requires exactly 20% replay")
    MultiChunkConfig(
        chunk_ms=parse_chunks(args.joint_chunks),
        right_context_ms=int(args.joint_right_context_ms),
    )


def _target_count(values, index: int) -> int | None:
    if values is None or index >= len(values) or values[index] is None:
        return None
    return int(values[index])


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building Phase3 Whisper StreamSpeech joint datasets ...")
    joint_base = JointAudioDataset(
        args.joint_train_manifest, args.joint_tokenizer_map_dir
    )
    joint_train = DirectionBalancedJointDataset(
        joint_base, args.joint_direction_index_dir, "train"
    )
    replay = IndexedPhase3ReplayDataset(
        args.joint_phase3_replay_packed,
        args.joint_phase3_replay_offsets,
        seq_length=int(args.seq_length),
        require_complete=not args.joint_allow_partial_replay_index,
    )
    target_train = _target_count(train_val_test_num_samples, 0)
    data_parallel_size = int(args.data_parallel_size)
    global_microbatch_size = data_parallel_size * int(args.micro_batch_size)
    cycles = None
    if target_train is not None:
        cycles = math.ceil(target_train / (5 * data_parallel_size))
    train = DeterministicReplaySchedule(
        joint_train,
        replay,
        joint_slots=4,
        replay_slots=1,
        data_parallel_group_size=global_microbatch_size,
        cycles=cycles,
    )
    if target_train is not None and target_train > len(train):
        train = RepeatToLengthDataset(train, target_train)
    train.split = "train"
    train.collate_fn = collate_joint_or_replay

    valid_base = JointAudioDataset(
        args.joint_valid_manifest, args.joint_tokenizer_map_dir
    )
    if args.joint_balance_validation:
        valid = DirectionBalancedJointDataset(
            valid_base, args.joint_direction_index_dir, "valid"
        )
    else:
        valid = valid_base
    target_valid = _target_count(train_val_test_num_samples, 1)
    if target_valid is not None and target_valid > len(valid):
        valid = RepeatToLengthDataset(valid, target_valid)
    valid.split = "valid"
    valid.collate_fn = collate_joint_or_replay
    runtime.print_rank_0(
        f"> joint={len(joint_train)} replay={len(replay)} train_virtual={len(train)} valid={len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


class JointMegatronFactory:
    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.pg_collection = pg_collection
                chunks = MultiChunkConfig(
                    chunk_ms=parse_chunks(args.joint_chunks),
                    right_context_ms=int(args.joint_right_context_ms),
                )
                weights = JointLossWeights(
                    bicodec_ctc=float(args.joint_bicodec_ctc_weight),
                    ar_s2tt=float(args.joint_ar_s2tt_weight),
                    asr_ctc=float(args.joint_asr_ctc_weight),
                    nar_s2tt_ctc=float(args.joint_nar_s2tt_ctc_weight),
                    phase3_replay=float(args.joint_phase3_replay_weight),
                    replay_probability=float(args.joint_replay_probability),
                )
                self.joint = Phase3WhisperStreamSpeechJointModel.from_pretrained(
                    whisper_path=args.joint_whisper_model,
                    phase3_model=args.joint_phase3_model,
                    tokenizer_map_dir=args.joint_tokenizer_map_dir,
                    chunk_config=chunks,
                    loss_weights=weights,
                    upsample_ratio=int(args.joint_unit_upsample_ratio),
                    gradient_checkpointing=not args.joint_disable_gradient_checkpointing,
                )
                if torch.distributed.get_rank() == 0:
                    grouped = {}
                    for name, parameter in self.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        group_name = next(
                            (
                                key
                                for key in lr_group_values(float(args.lr), float(args.min_lr))
                                if getattr(parameter, key, False)
                            ),
                            "untagged",
                        )
                        grouped[group_name] = grouped.get(group_name, 0) + parameter.numel()
                    print(
                        json.dumps(
                            {
                                "joint_model": "Phase3WhisperStreamSpeechJointModel",
                                "lr_groups": grouped,
                                "effective_lr": lr_group_values(float(args.lr), float(args.min_lr)),
                                "chunks_ms": chunks.chunk_ms,
                                "loss_weights": weights.__dict__,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            def set_input_tensor(self, input_tensor):
                self.input_tensor = input_tensor

            def forward(self, batch, chunk_ms):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.joint(batch, chunk_ms=chunk_ms)
                chunk_value = -1 if chunk_ms is None else int(chunk_ms)
                return torch.cat((output, output.new_tensor([chunk_value])))

        return Composite()


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    del pre_process, post_process, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    return JointMegatronFactory.build(
        config or _transformer_config(args), args, pg_collection=pg_collection
    )


def _cuda_batch(batch: dict[str, object]) -> dict[str, object]:
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _prepare_joint_batch(batch: dict[str, object]) -> dict[str, object]:
    batch = _cuda_batch(batch)
    waveform_lengths = batch.get("waveform_lengths", batch.get("waveform_length"))
    direction_ids = batch.get("direction_ids", batch.get("direction_id"))
    if not isinstance(waveform_lengths, torch.Tensor) or not isinstance(
        direction_ids, torch.Tensor
    ):
        raise TypeError("joint waveform/direction lengths are malformed")
    result: dict[str, object] = {
        "sample_kind": batch["sample_kind"],
        "waveform": batch["waveform"],
        "waveform_lengths": waveform_lengths.reshape(-1),
        "direction_ids": direction_ids.reshape(-1),
        "phase3_record_json": batch["phase3_record_json"],
    }
    for name in (
        "source_ctc_ids",
        "target_ctc_ids",
        "source_qwen_ids",
        "target_qwen_ids",
        "source_glm",
        "target_bicodec",
    ):
        value = batch[name]
        if value.ndim == 1:
            value = value.unsqueeze(0)
        result[name] = value
        lengths = batch.get(f"{name}_lengths")
        if lengths is None:
            lengths = torch.full(
                (value.shape[0],),
                value.shape[1],
                dtype=torch.long,
                device=value.device,
            )
        if not isinstance(lengths, torch.Tensor):
            raise TypeError(f"{name} lengths are malformed")
        result[f"{name}_lengths"] = lengths.reshape(-1)
    result["bicodec_global"] = batch["bicodec_global"]
    return result


def loss_func(output_tensor):
    loss = output_tensor[0]
    metrics = {
        name: output_tensor[index + 1]
        for index, name in enumerate(METRIC_NAMES)
    }
    metrics["sampler/replay_fraction"] = 1.0 - metrics["sampler/joint_fraction"]
    metrics["chunk/ms_offline_minus_one"] = output_tensor[len(METRIC_NAMES) + 1]
    return loss, metrics


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = next(data_iterator)
    sample_kind = batch["sample_kind"]
    normalized_kind = sample_kind[0] if isinstance(sample_kind, (list, tuple)) else sample_kind
    if normalized_kind == "joint":
        prepared = _prepare_joint_batch(batch)
        consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        chunk_config = MultiChunkConfig(
            chunk_ms=parse_chunks(args.joint_chunks),
            right_context_ms=int(args.joint_right_context_ms),
        )
        chunk_ms = choose_chunk_ms(
            chunk_config, seed=int(args.seed), sample_index=consumed + rank
        )
    elif normalized_kind == "replay":
        prepared = _cuda_batch(batch)
        chunk_ms = None
    else:
        raise ValueError(f"unknown batch kind: {sample_kind}")
    output = model(prepared, chunk_ms)
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_joint_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_joint_args(args)
    install_megatron_lr_overrides()
    install_synchronized_task_sampler()
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
