"""Megatron entrypoint for weighted, packed Simul-UniSS SFT samples."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

from training import constants_uniss as c
from training.megatron_uniss_dataset import RepeatToLengthDataset
from training.pretrain_uniss_megatron import (
    Split,
    _has_eval_enabled,
    _target_sample_count,
    load_megatron_runtime,
    maybe_wrap_pretrain_after_parse,
)
from training.simul_uniss import PACKED_SCHEMA_VERSION
from training.simul_uniss.dataset import SimulPackedJsonlDataset


_PROGRAM_START_TIME = time.time()


def add_simul_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Simul-UniSS weighted packed data")
    group.add_argument("--simul-packed-train", required=True)
    group.add_argument("--simul-packed-valid", default=None)
    group.add_argument("--simul-packed-test", default=None)
    group.add_argument("--simul-schema-version", default=PACKED_SCHEMA_VERSION)
    return parser


def add_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    megatron_gpt = load_megatron_runtime().megatron_gpt
    if megatron_gpt.has_nvidia_modelopt:
        parser = megatron_gpt.add_modelopt_args(parser)
    return add_simul_args(parser)


def validate_simul_args(args: SimpleNamespace) -> None:
    if not getattr(args, "sft", False):
        raise ValueError("Simul-UniSS packed training requires --sft")
    if getattr(args, "create_attention_mask_in_dataloader", False):
        raise ValueError("omit --create-attention-mask-in-dataloader for packed Simul-UniSS")
    if int(getattr(args, "context_parallel_size", 1)) != 1:
        raise ValueError("Simul-UniSS currently supports context parallel size 1")
    if _has_eval_enabled(args) and not getattr(args, "simul_packed_valid", None):
        raise ValueError("pass --simul-packed-valid or disable evaluation")
    if getattr(args, "simul_schema_version", None) != PACKED_SCHEMA_VERSION:
        raise ValueError(f"expected --simul-schema-version {PACKED_SCHEMA_VERSION}")
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) != c.VOCAB_SIZE:
        raise ValueError(f"Simul-UniSS requires vocab size {c.VOCAB_SIZE}")
    if bool(getattr(args, "add_bias_linear", True)):
        raise ValueError("Qwen2.5 checkpoint requires --disable-bias-linear")
    if not bool(getattr(args, "add_qkv_bias", False)):
        raise ValueError("Qwen2.5 checkpoint requires --add-qkv-bias")


def _dataset(path: str | None, seq_length: int, split: Split, target_samples: int | None):
    if path is None:
        return None
    dataset = SimulPackedJsonlDataset(path, seq_length)
    dataset.split = split
    if target_samples is not None and target_samples > len(dataset):
        dataset = RepeatToLengthDataset(dataset, target_samples)
        dataset.split = split
    return dataset


def build_simul_datasets(args: SimpleNamespace, train_val_test_num_samples=None):
    seq_length = int(args.seq_length)
    return (
        _dataset(
            args.simul_packed_train,
            seq_length,
            Split.train,
            _target_sample_count(train_val_test_num_samples, 0),
        ),
        _dataset(
            args.simul_packed_valid,
            seq_length,
            Split.valid,
            _target_sample_count(train_val_test_num_samples, 1),
        ),
        _dataset(
            args.simul_packed_test,
            seq_length,
            Split.test,
            _target_sample_count(train_val_test_num_samples, 2),
        ),
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building weighted Simul-UniSS datasets ...")
    datasets = build_simul_datasets(args, train_val_test_num_samples)
    runtime.print_rank_0("> finished building weighted Simul-UniSS datasets ...")
    return datasets


def main() -> None:
    runtime = load_megatron_runtime()
    megatron_gpt = runtime.megatron_gpt
    runtime.set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=time.time())
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_extra_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_simul_args(args)
    wrapped_pretrain, store = maybe_wrap_pretrain_after_parse(args, runtime=runtime)
    if megatron_gpt.has_nvidia_modelopt:
        megatron_gpt.maybe_enable_modelopt(args)
    if megatron_gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = runtime.gpt_config_from_args(args, model_config_cls=megatron_gpt.ModelOptModelConfig)
    else:
        model_cfg = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_cfg)
    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        megatron_gpt.forward_step,
        store=store,
        get_embedding_ranks=megatron_gpt.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
