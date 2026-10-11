"""Megatron entrypoint for Stage 2: the five-task mixture.

Same shape as the Stage 1 entrypoint -- parse_and_validate_args ->
gpt_config_from_args -> pretrain_cfg_container_from_args -> pretrain with
a custom model_provider -- and the same reasons for each of the three
patches it installs (dataset collate, the C++ index builder, a config for
the provider when the container has none).

What differs is the data. A step draws one task for every rank from a
seeded schedule, so the ranks do equal-cost work and agree on the
schedule without communicating, and each pool is consumed in a full
permutation rather than a sliding window.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import OrderedDict
from pathlib import Path

import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

METRIC_NAMES = ("loss", "text_ce", "code_ce")
_MODEL = None
_PROCESSOR = None


def add_stage2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS Omni Stage 2")
    group.add_argument("--omni-model", type=str, required=True)
    group.add_argument("--omni-talker-state", type=str, required=True)
    group.add_argument("--omni-manifests", type=str, required=True,
                       help="glob for the manifest jsonl files")
    group.add_argument("--omni-parquet-root", type=str, required=True)
    group.add_argument("--omni-audio-root", type=str, required=True)
    group.add_argument("--omni-tts-cache", type=str, required=True)
    group.add_argument("--omni-dev-parquet", type=str, nargs="+", required=True)
    group.add_argument("--omni-dev-subset", type=str, required=True)
    group.add_argument("--omni-audio-limit", type=int, default=0)
    group.add_argument("--omni-lora-rank", type=int, default=32)
    group.add_argument("--omni-lora-alpha", type=float, default=64.0)
    group.add_argument("--omni-code-weight", type=float, default=1.0)
    group.add_argument("--omni-seed", type=int, default=20261011)
    # Qwen2.5-Omni pads every mel to chunk_length=300 s. Our source audio is
    # median 5.1 s and at most 24.9 s, so 30 truncates nothing and takes the
    # feature extraction for a batch of eight from 477 ms to 43.
    group.add_argument("--omni-max-audio-seconds", type=float, default=30.0)
    return parser


def add_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return add_stage2_args(parser)


def disable_dataset_helpers() -> None:
    from megatron.core.datasets import utils as dataset_utils

    dataset_utils.compile_helpers = lambda: None
    try:
        from megatron.training import initialize as megatron_initialize

        megatron_initialize.compile_helpers = lambda: None
    except (ImportError, AttributeError):
        pass


def model_provider(pre_process=True, post_process=True, vp_stage=None,
                   config=None, pg_collection=None):
    global _MODEL, _PROCESSOR
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Stage 2 is restricted to TP=PP=1")

    from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
        load_audio_text_processor,
    )
    from experiments.uniss_omni_s2st_v1.training.stage2_model import Stage2Model

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = Stage2Model(
        config,
        model_path=args.omni_model,
        talker_state=args.omni_talker_state,
        lora_rank=args.omni_lora_rank,
        lora_alpha=args.omni_lora_alpha,
        code_weight=args.omni_code_weight,
        seed=args.omni_seed,
    )
    _MODEL = model
    _PROCESSOR = load_audio_text_processor(
        args.omni_model, max_audio_seconds=args.omni_max_audio_seconds
    )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(json.dumps({"model": "omni_stage2", **model.trainable_summary()},
                         sort_keys=True), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index]) for index, name in enumerate(METRIC_NAMES)
    )


def forward_step(data_iterator, model):
    from experiments.uniss_omni_s2st_v1.training.stage2_batch import build_batch

    batch = next(data_iterator)
    inputs = build_batch(
        batch, _PROCESSOR, _MODEL.talker, _MODEL.layout,
        device=torch.cuda.current_device(),
    )
    return model(**inputs), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from experiments.uniss_omni_s2st_v1.training.stage2_data import (
        ParquetCodeIndex,
        Stage2Corpus,
        read_audio_rows,
    )
    from experiments.uniss_omni_s2st_v1.training.stage2_dataset import (
        Stage2Dataset,
        task_schedule,
    )
    from experiments.uniss_omni_s2st_v1.training.tts_data import CachedTtsCorpus

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()

    manifests = sorted(glob.glob(args.omni_manifests))
    if not manifests:
        raise SystemExit(f"no manifests matched {args.omni_manifests}")
    audio_rows = read_audio_rows(
        manifests, audio_root=args.omni_audio_root, limit=args.omni_audio_limit
    )
    corpus = Stage2Corpus(
        audio_rows,
        ParquetCodeIndex(args.omni_parquet_root, {r.parquet for r in audio_rows}),
        CachedTtsCorpus(args.omni_tts_cache),
        audio_root=args.omni_audio_root,
    )
    # One extra step of schedule so the final partial batch has a task.
    schedule = task_schedule(int(args.train_iters) + 2, seed=args.omni_seed)
    train = Stage2Dataset(
        corpus, schedule, global_batch=int(args.global_batch_size), seed=args.omni_seed
    )
    runtime.print_rank_0(
        f"stage 2: {len(audio_rows):,} audio rows, {len(corpus.text_corpus):,} cached"
        f" TTS rows; coverage audio {train.epochs['audio']:.2f} epochs,"
        f" text {train.epochs['text']:.2f}"
    )
    # Validation is a short S2ST-only schedule over the dev split, so the
    # curve measures the task the gates care about rather than the mixture.
    dev = _dev_dataset(args, corpus, runtime)
    return train, dev, None


def _dev_dataset(args, corpus, runtime):
    from experiments.uniss_omni_s2st_v1.training.stage2_dataset import Stage2Dataset

    affordable = max(1, len(corpus.audio_rows) // max(1, int(args.global_batch_size)))
    if int(args.eval_iters) > affordable:
        runtime.print_rank_0(
            f"stage 2: eval_iters {args.eval_iters} exceeds the pool;"
            f" using {affordable}"
        )
        args.eval_iters = affordable
    evaluations = int(args.train_iters) // max(1, int(args.eval_interval)) + 4
    schedule = ["s2st"] * (int(args.eval_iters) * evaluations + 4)
    return Stage2Dataset(
        corpus, schedule, global_batch=int(args.global_batch_size),
        seed=args.omni_seed + 1,
    )


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import (
        install_omni_collate,
    )

    args = runtime.parse_and_validate_args(
        extra_args_provider=add_extra_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    install_omni_collate()
    disable_dataset_helpers()
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
