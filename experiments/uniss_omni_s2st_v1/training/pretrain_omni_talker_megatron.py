"""Megatron entrypoint for the Stage 1 Talker warmup.

Follows the shape of the repository's other native Megatron entrypoints --
``parse_and_validate_args`` -> ``gpt_config_from_args`` ->
``pretrain_cfg_container_from_args`` -> ``pretrain`` with a custom
``model_provider`` -- but the model it provides is not a Megatron GPT. It
is Qwen2.5-Omni's Thinker and Talker wrapped in Megatron's own
``HuggingFaceModule``, because Megatron's built-in Qwen bridge only knows
``Qwen2ForCausalLM`` and Omni is three models in a trenchcoat.

``full_config.model = None`` for the same reason the phase-3 entrypoints
set it: the config container would otherwise try to build a GPT.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

METRIC_NAMES = ("code_ce", "supervised_tokens")


def disable_dataset_helpers() -> None:
    """Skip Megatron's C++ index builder, which this run has no use for.

    ``initialize_megatron`` compiles ``helpers.cpp`` on rank 0 before
    anything else, and the build needs pybind11 headers this box does not
    have. The extension only serves Megatron's own indexed datasets; ours
    is a plain torch Dataset, so the compile is pure cost and a hard
    failure. The phase-3 prefix-streaming trainer no-ops it the same way.
    """
    from megatron.core.datasets import utils as dataset_utils

    dataset_utils.compile_helpers = lambda: None
    try:
        from megatron.training import initialize as megatron_initialize

        megatron_initialize.compile_helpers = lambda: None
    except (ImportError, AttributeError):
        pass


def add_omni_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS Omni Talker warmup")
    group.add_argument("--omni-model", type=str, required=True)
    group.add_argument("--omni-parquet-root", type=str, required=True)
    group.add_argument("--omni-shards", type=str, nargs="+", default=[])
    # A cache built by build_tts_cache: the parquets cost half an hour per
    # rank to read and tokenise, and every rank pays it separately.
    group.add_argument("--omni-cache", type=str, default=None)
    group.add_argument("--omni-dev-parquet", type=str, nargs="+", required=True)
    group.add_argument("--omni-dev-subset", type=str, required=True)
    group.add_argument("--omni-rows-per-shard", type=int, default=0)
    group.add_argument("--omni-max-codes", type=int, default=600)
    group.add_argument("--omni-head-scale", type=float, default=0.1)
    group.add_argument("--omni-embed-scale", type=float, default=1.0)
    group.add_argument("--omni-seed", type=int, default=20261009)
    # UniSS's own TTS prompt carries the 32 bicodec_global tokens before the
    # semantic codes; off by default so the two layouts can be compared.
    group.add_argument("--omni-global-prefix", action="store_true")
    return parser


def add_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return add_omni_args(parser)


_MODEL = None


def model_provider(pre_process=True, post_process=True, vp_stage=None,
                   config=None, pg_collection=None):
    global _MODEL
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("the Omni Talker warmup is restricted to TP=PP=1")

    from experiments.uniss_omni_s2st_v1.training.omni_megatron_model import (
        OmniTalkerWarmupModel,
    )

    if config is None:
        # With ``full_config.model = None`` Megatron does not hand a config
        # to the provider, but MegatronModule needs one -- the optimiser and
        # the DDP wrapper both reach for ``model.config``.
        config = runtime.gpt_config_from_args(args)

    model = OmniTalkerWarmupModel(
        config,
        model_path=args.omni_model,
        head_scale=args.omni_head_scale,
        embed_scale=args.omni_embed_scale,
        seed=args.omni_seed,
        global_prefix=bool(args.omni_global_prefix),
    )
    _MODEL = model
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            json.dumps(
                {
                    "model": "omni_talker_warmup_megatron",
                    "trainable_parameters_m": round(trainable / 1e6, 1),
                    "frozen_parameters_m": round(frozen / 1e6, 1),
                    "head_scale": args.omni_head_scale,
                    "embed_scale": args.omni_embed_scale,
                    "global_prefix": bool(args.omni_global_prefix),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index]) for index, name in enumerate(METRIC_NAMES)
    )


def forward_step(data_iterator, model):
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import (
        unwrap_to_device,
    )

    batch = unwrap_to_device(next(data_iterator), torch.cuda.current_device())
    output = model(
        thinker_input_ids=batch["thinker_input_ids"],
        thinker_attention_mask=batch["thinker_attention_mask"],
        reply_ids=batch["reply_ids"],
        reply_mask=batch["reply_mask"],
        reply_start=batch["reply_start"],
        codec_input_ids=batch["codec_input_ids"],
        codec_labels=batch["codec_labels"],
        codec_mask=batch["codec_mask"],
        bicodec_global=batch.get("bicodec_global"),
    )
    return output, loss_func


def _rows(args, *, split: str):
    from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
        load_audio_text_processor,
    )
    from experiments.uniss_omni_s2st_v1.training.stage1_talker_warmup import (
        encode_prompts,
        load_dev,
    )
    from experiments.uniss_omni_s2st_v1.training.tts_data import (
        CachedTtsCorpus,
        read_rows,
    )

    processor = load_audio_text_processor(args.omni_model)
    if split == "train":
        if args.omni_cache:
            # Already tokenised and memory-mapped; nothing to encode.
            return processor, CachedTtsCorpus(args.omni_cache)
        if not args.omni_shards:
            raise ValueError("pass --omni-cache or --omni-shards")
        paths = [Path(args.omni_parquet_root) / name for name in args.omni_shards]
        rows = read_rows(
            paths,
            limit_per_file=args.omni_rows_per_shard,
            max_codes=args.omni_max_codes,
        )
    else:
        rows = load_dev(args.omni_dev_parquet, args.omni_dev_subset, args.omni_max_codes)
    return processor, encode_prompts(processor, rows)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import (
        OmniTtsDataset,
    )

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    model = _MODEL
    if model is None:
        raise RuntimeError("datasets requested before the model was built")

    processor, train_rows = _rows(args, split="train")
    _, dev_rows = _rows(args, split="dev")
    runtime.print_rank_0(
        f"omni warmup: {len(train_rows)} train rows, {len(dev_rows)} dev rows"
    )

    # The dev split is a fixed 1,994 utterances, so the number of eval
    # iterations it can supply depends on the batch size. Megatron does not
    # check: asking for more raises StopIteration from inside the eval loop,
    # a hundred iterations into the run. Clamping here means changing
    # --micro-batch-size cannot reintroduce that.
    affordable = max(1, len(dev_rows) // max(1, int(args.global_batch_size)))
    if int(args.eval_iters) > affordable:
        runtime.print_rank_0(
            f"omni warmup: eval_iters {args.eval_iters} exceeds what"
            f" {len(dev_rows)} dev rows supply at global batch"
            f" {args.global_batch_size}; using {affordable}"
        )
        args.eval_iters = affordable

    def build(rows, seed, **kwargs):
        return OmniTtsDataset(
            rows,
            bos_id=model.talker.codec_bos_token,
            pad_id=model.talker.codec_pad_token,
            eos_id=model.layout.special_ids["eos"],
            text_pad_id=processor.tokenizer.pad_token_id or 0,
            micro_batch=int(args.micro_batch_size),
            seed=seed,
            **kwargs,
        )

    # One evaluation's worth of rows, repeated once per evaluation the run
    # will perform (plus a margin for the one at the end and any extra
    # Megatron decides to do). Truncating rather than cycling over the whole
    # dev split keeps every evaluation on identical utterances.
    per_eval = int(args.eval_iters) * int(args.global_batch_size)
    evaluations = int(args.train_iters) // max(1, int(args.eval_interval)) + 4
    # The training iterator is single-pass too. One epoch of 816k rows is
    # iteration 2,128 at global batch 384, so a 5,000-iteration run has to
    # be handed its epochs up front or it stops a little over a third of
    # the way in.
    needed = int(args.train_iters) * int(args.global_batch_size)
    epochs = -(-needed // max(1, len(train_rows))) + 1
    runtime.print_rank_0(
        f"omni warmup: {needed} train samples wanted from {len(train_rows)}"
        f" rows -> {epochs} epochs"
    )
    return (
        build(train_rows, args.omni_seed, epochs=epochs),
        build(dev_rows, args.omni_seed + 1, truncate_to=per_eval, repeat=evaluations),
        None,
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
