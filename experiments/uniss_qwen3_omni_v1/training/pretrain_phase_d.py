"""Megatron entry point for Phase D.

Stage 1's trainer with three things changed: the Thinker is Qwen3's and
stays frozen, the Talker starts from Stage 1's checkpoint instead of the
UniSS transfer, and the ordering is a true permutation with the sort moved
inside the global batch. Everything else -- the collate, the code sequence
builder, the dev-split truncation that keeps every evaluation on identical
utterances -- is imported from there rather than copied.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict

import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import (
    disable_dataset_helpers,
)

METRIC_NAMES = ("code_ce", "supervised_tokens")
_MODEL = None


def add_phase_d_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Phase D: Talker onto Qwen3")
    group.add_argument("--qwen3-path", type=str,
                       default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    group.add_argument("--qwen25-path", type=str,
                       default="pretrained_models/Qwen2.5-Omni-3B")
    group.add_argument("--talker-checkpoint", type=str, required=True)
    group.add_argument("--omni-cache", type=str, required=True)
    group.add_argument("--omni-dev-cache", type=str, default=None,
                       help="defaults to a tail slice of --omni-cache")
    group.add_argument("--omni-dev-rows", type=int, default=4096)
    group.add_argument("--omni-max-text-tokens", type=int, default=256)
    group.add_argument("--omni-head-scale", type=float, default=0.1)
    group.add_argument("--omni-seed", type=int, default=20261012)
    group.add_argument("--omni-no-recompute", action="store_true")
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage=None,
                   config=None, pg_collection=None):
    global _MODEL
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Phase D is restricted to TP=PP=1")

    from experiments.uniss_qwen3_omni_v1.training.phase_d_model import PhaseDModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = PhaseDModel(
        config,
        qwen3_path=args.qwen3_path,
        qwen25_path=args.qwen25_path,
        talker_checkpoint=args.talker_checkpoint,
        head_scale=args.omni_head_scale,
        seed=args.omni_seed,
        recompute=not args.omni_no_recompute,
    )
    _MODEL = model
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(json.dumps({"model": "phase_d", **model.trainable_report()},
                         sort_keys=True), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index]) for index, name in enumerate(METRIC_NAMES))


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
        bicodec_global=batch["bicodec_global"],
    )
    return output, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from transformers import Qwen3OmniMoeProcessor

    from experiments.uniss_omni_s2st_v1.training.tts_data import CachedTtsCorpus
    from experiments.uniss_qwen3_omni_v1.training.phase_d_dataset import PhaseDDataset

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    model = _MODEL
    if model is None:
        raise RuntimeError("datasets requested before the model was built")

    corpus = CachedTtsCorpus(args.omni_cache)
    tokenizer = Qwen3OmniMoeProcessor.from_pretrained(args.qwen3_path).tokenizer
    gbs = int(args.global_batch_size)

    # Dev comes from the tail of the same cache, held out by index so it
    # cannot also be drawn for training.
    dev_rows = min(int(args.omni_dev_rows), len(corpus) // 100)
    held_out = len(corpus) - dev_rows

    class _Slice:
        def __init__(self, source, start, stop):
            self.source, self.start, self.stop = source, start, stop
        def __len__(self):
            return self.stop - self.start
        def __getitem__(self, i):
            return self.source[self.start + int(i)]
        def code_lengths(self):
            return self.source.code_lengths()[self.start : self.stop]
        def text_lengths(self):
            return self.source.text_lengths()[self.start : self.stop]

    train_source = _Slice(corpus, 0, held_out)
    dev_source = _Slice(corpus, held_out, len(corpus))
    runtime.print_rank_0(
        f"phase D: {len(train_source):,} train rows, {len(dev_source):,} dev rows")

    affordable = max(1, len(dev_source) // gbs)
    if int(args.eval_iters) > affordable:
        runtime.print_rank_0(
            f"phase D: eval_iters {args.eval_iters} exceeds what {len(dev_source)}"
            f" dev rows supply at global batch {gbs}; using {affordable}")
        args.eval_iters = affordable

    def build(rows, seed, **kwargs):
        return PhaseDDataset(
            rows,
            global_batch=gbs,
            bos_id=model.talker.codec_bos_token,
            pad_id=model.talker.codec_pad_token,
            eos_id=model.layout.special_ids["eos"],
            text_pad_id=tokenizer.pad_token_id or 0,
            seed=seed,
            max_text_tokens=int(args.omni_max_text_tokens),
            **kwargs,
        )

    needed = int(args.train_iters) * gbs
    epochs = -(-needed // max(1, len(train_source))) + 1
    runtime.print_rank_0(
        f"phase D: {needed:,} samples wanted from {len(train_source):,} rows"
        f" -> {epochs} epochs")
    train_dataset = build(train_source, args.omni_seed, epochs=epochs)
    per_eval = int(args.eval_iters) * gbs
    evaluations = int(args.train_iters) // max(1, int(args.eval_interval)) + 4
    dev_dataset = build(dev_source, args.omni_seed + 1,
                        truncate_to=per_eval, repeat=evaluations)
    return train_dataset, dev_dataset, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import (
        install_omni_collate,
    )

    args = runtime.parse_and_validate_args(
        extra_args_provider=add_phase_d_args,
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
        model_provider_func=model_provider,
    )
    print("PHASE_D_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()
