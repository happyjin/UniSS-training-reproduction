"""Megatron entry point for T1: UniSS as a streaming Talker (training/si_model.py).

E2's trainer shape -- TP=PP=1, the model a HuggingFaceModule, only the
trainable talker checkpointed, the batch built in forward_step -- with T1's
model, T1's rows (training/si_data.py) and T1's batch (training/si_batch.py).
Rows are a real permutation per pass, the pool drawn per step.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict

import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers
from experiments.uniss_qwen3_omni_v1.training.pretrain_e2 import keep_lr_groups_apart
from experiments.uniss_qwen3_omni_v1.training.si_model import METRIC_NAMES

_TOKENIZER = None


def add_si_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="T1: streaming Talker")
    g.add_argument("--qwen3-path", type=str, default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct",
                   help="for the text tokenizer only")
    g.add_argument("--uniss-path", type=str, required=True, help="the talker's start (R2's export)")
    g.add_argument("--teacher-path", type=str, default="", help="the offline teacher; default --uniss-path")
    g.add_argument("--anchor-path", type=str, default="pretrained_models/UniSS_full")
    g.add_argument("--si-index", type=str, required=True)
    g.add_argument("--omni-audio-index", type=str, required=True)
    g.add_argument("--omni-parquet-root", type=str, required=True)
    g.add_argument("--omni-audio-root", type=str, required=True)
    g.add_argument("--omni-seed", type=int, default=20261021)
    g.add_argument("--omni-no-recompute", action="store_true")
    g.add_argument("--si-share", type=float, default=0.7, help="share of steps drawn from the si pool")
    g.add_argument("--si-delta-ms", type=int, default=200)
    g.add_argument("--si-holdback", type=int, default=12, help="GLM tokens behind the audio a block may read")
    g.add_argument("--si-m-max", type=int, default=6, help="read windows of 1..m seconds")
    g.add_argument("--si-read-cap", type=float, default=0.25)
    g.add_argument("--si-w-future", type=float, default=0.5)
    g.add_argument("--si-w-anchor", type=float, default=0.5)
    g.add_argument("--si-anchor-cmn-mult", type=float, default=2.0)
    g.add_argument("--si-init-wait-from-end", action="store_true",
                   help="start <|wait_read|>'s (tied) row as a copy of the end token's")
    g.add_argument("--si-close-weight", type=float, default=1.0, help="CE weight of the closing positions")
    g.add_argument("--si-word-share", type=float, default=0.0,
                   help="share of si draws cut as word-by-word release (needs an index with word times)")
    g.add_argument("--si-word-windows", type=str, default="1,2,3", help="read windows (s) for word blocks")
    g.add_argument("--si-word-min", type=str, default="1,2,3", help="words a word block waits for")
    g.add_argument("--si-block-speed", action="store_true", help="a speed token per block (source since last write / own codes)")
    g.add_argument("--si-lookahead-max", type=int, default=0,
                   help="show each non-final block up to this many next words (2x Chinese characters) it does not speak (0: off)")
    g.add_argument("--si-lookahead-noise", type=float, default=0.3, help="share of lookaheads drawn from elsewhere (a wrong tail)")
    g.add_argument("--si-lookahead-none", type=float, default=0.2, help="share of blocks shown no lookahead")
    g.add_argument("--si-init-look", action="store_true", help="start the lookahead brackets' rows as the content brackets'")
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    global _TOKENIZER
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("T1 is restricted to TP=PP=1")
    from transformers import AutoTokenizer

    from experiments.uniss_qwen3_omni_v1.training.si_model import SIModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = SIModel(config, uniss_path=args.uniss_path, teacher_path=args.teacher_path or args.uniss_path,
                    anchor_path=args.anchor_path, w_future=args.si_w_future, w_anchor=args.si_w_anchor,
                    recompute=not args.omni_no_recompute, init_wait_from_end=args.si_init_wait_from_end,
                    close_weight=args.si_close_weight, init_look_from_content=args.si_init_look)
    # the Thinker's text tokenizer: the translation ids UniSS reads are Qwen3's (the vocabularies agree)
    _TOKENIZER = AutoTokenizer.from_pretrained(args.qwen3_path)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "si_t1", **model.trainable_report()}, sort_keys=True), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    from experiments.uniss_qwen3_omni_v1.training.si_batch import build_si_batch

    args = load_megatron_runtime().megatron_gpt.get_args()
    batch = next(data_iterator)
    inputs = build_si_batch(batch, _TOKENIZER, device=torch.cuda.current_device(),
                            anchor_cmn_mult=args.si_anchor_cmn_mult, block_speed=args.si_block_speed,
                            lookahead_max=args.si_lookahead_max, lookahead_noise=args.si_lookahead_noise,
                            lookahead_none=args.si_lookahead_none)
    return model(**inputs), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from experiments.uniss_omni_s2st_v1.training.stage2_data import ParquetCodeIndex, Stage2Corpus
    from experiments.uniss_qwen3_omni_v1.training.audio_index import AudioIndex
    from experiments.uniss_qwen3_omni_v1.training.si_data import SIDataset

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    rows = AudioIndex(args.omni_audio_index)
    if rows.source_ms is None or rows._teacher_glm is None:
        raise SystemExit(f"{args.omni_audio_index} carries no source_ms / teacher GLM; T1 needs the clean index")
    codes = ParquetCodeIndex(args.omni_parquet_root, rows.parquet_names())
    corpus = Stage2Corpus(rows, codes, [], audio_root=args.omni_audio_root)
    train = SIDataset(corpus, args.si_index, int(args.train_iters), global_batch=int(args.global_batch_size),
                      seed=args.omni_seed, si_share=args.si_share, delta_ms=args.si_delta_ms,
                      holdback=args.si_holdback, m_max=args.si_m_max, read_cap=args.si_read_cap,
                      word_share=args.si_word_share,
                      word_windows=tuple(float(x) for x in args.si_word_windows.split(",")),   # seconds, 0.5 allowed
                      word_min=tuple(int(x) for x in args.si_word_min.split(",")))
    runtime.print_rank_0(f"T1: {len(train.si):,} si rows, {len(rows):,} offline rows; "
                         f"{train.schedule.count('si'):,} si steps and {train.schedule.count('offline'):,} offline "
                         f"of {int(args.global_batch_size)}; passes {json.dumps(train.epochs)}")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import install_omni_collate

    args = runtime.parse_and_validate_args(extra_args_provider=add_si_args,
                                           args_defaults={"tokenizer_type": "NullTokenizer"})
    keep_lr_groups_apart()
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("SI_T1_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()
