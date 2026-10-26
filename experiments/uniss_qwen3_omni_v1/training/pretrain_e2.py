"""Megatron entry point for E2: Qwen3-Omni's Thinker with UniSS as its Talker.

Phase E's trainer shape -- the frozen Thinker inside the model, only the
trainable part checkpointed, one batch builder in forward_step -- with
E2's model, E2's rows and E2's batch. Data is UniST's S2ST audio pool in
Stage2Dataset's order: a full permutation, reshuffled each pass when
UNISS_RESHUFFLE_EACH_EPOCH=1.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict

import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers

# loss = code_ce + kl_weight * kl. The metrics are clones: Megatron divides the
# returned loss by the micro-batch count in place, and runs before this logged
# code_ce at half its value because the metric shared that tensor.
METRIC_NAMES = ("loss", "code_ce", "kl", "supervised_tokens")
_MODEL = None
_PROCESSOR = None
_INSTRUCTION = None
_RNG = None


def add_e2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="E2: Qwen3-Omni Thinker + UniSS Talker")
    g.add_argument("--qwen3-path", type=str, default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    g.add_argument("--uniss-path", type=str, default="pretrained_models/UniSS_full")
    g.add_argument("--omni-audio-index", type=str, required=True)
    g.add_argument("--omni-parquet-root", type=str, required=True)
    g.add_argument("--omni-audio-root", type=str, required=True)
    g.add_argument("--omni-max-audio-seconds", type=float, default=30.0)
    g.add_argument("--omni-seed", type=int, default=20261020)
    g.add_argument("--e2-duration-dropout", type=float, default=0.2)
    g.add_argument("--omni-no-recompute", action="store_true")
    g.add_argument("--e2-no-added", action="store_true",
                   help="UniSS alone: no Thinker, no added inputs, no audio decoded when the index carries durations")
    g.add_argument("--e2-kl-weight", type=float, default=0.0, help="KL to a frozen UniSS over code positions")
    g.add_argument("--e2-kl-anchor", type=str, default="", help="the anchor's weights; default --uniss-path")
    g.add_argument("--e2-glm-source", choices=("released", "teacher"), default="released",
                   help="teacher: the index's source GLM re-tokenised from local audio, as inference sees it")
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    global _MODEL, _PROCESSOR, _INSTRUCTION, _RNG
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("E2 is restricted to TP=PP=1")
    from transformers import Qwen3OmniMoeProcessor

    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    from experiments.uniss_qwen3_omni_v1.training.e2_model import E2Model

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = E2Model(config, qwen3_path=args.qwen3_path, uniss_path=args.uniss_path,
                    recompute=not args.omni_no_recompute, added_inputs=not args.e2_no_added,
                    kl_anchor=args.e2_kl_anchor or args.uniss_path, kl_weight=args.e2_kl_weight)
    _MODEL = model
    _PROCESSOR = Qwen3OmniMoeProcessor.from_pretrained(args.qwen3_path)
    fe = getattr(_PROCESSOR, "feature_extractor", None)
    if fe is not None and args.omni_max_audio_seconds and hasattr(fe, "chunk_length"):
        fe.chunk_length = float(args.omni_max_audio_seconds)
    _INSTRUCTION = inference_instruction()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _RNG = random.Random(args.omni_seed * 1000 + rank)   # duration dropout, per rank
    if rank == 0:
        print(json.dumps({"model": "e2", **model.trainable_report()}, sort_keys=True), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import build_e2_batch

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = next(data_iterator)
    inputs = build_e2_batch(batch, _PROCESSOR, device=torch.cuda.current_device(), instruction=_INSTRUCTION,
                            duration_dropout=args.e2_duration_dropout, rng=_RNG, light=args.e2_no_added)
    return model(**inputs), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from experiments.uniss_omni_s2st_v1.training.stage2_data import ParquetCodeIndex, Stage2Corpus
    from experiments.uniss_qwen3_omni_v1.training.audio_index import AudioIndex
    from experiments.uniss_qwen3_omni_v1.training.e2_data import E2Dataset

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    audio_rows = AudioIndex(args.omni_audio_index)
    codes = ParquetCodeIndex(args.omni_parquet_root, audio_rows.parquet_names(), with_glm=True)
    corpus = Stage2Corpus(audio_rows, codes, [], audio_root=args.omni_audio_root)
    train = E2Dataset(corpus, ["s2st"] * int(args.train_iters), global_batch=int(args.global_batch_size),
                      seed=args.omni_seed)
    # The Thinker's mel and prompt are built in the dataloader workers from here.
    train.processor, train.instruction = _PROCESSOR, _INSTRUCTION
    train.light, train.glm_source = bool(args.e2_no_added), args.e2_glm_source
    runtime.print_rank_0(f"E2: {len(audio_rows):,} audio rows; {train.epochs['audio']:.3f} epochs over"
                         f" {int(args.train_iters):,} steps of {int(args.global_batch_size)}")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def keep_lr_groups_apart() -> None:
    """On a resume Megatron matches optimizer groups by (wd_mult, lr_mult,
    is_expert_parallel, is_decoupled_lr), and --decoupled-lr sets only
    max_lr/min_lr on its groups, so UniSS's groups and the added inputs'
    share a key and the later one's hyperparameters go to both. e2_main3,
    resumed at step 100, trained UniSS's matrices at 5e-4 and the gate at
    1e-5; by step 500 UniSS had moved 12% from where it started (the
    uninterrupted run, 0.6% in 1,000 steps) and ASR-BLEU was down 2.2.
    The rates themselves join the key, for this entry point only, and each
    group's rates are printed after every load."""
    import megatron.core.optimizer as package
    import megatron.core.optimizer.distrib_optimizer as distrib
    import megatron.core.optimizer.optimizer as base
    import megatron.training.training as training

    keys = (*base.param_group_identifier_keys, "max_lr", "min_lr")
    for module in (base, distrib, package, training):
        module.param_group_identifier_keys = keys

    load = distrib.DistributedOptimizer.load_state_dict

    def load_state_dict(self, state_dict):
        load(self, state_dict)
        if torch.distributed.get_rank() == 0:
            for g in self.optimizer.param_groups:
                print(json.dumps({"resumed_group": {"max_lr": g["max_lr"], "min_lr": g["min_lr"],
                                                    "wd_mult": g["wd_mult"], "step": g.get("step"),
                                                    "shard_m": round(sum(p.numel() for p in g["params"]) / 1e6, 2)}}),
                      flush=True)

    distrib.DistributedOptimizer.load_state_dict = load_state_dict


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import install_omni_collate

    args = runtime.parse_and_validate_args(extra_args_provider=add_e2_args,
                                           args_defaults={"tokenizer_type": "NullTokenizer"})
    keep_lr_groups_apart()
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("E2_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()
