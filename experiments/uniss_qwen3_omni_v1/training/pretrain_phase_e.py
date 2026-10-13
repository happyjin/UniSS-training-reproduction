"""Megatron entry point for Phase E.

Stage 2's trainer with the Thinker frozen, the objective cut to the code
cross-entropy, and the mixture reduced to the two tasks that supervise
codes. Everything else -- the corpus, the batch builder, the schedule
that keeps every rank on one task per step -- is imported from stage 2
rather than copied.
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
_PROCESSOR = None


def add_phase_e_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="Phase E: Talker on Qwen3 S2ST states")
    g.add_argument("--qwen3-path", type=str,
                   default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    g.add_argument("--qwen25-path", type=str, default="pretrained_models/Qwen2.5-Omni-3B")
    g.add_argument("--phase-d-checkpoint", type=str, required=True)
    g.add_argument("--omni-manifests", type=str, nargs="+", default=[])
    # Built once by build_audio_index. Scanning the manifests instead costs
    # every rank a parse of 19,186,799 JSON lines -- measured at over
    # sixteen minutes with the GPUs idle, paid again on every restart.
    g.add_argument("--omni-audio-index", type=str, default=None)
    g.add_argument("--omni-parquet-root", type=str, required=True)
    g.add_argument("--omni-audio-root", type=str, required=True)
    g.add_argument("--omni-tts-cache", type=str, required=True)
    g.add_argument("--omni-dev-parquet", type=str, nargs="+", required=True)
    g.add_argument("--omni-dev-subset", type=str, required=True)
    g.add_argument("--omni-audio-limit", type=int, default=0)
    g.add_argument("--omni-max-audio-seconds", type=float, default=30.0)
    g.add_argument("--omni-max-text-tokens", type=int, default=256)
    g.add_argument("--omni-head-scale", type=float, default=0.1)
    g.add_argument("--omni-seed", type=int, default=20261013)
    g.add_argument("--omni-no-recompute", action="store_true")
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage=None,
                   config=None, pg_collection=None):
    global _MODEL, _PROCESSOR
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Phase E is restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.phase_e_model import PhaseEModel
    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = PhaseEModel(
        config, qwen3_path=args.qwen3_path, qwen25_path=args.qwen25_path,
        phase_d_checkpoint=args.phase_d_checkpoint,
        head_scale=args.omni_head_scale, seed=args.omni_seed,
        recompute=not args.omni_no_recompute)
    _MODEL = model
    from transformers import Qwen3OmniMoeProcessor
    # Qwen3's processor, not Qwen2.5's: build_batch templates the
    # conversation and extracts the mel, and both belong to the Thinker
    # that will read them.
    _PROCESSOR = Qwen3OmniMoeProcessor.from_pretrained(args.qwen3_path)
    fe = getattr(_PROCESSOR, "feature_extractor", None)
    if fe is not None and args.omni_max_audio_seconds:
        for attr, val in (("chunk_length", float(args.omni_max_audio_seconds)),):
            if hasattr(fe, attr):
                setattr(fe, attr, val)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(json.dumps({"model": "phase_e", **model.trainable_report()},
                         sort_keys=True), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (n, output_tensor[i]) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    from experiments.uniss_omni_s2st_v1.training.stage2_batch import build_batch

    batch = next(data_iterator)
    inputs = build_batch(batch, _PROCESSOR, _MODEL.talker, _MODEL.layout,
                         device=torch.cuda.current_device())
    # build_batch is stage 2's and still carries its text-loss arguments.
    # Phase E freezes the Thinker, so there is nothing for a text loss to
    # train and PhaseEModel.forward does not accept them.
    for key in ("text_labels", "supervise_text"):
        inputs.pop(key, None)
    return model(**inputs), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from transformers import Qwen3OmniMoeProcessor

    from experiments.uniss_omni_s2st_v1.training.stage2_data import (
        ParquetCodeIndex, Stage2Corpus, read_audio_rows,
    )
    from experiments.uniss_omni_s2st_v1.training.stage2_dataset import (
        Stage2Dataset, task_schedule,
    )
    from experiments.uniss_omni_s2st_v1.training.tts_data import CachedTtsCorpus
    from experiments.uniss_qwen3_omni_v1.training.phase_e_model import MIXTURE

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    model = _MODEL
    if model is None:
        raise RuntimeError("datasets requested before the model was built")

    if args.omni_audio_index:
        from experiments.uniss_qwen3_omni_v1.training.audio_index import AudioIndex
        audio_rows = AudioIndex(args.omni_audio_index)
    else:
        import glob
        manifests = sorted(p for pat in args.omni_manifests for p in glob.glob(pat))
        audio_rows = read_audio_rows(manifests, audio_root=args.omni_audio_root,
                                     limit=args.omni_audio_limit)
    # ParquetCodeIndex loads one flat array per parquet it is named, so it
    # needs the set of names the rows actually point at. Measured at 9.8
    # GiB for all 198 against 1,992 GiB of memory, so all of them is fine.
    if hasattr(audio_rows, "parquet_names"):
        names = audio_rows.parquet_names()
    else:
        names = {r.parquet for r in audio_rows}
    corpus = Stage2Corpus(audio_rows,
                          ParquetCodeIndex(args.omni_parquet_root, names),
                          CachedTtsCorpus(args.omni_tts_cache),
                          audio_root=args.omni_audio_root)
    gbs = int(args.global_batch_size)
    schedule = task_schedule(int(args.train_iters) + int(args.eval_iters) * 8,
                             seed=args.omni_seed, mixture=MIXTURE)
    train = Stage2Dataset(corpus, schedule[: int(args.train_iters)],
                          global_batch=gbs, seed=args.omni_seed)
    runtime.print_rank_0(
        f"phase E: {len(audio_rows):,} audio rows, {len(corpus.text_corpus):,}"
        f" cached TTS rows; mixture {MIXTURE};"
        f" coverage audio {train.epochs['audio']:.2f} epochs,"
        f" text {train.epochs['text']:.2f}")
    # Dev is S2ST only, so every evaluation scores the same kind of batch.
    dev_steps = max(1, int(args.eval_iters)) * (
        int(args.train_iters) // max(1, int(args.eval_interval)) + 4)
    dev = Stage2Dataset(corpus, ["s2st"] * dev_steps, global_batch=gbs,
                        seed=args.omni_seed + 1)
    return train, dev, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import (
        install_omni_collate,
    )
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_phase_e_args,
        args_defaults={"tokenizer_type": "NullTokenizer"})
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider,
                     runtime.ModelType.encoder_or_decoder, forward_step,
                     model_provider=model_provider)
    print("PHASE_E_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()
