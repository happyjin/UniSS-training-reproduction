#!/usr/bin/env python3
"""Research-only Stage08 Step2 Qwen-LoRA plus offline Phase3 replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MEGATRON_ROOT = ROOT / "third_party" / "Megatron-LM"
TREE = Path(__file__).resolve().parents[2]
STAGE02 = TREE / "stage02_ctc_probe"
STAGE03 = TREE / "stage03_multitask_encoder"
STAGE03_AR = STAGE03 / "ar_s2tt_v1"
STAGE04 = TREE / "stage04_b2_discrete_bridge"
STAGE07 = TREE / "stage07_end_to_end_eval"
STEP1 = Path(__file__).resolve().parents[1] / "step1_frozen_qwen"
STEP2 = Path(__file__).resolve().parent
for path in (
    ROOT,
    MEGATRON_ROOT,
    STAGE02,
    STAGE03,
    STAGE03_AR,
    STAGE04,
    STAGE07,
    STEP1,
    STEP2,
):
    sys.path.insert(0, str(path))

import numpy as np
import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.checkpoint_io import (
    load_step1_inference_into_model,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.model import (
    JointEmformerB1,
)
from lora import inject_lora, lora_update_rms, set_lora_training
from phase3_batches import offline_replay_lm_batch, streaming_lm_batch
from replay_data import ReplayB2BridgeAudioDataset
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.pretrain_uniss_megatron import load_megatron_runtime


class Step2MegatronDataset(torch.utils.data.Dataset):
    def __init__(self, args, split: str) -> None:
        self.dataset = ReplayB2BridgeAudioDataset(
            args.step2_dataset_index,
            split,
            args.step2_source_manifest,
            args.step2_source_offsets,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.row_to_item(self.dataset[index])

    @staticmethod
    def row_to_item(row: dict[str, object]) -> dict[str, object]:
        return {
            "waveform": row["waveform"],
            "waveform_length": torch.tensor(len(row["waveform"]), dtype=torch.long),
            "direction_id": torch.tensor(row["direction_id"], dtype=torch.long),
            "record_json": json.dumps(row["phase3_record"], ensure_ascii=False),
        }


class DirectionBalancedStep2Dataset(Step2MegatronDataset):
    def __init__(self, args, split: str) -> None:
        super().__init__(args, split)
        root = Path(args.step2_direction_index_dir)
        self.direction_indices = {
            direction: np.load(root / f"{split}_direction_{direction}.npy", mmap_mode="r")
            for direction in (0, 1)
        }
        if any(len(values) == 0 for values in self.direction_indices.values()):
            raise ValueError(f"empty direction index for {split}: {root}")
        self.pairs = max(len(values) for values in self.direction_indices.values())

    def __len__(self) -> int:
        return 2 * self.pairs

    def __getitem__(self, index: int) -> dict[str, object]:
        direction = index % 2
        values = self.direction_indices[direction]
        source_index = int(values[(index // 2) % len(values)])
        item = self.row_to_item(self.dataset[source_index])
        if int(item["direction_id"]) != direction:
            raise ValueError(
                f"direction mismatch at virtual={index}, source={source_index}"
            )
        return item


def add_step2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS Stage08 Step2 research validation")
    for name in (
        "dataset-index",
        "source-manifest",
        "source-offsets",
        "direction-index-dir",
        "ctc-tokenizer-dir",
        "stage03b-checkpoint",
        "historical-stage-b-checkpoint",
        "stage04-checkpoint",
        "stage06-checkpoint",
        "step1-checkpoint",
        "codebook-model",
        "phase3-model",
    ):
        group.add_argument(f"--step2-{name}", required=True)
    group.add_argument("--step2-unfreeze-encoder-layers", type=int, default=4)
    group.add_argument("--step2-lora-rank", type=int, default=8)
    group.add_argument("--step2-lora-alpha", type=float, default=16.0)
    group.add_argument("--step2-lora-dropout", type=float, default=0.05)
    group.add_argument("--step2-lora-targets", default="q_proj,v_proj")
    group.add_argument("--step2-offline-replay-ratio", type=float, default=0.30)
    group.add_argument("--step2-research-only-override", action="store_true")
    return parser


def validate_step2_args(args) -> None:
    if not args.step2_research_only_override:
        raise ValueError(
            "Step1-R did not pass the hard gate; --step2-research-only-override is required"
        )
    if int(args.micro_batch_size) != 1 or int(args.global_batch_size) != 128:
        raise ValueError("Step2 requires micro/global batch sizes 1/128")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("Step2 currently requires TP=PP=1")
    if not 0.0 < float(args.step2_offline_replay_ratio) < 1.0:
        raise ValueError("offline replay ratio must be in (0,1)")
    if int(args.step2_lora_rank) <= 0 or float(args.step2_lora_alpha) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= float(args.step2_lora_dropout) < 1.0:
        raise ValueError("LoRA dropout must be in [0,1)")
    targets = [value.strip() for value in args.step2_lora_targets.split(",") if value.strip()]
    if not targets:
        raise ValueError("at least one LoRA target is required")
    for name in (
        "step2_dataset_index",
        "step2_source_manifest",
        "step2_source_offsets",
        "step2_direction_index_dir",
        "step2_ctc_tokenizer_dir",
        "step2_stage03b_checkpoint",
        "step2_historical_stage_b_checkpoint",
        "step2_stage04_checkpoint",
        "step2_stage06_checkpoint",
        "step2_step1_checkpoint",
        "step2_codebook_model",
        "step2_phase3_model",
    ):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {getattr(args, name)}")
    root = Path(args.step2_direction_index_dir)
    for split in ("train", "valid"):
        for direction in (0, 1):
            path = root / f"{split}_direction_{direction}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"missing balanced direction index: {path}")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building Stage08 Step2 full-replay audio datasets ...")
    datasets = (
        DirectionBalancedStep2Dataset(args, "train"),
        DirectionBalancedStep2Dataset(args, "valid"),
        None,
    )
    runtime.print_rank_0("> finished Stage08 Step2 full-replay audio datasets ...")
    return datasets


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


class Step2MegatronModel:
    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.pg_collection = pg_collection
                phase3 = Path(args.step2_phase3_model)
                self.tokenizer = AutoTokenizer.from_pretrained(phase3, local_files_only=True)
                self.text_encoder = load_hf_text_encoder(self.tokenizer)
                self.qwen = AutoModelForCausalLM.from_pretrained(
                    phase3, local_files_only=True, torch_dtype=torch.bfloat16
                ).cuda().eval()
                self.qwen.requires_grad_(False)
                qwen_glm_embeddings = self.qwen.get_input_embeddings().weight[
                    c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
                ].detach().float().cpu()
                tokenizer_dir = Path(args.step2_ctc_tokenizer_dir)
                vocab = {
                    language: spm.SentencePieceProcessor(
                        model_file=str(tokenizer_dir / f"ctc_{language}.model")
                    ).vocab_size()
                    for language in ("eng", "cmn")
                }
                self.joint, self.initialization = JointEmformerB1.from_checkpoints(
                    stage03b_checkpoint=args.step2_stage03b_checkpoint,
                    historical_stage_b_checkpoint=args.step2_historical_stage_b_checkpoint,
                    stage04_checkpoint=args.step2_stage04_checkpoint,
                    stage06_checkpoint=args.step2_stage06_checkpoint,
                    codebook_model=args.step2_codebook_model,
                    qwen_glm_embeddings=qwen_glm_embeddings,
                    eng_vocab_size=vocab["eng"],
                    cmn_vocab_size=vocab["cmn"],
                    unfreeze_encoder_layers=args.step2_unfreeze_encoder_layers,
                )
                self.step1 = load_step1_inference_into_model(
                    self.joint,
                    args.step2_step1_checkpoint,
                    unfreeze_encoder_layers=args.step2_unfreeze_encoder_layers,
                )
                self.joint.requires_grad_(False)
                self.joint.cuda().eval()
                targets = [
                    value.strip()
                    for value in args.step2_lora_targets.split(",")
                    if value.strip()
                ]
                self.lora = inject_lora(
                    self.qwen,
                    target_modules=targets,
                    rank=args.step2_lora_rank,
                    alpha=args.step2_lora_alpha,
                    dropout=args.step2_lora_dropout,
                )
                self.replay_ratio = float(args.step2_offline_replay_ratio)
                if torch.distributed.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                "research_only": True,
                                "stage08_initialization": self.initialization.__dict__,
                                "step1_initialization": self.step1,
                                "lora": self.lora.__dict__,
                                "offline_replay_ratio": self.replay_ratio,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            def train(self, mode: bool = True):
                super().train(mode)
                self.joint.eval()
                self.qwen.eval()
                set_lora_training(self.qwen, mode)
                return self

            def set_input_tensor(self, input_tensor):
                self.input_tensor = input_tensor

            def forward(self, waveform, waveform_length, direction_id, record_json):
                records = [json.loads(value) for value in record_json]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    bridge_output = self.joint.encode_to_b1(waveform, waveform_length)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    stream_inputs, stream_attention, stream_labels, stream_tokens = (
                        streaming_lm_batch(
                            self.qwen,
                            self.text_encoder,
                            records,
                            bridge_output,
                            waveform.device,
                        )
                    )
                    replay_inputs, replay_attention, replay_labels, replay_tokens = (
                        offline_replay_lm_batch(
                            self.qwen,
                            self.text_encoder,
                            records,
                            waveform.device,
                        )
                    )
                    stream = self.qwen(
                        inputs_embeds=stream_inputs,
                        attention_mask=stream_attention,
                        labels=stream_labels,
                        use_cache=False,
                    ).loss
                    replay = self.qwen(
                        inputs_embeds=replay_inputs,
                        attention_mask=replay_attention,
                        labels=replay_labels,
                        use_cache=False,
                    ).loss
                    total = (1.0 - self.replay_ratio) * stream + self.replay_ratio * replay
                    en_zh = (direction_id == 0).float().mean()
                    zh_en = (direction_id == 1).float().mean()
                    update_rms = lora_update_rms(self.qwen).detach()
                values = (
                    total.float(),
                    stream.detach().float(),
                    replay.detach().float(),
                    bridge_output.residual_rms.detach().float(),
                    update_rms.float(),
                    en_zh.float(),
                    zh_en.float(),
                    (stream.detach() * en_zh).float(),
                    (stream.detach() * zh_en).float(),
                    (replay.detach() * en_zh).float(),
                    (replay.detach() * zh_en).float(),
                    torch.tensor(float(stream_tokens), device=waveform.device),
                    torch.tensor(float(replay_tokens), device=waveform.device),
                )
                if not all(torch.isfinite(value).all() for value in values):
                    raise FloatingPointError("non-finite Stage08 Step2 loss component")
                return torch.stack(values)

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
    return Step2MegatronModel.build(
        config or _transformer_config(args), args, pg_collection=pg_collection
    )


def loss_func(output_tensor):
    from megatron.core import parallel_state
    from megatron.training.utils import average_losses_across_data_parallel_group

    loss = output_tensor[0]
    averaged = average_losses_across_data_parallel_group(
        list(output_tensor[1:]),
        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    metrics = dict(
        zip(
            (
                "streaming_phase3_nll",
                "offline_replay_nll",
                "b1_residual_rms",
                "lora_b_rms",
            ),
            averaged[:4],
        )
    )
    en_zh, zh_en = averaged[4], averaged[5]
    metrics.update(
        {
            "direction/en_zh_fraction": en_zh,
            "direction/zh_en_fraction": zh_en,
            "direction/streaming_nll_en_zh": averaged[6] / en_zh.clamp_min(1e-8),
            "direction/streaming_nll_zh_en": averaged[7] / zh_en.clamp_min(1e-8),
            "direction/replay_nll_en_zh": averaged[8] / en_zh.clamp_min(1e-8),
            "direction/replay_nll_zh_en": averaged[9] / zh_en.clamp_min(1e-8),
            "streaming_target_tokens": averaged[10],
            "offline_target_tokens": averaged[11],
        }
    )
    return loss, metrics


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    waveform = batch["waveform"].cuda(non_blocking=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform_length = batch["waveform_length"].cuda(non_blocking=True).reshape(-1)
    direction_id = batch["direction_id"].cuda(non_blocking=True).reshape(-1)
    output = model(
        waveform,
        waveform_length,
        direction_id,
        batch["record_json"],
    )
    return output, loss_func


def main():
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_step2_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_step2_args(args)
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
