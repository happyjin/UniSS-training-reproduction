#!/usr/bin/env python3
"""Megatron-Core entrypoint for the Stage06 B1 continuous residual."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = ROOT / "third_party" / "Megatron-LM"
TREE = Path(__file__).resolve().parents[1]
STAGE03 = TREE / "stage03_multitask_encoder"
STAGE02 = TREE / "stage02_ctc_probe"
STAGE04 = TREE / "stage04_b2_discrete_bridge"
for path in (ROOT, MEGATRON_ROOT, STAGE03, STAGE02, STAGE04, Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bridge_data import B2BridgeAudioDataset
from model import FrozenB2ResidualBridge
from train_b2 import lm_batch
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.pretrain_uniss_megatron import load_megatron_runtime


class B1MegatronDataset(torch.utils.data.Dataset):
    """Micro-batch-one dataset compatible with Megatron's default collator."""

    def __init__(self, args, split: str) -> None:
        self.dataset = B2BridgeAudioDataset(
            args.b1_dataset_index,
            split,
            args.b1_source_manifest,
            args.b1_source_offsets,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.dataset[index]
        return {
            "waveform": row["waveform"],
            "waveform_length": torch.tensor(len(row["waveform"]), dtype=torch.long),
            "record_json": json.dumps(row["phase3_record"], ensure_ascii=False),
        }


def add_b1_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS StreamSpeech B1")
    for name in (
        "dataset-index",
        "source-manifest",
        "source-offsets",
        "ctc-tokenizer-dir",
        "endpoint-checkpoint",
        "historical-stage-b-checkpoint",
        "stage04-b2-checkpoint",
        "codebook-model",
        "phase3-model",
    ):
        group.add_argument(f"--b1-{name}", required=True)
    group.add_argument("--b1-residual-weight", type=float, default=1e-4)
    return parser


def validate_b1_args(args) -> None:
    if int(args.micro_batch_size) != 1:
        raise ValueError("B1 Megatron path requires micro-batch-size 1 for finite HF input gradients")
    if int(args.global_batch_size) != 128:
        raise ValueError("B1 Megatron path preserves the Phase3 global batch size 128")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("B1 composite model currently requires TP=PP=1")
    for name in (
        "b1_dataset_index",
        "b1_source_manifest",
        "b1_source_offsets",
        "b1_ctc_tokenizer_dir",
        "b1_endpoint_checkpoint",
        "b1_historical_stage_b_checkpoint",
        "b1_stage04_b2_checkpoint",
        "b1_codebook_model",
        "b1_phase3_model",
    ):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {getattr(args, name)}")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building Stage06 B1 audio datasets ...")
    datasets = (
        B1MegatronDataset(args, "train"),
        B1MegatronDataset(args, "valid"),
        None,
    )
    runtime.print_rank_0("> finished Stage06 B1 audio datasets ...")
    return datasets


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


class B1MegatronModel:
    """Factory namespace kept outside model_provider for testable imports."""

    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                # Newer Megatron training loops retrieve this attribute from
                # every model (including custom MegatronModule subclasses) for
                # logging and process-group-aware bookkeeping.
                self.pg_collection = pg_collection
                phase3 = Path(args.b1_phase3_model)
                self.tokenizer = AutoTokenizer.from_pretrained(phase3, local_files_only=True)
                self.text_encoder = load_hf_text_encoder(self.tokenizer)
                self.qwen = AutoModelForCausalLM.from_pretrained(
                    phase3, local_files_only=True, torch_dtype=torch.bfloat16
                ).cuda().eval()
                self.qwen.requires_grad_(False)
                qwen_glm_embeddings = self.qwen.get_input_embeddings().weight[
                    c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
                ].detach().float().cpu()
                tokenizer_dir = Path(args.b1_ctc_tokenizer_dir)
                eng_vocab = spm.SentencePieceProcessor(
                    model_file=str(tokenizer_dir / "ctc_eng.model")
                ).vocab_size()
                cmn_vocab = spm.SentencePieceProcessor(
                    model_file=str(tokenizer_dir / "ctc_cmn.model")
                ).vocab_size()
                self.bridge = FrozenB2ResidualBridge.from_checkpoints(
                    endpoint_checkpoint=args.b1_endpoint_checkpoint,
                    historical_stage_b_checkpoint=args.b1_historical_stage_b_checkpoint,
                    stage04_b2_checkpoint=args.b1_stage04_b2_checkpoint,
                    codebook_model=args.b1_codebook_model,
                    qwen_glm_embeddings=qwen_glm_embeddings,
                    eng_vocab_size=eng_vocab,
                    cmn_vocab_size=cmn_vocab,
                ).cuda()
                self.residual_weight = float(args.b1_residual_weight)

            def train(self, mode: bool = True):
                super().train(mode)
                self.qwen.eval()
                self.bridge.base.eval()
                return self

            def set_input_tensor(self, input_tensor):
                """Megatron pipeline interface; PP=1 does not consume it."""
                self.input_tensor = input_tensor

            def forward(self, waveform, waveform_length, record_json):
                records = [json.loads(value) for value in record_json]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.bridge(waveform, waveform_length)
                    inputs, attention, labels, _ = lm_batch(
                        self.qwen, self.text_encoder, records, output, waveform.device
                    )
                    phase3 = self.qwen(
                        inputs_embeds=inputs,
                        attention_mask=attention,
                        labels=labels,
                        use_cache=False,
                    )
                    total = phase3.loss + self.residual_weight * output.residual_mse
                return torch.stack((total.float(), phase3.loss.detach().float(), output.residual_rms.float()))

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
    return B1MegatronModel.build(
        config or _transformer_config(args), args, pg_collection=pg_collection
    )


def loss_func(output_tensor):
    from megatron.core import parallel_state
    from megatron.training.utils import average_losses_across_data_parallel_group

    loss = output_tensor[0]
    averaged = average_losses_across_data_parallel_group(
        [output_tensor[1], output_tensor[2]],
        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    return loss, {
        "phase3_nll": averaged[0],
        "b1_residual_rms": averaged[1],
    }


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    waveform = batch["waveform"].cuda(non_blocking=True)
    waveform_length = batch["waveform_length"].cuda(non_blocking=True).reshape(-1)
    output = model(waveform, waveform_length, batch["record_json"])
    return output, loss_func


def main():
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_b1_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_b1_args(args)
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    # The new config-container API otherwise prioritizes its stock GPT builder
    # and silently ignores model_provider. Keep its DDP/optimizer/checkpoint
    # configs but delegate model construction to the B1 composite below.
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
