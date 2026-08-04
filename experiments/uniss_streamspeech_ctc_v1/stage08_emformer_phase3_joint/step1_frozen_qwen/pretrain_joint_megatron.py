#!/usr/bin/env python3
"""Megatron-Core entrypoint for Stage08 Step1 shared-Emformer training."""

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
STEP = Path(__file__).resolve().parent
for path in (ROOT, MEGATRON_ROOT, STAGE02, STAGE03, STAGE03_AR, STAGE04, STAGE07, STEP):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bridge_data import B2BridgeAudioDataset
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.checkpoint_io import (
    load_step1_trainable_into_model,
)
from model import JointEmformerB1
from train_b2 import lm_batch
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.pretrain_uniss_megatron import load_megatron_runtime


DIRECTION = {
    0: ("asr_eng", "nar_s2tt_cmn", "eng", "cmn"),
    1: ("asr_cmn", "nar_s2tt_eng", "cmn", "eng"),
}


class JointMegatronDataset(torch.utils.data.Dataset):
    """Micro-batch-one view retaining all Stage03 and Phase3 targets."""

    def __init__(self, args, split: str) -> None:
        self.dataset = B2BridgeAudioDataset(
            args.joint_dataset_index,
            split,
            args.joint_source_manifest,
            args.joint_source_offsets,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.row_to_item(self.dataset[index])

    @staticmethod
    def row_to_item(row: dict[str, object]) -> dict[str, object]:
        target = row["target_token_ids"]
        return {
            "waveform": row["waveform"],
            "waveform_length": torch.tensor(len(row["waveform"]), dtype=torch.long),
            "source_targets": row["source_token_ids"],
            "source_length": torch.tensor(len(row["source_token_ids"]), dtype=torch.long),
            "target_targets": target,
            "target_padded": target,
            "target_length": torch.tensor(len(target), dtype=torch.long),
            "direction_id": torch.tensor(row["direction_id"], dtype=torch.long),
            "record_json": json.dumps(row["phase3_record"], ensure_ascii=False),
        }


class DirectionBalancedJointDataset(JointMegatronDataset):
    """Virtual 50:50 direction dataset without copying audio or manifests."""

    def __init__(self, args, split: str) -> None:
        super().__init__(args, split)
        root = Path(args.joint_direction_index_dir)
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
                f"direction index mismatch at virtual={index}, source={source_index}"
            )
        return item


def add_joint_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS Stage08 Step1 joint Emformer")
    for name in (
        "dataset-index",
        "source-manifest",
        "source-offsets",
        "ctc-tokenizer-dir",
        "stage03b-checkpoint",
        "historical-stage-b-checkpoint",
        "stage04-checkpoint",
        "stage06-checkpoint",
        "codebook-model",
        "phase3-model",
    ):
        group.add_argument(f"--joint-{name}", required=True)
    group.add_argument("--joint-unfreeze-encoder-layers", type=int, default=4)
    group.add_argument("--joint-asr-weight", type=float, default=4.0)
    group.add_argument("--joint-nar-weight", type=float, default=4.0)
    group.add_argument("--joint-ar-weight", type=float, default=8.0)
    group.add_argument("--joint-phase3-weight", type=float, default=0.5)
    group.add_argument("--joint-residual-weight", type=float, default=1e-4)
    group.add_argument("--joint-zh-en-weight", type=float, default=1.0)
    group.add_argument("--joint-step1-initialize-checkpoint")
    group.add_argument("--joint-direction-index-dir")
    return parser


def validate_joint_args(args) -> None:
    if int(args.micro_batch_size) != 1:
        raise ValueError("Stage08 Step1 requires micro-batch-size 1")
    if int(args.global_batch_size) != 128:
        raise ValueError("Stage08 Step1 preserves Phase3 global batch size 128")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("Stage08 composite model currently requires TP=PP=1")
    if not 1 <= int(args.joint_unfreeze_encoder_layers) <= 16:
        raise ValueError("joint-unfreeze-encoder-layers must be in [1,16]")
    for name in (
        "joint_dataset_index",
        "joint_source_manifest",
        "joint_source_offsets",
        "joint_ctc_tokenizer_dir",
        "joint_stage03b_checkpoint",
        "joint_historical_stage_b_checkpoint",
        "joint_stage04_checkpoint",
        "joint_stage06_checkpoint",
        "joint_codebook_model",
        "joint_phase3_model",
    ):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {getattr(args, name)}")
    for name in (
        "joint_asr_weight",
        "joint_nar_weight",
        "joint_ar_weight",
        "joint_phase3_weight",
        "joint_residual_weight",
        "joint_zh_en_weight",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("joint_step1_initialize_checkpoint", "joint_direction_index_dir"):
        value = getattr(args, name)
        if value is not None and not Path(value).exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {value}")
    if args.joint_direction_index_dir is not None:
        root = Path(args.joint_direction_index_dir)
        for split in ("train", "valid"):
            for direction in (0, 1):
                path = root / f"{split}_direction_{direction}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"missing balanced direction index: {path}")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building Stage08 Step1 joint audio datasets ...")
    dataset_class = (
        DirectionBalancedJointDataset
        if args.joint_direction_index_dir is not None
        else JointMegatronDataset
    )
    datasets = (
        dataset_class(args, "train"),
        dataset_class(args, "valid"),
        None,
    )
    runtime.print_rank_0("> finished Stage08 Step1 joint audio datasets ...")
    return datasets


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def _select_flat_targets(flat, lengths, indices):
    pieces = torch.split(flat, lengths.tolist())
    chosen = [pieces[index] for index in indices.tolist()]
    return torch.cat(chosen), lengths[indices]


def _ctc_loss(logits, targets, input_lengths, target_lengths, blank):
    return F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=False,
    )


def endpoint_multitask_losses(
    output,
    *,
    source_targets,
    source_lengths,
    target_targets,
    target_padded,
    target_lengths,
    direction_ids,
    vocab,
    asr_weight,
    nar_weight,
    ar_weight,
):
    """Return the weighted Stage03b loss and unweighted components."""

    logits = output["logits"]
    lengths = output["output_lengths"]
    total = output["ar_anchor"] + sum(value.sum() * 0.0 for value in logits.values())
    asr_sum = total.detach() * 0.0
    nar_sum = total.detach() * 0.0
    ar_sum = total.detach() * 0.0
    samples = 0
    for direction_id, (source_head, target_head, src_lang, tgt_lang) in DIRECTION.items():
        indices = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        selected_source, selected_source_lengths = _select_flat_targets(
            source_targets, source_lengths, indices
        )
        selected_target, selected_target_lengths = _select_flat_targets(
            target_targets, target_lengths, indices
        )
        source = _ctc_loss(
            logits[source_head][indices],
            selected_source,
            lengths[indices],
            selected_source_lengths,
            vocab[src_lang],
        )
        target = _ctc_loss(
            logits[target_head][indices],
            selected_target,
            lengths[indices],
            selected_target_lengths,
            vocab[tgt_lang],
        )
        ar_values, ar_rows = output["ar_logits"][tgt_lang]
        references = target_padded[ar_rows]
        ar = F.cross_entropy(
            ar_values.float().reshape(-1, ar_values.shape[-1]),
            references.reshape(-1),
            ignore_index=-1,
        )
        count = len(indices)
        total = total + (asr_weight * source + nar_weight * target + ar_weight * ar) * count
        asr_sum = asr_sum + source.detach() * count
        nar_sum = nar_sum + target.detach() * count
        ar_sum = ar_sum + ar.detach() * count
        samples += count
    if not samples:
        raise ValueError("joint micro-batch has no recognized direction")
    return total / samples, asr_sum / samples, nar_sum / samples, ar_sum / samples


class JointMegatronModel:
    """Factory namespace kept outside model_provider for testable imports."""

    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.pg_collection = pg_collection
                phase3 = Path(args.joint_phase3_model)
                self.tokenizer = AutoTokenizer.from_pretrained(phase3, local_files_only=True)
                self.text_encoder = load_hf_text_encoder(self.tokenizer)
                self.qwen = AutoModelForCausalLM.from_pretrained(
                    phase3, local_files_only=True, torch_dtype=torch.bfloat16
                ).cuda().eval()
                self.qwen.requires_grad_(False)
                qwen_glm_embeddings = self.qwen.get_input_embeddings().weight[
                    c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
                ].detach().float().cpu()
                tokenizer_dir = Path(args.joint_ctc_tokenizer_dir)
                self.vocab = {
                    language: spm.SentencePieceProcessor(
                        model_file=str(tokenizer_dir / f"ctc_{language}.model")
                    ).vocab_size()
                    for language in ("eng", "cmn")
                }
                self.joint, self.initialization = JointEmformerB1.from_checkpoints(
                    stage03b_checkpoint=args.joint_stage03b_checkpoint,
                    historical_stage_b_checkpoint=args.joint_historical_stage_b_checkpoint,
                    stage04_checkpoint=args.joint_stage04_checkpoint,
                    stage06_checkpoint=args.joint_stage06_checkpoint,
                    codebook_model=args.joint_codebook_model,
                    qwen_glm_embeddings=qwen_glm_embeddings,
                    eng_vocab_size=self.vocab["eng"],
                    cmn_vocab_size=self.vocab["cmn"],
                    unfreeze_encoder_layers=args.joint_unfreeze_encoder_layers,
                )
                self.repair_initialization = None
                if args.joint_step1_initialize_checkpoint is not None:
                    self.repair_initialization = load_step1_trainable_into_model(
                        self.joint, args.joint_step1_initialize_checkpoint
                    )
                self.joint.cuda()
                self.weights = {
                    "asr": float(args.joint_asr_weight),
                    "nar": float(args.joint_nar_weight),
                    "ar": float(args.joint_ar_weight),
                    "phase3": float(args.joint_phase3_weight),
                    "residual": float(args.joint_residual_weight),
                    "zh_en": float(args.joint_zh_en_weight),
                }
                if torch.distributed.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                "stage08_initialization": self.initialization.__dict__,
                                "step1_repair_initialization": self.repair_initialization,
                                "trainable_parameters": self.joint.trainable_parameter_counts(),
                                "loss_weights": self.weights,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            def train(self, mode: bool = True):
                super().train(mode)
                self.qwen.eval()
                self.joint.bridge.eval()
                return self

            def set_input_tensor(self, input_tensor):
                self.input_tensor = input_tensor

            def forward(
                self,
                waveform,
                waveform_length,
                source_targets,
                source_length,
                target_targets,
                target_padded,
                target_length,
                direction_id,
                record_json,
            ):
                records = [json.loads(value) for value in record_json]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    endpoint, b1 = self.joint(
                        waveform,
                        waveform_length,
                        target_padded,
                        target_length,
                        direction_id,
                    )
                    multitask, asr, nar, ar = endpoint_multitask_losses(
                        endpoint,
                        source_targets=source_targets,
                        source_lengths=source_length,
                        target_targets=target_targets,
                        target_padded=target_padded,
                        target_lengths=target_length,
                        direction_ids=direction_id,
                        vocab=self.vocab,
                        asr_weight=self.weights["asr"],
                        nar_weight=self.weights["nar"],
                        ar_weight=self.weights["ar"],
                    )
                    inputs, attention, labels, _ = lm_batch(
                        self.qwen,
                        self.text_encoder,
                        records,
                        b1,
                        waveform.device,
                    )
                    phase3 = self.qwen(
                        inputs_embeds=inputs,
                        attention_mask=attention,
                        labels=labels,
                        use_cache=False,
                    )
                    total = (
                        multitask
                        + self.weights["phase3"] * phase3.loss
                        + self.weights["residual"] * b1.residual_mse
                    )
                    en_zh = (direction_id == 0).float().mean()
                    zh_en = (direction_id == 1).float().mean()
                    direction_scale = en_zh + self.weights["zh_en"] * zh_en
                    total = total * direction_scale
                values = (
                    total.float(),
                    multitask.detach().float(),
                    asr.float(),
                    nar.float(),
                    ar.float(),
                    phase3.loss.detach().float(),
                    b1.residual_rms.float(),
                    en_zh.float(),
                    zh_en.float(),
                    (asr * en_zh).float(),
                    (asr * zh_en).float(),
                    (nar * en_zh).float(),
                    (nar * zh_en).float(),
                    (ar * en_zh).float(),
                    (ar * zh_en).float(),
                    (phase3.loss.detach() * en_zh).float(),
                    (phase3.loss.detach() * zh_en).float(),
                    direction_scale.float(),
                )
                if not all(torch.isfinite(value).all() for value in values):
                    raise FloatingPointError("non-finite Stage08 Step1 loss component")
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
    return JointMegatronModel.build(
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
    names = (
        "joint_multitask",
        "asr_ctc",
        "nar_s2tt_ctc",
        "ar_s2tt_ce",
        "phase3_nll",
        "b1_residual_rms",
    )
    metrics = dict(zip(names, averaged[: len(names)]))
    en_zh, zh_en = averaged[6], averaged[7]
    metrics.update(
        {
            "direction/en_zh_fraction": en_zh,
            "direction/zh_en_fraction": zh_en,
            "direction/asr_ctc_en_zh": averaged[8] / en_zh.clamp_min(1e-8),
            "direction/asr_ctc_zh_en": averaged[9] / zh_en.clamp_min(1e-8),
            "direction/nar_ctc_en_zh": averaged[10] / en_zh.clamp_min(1e-8),
            "direction/nar_ctc_zh_en": averaged[11] / zh_en.clamp_min(1e-8),
            "direction/ar_ce_en_zh": averaged[12] / en_zh.clamp_min(1e-8),
            "direction/ar_ce_zh_en": averaged[13] / zh_en.clamp_min(1e-8),
            "direction/phase3_nll_en_zh": averaged[14] / en_zh.clamp_min(1e-8),
            "direction/phase3_nll_zh_en": averaged[15] / zh_en.clamp_min(1e-8),
            "direction/weight_scale": averaged[16],
        }
    )
    return loss, metrics


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    waveform = batch["waveform"].cuda(non_blocking=True)
    waveform_length = batch["waveform_length"].cuda(non_blocking=True).reshape(-1)
    source_targets = batch["source_targets"].cuda(non_blocking=True).reshape(-1)
    source_length = batch["source_length"].cuda(non_blocking=True).reshape(-1)
    target_targets = batch["target_targets"].cuda(non_blocking=True).reshape(-1)
    target_padded = batch["target_padded"].cuda(non_blocking=True)
    if target_padded.ndim == 1:
        target_padded = target_padded.unsqueeze(0)
    target_length = batch["target_length"].cuda(non_blocking=True).reshape(-1)
    direction_id = batch["direction_id"].cuda(non_blocking=True).reshape(-1)
    output = model(
        waveform,
        waveform_length,
        source_targets,
        source_length,
        target_targets,
        target_padded,
        target_length,
        direction_id,
        batch["record_json"],
    )
    return output, loss_func


def main():
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_joint_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_joint_args(args)
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
