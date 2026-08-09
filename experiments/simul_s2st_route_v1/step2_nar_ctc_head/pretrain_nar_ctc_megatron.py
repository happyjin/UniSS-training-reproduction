#!/usr/bin/env python3
"""Megatron entry: freeze Phase3 Qwen, train a duration-anchored causal NAR CTC head.

Isolated under ``experiments/simul_s2st_route_v1/`` — does not modify joint V6 or
``training/phase3_whisper_streamspeech_joint/``. The head is the only module with
``requires_grad=True``. CTC infeasibility is a hard error, not a silent drop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = ROOT / "third_party" / "Megatron-LM"
HERE = Path(__file__).resolve().parent
for path in (str(ROOT), str(MEGATRON_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.dataset import NarCtcJointDataset
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
    required_ctc_frames,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.teacher_forced import (
    batch_fields,
    target_text_hidden,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.phase3_whisper_streamspeech_joint.losses import ctc_normalized_loss
from training.pretrain_uniss_megatron import load_megatron_runtime


def add_nar_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Simul-S2ST route Step2 NAR CTC")
    group.add_argument("--nar-train-manifest", required=True)
    group.add_argument("--nar-valid-manifest", required=True)
    group.add_argument("--nar-phase3-model", required=True)
    group.add_argument("--nar-frames-per-second", type=float, default=75.0)
    group.add_argument("--nar-max-frames", type=int, default=1500)
    group.add_argument("--nar-model-size", type=int, default=512)
    group.add_argument("--nar-t2u-layers", type=int, default=2)
    group.add_argument("--nar-decoder-layers", type=int, default=2)
    group.add_argument("--nar-num-heads", type=int, default=8)
    group.add_argument("--nar-dropout", type=float, default=0.1)
    group.add_argument("--nar-max-audio-seconds", type=float, default=12.0)
    group.add_argument("--nar-min-audio-seconds", type=float, default=0.4)
    group.add_argument("--nar-max-unit-tokens", type=int, default=1200)
    group.add_argument("--nar-degenerate-ratio-limit", type=float, default=100.0)
    group.add_argument("--nar-max-train-samples", type=int, default=0)
    group.add_argument("--nar-max-valid-samples", type=int, default=0)
    group.add_argument("--nar-fail-on-infeasible", action="store_true", default=True)
    group.add_argument("--nar-allow-infeasible", action="store_true")
    return parser


def validate_nar_args(args) -> None:
    if int(args.micro_batch_size) != 1:
        raise ValueError("Step2 NAR CTC requires --micro-batch-size 1")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("Step2 NAR CTC requires TP=PP=1")
    for name in ("nar_train_manifest", "nar_valid_manifest", "nar_phase3_model"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"missing --{name.replace('_', '-')}: {path}")
    if float(args.nar_frames_per_second) <= 0:
        raise ValueError("--nar-frames-per-second must be positive")
    if int(args.nar_max_frames) <= 0:
        raise ValueError("--nar-max-frames must be positive")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building Step2 NAR CTC datasets ...")
    train = NarCtcJointDataset(
        args.nar_train_manifest,
        max_audio_seconds=float(args.nar_max_audio_seconds),
        min_audio_seconds=float(args.nar_min_audio_seconds),
        max_unit_tokens=int(args.nar_max_unit_tokens),
        degenerate_ratio_limit=float(args.nar_degenerate_ratio_limit),
        max_samples=int(args.nar_max_train_samples) or None,
    )
    valid = NarCtcJointDataset(
        args.nar_valid_manifest,
        max_audio_seconds=float(args.nar_max_audio_seconds),
        min_audio_seconds=float(args.nar_min_audio_seconds),
        max_unit_tokens=int(args.nar_max_unit_tokens),
        degenerate_ratio_limit=float(args.nar_degenerate_ratio_limit),
        max_samples=int(args.nar_max_valid_samples) or None,
    )
    runtime.print_rank_0(
        f"> Step2 NAR CTC datasets ready: train={len(train)} valid={len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


class NarCtcMegatronFactory:
    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.pg_collection = pg_collection
                phase3 = Path(args.nar_phase3_model)
                self.tokenizer = AutoTokenizer.from_pretrained(phase3, local_files_only=True)
                self.text_encoder = load_hf_text_encoder(self.tokenizer)
                self.qwen = (
                    AutoModelForCausalLM.from_pretrained(
                        phase3,
                        local_files_only=True,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                    )
                    .cuda()
                    .eval()
                )
                self.qwen.requires_grad_(False)
                self.qwen.config.use_cache = False
                self.head = DurationAnchoredCausalNARCTC(
                    qwen_hidden_size=int(self.qwen.config.hidden_size),
                    model_size=int(args.nar_model_size),
                    semantic_vocab_size=c.BICODEC_SEMANTIC_SIZE,
                    frames_per_second=float(args.nar_frames_per_second),
                    num_heads=int(args.nar_num_heads),
                    t2u_layers=int(args.nar_t2u_layers),
                    decoder_layers=int(args.nar_decoder_layers),
                    dropout=float(args.nar_dropout),
                    max_frames=int(args.nar_max_frames),
                ).cuda()
                self.fail_on_infeasible = bool(args.nar_fail_on_infeasible) and not bool(
                    args.nar_allow_infeasible
                )
                trainable = sum(parameter.numel() for parameter in self.head.parameters())
                frozen = sum(parameter.numel() for parameter in self.qwen.parameters())
                if torch.distributed.get_rank() == 0:
                    print(
                        {
                            "step2_nar_ctc": {
                                "trainable_params": trainable,
                                "frozen_qwen_params": frozen,
                                "frames_per_second": float(args.nar_frames_per_second),
                                "max_frames": int(args.nar_max_frames),
                                "blank_id": self.head.blank_id,
                            }
                        },
                        flush=True,
                    )

            def train(self, mode: bool = True):
                super().train(mode)
                self.qwen.eval()
                return self

            def set_input_tensor(self, input_tensor):
                self.input_tensor = input_tensor

            def forward(self, batch: dict[str, object]):
                fields = batch_fields(batch)
                device = next(self.head.parameters()).device
                text_hidden, text_lengths, _ = target_text_hidden(
                    self.qwen,
                    self.text_encoder,
                    source_glm=fields["source_glm"],
                    bicodec_global=fields["bicodec_global"],
                    tgt_lang=fields["tgt_lang"],
                    translation=fields["translation"],
                    target_bicodec=fields["target_bicodec"],
                    source_id=fields["id"],
                    device=device,
                )
                duration = fields["source_duration_ms"].to(device).reshape(1)
                units = fields["target_bicodec_tensor"].to(device)
                unit_lengths = torch.tensor([units.numel()], dtype=torch.long, device=device)
                unit_repeats = fields["unit_repeats"].to(device).reshape(1)
                required = required_ctc_frames(int(unit_lengths), int(unit_repeats))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, frame_lengths = self.head(
                        text_hidden,
                        text_lengths,
                        duration,
                        unit_lengths=unit_lengths,
                        unit_repeats=unit_repeats,
                    )
                    loss, infeasible = ctc_normalized_loss(
                        logits,
                        units,
                        frame_lengths,
                        unit_lengths,
                        blank_id=self.head.blank_id,
                    )
                if self.fail_on_infeasible and int(infeasible.item()) > 0:
                    raise FloatingPointError(
                        "CTC path infeasible after duration anchoring: "
                        f"id={fields['id']} units={int(unit_lengths)} "
                        f"repeats={int(unit_repeats)} required={required} "
                        f"frames={int(frame_lengths)} "
                        f"duration_ms={int(duration)} "
                        f"max_frames={self.head.max_frames}"
                    )
                mean = loss.mean
                occupancy = required / max(1, int(frame_lengths.item()))
                return torch.stack(
                    (
                        mean.float(),
                        mean.detach().float(),
                        infeasible.detach().float(),
                        frame_lengths.detach().float().mean(),
                        text_lengths.detach().float().mean(),
                        unit_lengths.detach().float().mean(),
                        torch.tensor(occupancy, device=device),
                    )
                )

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
    return NarCtcMegatronFactory.build(
        config or _transformer_config(args), args, pg_collection=pg_collection
    )


def loss_func(output_tensor):
    from megatron.core import parallel_state
    from megatron.training.utils import average_losses_across_data_parallel_group

    loss = output_tensor[0]
    averaged = average_losses_across_data_parallel_group(
        [output_tensor[i] for i in range(1, output_tensor.numel())],
        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    return loss, {
        "nar_ctc": averaged[0],
        "nar_infeasible": averaged[1],
        "nar_frames": averaged[2],
        "nar_text_tokens": averaged[3],
        "nar_unit_tokens": averaged[4],
        "nar_occupancy": averaged[5],
    }


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    # Move tensor fields to CUDA; strings stay on CPU.
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.cuda(non_blocking=True)
        else:
            prepared[key] = value
    return model(prepared), loss_func


def main():
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_nar_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_nar_args(args)
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
