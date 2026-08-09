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

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.dataset import (
    NarCtcJointDataset,
    collate_nar_ctc,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
    required_ctc_frames,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.teacher_forced import (
    batch_fields,
    batched_target_text_hidden,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.phase3_whisper_streamspeech_joint.losses import ctc_normalized_loss
from training.pretrain_uniss_megatron import load_megatron_runtime


def install_nar_collate() -> None:
    """Attach variable-length collate without patching Megatron source files."""

    import megatron.training.datasets.data_samplers as data_samplers
    import megatron.training.training as megatron_training

    original = data_samplers.build_pretraining_data_loader
    if getattr(original, "_uniss_nar_ctc_collate", False):
        return

    def build_with_nar_collate(dataset, *args, **kwargs):
        collate = getattr(dataset, "collate_fn", None)
        if not callable(collate):
            return original(dataset, *args, **kwargs)

        original_loader = torch.utils.data.DataLoader

        def data_loader(*loader_args, **loader_kwargs):
            loader_kwargs.setdefault("collate_fn", collate)
            return original_loader(*loader_args, **loader_kwargs)

        torch.utils.data.DataLoader = data_loader
        try:
            return original(dataset, *args, **kwargs)
        finally:
            torch.utils.data.DataLoader = original_loader

    build_with_nar_collate._uniss_nar_ctc_collate = True
    data_samplers.build_pretraining_data_loader = build_with_nar_collate
    megatron_training.build_pretraining_data_loader = build_with_nar_collate


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
    group.add_argument(
        "--nar-blank-penalty",
        type=float,
        default=0.0,
        help="Add λ * mean blank softmax mass on valid frames (fights CTC blank collapse).",
    )
    group.add_argument(
        "--nar-guided-ce-weight",
        type=float,
        default=0.0,
        help="Weight for duration-stretched unit CE (peaks non-blank mass; 0 disables).",
    )
    group.add_argument(
        "--nar-ctc-weight",
        type=float,
        default=1.0,
        help="Scale on CTC loss (default 1.0 preserves v1–v4). Lower in v5 when guided CE should dominate.",
    )
    return parser


def blank_probability_penalty(
    logits: torch.Tensor,
    frame_lengths: torch.Tensor,
    *,
    blank_id: int,
) -> torch.Tensor:
    """Mean blank probability over valid frames (differentiable)."""

    if logits.ndim != 3:
        raise ValueError("logits must be [B,T,V]")
    if frame_lengths.shape[0] != logits.shape[0]:
        raise ValueError("frame_lengths must match batch")
    positions = torch.arange(logits.shape[1], device=logits.device)
    mask = positions[None, :] < frame_lengths[:, None]
    probs = torch.softmax(logits.float(), dim=-1)[..., int(blank_id)]
    denom = mask.float().sum().clamp_min(1.0)
    return (probs * mask.float()).sum() / denom


def guided_duration_ce(
    logits: torch.Tensor,
    units_padded: torch.Tensor,
    frame_lengths: torch.Tensor,
    unit_lengths: torch.Tensor,
) -> torch.Tensor:
    """Uniformly stretch target units onto frames and take token CE.

    Blank-only CTC optima fragment non-blank mass across the unit vocabulary so
    argmax stays blank. A cheap duration-guided CE peaks one unit per frame and
    is disabled when ``--nar-guided-ce-weight`` is 0 (default preserves v1–v3).
    """

    if logits.ndim != 3:
        raise ValueError("logits must be [B,T,V]")
    batch, max_frames, _ = logits.shape
    device = logits.device
    targets = torch.zeros(batch, max_frames, dtype=torch.long, device=device)
    mask = torch.zeros(batch, max_frames, dtype=torch.bool, device=device)
    for index in range(batch):
        frames = int(frame_lengths[index])
        units = int(unit_lengths[index])
        if frames <= 0 or units <= 0:
            continue
        unit_ids = units_padded[index, :units]
        # Map frame t -> unit floor(t * units / frames), last frame gets last unit.
        positions = torch.arange(frames, device=device)
        mapped = (positions * units) // frames
        mapped = mapped.clamp(max=units - 1)
        targets[index, :frames] = unit_ids[mapped]
        mask[index, :frames] = True
    if not bool(mask.any()):
        return logits.new_zeros(())
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        flat_logits[flat_mask], flat_targets[flat_mask], reduction="mean"
    )
    return loss


def validate_nar_args(args) -> None:
    if int(args.micro_batch_size) < 1:
        raise ValueError("--micro-batch-size must be >= 1")
    if float(args.nar_blank_penalty) < 0:
        raise ValueError("--nar-blank-penalty must be >= 0")
    if float(args.nar_guided_ce_weight) < 0:
        raise ValueError("--nar-guided-ce-weight must be >= 0")
    if float(args.nar_ctc_weight) < 0:
        raise ValueError("--nar-ctc-weight must be >= 0")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("Step2 NAR CTC requires TP=PP=1")
    if int(args.global_batch_size) % (
        int(args.micro_batch_size) * max(1, int(getattr(args, "data_parallel_size", 1) or 1))
    ) != 0 and torch.distributed.is_initialized():
        # Megatron validates this more carefully after DP is known; keep a soft check here.
        pass
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
    train.collate_fn = collate_nar_ctc
    valid.collate_fn = collate_nar_ctc
    runtime.print_rank_0(
        f"> Step2 NAR CTC datasets ready: train={len(train)} valid={len(valid)} "
        f"micro_batch_size={int(args.micro_batch_size)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def _flatten_targets(padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    pieces = [padded[row, : int(lengths[row])] for row in range(int(lengths.numel()))]
    return torch.cat(pieces) if pieces else padded.new_zeros((0,), dtype=torch.long)


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
                self.blank_penalty = float(args.nar_blank_penalty)
                self.guided_ce_weight = float(args.nar_guided_ce_weight)
                self.ctc_weight = float(args.nar_ctc_weight)
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
                                "micro_batch_size": int(args.micro_batch_size),
                                "global_batch_size": int(args.global_batch_size),
                                "blank_penalty": self.blank_penalty,
                                "guided_ce_weight": self.guided_ce_weight,
                                "ctc_weight": self.ctc_weight,
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
                text_hidden, text_lengths, _ = batched_target_text_hidden(
                    self.qwen,
                    self.text_encoder,
                    source_glm=fields["source_glm"],
                    bicodec_global=fields["bicodec_global"],
                    tgt_lang=fields["tgt_lang"],
                    translation=fields["translation"],
                    target_bicodec=fields["target_bicodec"],
                    source_id=fields["ids"],
                    device=device,
                )
                duration = fields["source_duration_ms"].to(device)
                unit_lengths = fields["target_bicodec_lengths"].to(device)
                unit_repeats = fields["unit_repeats"].to(device)
                units_padded = fields["target_bicodec_tensor"].to(device)
                units_flat = _flatten_targets(units_padded, unit_lengths)
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
                        units_flat,
                        frame_lengths,
                        unit_lengths,
                        blank_id=self.head.blank_id,
                    )
                    blank_mass = blank_probability_penalty(
                        logits, frame_lengths, blank_id=self.head.blank_id
                    )
                    if self.guided_ce_weight > 0:
                        guided = guided_duration_ce(
                            logits, units_padded, frame_lengths, unit_lengths
                        )
                    else:
                        guided = logits.new_zeros(())
                if self.fail_on_infeasible and int(infeasible.item()) > 0:
                    bad = int(torch.nonzero(unit_lengths + unit_repeats > frame_lengths)[0])
                    raise FloatingPointError(
                        "CTC path infeasible after duration anchoring: "
                        f"id={fields['ids'][bad]} units={int(unit_lengths[bad])} "
                        f"repeats={int(unit_repeats[bad])} "
                        f"required={required_ctc_frames(int(unit_lengths[bad]), int(unit_repeats[bad]))} "
                        f"frames={int(frame_lengths[bad])} "
                        f"duration_ms={int(duration[bad])} "
                        f"max_frames={self.head.max_frames}"
                    )
                mean = (
                    self.ctc_weight * loss.mean
                    + self.blank_penalty * blank_mass
                    + self.guided_ce_weight * guided
                )
                required = (unit_lengths + unit_repeats).float()
                occupancy = (required / frame_lengths.float().clamp_min(1.0)).mean()
                return torch.stack(
                    (
                        mean.float(),
                        loss.mean.detach().float(),
                        infeasible.detach().float(),
                        frame_lengths.detach().float().mean(),
                        text_lengths.detach().float().mean(),
                        unit_lengths.detach().float().mean(),
                        occupancy.detach().float(),
                        blank_mass.detach().float(),
                        guided.detach().float(),
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
    metrics = {
        "nar_ctc": averaged[0],
        "nar_infeasible": averaged[1],
        "nar_frames": averaged[2],
        "nar_text_tokens": averaged[3],
        "nar_unit_tokens": averaged[4],
        "nar_occupancy": averaged[5],
    }
    if len(averaged) > 6:
        metrics["nar_blank_mass"] = averaged[6]
    if len(averaged) > 7:
        metrics["nar_guided_ce"] = averaged[7]
    return loss, metrics


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
    install_nar_collate()
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
