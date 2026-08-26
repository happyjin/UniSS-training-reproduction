#!/usr/bin/env python3
"""Megatron LoRA update from real free-running long-episode trajectories."""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as e2e
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.dataset import (
    EpisodeGRPOPackedDataset,
    collate_episode_grpo,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.model.dual_lora import (
    DualLoRAController,
    inject_top_layer_dual_lora,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.pretrain_megatron import (
    audit_stage_a_handoff,
    install_family_sampler,
    install_lr_overrides,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.schedule import (
    OneFamilyCoverageSchedule,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


METRIC_NAMES = (
    "loss/total",
    "loss/policy",
    "loss/reference_kl",
    "loss/phase3_replay",
    "loss/reference_anchor",
    "diagnostic/rl_tokens",
    "diagnostic/replay_tokens",
    "diagnostic/ratio_mean",
    "diagnostic/ratio_clipped_fraction",
    "diagnostic/advantage_mean",
    "diagnostic/policy_update_rms",
)


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="free-running long-episode GRPO")
    group.add_argument("--episode-grpo-train", type=Path, required=True)
    group.add_argument("--episode-grpo-valid", type=Path, required=True)
    group.add_argument("--episode-whispervq-model", type=Path, required=True)
    group.add_argument("--episode-lora-rank", type=int, default=16)
    group.add_argument("--episode-lora-alpha", type=float, default=32.0)
    group.add_argument("--episode-lora-dropout", type=float, default=0.05)
    group.add_argument("--episode-top-layers", type=int, default=8)
    group.add_argument("--episode-adapter-lr", type=float, default=1e-5)
    group.add_argument("--episode-clip-epsilon", type=float, default=0.20)
    group.add_argument("--episode-kl-beta", type=float, default=0.02)
    group.add_argument("--episode-replay-weight", type=float, default=0.25)
    group.add_argument("--episode-anchor-weight", type=float, default=0.02)
    group.add_argument("--episode-smoke", action="store_true")
    return parser


def validate_args(args) -> None:
    if not bool(args.sft):
        raise ValueError("packed episode GRPO requires Megatron SFT plumbing")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("episode GRPO is restricted to TP=PP=1")
    if int(args.seq_length) != 18_000 or int(args.micro_batch_size) != 1:
        raise ValueError("validated geometry is seq=18000, MBS=1")
    if not bool(args.finetune) or not bool(args.no_load_optim) or not bool(args.no_load_rng):
        raise ValueError("fresh Phase-A handoff requires finetune/no-load-optim/no-load-rng")
    for path in (
        args.episode_grpo_train,
        args.episode_grpo_valid,
        args.episode_whispervq_model,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if not 0.0 < float(args.episode_clip_epsilon) < 1.0:
        raise ValueError("invalid PPO clip epsilon")
    if any(
        float(value) < 0.0
        for value in (
            args.episode_kl_beta,
            args.episode_replay_weight,
            args.episode_anchor_weight,
        )
    ):
        raise ValueError("regularization weights must be non-negative")
    if bool(args.episode_smoke) and not 1 <= int(args.train_iters) <= 2:
        raise ValueError("smoke is restricted to one or two updates")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    train_source = EpisodeGRPOPackedDataset(
        args.episode_grpo_train, args.seq_length
    )
    train = OneFamilyCoverageSchedule(
        train_source,
        total_samples=int(train_val_test_num_samples[0]),
        global_batch_size=int(args.global_batch_size),
        data_parallel_group_size=int(args.data_parallel_size) * int(args.micro_batch_size),
        shuffle_seed=int(args.seed),
        split="train",
        require_full_coverage=not bool(args.episode_smoke),
    )
    train.collate_fn = collate_episode_grpo
    valid = None
    if int(train_val_test_num_samples[1]) > 0:
        valid_source = EpisodeGRPOPackedDataset(
            args.episode_grpo_valid, args.seq_length
        )
        eval_global = int(getattr(args, "eval_global_batch_size", 0) or args.global_batch_size)
        valid = OneFamilyCoverageSchedule(
            valid_source,
            total_samples=int(train_val_test_num_samples[1]),
            global_batch_size=eval_global,
            data_parallel_group_size=int(args.data_parallel_size)
            * int(getattr(args, "eval_micro_batch_size", 0) or args.micro_batch_size),
            shuffle_seed=int(args.seed) + 1,
            split="valid",
            require_full_coverage=False,
        )
        valid.collate_fn = collate_episode_grpo
    runtime.print_rank_0(
        f"> free-running episode GRPO packs: train_source={len(train_source)} "
        f"train_scheduled={len(train)} valid={0 if valid is None else len(valid_source)} "
        "strict_global_shuffle=true"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _reduce(metrics: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    names = tuple(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return OrderedDict((name, values[index]) for index, name in enumerate(names))


def _selected_log_probs(**kwargs) -> torch.Tensor:
    logits, _ = kwargs["output_layer"](
        kwargs["hidden_states"],
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    labels = kwargs["labels"].reshape(-1).long()
    return F.log_softmax(logits[:, 0].float(), dim=-1).gather(1, labels[:, None]).squeeze(1)


def _policy_objective(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    batch = context["batch"]
    current = _selected_log_probs(**kwargs)
    response = batch["response_mask"].reshape(-1).float()
    replay = batch["replay_mask"].reshape(-1).float()
    old = batch["old_log_probs"].reshape(-1).float()
    advantage = batch["advantages"].reshape(-1).float()
    reference = batch["reference_log_probs"].reshape(-1).float()
    rl_denominator = response.sum().clamp_min(1.0)
    replay_denominator = replay.sum().clamp_min(1.0)
    log_ratio = (current - old).clamp(-10.0, 10.0)
    ratio = log_ratio.exp()
    epsilon = float(context["clip_epsilon"])
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    policy = -(surrogate * response).sum() / rl_denominator
    # Non-negative sampled-action KL estimator, Schulman k3.
    ref_delta = (reference - current).clamp(-10.0, 10.0)
    kl_values = ref_delta.exp() - ref_delta - 1.0
    reference_kl = (kl_values * response).sum() / rl_denominator
    replay_ce = (-(current) * replay).sum() / replay_denominator
    anchor = batch["reference_anchor"].float()
    total = (
        policy
        + float(context["kl_beta"]) * reference_kl
        + float(context["replay_weight"]) * replay_ce
        + float(context["anchor_weight"]) * anchor
    )
    metrics = _reduce(
        OrderedDict(
            (
                ("loss/total", total.detach()),
                ("loss/policy", policy.detach()),
                ("loss/reference_kl", reference_kl.detach()),
                ("loss/phase3_replay", replay_ce.detach()),
                ("loss/reference_anchor", anchor.detach()),
                ("diagnostic/rl_tokens", response.sum().detach()),
                ("diagnostic/replay_tokens", replay.sum().detach()),
                (
                    "diagnostic/ratio_mean",
                    (ratio * response).sum().detach() / rl_denominator,
                ),
                (
                    "diagnostic/ratio_clipped_fraction",
                    (((ratio - 1.0).abs() > epsilon).float() * response).sum().detach()
                    / rl_denominator,
                ),
                (
                    "diagnostic/advantage_mean",
                    (advantage * response).sum().detach() / rl_denominator,
                ),
                (
                    "diagnostic/policy_update_rms",
                    batch["policy_update_rms"].detach(),
                ),
            )
        )
    )
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("episode GRPO metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite episode GRPO objective")
    return torch.stack(values)


def attach_episode_forward(model) -> None:
    raw_forward = model.forward
    controller = model.quality_grpo_lora
    if not isinstance(controller, DualLoRAController):
        raise TypeError("missing episode policy LoRA")

    def forward_with_episode(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        padding_mask=None,
        output_processor=None,
        output_processor_context=None,
        episode_batch=None,
    ):
        if decoder_input is not None or output_processor is not None:
            raise ValueError("episode entrypoint owns decoder/output processor")
        if episode_batch is None:
            raise ValueError("missing episode sidecar batch")
        route = episode_batch["family_ids"].reshape(-1) > 0
        controller.set_active_mask(route)
        controller.snapshot_reference()
        with controller.use("reference"), torch.no_grad():
            reference = raw_forward(
                input_ids,
                position_ids,
                attention_mask,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
                output_processor=_selected_log_probs,
            )
        episode_batch["reference_log_probs"] = reference
        episode_batch["reference_anchor"] = controller.reference_anchor()
        episode_batch["policy_update_rms"] = controller.policy_update_rms()
        context = {
            "batch": episode_batch,
            "clip_epsilon": float(episode_batch["clip_epsilon"].item()),
            "kl_beta": float(episode_batch["kl_beta"].item()),
            "replay_weight": float(episode_batch["replay_weight"].item()),
            "anchor_weight": float(episode_batch["anchor_weight"].item()),
        }
        with controller.use("policy"):
            return raw_forward(
                input_ids,
                position_ids,
                attention_mask,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
                output_processor=_policy_objective,
                output_processor_context=context,
            )

    model.forward = types.MethodType(forward_with_episode, model)


def augment_model(model, args) -> dict[str, object]:
    embedding = e2e.base._embedding_weight(model)
    frontend = TrainableSharedCausalWhisperVQ(
        args.episode_whispervq_model, gradient_checkpointing=False
    ).to(device=embedding.device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    model.add_module(
        "stage_a_objective",
        StageAObjective(frontend, qwen_hidden_size=int(args.hidden_size)),
    )
    summary = inject_top_layer_dual_lora(
        model,
        top_layers=int(args.episode_top_layers),
        rank=int(args.episode_lora_rank),
        alpha=float(args.episode_lora_alpha),
        dropout=float(args.episode_lora_dropout),
    )
    attach_episode_forward(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != summary.trainable_parameters:
        raise RuntimeError("trainable scope escaped policy LoRA")
    return {
        "top_layers": [summary.first_layer, summary.last_layer],
        "targets": len(summary.module_names),
        "trainable_parameters": summary.trainable_parameters,
    }


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    from gpt_builders import gpt_builder

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("episode GRPO is restricted to TP=PP=1")
    model = gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    parameters = augment_model(model, args)
    handoff = audit_stage_a_handoff(model, args.load)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "model": "phase_a_iter381_free_running_longepisode_grpo_v1",
                    "parameters": parameters,
                    "handoff": handoff,
                    "quality_gate_controls_training": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def _cuda_batch(batch):
    from megatron.core.utils import flatten_batch_for_packed_sequences

    result = {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    primary = {
        key: result.get(key)
        for key in ("tokens", "labels", "loss_mask", "position_ids", "cu_seqlens", "max_seqlen")
    }
    primary["attention_mask"] = None
    primary["cu_seqlens_padded"] = None
    result.update(flatten_batch_for_packed_sequences(primary))
    for key in (
        "response_mask",
        "old_log_probs",
        "advantages",
        "replay_mask",
        "family_ids",
    ):
        result[key] = result[key].reshape(-1)
    return result


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index + 1]) for index, name in enumerate(METRIC_NAMES)
    )


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = _cuda_batch(next(data_iterator))
    for name, value in (
        ("clip_epsilon", args.episode_clip_epsilon),
        ("kl_beta", args.episode_kl_beta),
        ("replay_weight", args.episode_replay_weight),
        ("anchor_weight", args.episode_anchor_weight),
    ):
        batch[name] = torch.tensor(float(value), device=batch["tokens"].device)
    packed = e2e.base.build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed,
        episode_batch=batch,
    )
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_args(args)
    # Reuse the validated adapter LR tagging and synchronized random sampler;
    # no historical Megatron source is edited.
    args.joint_adapter_lr = args.episode_adapter_lr
    install_lr_overrides(args)
    install_family_sampler()
    e2e.base.install_joint_collate()
    e2e.base.install_rerun_checkpoint_compatibility()
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
