#!/usr/bin/env python3
"""Native-Megatron event-level constrained GRPO from one fresh rollout round."""

from __future__ import annotations

import argparse
import json
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
from training import constants_uniss as c
from training.pretrain_uniss_megatron import load_megatron_runtime


METRICS = (
    "loss/total",
    "loss/policy",
    "loss/reference_kl",
    "loss/phase3_replay",
    "loss/reference_anchor",
    "diagnostic/rl_tokens",
    "diagnostic/control_tokens",
    "diagnostic/mt_tokens",
    "diagnostic/tts_tokens",
    "diagnostic/replay_tokens",
    "diagnostic/ratio_mean",
    "diagnostic/ratio_clipped_fraction",
    "diagnostic/control_ratio_clipped_fraction",
    "diagnostic/advantage_mean",
    "diagnostic/control_advantage_mean",
    "diagnostic/policy_update_rms",
)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("event-level constrained GRPO")
    group.add_argument("--event-grpo-train", type=Path, required=True)
    group.add_argument("--event-grpo-valid", type=Path, required=True)
    group.add_argument("--event-whispervq-model", type=Path, required=True)
    group.add_argument("--event-lora-rank", type=int, default=16)
    group.add_argument("--event-lora-alpha", type=float, default=32.0)
    group.add_argument("--event-lora-dropout", type=float, default=0.05)
    group.add_argument("--event-top-layers", type=int, default=8)
    group.add_argument("--event-adapter-lr", type=float, default=5e-6)
    group.add_argument("--event-clip-epsilon", type=float, default=0.20)
    group.add_argument("--event-kl-beta", type=float, default=0.03)
    group.add_argument("--event-replay-weight", type=float, default=0.50)
    group.add_argument("--event-anchor-weight", type=float, default=0.03)
    group.add_argument("--event-smoke", action="store_true")
    return parser


def validate(args) -> None:
    if not bool(args.sft):
        raise ValueError("event GRPO requires Megatron SFT plumbing")
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("event GRPO is validated only for TP=PP=1")
    if int(args.seq_length) != 18_000 or int(args.micro_batch_size) != 1:
        raise ValueError("validated geometry is seq=18000, MBS=1")
    if not bool(args.finetune) or not bool(args.no_load_optim) or not bool(args.no_load_rng):
        raise ValueError("fresh round requires finetune/no-load-optim/no-load-rng")
    for path in (args.event_grpo_train, args.event_grpo_valid, args.event_whispervq_model):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if not 0.0 < float(args.event_clip_epsilon) < 1.0:
        raise ValueError("invalid PPO clip epsilon")
    if any(
        float(value) < 0
        for value in (args.event_kl_beta, args.event_replay_weight, args.event_anchor_weight)
    ):
        raise ValueError("regularization weights must be non-negative")
    if bool(args.event_smoke) and not 1 <= int(args.train_iters) <= 2:
        raise ValueError("event GRPO smoke supports one or two updates")


def datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    train_source = EpisodeGRPOPackedDataset(args.event_grpo_train, args.seq_length)
    train = OneFamilyCoverageSchedule(
        train_source,
        total_samples=int(train_val_test_num_samples[0]),
        global_batch_size=int(args.global_batch_size),
        data_parallel_group_size=int(args.data_parallel_size) * int(args.micro_batch_size),
        shuffle_seed=int(args.seed),
        split="train",
        require_full_coverage=not bool(args.event_smoke),
    )
    train.collate_fn = collate_episode_grpo
    valid_source = EpisodeGRPOPackedDataset(args.event_grpo_valid, args.seq_length)
    valid = OneFamilyCoverageSchedule(
        valid_source,
        total_samples=int(train_val_test_num_samples[1]),
        global_batch_size=int(getattr(args, "eval_global_batch_size", 0) or args.global_batch_size),
        data_parallel_group_size=int(args.data_parallel_size)
        * int(getattr(args, "eval_micro_batch_size", 0) or args.micro_batch_size),
        shuffle_seed=int(args.seed) + 1,
        split="valid",
        require_full_coverage=False,
    )
    valid.collate_fn = collate_episode_grpo
    runtime.print_rank_0(
        f"> event GRPO packs: train={len(train_source)} valid={len(valid_source)} "
        "strict_global_shuffle=true fresh_round_epochs=1"
    )
    return train, valid, None


datasets_provider.is_distributed = True


def reduce(metrics: OrderedDict[str, torch.Tensor]):
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    names = tuple(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return OrderedDict((name, values[index]) for index, name in enumerate(names))


def policy_log_probs(**kwargs):
    logits, _ = kwargs["output_layer"](
        kwargs["hidden_states"],
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    labels = kwargs["labels"].reshape(-1).long()
    selected = F.log_softmax(logits[:, 0].float(), dim=-1).gather(1, labels[:, None]).squeeze(1)
    batch = kwargs["context"]["batch"]
    family = batch["family_ids"].reshape(-1).long()
    control = (family == 4) & (batch["response_mask"].reshape(-1) > 0)
    if bool(control.any()):
        control_labels = labels[control]
        valid = (control_labels == c.TOKEN_WAIT_READ) | (
            control_labels == c.TOKEN_WRITE_GENERATE
        )
        if not bool(valid.all()):
            raise ValueError("control response label is not WAIT/WRITE")
        pair = logits[control, 0].float()[:, [c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE]]
        pair_target = (control_labels == c.TOKEN_WRITE_GENERATE).long()
        pair_selected = F.log_softmax(pair, dim=-1).gather(1, pair_target[:, None]).squeeze(1)
        selected = selected.clone()
        selected[control] = pair_selected
    return selected


def objective(**kwargs):
    context = kwargs["context"]
    batch = context["batch"]
    current = policy_log_probs(**kwargs)
    response = batch["response_mask"].reshape(-1).float()
    replay = batch["replay_mask"].reshape(-1).float()
    old = batch["old_log_probs"].reshape(-1).float()
    advantage = batch["advantages"].reshape(-1).float()
    reference = batch["reference_log_probs"].reshape(-1).float()
    family = batch["family_ids"].reshape(-1).long()
    rl_denom = response.sum().clamp_min(1.0)
    replay_denom = replay.sum().clamp_min(1.0)
    log_ratio = (current - old).clamp(-10.0, 10.0)
    ratio = log_ratio.exp()
    epsilon = float(context["clip_epsilon"])
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    policy = -(surrogate * response).sum() / rl_denom
    ref_delta = (reference - current).clamp(-10.0, 10.0)
    kl_values = ref_delta.exp() - ref_delta - 1.0
    reference_kl = (kl_values * response).sum() / rl_denom
    replay_ce = (-(current) * replay).sum() / replay_denom
    anchor = batch["reference_anchor"].float()
    total = (
        policy
        + float(context["kl_beta"]) * reference_kl
        + float(context["replay_weight"]) * replay_ce
        + float(context["anchor_weight"]) * anchor
    )

    def count(kind: int):
        return (response * (family == kind).float()).sum()

    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    control_mask = response * (family == 4).float()
    clipped_mask = ((ratio - 1.0).abs() > epsilon).float()
    metrics = reduce(
        OrderedDict(
            (
                ("loss/total", total.detach()),
                ("loss/policy", policy.detach()),
                ("loss/reference_kl", reference_kl.detach()),
                ("loss/phase3_replay", replay_ce.detach()),
                ("loss/reference_anchor", anchor.detach()),
                ("diagnostic/rl_tokens", response.sum().detach()),
                ("diagnostic/control_tokens", count(4).detach()),
                ("diagnostic/mt_tokens", count(2).detach()),
                ("diagnostic/tts_tokens", count(3).detach()),
                ("diagnostic/replay_tokens", replay.sum().detach()),
                ("diagnostic/ratio_mean", masked_mean(ratio, response).detach()),
                (
                    "diagnostic/ratio_clipped_fraction",
                    masked_mean(clipped_mask, response).detach(),
                ),
                (
                    "diagnostic/control_ratio_clipped_fraction",
                    masked_mean(clipped_mask, control_mask).detach(),
                ),
                ("diagnostic/advantage_mean", masked_mean(advantage, response).detach()),
                (
                    "diagnostic/control_advantage_mean",
                    masked_mean(advantage, control_mask).detach(),
                ),
                ("diagnostic/policy_update_rms", batch["policy_update_rms"].detach()),
            )
        )
    )
    if tuple(metrics) != METRICS:
        raise AssertionError("event GRPO metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRICS])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite event GRPO objective")
    return torch.stack(values)


def attach_forward(model) -> None:
    raw_forward = model.forward
    controller = model.quality_grpo_lora
    if not isinstance(controller, DualLoRAController):
        raise TypeError("missing event GRPO LoRA")
    round_reference_ready = False

    def forward_with_event(
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
        event_batch=None,
    ):
        nonlocal round_reference_ready
        if decoder_input is not None or output_processor is not None or event_batch is None:
            raise ValueError("event GRPO entrypoint owns decoder/output processor")
        route = event_batch["family_ids"].reshape(-1) > 0
        controller.set_active_mask(route)
        # A DCP handoff also restores the previous run's reference_ready flag.
        # Every fresh rollout round must compare against the policy that was
        # loaded at the start of *this* round, not against the stale Phase-A or
        # warm-up snapshot stored in the checkpoint.
        if not round_reference_ready:
            controller.reference_ready.fill_(False)
            controller.snapshot_reference()
            round_reference_ready = True
        reference_context = {"batch": event_batch}
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
                output_processor=policy_log_probs,
                output_processor_context=reference_context,
            )
        event_batch["reference_log_probs"] = reference
        event_batch["reference_anchor"] = controller.reference_anchor()
        event_batch["policy_update_rms"] = controller.policy_update_rms()
        context = {
            "batch": event_batch,
            "clip_epsilon": float(event_batch["clip_epsilon"].item()),
            "kl_beta": float(event_batch["kl_beta"].item()),
            "replay_weight": float(event_batch["replay_weight"].item()),
            "anchor_weight": float(event_batch["anchor_weight"].item()),
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
                output_processor=objective,
                output_processor_context=context,
            )

    model.forward = types.MethodType(forward_with_event, model)


def augment_model(model, args):
    embedding = e2e.base._embedding_weight(model)
    frontend = TrainableSharedCausalWhisperVQ(
        args.event_whispervq_model, gradient_checkpointing=False
    ).to(device=embedding.device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    model.add_module(
        "stage_a_objective", StageAObjective(frontend, qwen_hidden_size=int(args.hidden_size))
    )
    summary = inject_top_layer_dual_lora(
        model,
        top_layers=int(args.event_top_layers),
        rank=int(args.event_lora_rank),
        alpha=float(args.event_lora_alpha),
        dropout=float(args.event_lora_dropout),
    )
    attach_forward(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != summary.trainable_parameters:
        raise RuntimeError("trainable scope escaped event GRPO LoRA")
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
        raise ValueError("event GRPO is restricted to TP=PP=1")
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
                    "model": "phase_a_coverage_constrained_fresh_grpo_v3",
                    "parameters": parameters,
                    "handoff": handoff,
                    "asr_adapter_route": "disabled",
                    "fresh_rollout_epochs": 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return model


def cuda_batch(batch):
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
    for key in ("response_mask", "old_log_probs", "advantages", "replay_mask", "family_ids"):
        result[key] = result[key].reshape(-1)
    return result


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict(
        (name, output_tensor[index + 1]) for index, name in enumerate(METRICS)
    )


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = cuda_batch(next(data_iterator))
    for name, value in (
        ("clip_epsilon", args.event_clip_epsilon),
        ("kl_beta", args.event_kl_beta),
        ("replay_weight", args.event_replay_weight),
        ("anchor_weight", args.event_anchor_weight),
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
        event_batch=batch,
    )
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate(args)
    args.joint_adapter_lr = args.event_adapter_lr
    install_lr_overrides(args)
    install_family_sampler()
    e2e.base.install_joint_collate()
    e2e.base.install_rerun_checkpoint_compatibility()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(
        full_config,
        datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )


if __name__ == "__main__":
    main()
