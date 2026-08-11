#!/usr/bin/env python3
"""Train only the v12 causal microblock semantic head with Megatron."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.microblock import (
    CausalMicroblockSemanticHead,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    ObjectiveOutput,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    zero_term,
)


REPLAY_FRACTION = 0.01
V12_TERM_NAMES = (
    *TERM_NAMES,
    "microblock_semantic_content",
    "microblock_final_length",
    "microblock_continue",
)
V12_DIAGNOSTIC_NAMES = (
    *DIAGNOSTIC_NAMES,
    "microblock_token_accuracy",
    "microblock_first_slot_accuracy",
    "microblock_final_length_accuracy",
    "microblock_final_length_mae",
    "microblock_continue_accuracy",
    "microblock_predicted_continue_fraction",
    "microblock_target_continue_fraction",
    "microblock_predicted_unique_fraction",
    "microblock_blocks",
)
V12_METRIC_NAMES = (
    *V12_TERM_NAMES,
    *V12_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)
V12_WEIGHTS = OrderedDict((name, 0.0) for name in TERM_NAMES)
V12_WEIGHTS.update(
    (
        ("microblock_semantic_content", 1.0),
        ("microblock_final_length", 0.5),
        ("microblock_continue", 1.0),
    )
)


class RuntimeParityGeneralize12Objective(v2.RuntimeParityOverfit2Objective):
    def __init__(self, hidden_size: int, codebook_weight: torch.Tensor, **kwargs) -> None:
        super().__init__(hidden_size, codebook_weight, **kwargs)
        self.semantic_microblock_head = CausalMicroblockSemanticHead(
            hidden_size, block_size=4
        )

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        anchor = logits.sum() * 0.0
        for parameter in self.semantic_microblock_head.parameters():
            anchor = anchor + parameter.reshape(-1)[0] * 0.0
        terms = OrderedDict(output.terms)
        for name in V12_TERM_NAMES[len(TERM_NAMES) :]:
            terms[name] = zero_term(anchor)
        diagnostics = OrderedDict(output.diagnostics)
        zero = anchor.detach().new_zeros(())
        for name in V12_DIAGNOSTIC_NAMES[len(DIAGNOSTIC_NAMES) :]:
            diagnostics[name] = zero
        return ObjectiveOutput(terms, diagnostics)

    def trajectory(
        self,
        hidden,
        logits,
        labels,
        loss_mask,
        token_roles,
        word_embedding_weight,
        batch,
        *,
        frontend_residual_rms,
    ) -> ObjectiveOutput:
        output = super().trajectory(
            hidden,
            logits,
            labels,
            loss_mask,
            token_roles,
            word_embedding_weight,
            batch,
            frontend_residual_rms=frontend_residual_rms,
        )
        microblock = self.semantic_microblock_head.training_output(
            hidden, labels, token_roles, loss_mask, word_embedding_weight
        )
        terms = OrderedDict(output.terms)
        terms.update(
            (
                ("microblock_semantic_content", microblock.content_term),
                ("microblock_final_length", microblock.final_length_term),
                ("microblock_continue", microblock.continue_term),
            )
        )
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics.update(
            (
                ("microblock_token_accuracy", microblock.token_accuracy),
                ("microblock_first_slot_accuracy", microblock.first_slot_accuracy),
                ("microblock_final_length_accuracy", microblock.final_length_accuracy),
                ("microblock_final_length_mae", microblock.final_length_mae),
                ("microblock_continue_accuracy", microblock.continue_accuracy),
                (
                    "microblock_predicted_continue_fraction",
                    microblock.predicted_continue_fraction,
                ),
                (
                    "microblock_target_continue_fraction",
                    microblock.target_continue_fraction,
                ),
                (
                    "microblock_predicted_unique_fraction",
                    microblock.predicted_unique_fraction,
                ),
                ("microblock_blocks", microblock.blocks),
            )
        )
        return ObjectiveOutput(terms, diagnostics)


def distributed_generalize12_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != V12_TERM_NAMES:
        raise ValueError("generalize12 objective term order changed")
    if tuple(output.diagnostics) != V12_DIAGNOSTIC_NAMES:
        raise ValueError("generalize12 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V12_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V12_TERM_NAMES]
    )
    global_numerators = numerators.detach().clone()
    global_denominators = denominators.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_numerators)
        dist.all_reduce(global_denominators)
        world_size = dist.get_world_size()
    active = global_denominators > 0
    local_means = torch.where(
        active,
        world_size * numerators / global_denominators.clamp_min(1),
        numerators * 0.0,
    )
    total = (
        local_means * numerators.new_tensor(list(V12_WEIGHTS.values()))
    ).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V12_TERM_NAMES)
    )
    diagnostics = torch.stack(
        [output.diagnostics[name].detach().float() for name in V12_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index])
        for index, name in enumerate(V12_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_zeros(())
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(REPLAY_FRACTION)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_ones(())
    return total, metrics


def dense_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    objective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("generalize12 TP=PP=1 expects flattened [tokens,1,*]")
    hidden = hidden[:, 0]
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
    batch = context["batch"]
    if context["sample_kind"] == "replay":
        output = objective.replay(logits, labels, loss_mask)
    elif context["sample_kind"] == "trajectory":
        output = objective.trajectory(
            hidden,
            logits,
            labels,
            loss_mask,
            batch["token_roles"].reshape(-1),
            context["word_embedding_weight"],
            batch,
            frontend_residual_rms=context["frontend_residual_rms"],
        )
    else:
        raise ValueError(
            f"unknown generalize12 sample kind: {context['sample_kind']}"
        )
    total, metrics = distributed_generalize12_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != V12_METRIC_NAMES:
        raise AssertionError("generalize12 metric order changed")
    values = (total.float(), *[metrics[name].float() for name in V12_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite generalize12 loss component")
    return torch.stack(values)


def is_generalize12_trainable_parameter(name: str) -> bool:
    return "true_subsecond_objective.semantic_microblock_head." in name


def freeze_base_and_train_microblock_head() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = is_generalize12_trainable_parameter(name)
            parameter.requires_grad_(keep)
            if keep:
                parameter.uniss_lr_new_heads = True
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("generalize12 found no microblock-head parameters")
        model._generalize12_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = RuntimeParityGeneralize12Objective
    dense._distributed_dense_objective = distributed_generalize12_objective
    dense._dense_output_processor = dense_output_processor
    dense.METRIC_NAMES = V12_METRIC_NAMES
    dense.base.METRIC_NAMES = V12_METRIC_NAMES
    freeze_base_and_train_microblock_head()
    dense.main()


if __name__ == "__main__":
    main()
