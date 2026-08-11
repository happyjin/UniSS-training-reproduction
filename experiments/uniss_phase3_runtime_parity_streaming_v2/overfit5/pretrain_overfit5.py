#!/usr/bin/env python3
"""Megatron overfit v5 with natural-length parallel semantic microblocks."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.semantic_block import (
    ParallelSemanticBlockHead,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    ObjectiveOutput,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    zero_term,
)


V5_TERM_NAMES = (*TERM_NAMES, "nar_semantic_block")
V5_DIAGNOSTIC_NAMES = (
    *DIAGNOSTIC_NAMES,
    "nar_semantic_accuracy",
    "nar_semantic_end_accuracy",
    "nar_semantic_length_mae",
    "nar_semantic_blocks",
)
V5_METRIC_NAMES = (
    *V5_TERM_NAMES,
    *V5_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)
V5_WEIGHTS = OrderedDict(v2.OVERFIT_WEIGHTS)
V5_WEIGHTS["boundary_continuity"] = 0.5
V5_WEIGHTS["ar_semantic_microblock"] = 1.0
V5_WEIGHTS["nar_semantic_block"] = 4.0


class RuntimeParityOverfit5Objective(v2.RuntimeParityOverfit2Objective):
    """Preserve the proven v4 grammar and add one parallel semantic head."""

    def __init__(self, hidden_size: int, codebook_weight: torch.Tensor, **kwargs) -> None:
        super().__init__(hidden_size, codebook_weight, **kwargs)
        self.semantic_block_head = ParallelSemanticBlockHead(
            hidden_size, maximum_semantic_tokens=24
        )

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        anchor = logits.sum() * 0.0
        for parameter in self.semantic_block_head.parameters():
            anchor = anchor + parameter.reshape(-1)[0] * 0.0
        terms = OrderedDict(output.terms)
        terms["nar_semantic_block"] = zero_term(anchor)
        diagnostics = OrderedDict(output.diagnostics)
        zero = anchor.detach().new_zeros(())
        for name in V5_DIAGNOSTIC_NAMES[len(DIAGNOSTIC_NAMES) :]:
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
        block = self.semantic_block_head.training_output(
            hidden, labels, token_roles, loss_mask, word_embedding_weight
        )
        terms = OrderedDict(output.terms)
        terms["nar_semantic_block"] = block.term
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics.update(
            (
                ("nar_semantic_accuracy", block.token_accuracy),
                ("nar_semantic_end_accuracy", block.end_accuracy),
                ("nar_semantic_length_mae", block.length_mae),
                ("nar_semantic_blocks", block.blocks),
            )
        )
        return ObjectiveOutput(terms, diagnostics)


def distributed_overfit5_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != V5_TERM_NAMES:
        raise ValueError("overfit5 objective term order changed")
    if tuple(output.diagnostics) != V5_DIAGNOSTIC_NAMES:
        raise ValueError("overfit5 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V5_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V5_TERM_NAMES]
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
    total = (local_means * numerators.new_tensor(list(V5_WEIGHTS.values()))).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V5_TERM_NAMES)
    )
    diagnostics = torch.stack(
        [output.diagnostics[name].detach().float() for name in V5_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index])
        for index, name in enumerate(V5_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_zeros(())
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(0.10)
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
        raise ValueError("overfit5 TP=PP=1 expects flattened [tokens,1,*]")
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
        raise ValueError(f"unknown overfit5 sample kind: {context['sample_kind']}")
    total, metrics = distributed_overfit5_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != V5_METRIC_NAMES:
        raise AssertionError("overfit5 metric order changed")
    values = (total.float(), *[metrics[name].float() for name in V5_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite overfit5 loss component")
    return torch.stack(values)


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = RuntimeParityOverfit5Objective
    dense._distributed_dense_objective = distributed_overfit5_objective
    dense._dense_output_processor = dense_output_processor
    dense.METRIC_NAMES = V5_METRIC_NAMES
    dense.base.METRIC_NAMES = V5_METRIC_NAMES
    dense.main()


if __name__ == "__main__":
    main()

