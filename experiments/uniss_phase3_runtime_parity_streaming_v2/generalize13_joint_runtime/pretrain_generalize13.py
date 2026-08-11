#!/usr/bin/env python3
"""Jointly adapt runtime text/action and v12 causal semantic microblocks.

V12 proved that a causal microblock decoder can emit diverse semantic units at
sub-second latency, but freezing the runtime text path left both seen and
held-out translations at generic one-token fragments.  V13 keeps the Phase3
base and causal frontend frozen while training the existing Qwen LoRA,
runtime policy heads, critical grammar boundaries and the semantic microblock
head in one Megatron objective.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn import functional as F

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_BOUNDARY,
    ROLE_TEXT,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.microblock import (
    _balanced_example_weights,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
    RuntimeParityGeneralize12Objective,
    V12_DIAGNOSTIC_NAMES,
    V12_TERM_NAMES,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    ObjectiveOutput,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    token_cross_entropy_values,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


REPLAY_FRACTION = 0.10
V13_EXTRA_TERMS = (
    "runtime_text_content",
    "runtime_critical_boundary",
    "runtime_action",
)
V13_EXTRA_DIAGNOSTICS = (
    "runtime_text_token_accuracy",
    "runtime_boundary_accuracy",
    "runtime_end_content_accuracy",
    "runtime_action_accuracy",
    "runtime_text_supervised_tokens",
)
V13_TERM_NAMES = (*V12_TERM_NAMES, *V13_EXTRA_TERMS)
V13_DIAGNOSTIC_NAMES = (*V12_DIAGNOSTIC_NAMES, *V13_EXTRA_DIAGNOSTICS)
V13_METRIC_NAMES = (
    *V13_TERM_NAMES,
    *V13_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)

# Preserve Phase3 with replay while giving the previously frozen runtime text
# and action routes enough mass to move.  The legacy interleaved and semantic
# CE terms remain as low-weight grammar/context regularizers.
V13_WEIGHTS = OrderedDict((name, 0.0) for name in V13_TERM_NAMES)
V13_WEIGHTS.update(
    (
        ("phase3_replay", 1.0),
        ("interleaved_trajectory", 0.25),
        ("support_ordinal", 0.05),
        ("token_safe_commit", 0.05),
        ("ar_semantic_microblock", 0.25),
        ("boundary_continuity", 0.25),
        ("microblock_semantic_content", 1.0),
        ("microblock_final_length", 0.25),
        ("microblock_continue", 0.50),
        ("runtime_text_content", 4.0),
        ("runtime_critical_boundary", 1.0),
        ("runtime_action", 1.0),
    )
)


def _masked_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    active = mask.bool()
    if not bool(active.any()):
        return anchor.detach().new_zeros(())
    return (prediction[active] == target[active]).float().mean()


class RuntimeParityGeneralize13Objective(RuntimeParityGeneralize12Objective):
    """V12 semantic decoder plus explicit runtime text/action supervision."""

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        anchor = logits.sum() * 0.0
        terms = OrderedDict(output.terms)
        for name in V13_EXTRA_TERMS:
            terms[name] = zero_term(anchor)
        diagnostics = OrderedDict(output.diagnostics)
        zero = anchor.detach().new_zeros(())
        for name in V13_EXTRA_DIAGNOSTICS:
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
        active = loss_mask > 0
        token_losses = token_cross_entropy_values(logits, labels)
        text_mask = (token_roles == ROLE_TEXT) & active
        text_weights = _balanced_example_weights(
            labels,
            text_mask,
            classes=logits.shape[-1],
            minimum=0.5,
            maximum=4.0,
        )
        text_term = values_to_term(token_losses, text_weights)

        boundary_mask = (token_roles == ROLE_BOUNDARY) & active
        boundary_weights = boundary_mask.float()
        boundary_weights = torch.where(
            labels == c.TOKEN_END_CONTENT,
            boundary_weights * 4.0,
            boundary_weights,
        )
        boundary_weights = torch.where(
            labels == c.TOKEN_END_SEMANTIC,
            boundary_weights * 2.0,
            boundary_weights,
        )
        boundary_weights = torch.where(
            labels == c.TOKEN_EOS,
            boundary_weights * 4.0,
            boundary_weights,
        )
        boundary_term = values_to_term(token_losses, boundary_weights)

        original_seq_length = int(batch["original_seq_length"].item())
        action_flat = (
            batch["action_batch"].long() * original_seq_length
            + batch["action_position"].long()
        )
        action_logits = self.action_head(hidden[action_flat])
        action_targets = batch["natural_action"].long()
        action_losses = F.cross_entropy(
            action_logits.float(), action_targets, reduction="none"
        )
        action_term = values_to_term(
            action_losses, torch.ones_like(action_losses, dtype=torch.float32)
        )

        prediction = logits.float().argmax(dim=-1)
        end_content_mask = (labels == c.TOKEN_END_CONTENT) & active
        terms = OrderedDict(output.terms)
        terms.update(
            (
                ("runtime_text_content", text_term),
                ("runtime_critical_boundary", boundary_term),
                ("runtime_action", action_term),
            )
        )
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics.update(
            (
                (
                    "runtime_text_token_accuracy",
                    _masked_accuracy(prediction, labels, text_mask, logits),
                ),
                (
                    "runtime_boundary_accuracy",
                    _masked_accuracy(prediction, labels, boundary_mask, logits),
                ),
                (
                    "runtime_end_content_accuracy",
                    _masked_accuracy(prediction, labels, end_content_mask, logits),
                ),
                (
                    "runtime_action_accuracy",
                    (action_logits.float().argmax(dim=-1) == action_targets)
                    .float()
                    .mean(),
                ),
                (
                    "runtime_text_supervised_tokens",
                    text_mask.sum().detach().float(),
                ),
            )
        )
        return ObjectiveOutput(terms, diagnostics)


def distributed_generalize13_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != V13_TERM_NAMES:
        raise ValueError("generalize13 objective term order changed")
    if tuple(output.diagnostics) != V13_DIAGNOSTIC_NAMES:
        raise ValueError("generalize13 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V13_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V13_TERM_NAMES]
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
        local_means * numerators.new_tensor(list(V13_WEIGHTS.values()))
    ).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V13_TERM_NAMES)
    )
    diagnostic_values = torch.stack(
        [output.diagnostics[name].detach().float() for name in V13_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostic_values)
        diagnostic_values /= dist.get_world_size()
    metrics.update(
        (name, diagnostic_values[index])
        for index, name in enumerate(V13_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_zeros(())
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(REPLAY_FRACTION)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_zeros(())
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
        raise ValueError("generalize13 TP=PP=1 expects flattened [tokens,1,*]")
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
            f"unknown generalize13 sample kind: {context['sample_kind']}"
        )
    total, metrics = distributed_generalize13_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != V13_METRIC_NAMES:
        raise AssertionError("generalize13 metric order changed")
    values = (total.float(), *[metrics[name].float() for name in V13_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite generalize13 loss component")
    return torch.stack(values)


def is_generalize13_trainable_parameter(name: str) -> bool:
    if "true_subsecond_lora." in name:
        return True
    return any(
        value in name
        for value in (
            "true_subsecond_objective.support_head.",
            "true_subsecond_objective.action_head.",
            "true_subsecond_objective.safe_commit_head.",
            "true_subsecond_objective.semantic_microblock_head.",
        )
    )


def freeze_base_and_train_runtime_joint() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = is_generalize13_trainable_parameter(name)
            parameter.requires_grad_(keep)
            if keep:
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("generalize13 found no joint runtime parameters")
        model._generalize13_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = RuntimeParityGeneralize13Objective
    dense._distributed_dense_objective = distributed_generalize13_objective
    dense._dense_output_processor = dense_output_processor
    dense.METRIC_NAMES = V13_METRIC_NAMES
    dense.base.METRIC_NAMES = V13_METRIC_NAMES
    dense.JointValidationDataset = SynchronizedValidationDataset
    freeze_base_and_train_runtime_joint()
    dense.main()


if __name__ == "__main__":
    main()
