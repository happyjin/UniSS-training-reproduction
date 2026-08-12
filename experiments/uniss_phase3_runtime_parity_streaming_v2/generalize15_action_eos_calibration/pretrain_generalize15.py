#!/usr/bin/env python3
"""Calibrate natural WRITE precision and post-source continuation.

Generalize14 proves that deadline-aware model-prefix training can produce a
sub-second natural WRITE, but its WRITE false positives fill the persistent KV
history with hallucinated text and it selects EOS at source end before the
oracle drain trajectory is complete.  This stage freezes every content
parameter at the best Generalize14 checkpoint and trains only:

* the existing action head with extra WAIT/false-positive mass; and
* a new binary continuation head for START_GLM versus EOS.

A bounded one-pass prefix roll-in exposes both heads to imperfect text and
semantic history without allowing the corruption schedule to destroy the
content model again.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix.pretrain_generalize14 as g14
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix.prefix_rollout import (
    PrefixSchedule,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    ObjectiveOutput,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


REPLAY_FRACTION = 0.01
WAIT_FALSE_POSITIVE_WEIGHT = 2.0
EOS_CLASS_WEIGHT = 8.0
V15_EXTRA_TERMS = ("runtime_continuation",)
V15_EXTRA_DIAGNOSTICS = (
    "runtime_action_write_precision",
    "runtime_action_false_positive_rate",
    "runtime_continuation_accuracy",
    "runtime_eos_precision",
    "runtime_eos_recall",
    "runtime_predicted_eos_fraction",
    "runtime_target_eos_fraction",
)
V15_TERM_NAMES = (*g14.V14_TERM_NAMES, *V15_EXTRA_TERMS)
V15_DIAGNOSTIC_NAMES = (*g14.V14_DIAGNOSTIC_NAMES, *V15_EXTRA_DIAGNOSTICS)
V15_METRIC_NAMES = (
    *V15_TERM_NAMES,
    *V15_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)

# Content parameters are frozen.  Only policy precision, the first-WRITE
# deadline and natural drain/EOS are optimized in this isolated calibration.
V15_WEIGHTS = OrderedDict((name, 0.0) for name in V15_TERM_NAMES)
V15_WEIGHTS.update(
    (
        ("deadline_survival", 1.0),
        ("runtime_action", 2.0),
        ("runtime_continuation", 2.0),
    )
)


def calibration_prefix_schedule(progress: float) -> PrefixSchedule:
    """Keep model-prefix exposure bounded after a short clean warm-up."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.10:
        return PrefixSchedule(0, 0.0)
    return PrefixSchedule(1, 0.10)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1.0)


def continuation_supervision_mask(labels: torch.Tensor) -> torch.Tensor:
    """Supervise transcript continuation even when START_GLM is observed-role.

    Historical dense packs intentionally mask START_GLM from the vocabulary CE,
    but the new binary continuation head must see every legal next-tick marker.
    EOS remains the only positive class.
    """

    return (labels == c.TOKEN_START_GLM) | (labels == c.TOKEN_EOS)


def runtime_equivalent_action_targets(batch) -> torch.Tensor:
    """Collapse semantic-only continuation ticks into their preceding WRITE.

    Dense data emitted long target words across several wall-clock WRITE
    events.  The deployed microblock decoder instead generates all semantic
    continuation blocks inside one natural WRITE.  A later packed WRITE whose
    stable text prefix did not grow is therefore a WAIT target for the runtime
    action head, not another top-level WRITE.
    """

    natural = batch["natural_action"].long()
    previous = batch["previous_committed_length"].long()
    stable = batch["stable_target_length"].long()
    if not (natural.shape == previous.shape == stable.shape):
        raise ValueError("runtime action target tensors have different shapes")
    return natural & (stable > previous).long()


class RuntimeParityGeneralize15Objective(g14.RuntimeParityGeneralize14Objective):
    """Generalize14 content plus calibrated action and continuation heads."""

    def __init__(self, *args, **kwargs) -> None:
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None and args:
            hidden_size = args[0]
        if hidden_size is None:
            raise ValueError("generalize15 requires hidden_size")
        super().__init__(*args, **kwargs)
        self.continuation_head = nn.Linear(int(hidden_size), 2)
        nn.init.zeros_(self.continuation_head.weight)
        with torch.no_grad():
            self.continuation_head.bias.copy_(torch.tensor([1.0, -1.0]))

    def _trainable_anchor(self) -> torch.Tensor:
        values = [
            parameter.reshape(-1)[0] * 0.0
            for module in (self.action_head, self.continuation_head)
            for parameter in module.parameters()
        ]
        return torch.stack(values).sum()

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        terms = OrderedDict(output.terms)
        terms["runtime_continuation"] = zero_term(self._trainable_anchor())
        diagnostics = OrderedDict(output.diagnostics)
        zero = logits.detach().new_zeros(())
        for name in V15_EXTRA_DIAGNOSTICS:
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
        terms = OrderedDict(output.terms)
        diagnostics = OrderedDict(output.diagnostics)
        active = loss_mask > 0
        original_seq_length = int(batch["original_seq_length"].item())

        action_flat = (
            batch["action_batch"].long() * original_seq_length
            + batch["action_position"].long()
        )
        action_logits = self.action_head(hidden[action_flat])
        action_targets = runtime_equivalent_action_targets(batch)
        action_losses = F.cross_entropy(
            action_logits.float(), action_targets, reduction="none"
        )
        example_weights = torch.where(
            action_targets == 0,
            torch.full_like(action_losses, WAIT_FALSE_POSITIVE_WEIGHT),
            torch.ones_like(action_losses),
        )
        action_term = values_to_term(action_losses, example_weights)
        terms["runtime_action"] = action_term

        action_prediction = action_logits.float().argmax(dim=-1)
        predicted_write = action_prediction == 1
        target_write = action_targets == 1
        true_positive = (predicted_write & target_write).sum()
        false_positive = (predicted_write & ~target_write).sum()
        diagnostics["predicted_write_fraction"] = predicted_write.float().mean()
        diagnostics["natural_write_fraction"] = target_write.float().mean()
        diagnostics["runtime_action_accuracy"] = (
            action_prediction == action_targets
        ).float().mean()

        continuation_mask = continuation_supervision_mask(labels)
        continuation_hidden = hidden[continuation_mask]
        continuation_targets = (labels[continuation_mask] == c.TOKEN_EOS).long()
        if continuation_hidden.numel():
            continuation_logits = self.continuation_head(continuation_hidden)
            class_weight = continuation_logits.new_tensor([1.0, EOS_CLASS_WEIGHT])
            continuation_losses = F.cross_entropy(
                continuation_logits.float(),
                continuation_targets,
                weight=class_weight.float(),
                reduction="none",
            )
            continuation_term = values_to_term(
                continuation_losses,
                torch.ones_like(continuation_losses, dtype=torch.float32),
            )
            continuation_prediction = continuation_logits.float().argmax(dim=-1)
            predicted_eos = continuation_prediction == 1
            target_eos = continuation_targets == 1
            eos_true_positive = (predicted_eos & target_eos).sum()
            eos_false_positive = (predicted_eos & ~target_eos).sum()
            eos_false_negative = (~predicted_eos & target_eos).sum()
            continuation_accuracy = (
                continuation_prediction == continuation_targets
            ).float().mean()
            predicted_eos_fraction = predicted_eos.float().mean()
            target_eos_fraction = target_eos.float().mean()
        else:
            continuation_term = zero_term(self._trainable_anchor())
            zero = hidden.detach().new_zeros(())
            eos_true_positive = zero
            eos_false_positive = zero
            eos_false_negative = zero
            continuation_accuracy = zero
            predicted_eos_fraction = zero
            target_eos_fraction = zero
        terms["runtime_continuation"] = continuation_term

        diagnostics["runtime_action_write_precision"] = _ratio(
            true_positive, true_positive + false_positive
        )
        diagnostics["runtime_action_false_positive_rate"] = _ratio(
            false_positive, (~target_write).sum()
        )
        diagnostics["runtime_continuation_accuracy"] = continuation_accuracy
        diagnostics["runtime_eos_precision"] = _ratio(
            eos_true_positive, eos_true_positive + eos_false_positive
        )
        diagnostics["runtime_eos_recall"] = _ratio(
            eos_true_positive, eos_true_positive + eos_false_negative
        )
        diagnostics["runtime_predicted_eos_fraction"] = predicted_eos_fraction
        diagnostics["runtime_target_eos_fraction"] = target_eos_fraction

        if tuple(terms) != V15_TERM_NAMES:
            raise AssertionError("generalize15 trajectory term order changed")
        if tuple(diagnostics) != V15_DIAGNOSTIC_NAMES:
            raise AssertionError("generalize15 diagnostic order changed")
        return ObjectiveOutput(terms, diagnostics)


def distributed_generalize15_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != V15_TERM_NAMES:
        raise ValueError("generalize15 objective term order changed")
    if tuple(output.diagnostics) != V15_DIAGNOSTIC_NAMES:
        raise ValueError("generalize15 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V15_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V15_TERM_NAMES]
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
    total = (local_means * numerators.new_tensor(list(V15_WEIGHTS.values()))).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V15_TERM_NAMES)
    )
    diagnostic_values = torch.stack(
        [output.diagnostics[name].detach().float() for name in V15_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostic_values)
        diagnostic_values /= dist.get_world_size()
    metrics.update(
        (name, diagnostic_values[index])
        for index, name in enumerate(V15_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_tensor(
        V15_WEIGHTS["deadline_survival"]
    )
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(
        REPLAY_FRACTION
    )
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_zeros(())
    return total, metrics


def dense_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    if bool(context["batch"].get("runtime_prefix_probe", False)):
        return g14._probe_output_processor(**kwargs)
    objective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("generalize15 TP=PP=1 expects flattened [tokens,1,*]")
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
        raise ValueError(f"unknown generalize15 sample kind: {context['sample_kind']}")
    total, metrics = distributed_generalize15_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != V15_METRIC_NAMES:
        raise AssertionError("generalize15 metric order changed")
    values = (total.float(), *[metrics[name].float() for name in V15_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite generalize15 loss component")
    return torch.stack(values)


def is_generalize15_trainable_parameter(name: str) -> bool:
    return any(
        value in name
        for value in (
            "true_subsecond_objective.action_head.",
            "true_subsecond_objective.continuation_head.",
        )
    )


def freeze_content_and_train_policy_heads() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = is_generalize15_trainable_parameter(name)
            parameter.requires_grad_(keep)
            if keep:
                parameter.uniss_lr_new_heads = True
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("generalize15 found no policy-head parameters")
        model._generalize15_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    g14.prefix_schedule = calibration_prefix_schedule
    dense.base.TrueSubsecondObjective = RuntimeParityGeneralize15Objective
    dense._distributed_dense_objective = distributed_generalize15_objective
    dense._dense_output_processor = dense_output_processor
    dense.METRIC_NAMES = V15_METRIC_NAMES
    dense.base.METRIC_NAMES = V15_METRIC_NAMES
    dense.JointValidationDataset = g14.SynchronizedValidationDataset
    dense.base.forward_step = g14.forward_step
    freeze_content_and_train_policy_heads()
    dense.main()


if __name__ == "__main__":
    main()
