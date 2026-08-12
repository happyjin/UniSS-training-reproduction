"""Unified clean/replay/model-history objective for the one-stage run."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize13_joint_runtime.pretrain_generalize13 import (
    RuntimeParityGeneralize13Objective,
    V13_DIAGNOSTIC_NAMES,
    V13_TERM_NAMES,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    ObjectiveOutput,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


ROLLOUT_TERM_NAMES = (*V13_TERM_NAMES, "runtime_continuation")
ROLLOUT_DIAGNOSTIC_NAMES = (
    *V13_DIAGNOSTIC_NAMES,
    "runtime_continuation_accuracy",
    "runtime_eos_precision",
    "runtime_eos_recall",
    "runtime_predicted_eos_fraction",
    "runtime_target_eos_fraction",
    "event_rollout_recovery_fraction",
    "event_rollout_first_divergence",
    "event_rollout_all_wait_fraction",
    "event_rollout_false_write_fraction",
    "event_rollout_grammar_valid_fraction",
    "event_rollout_stopped_early_fraction",
)
ROLLOUT_METRIC_NAMES = (
    *ROLLOUT_TERM_NAMES,
    *ROLLOUT_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
    "curriculum_event_rollout_fraction",
)

# One optimizer/checkpoint chain.  Exact Phase3 replay remains the quality
# anchor; every streaming role remains active from the start, while the
# model-history exposure itself follows rollout_policy.rollout_schedule.
ROLLOUT_WEIGHTS = OrderedDict((name, 0.0) for name in ROLLOUT_TERM_NAMES)
ROLLOUT_WEIGHTS.update(
    (
        ("phase3_replay", 1.0),
        ("interleaved_trajectory", 0.25),
        ("support_ordinal", 0.05),
        ("token_safe_commit", 0.05),
        ("deadline_survival", 1.0),
        ("ar_semantic_microblock", 0.25),
        ("boundary_continuity", 0.25),
        ("microblock_semantic_content", 2.0),
        ("microblock_final_length", 1.0),
        ("microblock_continue", 1.0),
        ("runtime_text_content", 4.0),
        ("runtime_critical_boundary", 1.0),
        ("runtime_action", 1.0),
        ("runtime_continuation", 1.0),
    )
)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1.0)


def continuation_positions_and_targets(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ready-state hidden vectors and binary START_GLM/EOS targets."""

    if bool(batch.get("event_rollout_recovery", False)):
        sequence = int(batch["original_seq_length"].item())
        positions = (
            batch["continuation_batch"].long() * sequence
            + batch["continuation_position"].long()
        )
        return hidden[positions], batch["continuation_target"].long()
    mask = (labels == c.TOKEN_START_GLM) | (labels == c.TOKEN_EOS)
    return hidden[mask], (labels[mask] == c.TOKEN_EOS).long()


class EventRolloutJointObjective(RuntimeParityGeneralize13Objective):
    """Generalize13 content plus natural continuation and rollout diagnostics."""

    def __init__(self, *args, **kwargs) -> None:
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None and args:
            hidden_size = args[0]
        if hidden_size is None:
            raise ValueError("event-rollout objective requires hidden_size")
        super().__init__(*args, **kwargs)
        self.continuation_head = nn.Linear(int(hidden_size), 2)
        nn.init.zeros_(self.continuation_head.weight)
        with torch.no_grad():
            self.continuation_head.bias.copy_(torch.tensor([1.0, -1.0]))
        for parameter in self.continuation_head.parameters():
            parameter.uniss_lr_new_heads = True

    def _continuation_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        anchor = reference.sum() * 0.0
        for parameter in self.continuation_head.parameters():
            anchor = anchor + parameter.reshape(-1)[0] * 0.0
        return anchor

    @staticmethod
    def _rollout_diagnostics(batch, reference: torch.Tensor) -> OrderedDict:
        zero = reference.detach().new_zeros(())

        def value(name: str, default=0.0):
            raw = batch.get(name)
            if isinstance(raw, torch.Tensor):
                return raw.detach().float().mean()
            return zero.new_tensor(float(default if raw is None else raw))

        return OrderedDict(
            (
                ("event_rollout_recovery_fraction", value("event_rollout_recovery")),
                ("event_rollout_first_divergence", value("event_rollout_first_divergence", -1.0)),
                ("event_rollout_all_wait_fraction", value("event_rollout_all_wait")),
                ("event_rollout_false_write_fraction", value("event_rollout_false_write")),
                ("event_rollout_grammar_valid_fraction", value("event_rollout_grammar_valid", 1.0)),
                ("event_rollout_stopped_early_fraction", value("event_rollout_stopped_early")),
            )
        )

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        terms = OrderedDict(output.terms)
        terms["runtime_continuation"] = zero_term(
            self._continuation_anchor(logits)
        )
        diagnostics = OrderedDict(output.diagnostics)
        zero = logits.detach().new_zeros(())
        for name in ROLLOUT_DIAGNOSTIC_NAMES[len(V13_DIAGNOSTIC_NAMES) :]:
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
        continuation_hidden, continuation_targets = continuation_positions_and_targets(
            hidden, labels, batch
        )
        if continuation_hidden.numel():
            continuation_logits = self.continuation_head(continuation_hidden)
            class_weights = continuation_logits.new_tensor([1.0, 8.0])
            losses = F.cross_entropy(
                continuation_logits.float(),
                continuation_targets,
                weight=class_weights.float(),
                reduction="none",
            )
            continuation = values_to_term(losses, torch.ones_like(losses))
            prediction = continuation_logits.float().argmax(dim=-1)
            predicted_eos = prediction == 1
            target_eos = continuation_targets == 1
            true_positive = (predicted_eos & target_eos).sum()
            false_positive = (predicted_eos & ~target_eos).sum()
            false_negative = (~predicted_eos & target_eos).sum()
            continuation_accuracy = (prediction == continuation_targets).float().mean()
            eos_precision = _ratio(true_positive, true_positive + false_positive)
            eos_recall = _ratio(true_positive, true_positive + false_negative)
            predicted_eos_fraction = predicted_eos.float().mean()
            target_eos_fraction = target_eos.float().mean()
        else:
            continuation = zero_term(self._continuation_anchor(logits))
            zero = logits.detach().new_zeros(())
            continuation_accuracy = zero
            eos_precision = zero
            eos_recall = zero
            predicted_eos_fraction = zero
            target_eos_fraction = zero

        terms = OrderedDict(output.terms)
        terms["runtime_continuation"] = continuation
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics.update(
            (
                ("runtime_continuation_accuracy", continuation_accuracy),
                ("runtime_eos_precision", eos_precision),
                ("runtime_eos_recall", eos_recall),
                ("runtime_predicted_eos_fraction", predicted_eos_fraction),
                ("runtime_target_eos_fraction", target_eos_fraction),
            )
        )
        diagnostics.update(self._rollout_diagnostics(batch, logits))
        if tuple(terms) != ROLLOUT_TERM_NAMES:
            raise AssertionError("event-rollout term order changed")
        if tuple(diagnostics) != ROLLOUT_DIAGNOSTIC_NAMES:
            raise AssertionError("event-rollout diagnostic order changed")
        return ObjectiveOutput(terms, diagnostics)


def distributed_event_rollout_objective(output: ObjectiveOutput, *, progress: float):
    from experiments.uniss_phase3_event_rollout_joint_full198_v1.rollout_policy import (
        rollout_schedule,
    )

    if tuple(output.terms) != ROLLOUT_TERM_NAMES:
        raise ValueError("event-rollout objective term order changed")
    if tuple(output.diagnostics) != ROLLOUT_DIAGNOSTIC_NAMES:
        raise ValueError("event-rollout objective diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in ROLLOUT_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in ROLLOUT_TERM_NAMES]
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
    total = (local_means * numerators.new_tensor(list(ROLLOUT_WEIGHTS.values()))).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(ROLLOUT_TERM_NAMES)
    )
    values = torch.stack(
        [output.diagnostics[name].detach().float() for name in ROLLOUT_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
        values /= dist.get_world_size()
    metrics.update(
        (name, values[index]) for index, name in enumerate(ROLLOUT_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_tensor(
        ROLLOUT_WEIGHTS["deadline_survival"]
    )
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(0.35)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_ones(())
    metrics["curriculum_event_rollout_fraction"] = total.detach().new_tensor(
        rollout_schedule(progress).fraction
    )
    return total, metrics


__all__ = [
    "EventRolloutJointObjective",
    "ROLLOUT_DIAGNOSTIC_NAMES",
    "ROLLOUT_METRIC_NAMES",
    "ROLLOUT_TERM_NAMES",
    "ROLLOUT_WEIGHTS",
    "continuation_positions_and_targets",
    "distributed_event_rollout_objective",
]
