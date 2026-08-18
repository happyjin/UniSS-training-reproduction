"""Role-normalized Phase3 replay and exact-runtime AR objective for v3."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.rollout_policy import (
    rollout_schedule,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    ObjectiveOutput,
    TrueSubsecondObjective,
    _binary_metrics,
    _parameter_anchor,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    grouped_deadline_survival_term,
    token_cross_entropy_values,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


V3_TERM_NAMES = (
    "phase3_replay",
    "phase3_semantic_preservation",
    "trajectory_lm",
    "runtime_action",
    "runtime_text",
    "runtime_semantic",
    "semantic_end",
    "event_continuation",
    "support_ordinal",
    "safe_commit",
    "deadline_survival",
    "oracle_recovery",
    "rollout_consistency",
)
V3_DIAGNOSTIC_NAMES = (
    "runtime_action_accuracy",
    "runtime_write_precision",
    "runtime_write_recall",
    "runtime_write_f1",
    "runtime_text_token_accuracy",
    "runtime_semantic_token_accuracy",
    "runtime_semantic_end_accuracy",
    "runtime_continuation_accuracy",
    "runtime_eos_precision",
    "runtime_eos_recall",
    "runtime_predicted_write_fraction",
    "runtime_target_write_fraction",
    "runtime_predicted_eos_fraction",
    "runtime_target_eos_fraction",
    "safe_commit_precision",
    "safe_commit_recall",
    "safe_commit_f1",
    "support_accuracy",
    "frontend_residual_rms",
    "supervised_tokens",
    "event_rollout_recovery_fraction",
    "event_rollout_first_divergence",
    "event_rollout_all_wait_fraction",
    "event_rollout_false_write_fraction",
    "event_rollout_grammar_valid_fraction",
    "event_rollout_stopped_early_fraction",
    "event_rollout_corrupted_prefix_tokens",
    "event_rollout_action_recovery_fraction",
    "event_rollout_text_recovery_fraction",
    "event_rollout_semantic_recovery_fraction",
    "event_rollout_continuation_recovery_fraction",
)
V3_METRIC_NAMES = (
    *V3_TERM_NAMES,
    *V3_DIAGNOSTIC_NAMES,
    "curriculum_replay_fraction",
    "curriculum_event_rollout_fraction",
    "active_loss_weight_sum",
)

# These are relative contributions inside one active sample-kind group.  The
# distributed reducer divides by the sum of active weights, so a streaming
# batch cannot acquire a larger total scale merely because it exposes more
# named losses than a replay batch.
V3_WEIGHTS = OrderedDict(
    (
        ("phase3_replay", 0.85),
        ("phase3_semantic_preservation", 0.15),
        ("trajectory_lm", 0.30),
        ("runtime_action", 0.15),
        ("runtime_text", 0.08),
        ("runtime_semantic", 0.18),
        ("semantic_end", 0.06),
        ("event_continuation", 0.08),
        ("support_ordinal", 0.03),
        ("safe_commit", 0.05),
        ("deadline_survival", 0.02),
        ("oracle_recovery", 0.30),
        ("rollout_consistency", 0.05),
    )
)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1.0)


def _accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    active = mask.bool()
    if not bool(active.any()):
        return reference.detach().new_zeros(())
    return (prediction[active] == target[active]).float().mean()


def _binary_classification(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = mask.bool()
    if not bool(active.any()):
        zero = logits.detach().new_zeros(())
        return zero, zero, zero, zero
    prediction = logits.float().argmax(dim=-1)
    predicted = prediction == int(positive)
    target = targets == int(positive)
    true_positive = (predicted & target & active).sum()
    false_positive = (predicted & ~target & active).sum()
    false_negative = (~predicted & target & active).sum()
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    accuracy = (prediction[active] == targets[active]).float().mean()
    return accuracy, precision, recall, f1


class EventRolloutV3Objective(TrueSubsecondObjective):
    """Phase3-native text/semantic CE plus policy and exact recovery heads."""

    def __init__(self, *args, **kwargs) -> None:
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None and args:
            hidden_size = args[0]
        if hidden_size is None:
            raise ValueError("v3 objective requires hidden_size")
        super().__init__(*args, **kwargs)
        self.continuation_head = nn.Linear(int(hidden_size), 2)
        nn.init.zeros_(self.continuation_head.weight)
        with torch.no_grad():
            self.continuation_head.bias.copy_(torch.tensor([2.0, -2.0]))
        for parameter in self.continuation_head.parameters():
            parameter.uniss_lr_new_heads = True

    def _zero_output(self, anchor: torch.Tensor) -> ObjectiveOutput:
        terms = OrderedDict(
            (name, zero_term(anchor)) for name in V3_TERM_NAMES
        )
        zero = anchor.detach().new_zeros(())
        diagnostics = OrderedDict(
            (name, zero) for name in V3_DIAGNOSTIC_NAMES
        )
        return ObjectiveOutput(terms, diagnostics)

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        anchor = _parameter_anchor(self, logits)
        output = self._zero_output(anchor)
        losses = token_cross_entropy_values(logits, labels)
        active = loss_mask > 0
        output.terms["phase3_replay"] = values_to_term(losses, active)
        semantic = (
            (labels >= c.BICODEC_SEMANTIC_OFFSET)
            & (labels < c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE)
            & active
        )
        output.terms["phase3_semantic_preservation"] = values_to_term(
            losses, semantic
        )
        output.diagnostics["supervised_tokens"] = active.sum().detach().float()
        return output

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
        anchor = _parameter_anchor(self, logits)
        output = self._zero_output(anchor)
        active = loss_mask > 0
        losses = token_cross_entropy_values(logits, labels)
        prediction = logits.float().argmax(dim=-1)
        recovery = bool(batch.get("event_rollout_recovery", False))

        if recovery:
            output.terms["oracle_recovery"] = values_to_term(losses, active)
        else:
            role_weight = torch.zeros_like(loss_mask, dtype=torch.float32)
            role_weight = torch.where(token_roles == ROLE_ACTION, 2.0, role_weight)
            role_weight = torch.where(token_roles == ROLE_TEXT, 1.5, role_weight)
            role_weight = torch.where(token_roles == ROLE_SEMANTIC, 1.0, role_weight)
            role_weight = torch.where(token_roles == ROLE_BOUNDARY, 1.0, role_weight)
            output.terms["trajectory_lm"] = values_to_term(
                losses, role_weight * active
            )

        text_mask = (token_roles == ROLE_TEXT) & active
        semantic_mask = (token_roles == ROLE_SEMANTIC) & active
        semantic_end_mask = (labels == c.TOKEN_END_SEMANTIC) & active
        output.terms["runtime_text"] = values_to_term(losses, text_mask)
        output.terms["runtime_semantic"] = values_to_term(losses, semantic_mask)
        output.terms["semantic_end"] = values_to_term(
            losses, semantic_end_mask.float() * 2.0
        )

        sequence = int(batch["original_seq_length"].item())
        action_positions = (
            batch["action_batch"].long() * sequence
            + batch["action_position"].long()
        )
        action_hidden = hidden[action_positions]
        action_logits = self.action_head(action_hidden)
        action_targets = batch["natural_action"].long()
        action_mask = batch.get("action_supervised")
        if action_mask is None:
            action_mask = torch.ones_like(action_targets, dtype=torch.bool)
        else:
            action_mask = action_mask.bool()
        action_losses = F.cross_entropy(
            action_logits.float(),
            action_targets,
            weight=action_logits.new_tensor([1.0, self.action_write_weight]).float(),
            reduction="none",
        )
        output.terms["runtime_action"] = values_to_term(
            action_losses, action_mask
        )

        support_logits = self.support_head(action_hidden)
        support_losses = F.cross_entropy(
            support_logits.float(), batch["support_bucket"].long(), reduction="none"
        )
        output.terms["support_ordinal"] = values_to_term(
            support_losses, action_mask
        )

        target_ids = batch["translation_ids"].long()
        target_hidden = F.embedding(target_ids, word_embedding_weight)
        safe_logits = self.safe_commit_head(action_hidden, target_hidden)
        safe_mask = batch["translation_mask"].bool() & action_mask[:, None]
        safe_losses = F.binary_cross_entropy_with_logits(
            safe_logits.float(), batch["safe_commit_targets"].float(), reduction="none"
        )
        safe_probability = torch.sigmoid(safe_logits.float())
        safe_target = batch["safe_commit_targets"].float()
        pt = torch.where(
            safe_target > 0.5, safe_probability, 1.0 - safe_probability
        )
        alpha = torch.where(
            safe_target > 0.5,
            torch.full_like(safe_target, self.safe_positive_alpha),
            torch.full_like(safe_target, 1.0 - self.safe_positive_alpha),
        )
        output.terms["safe_commit"] = values_to_term(
            alpha * (1.0 - pt).square() * safe_losses, safe_mask
        )

        if bool(action_mask.any()):
            selected = action_mask.nonzero(as_tuple=False).flatten()
            output.terms["deadline_survival"] = grouped_deadline_survival_term(
                action_logits[selected],
                batch["sample_group"][selected],
                batch["chunk_end_ms"][selected],
                batch["soft_deadline_ms"][selected],
                batch["hard_deadline_ms"][selected],
                batch.get("deadline_loss_enabled")[selected]
                if batch.get("deadline_loss_enabled") is not None
                else None,
            )

        if recovery:
            continuation_positions = (
                batch["continuation_batch"].long() * sequence
                + batch["continuation_position"].long()
            )
            continuation_targets = batch["continuation_target"].long()
            continuation_mask = batch["continuation_supervised"].bool()
        else:
            continuation_mask_flat = (
                (labels == c.TOKEN_START_GLM) | (labels == c.TOKEN_EOS)
            ) & active
            continuation_positions = continuation_mask_flat.nonzero(
                as_tuple=False
            ).flatten()
            continuation_targets = (labels[continuation_positions] == c.TOKEN_EOS).long()
            continuation_mask = torch.ones_like(
                continuation_targets, dtype=torch.bool
            )
        continuation_hidden = hidden[continuation_positions]
        if continuation_hidden.numel():
            continuation_logits = self.continuation_head(continuation_hidden)
            continuation_losses = F.cross_entropy(
                continuation_logits.float(),
                continuation_targets,
                weight=continuation_logits.new_tensor([1.0, 4.0]).float(),
                reduction="none",
            )
            output.terms["event_continuation"] = values_to_term(
                continuation_losses, continuation_mask
            )
        else:
            continuation_logits = logits.new_empty((0, 2))

        action_accuracy, write_precision, write_recall, write_f1 = (
            _binary_classification(
                action_logits, action_targets, action_mask, positive=1
            )
        )
        if continuation_logits.numel():
            continuation_accuracy, eos_precision, eos_recall, _ = (
                _binary_classification(
                    continuation_logits,
                    continuation_targets,
                    continuation_mask,
                    positive=1,
                )
            )
            continuation_prediction = continuation_logits.float().argmax(dim=-1)
            predicted_eos_fraction = (
                (continuation_prediction[continuation_mask] == 1).float().mean()
                if bool(continuation_mask.any())
                else logits.detach().new_zeros(())
            )
            target_eos_fraction = (
                continuation_targets[continuation_mask].float().mean()
                if bool(continuation_mask.any())
                else logits.detach().new_zeros(())
            )
        else:
            zero = logits.detach().new_zeros(())
            continuation_accuracy = eos_precision = eos_recall = zero
            predicted_eos_fraction = target_eos_fraction = zero

        safe_precision, safe_recall, safe_f1 = _binary_metrics(
            safe_logits, batch["safe_commit_targets"], safe_mask
        )

        def value(name: str, default=0.0):
            raw = batch.get(name)
            if isinstance(raw, torch.Tensor):
                return raw.detach().float().mean()
            return logits.detach().new_tensor(float(default if raw is None else raw))

        kind = batch.get("event_rollout_divergence_kind")
        kind_values = kind.long() if isinstance(kind, torch.Tensor) else None
        zeros = logits.detach().new_zeros(())
        recovery_kind = lambda indices: (
            torch.isin(
                kind_values,
                torch.tensor(indices, device=kind_values.device),
            ).float().mean()
            if kind_values is not None and kind_values.numel()
            else zeros
        )
        output.diagnostics.update(
            (
                ("runtime_action_accuracy", action_accuracy),
                ("runtime_write_precision", write_precision),
                ("runtime_write_recall", write_recall),
                ("runtime_write_f1", write_f1),
                (
                    "runtime_text_token_accuracy",
                    _accuracy(prediction, labels, text_mask, logits),
                ),
                (
                    "runtime_semantic_token_accuracy",
                    _accuracy(prediction, labels, semantic_mask, logits),
                ),
                (
                    "runtime_semantic_end_accuracy",
                    _accuracy(prediction, labels, semantic_end_mask, logits),
                ),
                ("runtime_continuation_accuracy", continuation_accuracy),
                ("runtime_eos_precision", eos_precision),
                ("runtime_eos_recall", eos_recall),
                (
                    "runtime_predicted_write_fraction",
                    (action_logits.float().argmax(dim=-1)[action_mask] == 1)
                    .float()
                    .mean()
                    if bool(action_mask.any())
                    else zeros,
                ),
                (
                    "runtime_target_write_fraction",
                    action_targets[action_mask].float().mean()
                    if bool(action_mask.any())
                    else zeros,
                ),
                ("runtime_predicted_eos_fraction", predicted_eos_fraction),
                ("runtime_target_eos_fraction", target_eos_fraction),
                ("safe_commit_precision", safe_precision),
                ("safe_commit_recall", safe_recall),
                ("safe_commit_f1", safe_f1),
                (
                    "support_accuracy",
                    (
                        support_logits.argmax(dim=-1)[action_mask]
                        == batch["support_bucket"][action_mask]
                    )
                    .float()
                    .mean()
                    if bool(action_mask.any())
                    else zeros,
                ),
                ("frontend_residual_rms", frontend_residual_rms.float()),
                ("supervised_tokens", active.sum().detach().float()),
                ("event_rollout_recovery_fraction", value("event_rollout_recovery")),
                ("event_rollout_first_divergence", value("event_rollout_first_divergence", -1.0)),
                ("event_rollout_all_wait_fraction", value("event_rollout_all_wait")),
                ("event_rollout_false_write_fraction", value("event_rollout_false_write")),
                ("event_rollout_grammar_valid_fraction", value("event_rollout_grammar_valid", 1.0)),
                ("event_rollout_stopped_early_fraction", value("event_rollout_stopped_early")),
                ("event_rollout_corrupted_prefix_tokens", value("event_rollout_corrupted_prefix_tokens")),
                ("event_rollout_action_recovery_fraction", recovery_kind([0])),
                ("event_rollout_text_recovery_fraction", recovery_kind([1, 2])),
                ("event_rollout_semantic_recovery_fraction", recovery_kind([3, 4])),
                ("event_rollout_continuation_recovery_fraction", recovery_kind([5])),
            )
        )
        if tuple(output.terms) != V3_TERM_NAMES:
            raise AssertionError("v3 objective term order changed")
        if tuple(output.diagnostics) != V3_DIAGNOSTIC_NAMES:
            raise AssertionError("v3 diagnostic order changed")
        return output


def distributed_v3_objective(
    output: ObjectiveOutput, *, progress: float
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != V3_TERM_NAMES:
        raise ValueError("v3 objective term order changed")
    if tuple(output.diagnostics) != V3_DIAGNOSTIC_NAMES:
        raise ValueError("v3 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V3_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V3_TERM_NAMES]
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
    weights = numerators.new_tensor(list(V3_WEIGHTS.values()))
    active_weight = (weights * active).sum().clamp_min(1e-8)
    total = (local_means * weights).sum() / active_weight
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V3_TERM_NAMES)
    )
    diagnostics = torch.stack(
        [output.diagnostics[name].detach().float() for name in V3_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index])
        for index, name in enumerate(V3_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(0.35)
    metrics["curriculum_event_rollout_fraction"] = total.detach().new_tensor(
        rollout_schedule(progress).fraction
    )
    metrics["active_loss_weight_sum"] = active_weight.detach()
    return total, metrics


__all__ = [
    "EventRolloutV3Objective",
    "V3_DIAGNOSTIC_NAMES",
    "V3_METRIC_NAMES",
    "V3_TERM_NAMES",
    "V3_WEIGHTS",
    "distributed_v3_objective",
]
