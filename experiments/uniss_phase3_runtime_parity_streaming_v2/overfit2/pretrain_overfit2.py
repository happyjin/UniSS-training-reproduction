#!/usr/bin/env python3
"""Megatron overfit v2 with explicit content, semantic, and STOP weighting.

This entrypoint intentionally reuses the proven dense-aligned Megatron data,
sampler, checkpoint, and forward path.  Only the isolated capability-overfit
objective differs: dropout is controlled by the launch script, rare text and
terminal boundaries receive enough mass, and unrelated curriculum terms no
longer dilute the one-session grammar audit.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn import functional as F

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    ObjectiveOutput,
    TrueSubsecondObjective,
    _binary_metrics,
    _parameter_anchor,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    grouped_deadline_survival_term,
    token_cross_entropy_values,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


OVERFIT_WEIGHTS = OrderedDict(
    (
        ("phase3_replay", 0.25),
        ("interleaved_trajectory", 2.0),
        ("real_prefix_kd", 0.0),
        ("support_ordinal", 0.10),
        ("token_safe_commit", 0.10),
        ("deadline_survival", 0.0),
        ("prefix_stability", 0.0),
        ("ar_semantic_microblock", 2.0),
        ("speaker_consistency", 0.0),
        ("boundary_continuity", 4.0),
    )
)


def trajectory_token_weights(
    labels: torch.Tensor,
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return main-CE and boundary-CE weights for the strict overfit gate."""

    active = (loss_mask > 0).float()
    main = torch.zeros_like(loss_mask, dtype=torch.float32)
    main = torch.where(token_roles == ROLE_ACTION, 4.0, main)
    main = torch.where(token_roles == ROLE_TEXT, 8.0, main)
    main = torch.where(token_roles == ROLE_SEMANTIC, 2.0, main)
    main = torch.where(token_roles == ROLE_BOUNDARY, 2.0, main)
    main = main * active

    boundary = (token_roles == ROLE_BOUNDARY).float() * active * 2.0
    boundary = torch.where(labels == c.TOKEN_END_CONTENT, 4.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_END_SEMANTIC, 6.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_EOS, 12.0 * active, boundary)
    return main, boundary


class RuntimeParityOverfit2Objective(TrueSubsecondObjective):
    """One-sample diagnostic objective; not a replacement for canary training."""

    def trajectory(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        token_roles: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        batch,
        *,
        frontend_residual_rms: torch.Tensor,
    ) -> ObjectiveOutput:
        anchor = _parameter_anchor(self, logits)
        main_weights, boundary_weights = trajectory_token_weights(
            labels, token_roles, loss_mask
        )
        token_losses = token_cross_entropy_values(logits, labels)
        lm_trajectory = values_to_term(token_losses, main_weights)
        semantic = values_to_term(
            token_losses,
            (token_roles == ROLE_SEMANTIC).float() * (loss_mask > 0),
        )
        boundary = values_to_term(token_losses, boundary_weights)

        original_seq_length = int(batch["original_seq_length"].item())
        action_flat = (
            batch["action_batch"].long() * original_seq_length
            + batch["action_position"].long()
        )
        source_summary = hidden[action_flat]
        support_logits = self.support_head(source_summary)
        support_losses = F.cross_entropy(
            support_logits.float(), batch["support_bucket"].long(), reduction="none"
        )
        support = values_to_term(support_losses, torch.ones_like(support_losses))
        action_logits = self.action_head(source_summary)
        action_losses = F.cross_entropy(
            action_logits.float(),
            batch["natural_action"].long(),
            weight=action_logits.new_tensor([1.0, self.action_write_weight]).float(),
            reduction="none",
        )
        action = values_to_term(action_losses, torch.ones_like(action_losses))
        interleaved = LossTerm(
            lm_trajectory.mean + action.mean,
            lm_trajectory.denominator.new_ones(()),
        )

        target_ids = batch["translation_ids"].long()
        target_hidden = F.embedding(target_ids, word_embedding_weight)
        safe_logits = self.safe_commit_head(source_summary, target_hidden)
        safe_mask = batch["translation_mask"].bool()
        safe_losses = F.binary_cross_entropy_with_logits(
            safe_logits.float(), batch["safe_commit_targets"].float(), reduction="none"
        )
        probability = torch.sigmoid(safe_logits.float())
        target = batch["safe_commit_targets"].float()
        pt = torch.where(target > 0.5, probability, 1.0 - probability)
        safe_alpha = torch.where(
            target > 0.5,
            torch.full_like(target, self.safe_positive_alpha),
            torch.full_like(target, 1.0 - self.safe_positive_alpha),
        )
        safe = values_to_term(safe_alpha * (1.0 - pt).square() * safe_losses, safe_mask)
        deadline = grouped_deadline_survival_term(
            action_logits,
            batch["sample_group"],
            batch["chunk_end_ms"],
            batch["soft_deadline_ms"],
            batch["hard_deadline_ms"],
            batch.get("deadline_loss_enabled"),
        )
        kd, stability = self._teacher_terms(logits, batch, anchor)
        terms = OrderedDict(
            (
                ("phase3_replay", zero_term(anchor)),
                ("interleaved_trajectory", interleaved),
                ("real_prefix_kd", kd),
                ("support_ordinal", support),
                ("token_safe_commit", safe),
                ("deadline_survival", deadline),
                ("prefix_stability", stability),
                ("ar_semantic_microblock", semantic),
                ("speaker_consistency", zero_term(anchor)),
                ("boundary_continuity", boundary),
            )
        )
        support_prediction = support_logits.argmax(dim=-1)
        precision, recall, f1 = _binary_metrics(
            safe_logits, batch["safe_commit_targets"], safe_mask
        )
        diagnostics = OrderedDict(
            (
                ("support_accuracy", (support_prediction == batch["support_bucket"]).float().mean()),
                ("support_mae", (support_prediction - batch["support_bucket"]).abs().float().mean()),
                ("safe_commit_precision", precision),
                ("safe_commit_recall", recall),
                ("safe_commit_f1", f1),
                ("predicted_write_fraction", (action_logits.argmax(dim=-1) == 1).float().mean()),
                ("natural_write_fraction", batch["natural_action"].float().mean()),
                ("deadline_forced_fraction", batch["deadline_forced"].float().mean()),
                ("frontend_residual_rms", frontend_residual_rms.float()),
                ("supervised_tokens", main_weights.sum().detach().float()),
            )
        )
        return ObjectiveOutput(terms, diagnostics)


def distributed_overfit2_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != TERM_NAMES:
        raise ValueError("overfit2 objective term order changed")
    numerators = torch.stack([output.terms[name].numerator for name in TERM_NAMES])
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in TERM_NAMES]
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
    total = (local_means * numerators.new_tensor(list(OVERFIT_WEIGHTS.values()))).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(TERM_NAMES)
    )
    diagnostics = torch.stack(
        [value.detach().float() for value in output.diagnostics.values()]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index]) for index, name in enumerate(DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_zeros(())
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(0.10)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_ones(())
    return total, metrics


def main() -> None:
    dense.base.TrueSubsecondObjective = RuntimeParityOverfit2Objective
    dense._distributed_dense_objective = distributed_overfit2_objective
    dense.main()


if __name__ == "__main__":
    main()
