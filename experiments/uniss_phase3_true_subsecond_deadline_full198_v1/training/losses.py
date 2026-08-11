"""Normalized losses for the single-run replay/trajectory objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossTerm:
    """Unreduced numerator/denominator for correct DP normalization."""

    numerator: torch.Tensor
    denominator: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.numerator / self.denominator.to(self.numerator.dtype).clamp_min(1.0)


def zero_term(anchor: torch.Tensor) -> LossTerm:
    zero = anchor.sum() * 0.0
    return LossTerm(zero, zero.detach().new_zeros(()))


def values_to_term(values: torch.Tensor, mask: torch.Tensor) -> LossTerm:
    mask = mask.to(dtype=values.dtype)
    return LossTerm((values * mask).sum(), mask.sum())


def token_cross_entropy_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> LossTerm:
    """Return next-token CE normalized by the explicit supervision weights."""

    if logits.shape[:-1] != labels.shape or labels.shape != weights.shape:
        raise ValueError("token CE tensors have incompatible shapes")
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.long().reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    return values_to_term(losses, weights)


def topk_teacher_kl_term(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.5,
) -> LossTerm:
    """Forward KL on cached teacher top-k probabilities.

    Teacher probabilities are already normalized within the cached top-k. The
    student denominator remains the complete vocabulary, so probability mass
    outside the teacher candidates is penalized instead of silently ignored.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if teacher_indices.shape != teacher_probabilities.shape:
        raise ValueError("teacher top-k arrays must have identical shapes")
    if student_logits.shape[:-1] != teacher_indices.shape[:-1]:
        raise ValueError("student/teacher position geometry differs")
    if mask.shape != teacher_indices.shape[:-1]:
        raise ValueError("teacher mask geometry differs")
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    selected_log = student_log.gather(-1, teacher_indices.long())
    teacher = teacher_probabilities.float().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    divergence = (teacher * (teacher.log() - selected_log)).sum(dim=-1)
    return values_to_term(divergence * (temperature**2), mask)


def restricted_symmetric_topk_term(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> LossTerm:
    """Symmetric KL on the teacher candidate support for stability training."""

    selected = student_logits.float().gather(-1, teacher_indices.long())
    student_log = F.log_softmax(selected, dim=-1)
    student = student_log.exp()
    teacher = teacher_probabilities.float().clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    teacher_log = teacher.log()
    forward = (student * (student_log - teacher_log)).sum(dim=-1)
    backward = (teacher * (teacher_log - student_log)).sum(dim=-1)
    return values_to_term(0.5 * (forward + backward), mask)


def grouped_deadline_survival_term(
    action_logits: torch.Tensor,
    sample_group: torch.Tensor,
    chunk_end_ms: torch.Tensor,
    soft_deadline_ms: torch.Tensor,
    hard_deadline_ms: torch.Tensor,
    enabled: torch.Tensor | None = None,
    *,
    soft_weight: float = 0.7,
    hard_weight: float = 1.0,
    eps: float = 1e-6,
) -> LossTerm:
    """Survival objective over all observed ticks belonging to one utterance."""

    if action_logits.ndim != 2 or action_logits.shape[-1] != 2:
        raise ValueError("action_logits must be [N,2]")
    count = action_logits.shape[0]
    values = (sample_group, chunk_end_ms, soft_deadline_ms, hard_deadline_ms)
    if any(value.shape != (count,) for value in values):
        raise ValueError("deadline metadata must have shape [N]")
    if enabled is None:
        enabled = torch.ones(count, dtype=torch.bool, device=action_logits.device)
    elif enabled.shape != (count,):
        raise ValueError("deadline enabled mask must have shape [N]")
    else:
        enabled = enabled.bool()
    write_probability = action_logits.float().softmax(dim=-1)[:, 1].clamp(eps, 1 - eps)
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for group in torch.unique(sample_group):
        rows = torch.nonzero(sample_group == group, as_tuple=False).flatten()
        rows = rows[enabled[rows]]
        if not len(rows):
            continue
        for deadline_values, weight in (
            (soft_deadline_ms, soft_weight),
            (hard_deadline_ms, hard_weight),
        ):
            deadline = int(deadline_values[rows[0]].item())
            active = rows[chunk_end_ms[rows] <= deadline]
            if not len(active):
                continue
            survival = torch.log1p(-write_probability[active]).sum().exp()
            losses.append(-torch.log((1.0 - survival).clamp_min(eps)) * weight)
            weights.append(weight)
    if not losses:
        return zero_term(action_logits)
    return LossTerm(
        torch.stack(losses).sum(),
        action_logits.new_tensor(weights, dtype=torch.float32).sum(),
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def class_balanced_ordinal_loss(
    logits: torch.Tensor, targets: torch.Tensor, class_counts: torch.Tensor | None = None
) -> torch.Tensor:
    weight = None
    if class_counts is not None:
        weight = class_counts.float().clamp_min(1.0).rsqrt()
        weight = weight / weight.mean()
    return F.cross_entropy(logits.float(), targets.long(), weight=weight)


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    positive_alpha: float = 0.5,
) -> torch.Tensor:
    targets = targets.float()
    probability = torch.sigmoid(logits.float())
    ce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    pt = torch.where(targets > 0.5, probability, 1.0 - probability)
    alpha = torch.where(
        targets > 0.5,
        torch.full_like(targets, positive_alpha),
        torch.full_like(targets, 1.0 - positive_alpha),
    )
    return masked_mean(alpha * (1.0 - pt).pow(gamma) * ce, mask)


def deadline_survival_loss(
    write_logits: torch.Tensor,
    tick_mask: torch.Tensor,
    deadline_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Negative log probability of at least one WRITE by the deadline."""

    if write_logits.shape != tick_mask.shape or tick_mask.shape != deadline_mask.shape:
        raise ValueError("deadline tensors must have identical shapes")
    probability = torch.sigmoid(write_logits.float()).clamp(eps, 1.0 - eps)
    active = tick_mask.bool() & deadline_mask.bool()
    log_survival = torch.where(active, torch.log1p(-probability), torch.zeros_like(probability))
    survival = log_survival.sum(dim=-1).exp()
    has_deadline = active.any(dim=-1)
    losses = -torch.log((1.0 - survival).clamp_min(eps))
    return masked_mean(losses, has_deadline)


def symmetric_topk_kl(current: torch.Tensor, future: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if current.shape != future.shape:
        raise ValueError("stability logits must have identical shapes")
    current_log = F.log_softmax(current.float(), dim=-1)
    future_log = F.log_softmax(future.float(), dim=-1)
    current_probability = current_log.exp()
    future_probability = future_log.exp()
    forward = (current_probability * (current_log - future_log)).sum(dim=-1)
    backward = (future_probability * (future_log - current_log)).sum(dim=-1)
    return masked_mean(0.5 * (forward + backward), mask)


@dataclass(frozen=True)
class LossWeights:
    replay: float = 1.0
    trajectory: float = 1.0
    real_prefix_kd: float = 0.50
    support: float = 0.30
    safe_commit: float = 0.25
    deadline: float = 0.30
    stability: float = 0.20
    semantic: float = 0.50
    speaker: float = 0.05
    boundary: float = 0.05


def weighted_total(losses: Mapping[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    required = {
        "phase3_replay": weights.replay,
        "interleaved_trajectory": weights.trajectory,
        "real_prefix_kd": weights.real_prefix_kd,
        "support_ordinal": weights.support,
        "token_safe_commit": weights.safe_commit,
        "deadline_survival": weights.deadline,
        "prefix_stability": weights.stability,
        "ar_semantic_microblock": weights.semantic,
        "speaker_consistency": weights.speaker,
        "boundary_continuity": weights.boundary,
    }
    missing = sorted(set(required) - set(losses))
    if missing:
        raise KeyError(f"missing losses: {missing}")
    return sum(losses[name] * scale for name, scale in required.items())
