"""Normalized losses for the single-run replay/trajectory objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


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
