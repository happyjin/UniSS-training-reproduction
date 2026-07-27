"""Reward-v2 objectives for isolated Stage7A action-policy experiments.

The original Stage7A reward remains the default in :mod:`policy`.  This module
is opt-in and keeps every Reward-v2 experiment isolated from the reproducible
E1/E2/E3 path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RewardV2Config:
    """Interpretable, mutually-exclusive event and trajectory rewards."""

    name: str
    correct_wait: float = 0.5
    correct_write: float = 0.5
    premature_write: float = -1.5
    unnecessary_wait: float = -2.0
    final_wait: float = -5.0
    safe_commit: float = 0.5
    coverage: float = 1.0
    wait_excess: float = 0.5
    first_write_delta: float = 0.0
    mean_write_delta: float = 0.0
    write_area_delta: float = 0.0
    quality_proxy: float = 0.0
    balance_directions: bool = False


REWARD_V2_CONFIGS = {
    "r1": RewardV2Config(name="r1_rebalanced_coverage"),
    "r2": RewardV2Config(
        name="r2_explicit_latency",
        first_write_delta=1.0,
        mean_write_delta=0.75,
        write_area_delta=0.5,
    ),
    "r3": RewardV2Config(
        name="r3_bilingual_adaptive",
        coverage=1.25,
        wait_excess=0.75,
        first_write_delta=1.0,
        mean_write_delta=0.75,
        write_area_delta=0.5,
        quality_proxy=0.5,
        balance_directions=True,
    ),
}


def reward_v2_config(name: str) -> RewardV2Config:
    try:
        return REWARD_V2_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported Reward-v2 variant: {name}") from exc


def _scatter_sum(
    values: torch.Tensor, event_sample_ids: torch.Tensor, sample_count: int
) -> torch.Tensor:
    result = torch.zeros(
        (sample_count, values.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    result.index_add_(0, event_sample_ids, values)
    return result


def _balanced_event_mean(
    values: torch.Tensor,
    event_sample_ids: torch.Tensor,
    sample_target_language_ids: torch.Tensor,
) -> torch.Tensor:
    event_directions = sample_target_language_ids[event_sample_ids]
    direction_means = [
        values[event_directions == direction].mean()
        for direction in (0, 1)
        if bool((event_directions == direction).any())
    ]
    return torch.stack(direction_means).mean() if direction_means else values.mean()


def _sample_direction_weights(sample_target_language_ids: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(sample_target_language_ids, dtype=torch.float32)
    known = sample_target_language_ids >= 0
    directions = torch.unique(sample_target_language_ids[known])
    if directions.numel() < 2:
        return weights
    for direction in directions:
        mask = sample_target_language_ids == direction
        weights[mask] = float(sample_target_language_ids.numel()) / (
            float(directions.numel()) * mask.sum().float()
        )
    return weights


def rollout_rewards_v2(
    actions: torch.Tensor,
    labels: torch.Tensor,
    event_sample_ids: torch.Tensor,
    event_fractions: torch.Tensor,
    final_flags: torch.Tensor,
    *,
    sample_count: int,
    config: RewardV2Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return group rewards with explicit latency and coverage components."""

    if actions.ndim != 2:
        raise ValueError("actions must have shape [events, group]")
    group_size = actions.shape[1]
    expanded_labels = labels[:, None].expand_as(actions)
    expanded_final = final_flags[:, None].expand_as(actions)
    fractions = event_fractions[:, None].expand_as(actions)

    correct_wait = (actions == 0) & (expanded_labels == 0)
    correct_write = (actions == 1) & (expanded_labels == 1)
    premature = (actions == 1) & (expanded_labels == 0)
    final_wait = (actions == 0) & expanded_final
    unnecessary = (actions == 0) & (expanded_labels == 1) & ~expanded_final

    event_rewards = (
        correct_wait.float() * config.correct_wait
        + correct_write.float() * config.correct_write
        + premature.float() * config.premature_write
        + unnecessary.float() * config.unnecessary_wait
        + final_wait.float() * config.final_wait
        + correct_write.float() * (1.0 - fractions) * config.safe_commit
    )
    event_counts = _scatter_sum(
        torch.ones_like(actions, dtype=torch.float32), event_sample_ids, sample_count
    ).clamp_min(1.0)
    event_reward_mean = (
        _scatter_sum(event_rewards, event_sample_ids, sample_count) / event_counts
    )

    predicted_write_count = _scatter_sum(
        (actions == 1).float(), event_sample_ids, sample_count
    )
    reference_write_count = _scatter_sum(
        (expanded_labels == 1).float(), event_sample_ids, sample_count
    )
    coverage = -(
        predicted_write_count - reference_write_count
    ).abs() / reference_write_count.clamp_min(1.0)

    predicted_wait_count = event_counts - predicted_write_count
    reference_wait_count = event_counts - reference_write_count
    wait_excess = (reference_wait_count - predicted_wait_count) / event_counts

    inf = torch.full_like(fractions, 2.0)
    predicted_first_source = torch.where(actions == 1, fractions, inf)
    reference_first_source = torch.where(expanded_labels == 1, fractions, inf)
    predicted_first = torch.full(
        (sample_count, group_size),
        2.0,
        dtype=fractions.dtype,
        device=fractions.device,
    )
    reference_first = torch.full_like(predicted_first, 2.0)
    scatter_index = event_sample_ids[:, None].expand_as(actions)
    predicted_first.scatter_reduce_(
        0, scatter_index, predicted_first_source, reduce="amin", include_self=True
    )
    reference_first.scatter_reduce_(
        0, scatter_index, reference_first_source, reduce="amin", include_self=True
    )
    predicted_first = predicted_first.clamp_max(1.0)
    reference_first = reference_first.clamp_max(1.0)
    first_write_delta = (reference_first - predicted_first).clamp(-1.0, 1.0)

    predicted_write_fraction_sum = _scatter_sum(
        (actions == 1).float() * fractions, event_sample_ids, sample_count
    )
    reference_write_fraction_sum = _scatter_sum(
        (expanded_labels == 1).float() * fractions, event_sample_ids, sample_count
    )
    predicted_mean_write = torch.where(
        predicted_write_count > 0,
        predicted_write_fraction_sum / predicted_write_count.clamp_min(1.0),
        torch.ones_like(predicted_write_count),
    )
    reference_mean_write = torch.where(
        reference_write_count > 0,
        reference_write_fraction_sum / reference_write_count.clamp_min(1.0),
        torch.ones_like(reference_write_count),
    )
    mean_write_delta = (reference_mean_write - predicted_mean_write).clamp(-1.0, 1.0)

    predicted_area = _scatter_sum(
        (actions == 1).float() * (1.0 - fractions),
        event_sample_ids,
        sample_count,
    ) / reference_write_count.clamp_min(1.0)
    reference_area = _scatter_sum(
        (expanded_labels == 1).float() * (1.0 - fractions),
        event_sample_ids,
        sample_count,
    ) / reference_write_count.clamp_min(1.0)
    write_area_delta = (predicted_area - reference_area).clamp(-1.0, 1.0)

    correctness = (
        _scatter_sum(
            (actions == expanded_labels).float(), event_sample_ids, sample_count
        )
        / event_counts
    )
    rewards = (
        event_reward_mean
        + config.coverage * coverage
        + config.wait_excess * wait_excess
        + config.first_write_delta * first_write_delta
        + config.mean_write_delta * mean_write_delta
        + config.write_area_delta * write_area_delta
        + config.quality_proxy * correctness
    )
    components = {
        "event": event_reward_mean,
        "coverage": coverage,
        "wait_excess": wait_excess,
        "first_write_delta": first_write_delta,
        "mean_write_delta": mean_write_delta,
        "write_area_delta": write_area_delta,
        "quality_proxy": correctness,
        "predicted_write_count": predicted_write_count,
        "reference_write_count": reference_write_count,
    }
    return rewards, components


def grpo_action_loss_v2(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    event_sample_ids: torch.Tensor,
    event_fractions: torch.Tensor,
    final_flags: torch.Tensor,
    sample_target_language_ids: torch.Tensor,
    *,
    sample_count: int,
    group_size: int,
    kl_beta: float,
    sft_weight: float,
    config: RewardV2Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    distribution = torch.distributions.Categorical(logits=policy_logits)
    actions = distribution.sample((group_size,)).transpose(0, 1)
    rewards, components = rollout_rewards_v2(
        actions,
        labels,
        event_sample_ids,
        event_fractions,
        final_flags,
        sample_count=sample_count,
        config=config,
    )
    reward_mean = rewards.mean(dim=1, keepdim=True)
    reward_std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    advantages = (rewards - reward_mean) / reward_std

    log_probs = F.log_softmax(policy_logits, dim=-1)
    sampled_log_probs = (
        log_probs[:, None, :]
        .expand(-1, group_size, -1)
        .gather(2, actions[:, :, None])
        .squeeze(-1)
    )
    sample_log_probs = _scatter_sum(sampled_log_probs, event_sample_ids, sample_count)
    sample_counts = _scatter_sum(
        torch.ones_like(sampled_log_probs), event_sample_ids, sample_count
    ).clamp_min(1.0)
    sample_log_probs = sample_log_probs / sample_counts
    sample_weights = (
        _sample_direction_weights(sample_target_language_ids)
        if config.balance_directions
        else torch.ones(
            sample_count, dtype=policy_logits.dtype, device=policy_logits.device
        )
    )
    weighted_policy = advantages.detach() * sample_log_probs * sample_weights[:, None]
    policy_loss = -weighted_policy.sum() / (
        sample_weights.sum().clamp_min(1.0) * group_size
    )

    reference_probabilities = F.softmax(reference_logits.detach(), dim=-1)
    kl = F.kl_div(
        F.log_softmax(policy_logits, dim=-1),
        reference_probabilities,
        reduction="batchmean",
    ).clamp_min(0.0)
    event_sft = F.cross_entropy(policy_logits, labels, reduction="none")
    sft_loss = (
        _balanced_event_mean(event_sft, event_sample_ids, sample_target_language_ids)
        if config.balance_directions
        else event_sft.mean()
    )
    total = policy_loss + kl_beta * kl + sft_weight * sft_loss
    metrics = {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "sft_loss": sft_loss.detach(),
        "kl": kl.detach(),
        "kl_beta": torch.tensor(kl_beta, device=policy_logits.device),
        "reward_mean": rewards.mean().detach(),
        "reward_max": rewards.max().detach(),
        "reward_std": rewards.std(unbiased=False).detach(),
        "write_rate": (actions == 1).float().mean().detach(),
        "premature_write_rate": ((actions == 1) & (labels[:, None] == 0))
        .float()
        .mean()
        .detach(),
        "unnecessary_wait_rate": ((actions == 0) & (labels[:, None] == 1))
        .float()
        .mean()
        .detach(),
        "final_wait_rate": ((actions == 0) & final_flags[:, None])
        .float()
        .mean()
        .detach(),
        **{
            f"reward_{name}": value.mean().detach()
            for name, value in components.items()
        },
    }
    return total, metrics
