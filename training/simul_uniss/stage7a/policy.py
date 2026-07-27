"""Stage6-initialized binary action head and group-relative objectives."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from training import constants_uniss as c

ACTION_IDS = (c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE)


class ActionHead(nn.Module):
    """Binary WAIT/WRITE projection initialized from the Stage6 LM head rows."""

    def __init__(self, hidden_size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 2, bias=bias)

    @classmethod
    def from_lm_head(cls, lm_head: nn.Linear) -> ActionHead:
        head = cls(lm_head.in_features, bias=False)
        with torch.no_grad():
            head.projection.weight.copy_(
                lm_head.weight[list(ACTION_IDS)].detach().float().cpu()
            )
        return head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden.float())

    def frozen_copy(self) -> ActionHead:
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad = False
        return result


@dataclass(frozen=True)
class RewardWeights:
    correct: float = 1.0
    incorrect: float = -1.0
    premature_write: float = -2.0
    unnecessary_wait: float = -0.5
    final_wait: float = -5.0
    safe_early_write: float = 0.2


DEFAULT_REWARD_WEIGHTS = RewardWeights()


def rollout_rewards(
    actions: torch.Tensor,
    labels: torch.Tensor,
    event_sample_ids: torch.Tensor,
    event_fractions: torch.Tensor,
    final_flags: torch.Tensor,
    *,
    sample_count: int,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-sample, per-group rewards for sampled action trajectories.

    ``actions`` has shape ``[events, group]``. The initial Stage7A reward uses
    pseudo-alignment action labels plus an early-safe-WRITE bonus. It is a
    deliberate action-policy proof, not a replacement for contextual alignment.
    """

    if actions.ndim != 2:
        raise ValueError("actions must have shape [events, group]")
    group_size = actions.shape[1]
    expanded_labels = labels[:, None].expand_as(actions)
    correct = torch.where(
        actions == expanded_labels,
        torch.full_like(actions, weights.correct, dtype=torch.float32),
        torch.full_like(actions, weights.incorrect, dtype=torch.float32),
    )
    premature = (
        (actions == 1) & (expanded_labels == 0)
    ).float() * weights.premature_write
    unnecessary = (
        (actions == 0) & (expanded_labels == 1)
    ).float() * weights.unnecessary_wait
    final_wait = ((actions == 0) & final_flags[:, None]).float() * weights.final_wait
    early = (
        ((actions == 1) & (expanded_labels == 1)).float()
        * (1.0 - event_fractions[:, None])
        * weights.safe_early_write
    )
    event_rewards = correct + premature + unnecessary + final_wait + early

    sample_rewards = torch.zeros(
        (sample_count, group_size), dtype=torch.float32, device=actions.device
    )
    sample_counts = torch.zeros(
        (sample_count, 1), dtype=torch.float32, device=actions.device
    )
    sample_rewards.index_add_(0, event_sample_ids, event_rewards)
    sample_counts.index_add_(
        0,
        event_sample_ids,
        torch.ones((actions.shape[0], 1), dtype=torch.float32, device=actions.device),
    )
    sample_rewards = sample_rewards / sample_counts.clamp_min(1.0)
    components = {
        "correct": correct,
        "premature": premature,
        "unnecessary": unnecessary,
        "final_wait": final_wait,
        "early": early,
    }
    return sample_rewards, components


def grpo_action_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    event_sample_ids: torch.Tensor,
    event_fractions: torch.Tensor,
    final_flags: torch.Tensor,
    *,
    sample_count: int,
    group_size: int,
    kl_beta: float,
    sft_weight: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    distribution = torch.distributions.Categorical(logits=policy_logits)
    actions = distribution.sample((group_size,)).transpose(0, 1)
    if generator is not None:
        # Categorical.sample does not accept a generator; callers should set the
        # rank-specific global seed. Keep the parameter for an explicit API and
        # reject accidental assumptions about independent generators.
        del generator
    rewards, components = rollout_rewards(
        actions,
        labels,
        event_sample_ids,
        event_fractions,
        final_flags,
        sample_count=sample_count,
    )
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    advantages = (rewards - mean) / std

    log_probs = F.log_softmax(policy_logits, dim=-1)
    sampled_log_probs = (
        log_probs[:, None, :]
        .expand(-1, group_size, -1)
        .gather(2, actions[:, :, None])
        .squeeze(-1)
    )
    sample_log_probs = torch.zeros_like(rewards)
    sample_counts = torch.zeros(
        (sample_count, 1), dtype=torch.float32, device=policy_logits.device
    )
    sample_log_probs.index_add_(0, event_sample_ids, sampled_log_probs)
    sample_counts.index_add_(
        0,
        event_sample_ids,
        torch.ones(
            (sampled_log_probs.shape[0], 1),
            dtype=torch.float32,
            device=policy_logits.device,
        ),
    )
    sample_log_probs = sample_log_probs / sample_counts.clamp_min(1.0)

    reference_probabilities = F.softmax(reference_logits.detach(), dim=-1)
    kl = F.kl_div(
        F.log_softmax(policy_logits, dim=-1),
        reference_probabilities,
        reduction="batchmean",
    ).clamp_min(0.0)
    sft_loss = F.cross_entropy(policy_logits, labels)
    policy_loss = -(advantages.detach() * sample_log_probs).mean()
    total = policy_loss + kl_beta * kl + sft_weight * sft_loss
    metrics = {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "sft_loss": sft_loss.detach(),
        "kl": kl.detach(),
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
