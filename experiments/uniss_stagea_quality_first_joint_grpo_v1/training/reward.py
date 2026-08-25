"""Quality-first grouped token-trajectory reward for simultaneous S2ST.

Candidates are sampled jointly over every supervised MT/semantic/action/
boundary position inside a packed utterance.  Rewards and log probabilities
are reduced per utterance before group-relative normalization, so long samples
do not dominate merely because they contain more tokens.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_SEMANTIC,
)
from training import constants_uniss as c


GRPO_METRIC_NAMES = (
    "grpo/policy_loss",
    "grpo/reference_kl",
    "grpo/reward_mean",
    "grpo/reward_std",
    "grpo/reward_max",
    "grpo/quality",
    "grpo/prefix_support",
    "grpo/completeness",
    "grpo/semantic_validity",
    "grpo/boundary",
    "grpo/latency_gain",
    "grpo/premature_write",
    "grpo/unnecessary_wait",
    "grpo/write_coverage",
    "grpo/final_flush",
    "grpo/sampled_gold_rate",
    "grpo/samples",
    "grpo/positions",
)


@dataclass(frozen=True)
class GroupObjective:
    loss: torch.Tensor
    metrics: OrderedDict[str, torch.Tensor]


def eligible_mask(loss_kinds: torch.Tensor) -> torch.Tensor:
    return (
        (loss_kinds == LOSS_MT)
        | (loss_kinds == LOSS_SEMANTIC)
        | (loss_kinds == LOSS_BOUNDARY)
        | (loss_kinds == LOSS_EOS)
    )


def candidate_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = eligible_mask(loss_kinds)
    positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        raise ValueError("GRPO batch has no eligible positions")
    rows = logits.index_select(0, positions).float()
    values, indices = rows.topk(int(width) + 1, dim=-1)
    gold = labels.index_select(0, positions).long()
    duplicate = indices[:, :width] == gold[:, None]
    if bool(duplicate.any()):
        replacement = indices[:, width]
        replacement_value = values[:, width]
        first_duplicate = duplicate.float().argmax(dim=1)
        affected = duplicate.any(dim=1)
        indices[affected, first_duplicate[affected]] = replacement[affected]
        values[affected, first_duplicate[affected]] = replacement_value[affected]
    gold_values = rows.gather(1, gold[:, None])
    return positions, torch.cat((indices[:, :width], gold[:, None]), dim=1), torch.cat((values[:, :width], gold_values), dim=1)


def _position_sample_ids(
    positions: torch.Tensor,
    sample_boundaries: list[list[tuple[int, int]]],
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    row_count = len(sample_boundaries)
    if row_count <= 0:
        raise ValueError("GRPO requires packed sample boundaries")
    ids = torch.full(
        (row_count * sequence_length,),
        -1,
        dtype=torch.long,
        device=positions.device,
    )
    fractions = torch.zeros_like(ids, dtype=torch.float32)
    sample = 0
    for row, boundaries in enumerate(sample_boundaries):
        previous = 0
        for start, stop in boundaries:
            start, stop = int(start), int(stop)
            if start != previous or not start < stop <= sequence_length:
                raise ValueError("malformed packed sample boundaries")
            base = row * sequence_length
            ids[base + start : base + stop] = sample
            fractions[base + start : base + stop] = torch.linspace(
                0.0,
                1.0,
                stop - start,
                device=positions.device,
            )
            sample += 1
            previous = stop
    selected_ids = ids.index_select(0, positions)
    if bool((selected_ids < 0).any()):
        raise ValueError("eligible GRPO position falls outside an utterance")
    return selected_ids, sample, fractions.index_select(0, positions)


def _scatter_mean(
    values: torch.Tensor,
    sample_ids: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    output = values.new_zeros((sample_count, values.shape[1]))
    counts = values.new_zeros((sample_count, 1))
    output.index_add_(0, sample_ids, values)
    counts.index_add_(0, sample_ids, torch.ones_like(values[:, :1]))
    return output / counts.clamp_min(1.0)


def _trajectory_action_terms(
    sampled: torch.Tensor,
    gold: torch.Tensor,
    sample_ids: torch.Tensor,
    fractions: torch.Tensor,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    action = (gold == c.TOKEN_WAIT_READ) | (gold == c.TOKEN_WRITE_GENERATE)
    group = sampled.shape[1]
    zeros = sampled.new_zeros((sample_count, group), dtype=torch.float32)
    if not bool(action.any()):
        return {
            "latency_gain": zeros,
            "premature": zeros,
            "unnecessary": zeros,
            "coverage": zeros,
            "final_flush": zeros,
        }
    sid = sample_ids[action]
    candidate = sampled[action]
    target = gold[action, None].expand_as(candidate)
    frac = fractions[action, None].expand_as(candidate)
    premature = _scatter_mean(
        ((candidate == c.TOKEN_WRITE_GENERATE) & (target == c.TOKEN_WAIT_READ)).float(),
        sid,
        sample_count,
    )
    unnecessary = _scatter_mean(
        ((candidate == c.TOKEN_WAIT_READ) & (target == c.TOKEN_WRITE_GENERATE)).float(),
        sid,
        sample_count,
    )
    predicted_writes = sampled.new_zeros((sample_count, group), dtype=torch.float32)
    reference_writes = sampled.new_zeros((sample_count, group), dtype=torch.float32)
    predicted_writes.index_add_(0, sid, (candidate == c.TOKEN_WRITE_GENERATE).float())
    reference_writes.index_add_(0, sid, (target == c.TOKEN_WRITE_GENERATE).float())
    coverage = 1.0 - (predicted_writes - reference_writes).abs() / reference_writes.clamp_min(1.0)
    inf = torch.full_like(frac, 2.0)
    predicted_source = torch.where(candidate == c.TOKEN_WRITE_GENERATE, frac, inf)
    reference_source = torch.where(target == c.TOKEN_WRITE_GENERATE, frac, inf)
    scatter = sid[:, None].expand_as(candidate)
    predicted_first = torch.full_like(zeros, 2.0)
    reference_first = torch.full_like(zeros, 2.0)
    predicted_first.scatter_reduce_(0, scatter, predicted_source, reduce="amin", include_self=True)
    reference_first.scatter_reduce_(0, scatter, reference_source, reduce="amin", include_self=True)
    predicted_first = predicted_first.clamp_max(1.0)
    reference_first = reference_first.clamp_max(1.0)
    latency_gain = (reference_first - predicted_first).clamp(-1.0, 1.0)
    final_action = torch.zeros_like(candidate, dtype=torch.bool)
    for sample in range(sample_count):
        indexes = torch.nonzero(sid == sample, as_tuple=False).reshape(-1)
        if indexes.numel():
            final_action[indexes[-1]] = True
    final_flush = _scatter_mean(
        ((candidate == c.TOKEN_WRITE_GENERATE) & final_action).float(),
        sid,
        sample_count,
    )
    return {
        "latency_gain": latency_gain,
        "premature": premature,
        "unnecessary": unnecessary,
        "coverage": coverage,
        "final_flush": final_flush,
    }


def group_relative_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    sample_boundaries: list[list[tuple[int, int]]],
    reference_indices: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    sequence_length: int,
    group_size: int,
    progress: float,
    clip_epsilon: float,
) -> GroupObjective:
    if group_size < 2:
        raise ValueError("GRPO group size must be at least two")
    positions = torch.nonzero(eligible_mask(loss_kinds), as_tuple=False).reshape(-1)
    if positions.numel() != reference_indices.shape[0]:
        raise ValueError("policy/reference GRPO position counts differ")
    current = logits.index_select(0, positions).float().gather(1, reference_indices.long())
    current_log_probs = F.log_softmax(current, dim=-1)
    reference_probs = F.softmax(reference_logits.float(), dim=-1)
    sampled_choice = torch.multinomial(
        current_log_probs.detach().exp(),
        num_samples=int(group_size),
        replacement=True,
    )
    sampled = reference_indices.gather(1, sampled_choice)
    sampled_log_probs = current_log_probs.gather(1, sampled_choice)
    gold = labels.index_select(0, positions).long()
    kinds = loss_kinds.index_select(0, positions)
    sample_ids, sample_count, fractions = _position_sample_ids(
        positions,
        sample_boundaries,
        sequence_length=int(sequence_length),
    )
    exact = (sampled == gold[:, None]).float()
    mt = (kinds == LOSS_MT)[:, None]
    semantic = (kinds == LOSS_SEMANTIC)[:, None]
    boundary = ((kinds == LOSS_BOUNDARY) | (kinds == LOSS_EOS))[:, None]
    semantic_stop = c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
    semantic_valid = (
        (sampled >= c.BICODEC_SEMANTIC_OFFSET) & (sampled < semantic_stop)
    ).float()
    end_tokens = (
        (gold == c.TOKEN_END_CONTENT)
        | (gold == c.TOKEN_END_SEMANTIC)
        | (gold == c.TOKEN_EOS)
    )[:, None]
    quality = _scatter_mean(exact * (mt.float() + 0.6 * semantic.float()), sample_ids, sample_count)
    prefix = _scatter_mean(exact * mt.float(), sample_ids, sample_count)
    completeness = _scatter_mean(exact * end_tokens.float(), sample_ids, sample_count)
    semantic_score = _scatter_mean(semantic_valid * semantic.float(), sample_ids, sample_count)
    boundary_score = _scatter_mean(exact * boundary.float(), sample_ids, sample_count)
    actions = _trajectory_action_terms(sampled, gold, sample_ids, fractions, sample_count)
    late = float(progress) >= 0.60
    latency_weight = 0.20 if late else 0.05
    wait_weight = 0.15 if late else 0.05
    reward = (
        1.00 * quality
        + 0.40 * prefix
        + 0.60 * completeness
        + 0.45 * semantic_score
        + 0.20 * boundary_score
        + latency_weight * actions["latency_gain"]
        + 0.20 * actions["coverage"]
        + 0.20 * actions["final_flush"]
        - 0.30 * actions["premature"]
        - wait_weight * actions["unnecessary"]
    )
    reward_mean = reward.mean(dim=1, keepdim=True)
    reward_std = reward.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    advantage = (reward - reward_mean) / reward_std
    trajectory_log_probs = _scatter_mean(sampled_log_probs, sample_ids, sample_count)
    old = trajectory_log_probs.detach()
    ratio = (trajectory_log_probs - old).exp()
    clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    policy_loss = -torch.minimum(ratio * advantage.detach(), clipped * advantage.detach()).mean()
    kl = F.kl_div(current_log_probs, reference_probs, reduction="batchmean").clamp_min(0.0)
    metrics = OrderedDict(
        (
            ("grpo/policy_loss", policy_loss.detach()),
            ("grpo/reference_kl", kl.detach()),
            ("grpo/reward_mean", reward.mean().detach()),
            ("grpo/reward_std", reward.std(unbiased=False).detach()),
            ("grpo/reward_max", reward.max().detach()),
            ("grpo/quality", quality.mean().detach()),
            ("grpo/prefix_support", prefix.mean().detach()),
            ("grpo/completeness", completeness.mean().detach()),
            ("grpo/semantic_validity", semantic_score.mean().detach()),
            ("grpo/boundary", boundary_score.mean().detach()),
            ("grpo/latency_gain", actions["latency_gain"].mean().detach()),
            ("grpo/premature_write", actions["premature"].mean().detach()),
            ("grpo/unnecessary_wait", actions["unnecessary"].mean().detach()),
            ("grpo/write_coverage", actions["coverage"].mean().detach()),
            ("grpo/final_flush", actions["final_flush"].mean().detach()),
            ("grpo/sampled_gold_rate", exact.mean().detach()),
            ("grpo/samples", policy_loss.new_tensor(float(sample_count))),
            ("grpo/positions", policy_loss.new_tensor(float(positions.numel()))),
        )
    )
    return GroupObjective(policy_loss + kl * 0.0, metrics)


def zero_grpo_metrics(reference: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    zero = reference.detach().new_zeros(())
    return OrderedDict((name, zero) for name in GRPO_METRIC_NAMES)


__all__ = [
    "GRPO_METRIC_NAMES",
    "candidate_topk",
    "group_relative_objective",
    "zero_grpo_metrics",
]
