"""Explicit Bayesian source safe-commit gate for the Stage-C pilot."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StageCGateConfig:
    context_dim: int = 4
    evidence_dim: int = 8
    minimum_commit_tokens: int = 2
    likelihood_floor: float = 0.03

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageCGateConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BayesianSourceSafeCommitGate(nn.Module):
    """Bayes rule with a learned prior and diagonal Gaussian likelihoods.

    posterior odds = prior odds * p(evidence|safe) / p(evidence|unsafe)
    """

    def __init__(self, config: StageCGateConfig) -> None:
        super().__init__()
        self.config = config
        self.prior = nn.Linear(config.context_dim, 1)
        self.likelihood_mean = nn.Parameter(torch.stack((
            torch.full((config.evidence_dim,), 0.35),
            torch.full((config.evidence_dim,), 0.65),
        )))
        self.likelihood_log_scale = nn.Parameter(
            torch.full((2, config.evidence_dim), math.log(0.25))
        )

    def class_log_likelihood(self, evidence: torch.Tensor) -> torch.Tensor:
        value = evidence.unsqueeze(1)
        scale = self.likelihood_log_scale.exp().clamp_min(self.config.likelihood_floor)
        normalized = (value - self.likelihood_mean.unsqueeze(0)) / scale.unsqueeze(0)
        return (
            -0.5 * normalized.square()
            - self.likelihood_log_scale.unsqueeze(0)
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def forward(self, context: torch.Tensor, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        prior_logit = self.prior(context).squeeze(-1)
        likelihood = self.class_log_likelihood(evidence)
        log_likelihood_ratio = likelihood[:, 1] - likelihood[:, 0]
        posterior_logit = prior_logit + log_likelihood_ratio
        return {
            "prior_logit": prior_logit,
            "class_log_likelihood": likelihood,
            "log_likelihood_ratio": log_likelihood_ratio,
            "posterior_logit": posterior_logit,
            "posterior": torch.sigmoid(posterior_logit),
        }


def _collapsed_tokens(values: list[int]) -> list[int]:
    result: list[int] = []
    previous = 0
    for value in values:
        value = int(value)
        if value != 0 and value != previous:
            result.append(value - 1)
        previous = value
    return result


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


@torch.no_grad()
def extract_gate_inputs(
    student_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    minimum_commit_tokens: int,
    segment_frames: int,
) -> dict[str, torch.Tensor]:
    logits = student_output["teacher_glm_logits"].float()
    source_logits = student_output["source_ctc_logits"].float()
    lengths = student_output["output_lengths"]
    batch_size, time_steps, _ = logits.shape
    positions = torch.arange(time_steps, device=logits.device).unsqueeze(0)
    tail_mask = (positions < lengths.unsqueeze(1)) & (
        positions >= (lengths - segment_frames).clamp_min(0).unsqueeze(1)
    )
    divisor = tail_mask.sum(dim=1).clamp_min(1).float()

    log_normalizer = torch.logsumexp(logits, dim=-1)
    top_values, top_indices = logits.topk(2, dim=-1)
    top1_probability = torch.exp(top_values[..., 0] - log_normalizer)
    top2_probability = torch.exp(top_values[..., 1] - log_normalizer)
    blank_probability = torch.exp(logits[..., 0] - log_normalizer)
    uncertainty = (-torch.log(top1_probability.clamp_min(1e-7))) / math.log(logits.shape[-1])
    source_log_normalizer = torch.logsumexp(source_logits, dim=-1)
    source_top = source_logits.amax(dim=-1)
    source_confidence = torch.exp(source_top - source_log_normalizer)
    stability = torch.sigmoid(student_output["stability_logits"].float())

    def tail_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * tail_mask).sum(dim=1) / divisor

    final_positions = (lengths - 1).clamp_min(0)
    capacity = torch.sigmoid(
        student_output["target_capacity_logits"].float().gather(
            1, final_positions.unsqueeze(1)
        ).squeeze(1)
    )
    argmax = logits.argmax(dim=-1)
    repeated = (argmax[:, 1:] == argmax[:, :-1]).float()
    repeated_mask = tail_mask[:, 1:] & tail_mask[:, :-1]
    persistence = (repeated * repeated_mask).sum(dim=1) / repeated_mask.sum(dim=1).clamp_min(1)
    nonblank = ((argmax != 0) & tail_mask).sum(dim=1).float() / divisor

    prefix_fraction = (
        batch["utterance_sample_lengths"].float() / batch["full_samples"].float().clamp_min(1)
    ).clamp(0.0, 1.0)
    prefix_seconds = (batch["utterance_sample_lengths"].float() / 16000.0 / 8.0).clamp(0.0, 1.5)
    duration_seconds = (batch["full_samples"].float() / 16000.0 / 12.0).clamp(0.0, 2.0)
    direction = batch["direction"].float()
    context = torch.stack((prefix_fraction, prefix_seconds, duration_seconds, direction), dim=-1)
    evidence = torch.stack(
        (
            tail_mean(top1_probability),
            tail_mean(top1_probability - top2_probability),
            1.0 - tail_mean(blank_probability),
            1.0 - tail_mean(uncertainty),
            tail_mean(stability),
            tail_mean(source_confidence),
            persistence,
            capacity,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)

    predicted_cpu = argmax.cpu().tolist()
    references_cpu = batch["reference_glm"].cpu().tolist()
    reference_lengths = batch["reference_glm_lengths"].cpu().tolist()
    output_lengths = lengths.cpu().tolist()
    lcp_values: list[int] = []
    for row in range(batch_size):
        predicted = _collapsed_tokens(predicted_cpu[row][: int(output_lengths[row])])
        reference = [int(value) for value in references_cpu[row][: int(reference_lengths[row])]]
        lcp_values.append(_longest_common_prefix(predicted, reference))
    lcp = torch.tensor(lcp_values, dtype=torch.long, device=logits.device)
    support_ready = batch["support_count"] >= minimum_commit_tokens
    if "safe_label" in batch:
        labels = batch["safe_label"].bool()
    else:
        labels = support_ready & (lcp >= minimum_commit_tokens)
    return {
        "context": context,
        "evidence": evidence,
        "labels": labels.float(),
        "lcp": lcp,
        "support_ready": support_ready.float(),
        "predicted_top_class": top_indices[..., 0],
    }


def stage_c_losses(
    gate: nn.Module,
    context: torch.Tensor,
    evidence: torch.Tensor,
    labels: torch.Tensor,
    *,
    prior_weight: float = 0.2,
    likelihood_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    output = gate(context, evidence)
    posterior = F.binary_cross_entropy_with_logits(output["posterior_logit"], labels)
    prior = F.binary_cross_entropy_with_logits(output["prior_logit"], labels)
    class_likelihood = output["class_log_likelihood"].gather(
        1, labels.long().unsqueeze(1)
    ).squeeze(1)
    likelihood = -class_likelihood.mean() / evidence.shape[-1]
    total = posterior + prior_weight * prior + likelihood_weight * likelihood
    brier = (output["posterior"] - labels).square().mean()
    return {
        "total": total,
        "posterior": posterior,
        "prior": prior,
        "likelihood": likelihood,
        "brier": brier,
        "positive_rate": labels.mean(),
    }
