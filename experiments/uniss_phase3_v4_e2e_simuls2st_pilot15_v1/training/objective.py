"""Numerator/denominator-safe losses for the five-family E2E objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TeacherPosterior,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_REPLAY,
    LOSS_SEMANTIC,
)


@dataclass(frozen=True)
class LossTerm:
    numerator: torch.Tensor
    denominator: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return self.numerator / self.denominator.clamp_min(1.0)

    @property
    def active(self) -> bool:
        return bool(self.denominator.detach().item() > 0)


@dataclass(frozen=True)
class LogitConsistencyPair:
    previous_logits: torch.Tensor
    current_logits: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SpeakerContinuityPair:
    previous_embedding: torch.Tensor
    current_embedding: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class E2ELossWeights:
    asr_ce: float = 1.0
    mt_ce: float = 1.0
    semantic_ce: float = 1.0
    replay_ce: float = 0.50
    v1_asr_kl: float = 0.30
    phase3_kl: float = 0.25
    commit_consistency: float = 0.20
    boundary_eos: float = 0.10
    speaker_continuity: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("E2E loss weights must be non-negative")


@dataclass(frozen=True)
class E2EObjectiveOutput:
    total: torch.Tensor
    terms: Mapping[str, LossTerm]
    weighted: Mapping[str, torch.Tensor]

    def metrics(self) -> dict[str, torch.Tensor]:
        output = {"loss/total": self.total.detach()}
        for name, term in self.terms.items():
            output[f"loss/{name}"] = term.loss.detach()
            output[f"numerator/{name}"] = term.numerator.detach()
            output[f"denominator/{name}"] = term.denominator.detach()
        for name, value in self.weighted.items():
            output[f"weighted/{name}"] = value.detach()
        return output


def token_nll_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("E2E logits/labels geometry differs")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1).long(),
        reduction="none",
    ).reshape_as(labels)


def _zero(reference: torch.Tensor) -> LossTerm:
    value = reference.sum() * 0.0
    return LossTerm(value, value.detach())


def _masked_term(values: torch.Tensor, mask: torch.Tensor) -> LossTerm:
    if values.shape != mask.shape:
        raise ValueError("E2E loss values/mask geometry differs")
    selected = mask.to(dtype=torch.bool)
    denominator = selected.sum().to(dtype=torch.float32)
    numerator = values.masked_select(selected).float().sum()
    return LossTerm(numerator, denominator)


def token_ce_terms(
    token_nll: torch.Tensor, loss_kinds: torch.Tensor
) -> dict[str, LossTerm]:
    if token_nll.ndim != 2 or token_nll.shape != loss_kinds.shape:
        raise ValueError("E2E token NLL/loss-kind geometry differs")
    return {
        "asr_ce": _masked_term(token_nll, loss_kinds == LOSS_ASR),
        "mt_ce": _masked_term(token_nll, loss_kinds == LOSS_MT),
        "semantic_ce": _masked_term(token_nll, loss_kinds == LOSS_SEMANTIC),
        "boundary_ce": _masked_term(token_nll, loss_kinds == LOSS_BOUNDARY),
        "eos_ce": _masked_term(token_nll, loss_kinds == LOSS_EOS),
        "replay_ce": _masked_term(token_nll, loss_kinds == LOSS_REPLAY),
    }


def _balanced_terms(
    values: Sequence[LossTerm], reference: torch.Tensor
) -> LossTerm:
    if not values:
        return _zero(reference)
    active = torch.stack(
        [(value.denominator > 0).to(dtype=torch.float32) for value in values]
    )
    numerator = torch.stack(
        [value.loss * flag for value, flag in zip(values, active)]
    ).sum()
    return LossTerm(numerator, active.sum())


def topk_teacher_kl(
    resolved: Sequence[Mapping[str, object]],
    *,
    cache_kind: str,
    full_logits: torch.Tensor | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> LossTerm:
    selected_logits: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    reference: torch.Tensor | None = (
        full_logits if full_logits is not None else reference_tensor
    )
    for value in resolved:
        posterior = value.get("posterior")
        if not isinstance(posterior, TeacherPosterior):
            raise TypeError("resolved teacher binding has no posterior")
        if posterior.cache_kind != cache_kind:
            continue
        start = int(value["packed_start"])
        stop = int(value["packed_stop"])
        if stop - start != posterior.positions:
            raise ValueError("teacher KL packed/cache geometry differs")
        logits = value.get("student_logits")
        if logits is None:
            if full_logits is None:
                raise ValueError("teacher KL is missing selected student logits")
            if full_logits.ndim != 3:
                raise ValueError("teacher KL full logits must be [batch, seq, vocab]")
            batch_index = int(value.get("batch_index", 0))
            logits = full_logits[batch_index, start:stop]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise TypeError("teacher KL student logits are malformed")
        if logits.shape[0] != posterior.positions:
            raise ValueError("teacher KL selected student position count differs")
        reference = logits
        selected_logits.append(logits)
        indices.append(posterior.indices)
        probabilities.append(posterior.probabilities)
    if not selected_logits:
        if reference is None:
            reference = torch.zeros((), dtype=torch.float32)
        return _zero(reference)
    logits = torch.cat(selected_logits).float()
    teacher_indices = torch.cat(indices).to(device=logits.device, dtype=torch.long)
    teacher_probabilities = torch.cat(probabilities).to(
        device=logits.device, dtype=torch.float32
    )
    if teacher_indices.shape != teacher_probabilities.shape:
        raise ValueError("teacher KL top-k indices/probabilities differ")
    if teacher_indices.shape[0] != logits.shape[0]:
        raise ValueError("teacher KL posterior/student rows differ")
    if torch.any(teacher_indices >= logits.shape[1]):
        raise ValueError("teacher KL token is outside student logits")
    log_student = F.log_softmax(logits, dim=-1).gather(1, teacher_indices)
    log_teacher = teacher_probabilities.clamp_min(1e-8).log()
    per_position = (
        teacher_probabilities * (log_teacher - log_student)
    ).sum(dim=1)
    return LossTerm(
        per_position.sum(),
        per_position.new_tensor(float(per_position.numel())),
    )


def commit_consistency_kl(
    pairs: Sequence[LogitConsistencyPair],
    *,
    temperature: float = 1.0,
    reference_tensor: torch.Tensor | None = None,
) -> LossTerm:
    if temperature <= 0:
        raise ValueError("commit consistency temperature must be positive")
    values: list[torch.Tensor] = []
    reference: torch.Tensor | None = reference_tensor
    for pair in pairs:
        previous = pair.previous_logits
        current = pair.current_logits
        if previous.shape != current.shape or previous.ndim != 2:
            raise ValueError("commit consistency logits geometry differs")
        reference = current
        teacher = F.softmax(previous.detach().float() / temperature, dim=-1)
        student = F.log_softmax(current.float() / temperature, dim=-1)
        row = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
        if pair.mask is not None:
            mask = pair.mask.to(device=row.device, dtype=torch.bool)
            if mask.shape != row.shape:
                raise ValueError("commit consistency mask geometry differs")
            row = row[mask]
        values.append(row * (temperature**2))
    if not values:
        return _zero(reference if reference is not None else torch.zeros(()))
    merged = torch.cat(values)
    return LossTerm(merged.sum(), merged.new_tensor(float(merged.numel())))


def commit_pairs_from_full_logits(
    full_logits: torch.Tensor,
    bindings: Sequence[Mapping[str, object]],
) -> list[LogitConsistencyPair]:
    if full_logits.ndim != 3:
        raise ValueError("commit consistency full logits must be [batch, seq, vocab]")
    pairs: list[LogitConsistencyPair] = []
    for binding in bindings:
        batch_index = int(binding.get("batch_index", 0))
        previous_start = int(binding["previous_packed_start"])
        previous_stop = int(binding["previous_packed_stop"])
        current_start = int(binding["current_packed_start"])
        current_stop = int(binding["current_packed_stop"])
        if previous_stop - previous_start != current_stop - current_start:
            raise ValueError("commit consistency binding length differs")
        pairs.append(
            LogitConsistencyPair(
                full_logits[batch_index, previous_start:previous_stop],
                full_logits[batch_index, current_start:current_stop],
            )
        )
    return pairs


def speaker_continuity_loss(
    pairs: Sequence[SpeakerContinuityPair],
    *,
    reference_tensor: torch.Tensor | None = None,
) -> LossTerm:
    values: list[torch.Tensor] = []
    reference: torch.Tensor | None = reference_tensor
    for pair in pairs:
        previous = pair.previous_embedding
        current = pair.current_embedding
        if previous.shape != current.shape or previous.ndim != 2:
            raise ValueError("speaker continuity embedding geometry differs")
        reference = current
        row = 1.0 - F.cosine_similarity(
            previous.detach().float(), current.float(), dim=-1
        )
        if pair.mask is not None:
            mask = pair.mask.to(device=row.device, dtype=torch.bool)
            if mask.shape != row.shape:
                raise ValueError("speaker continuity mask geometry differs")
            row = row[mask]
        values.append(row)
    if not values:
        return _zero(reference if reference is not None else torch.zeros(()))
    merged = torch.cat(values)
    return LossTerm(merged.sum(), merged.new_tensor(float(merged.numel())))


def compute_e2e_objective(
    *,
    token_nll: torch.Tensor,
    loss_kinds: torch.Tensor,
    teacher_posteriors: Sequence[Mapping[str, object]] = (),
    full_logits: torch.Tensor | None = None,
    commit_pairs: Sequence[LogitConsistencyPair] = (),
    speaker_pairs: Sequence[SpeakerContinuityPair] = (),
    weights: E2ELossWeights | None = None,
) -> E2EObjectiveOutput:
    weights = weights or E2ELossWeights()
    terms = token_ce_terms(token_nll, loss_kinds)
    terms["boundary_eos"] = _balanced_terms(
        (terms["boundary_ce"], terms["eos_ce"]), token_nll
    )
    terms["v1_asr_kl"] = topk_teacher_kl(
        teacher_posteriors,
        cache_kind="v1_asr",
        full_logits=full_logits,
        reference_tensor=token_nll,
    )
    terms["phase3_kl"] = topk_teacher_kl(
        teacher_posteriors,
        cache_kind="phase3",
        full_logits=full_logits,
        reference_tensor=token_nll,
    )
    terms["commit_consistency"] = commit_consistency_kl(
        commit_pairs, reference_tensor=token_nll
    )
    terms["speaker_continuity"] = speaker_continuity_loss(
        speaker_pairs, reference_tensor=token_nll
    )
    weighted = {
        "asr_ce": terms["asr_ce"].loss * weights.asr_ce,
        "mt_ce": terms["mt_ce"].loss * weights.mt_ce,
        "semantic_ce": terms["semantic_ce"].loss * weights.semantic_ce,
        "replay_ce": terms["replay_ce"].loss * weights.replay_ce,
        "v1_asr_kl": terms["v1_asr_kl"].loss * weights.v1_asr_kl,
        "phase3_kl": terms["phase3_kl"].loss * weights.phase3_kl,
        "commit_consistency": terms["commit_consistency"].loss
        * weights.commit_consistency,
        "boundary_eos": terms["boundary_eos"].loss * weights.boundary_eos,
        "speaker_continuity": terms["speaker_continuity"].loss
        * weights.speaker_continuity,
    }
    total = torch.stack(list(weighted.values())).sum()
    return E2EObjectiveOutput(total=total, terms=terms, weighted=weighted)


__all__ = [
    "E2ELossWeights",
    "E2EObjectiveOutput",
    "LogitConsistencyPair",
    "LossTerm",
    "SpeakerContinuityPair",
    "commit_consistency_kl",
    "commit_pairs_from_full_logits",
    "compute_e2e_objective",
    "speaker_continuity_loss",
    "token_ce_terms",
    "token_nll_from_logits",
    "topk_teacher_kl",
]
