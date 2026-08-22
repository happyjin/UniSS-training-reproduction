"""Numerator/denominator-safe losses for the five-family E2E objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from training import constants_uniss as c

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
    content_end_ce: float = 0.0
    semantic_end_ce: float = 0.0
    semantic_end_margin: float = 0.0
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


E2E_TERM_NAMES = (
    "asr_ce",
    "mt_ce",
    "semantic_ce",
    "replay_ce",
    "v1_asr_kl",
    "phase3_kl",
    "commit_consistency",
    "boundary_ce",
    "eos_ce",
    "content_end_ce",
    "semantic_end_ce",
    "semantic_end_margin",
    "speaker_continuity",
)

E2E_WEIGHTED_NAMES = (
    "asr_ce",
    "mt_ce",
    "semantic_ce",
    "replay_ce",
    "v1_asr_kl",
    "phase3_kl",
    "commit_consistency",
    "boundary_eos",
    "content_end_ce",
    "semantic_end_ce",
    "semantic_end_margin",
    "speaker_continuity",
)


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


def flattened_token_ce_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
) -> dict[str, LossTerm]:
    """Compute CE only at supervised flattened Megatron positions."""

    if logits.ndim != 2 or labels.ndim != 1 or loss_kinds.ndim != 1:
        raise ValueError("flattened E2E token tensors have invalid rank")
    if logits.shape[0] != labels.numel() or labels.shape != loss_kinds.shape:
        raise ValueError("flattened E2E token tensors differ in length")
    output: dict[str, LossTerm] = {}
    for name, kind in (
        ("asr_ce", LOSS_ASR),
        ("mt_ce", LOSS_MT),
        ("semantic_ce", LOSS_SEMANTIC),
        ("boundary_ce", LOSS_BOUNDARY),
        ("eos_ce", LOSS_EOS),
        ("replay_ce", LOSS_REPLAY),
    ):
        mask = loss_kinds == kind
        denominator = mask.sum().float()
        if bool(mask.any()):
            numerator = F.cross_entropy(
                logits[mask].float(), labels[mask].long(), reduction="sum"
            )
        else:
            numerator = logits.sum() * 0.0
        output[name] = LossTerm(numerator, denominator)
    content_end_mask = (loss_kinds == LOSS_BOUNDARY) & (
        labels == c.TOKEN_END_CONTENT
    )
    content_end_denominator = content_end_mask.sum().float()
    if bool(content_end_mask.any()):
        content_end_numerator = F.cross_entropy(
            logits[content_end_mask].float(),
            labels[content_end_mask].long(),
            reduction="sum",
        )
    else:
        content_end_numerator = logits.sum() * 0.0
    output["content_end_ce"] = LossTerm(
        content_end_numerator, content_end_denominator
    )
    semantic_end_mask = (loss_kinds == LOSS_BOUNDARY) & (
        labels == c.TOKEN_END_SEMANTIC
    )
    semantic_end_denominator = semantic_end_mask.sum().float()
    if bool(semantic_end_mask.any()):
        semantic_end_numerator = F.cross_entropy(
            logits[semantic_end_mask].float(),
            labels[semantic_end_mask].long(),
            reduction="sum",
        )
    else:
        semantic_end_numerator = logits.sum() * 0.0
    output["semantic_end_ce"] = LossTerm(
        semantic_end_numerator, semantic_end_denominator
    )
    return output


def flattened_semantic_end_margin_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    margin: float,
) -> LossTerm:
    """Make END_SEMANTIC beat every legal semantic continuation token."""

    if margin < 0:
        raise ValueError("semantic end logit margin must be non-negative")
    if logits.ndim != 2 or labels.ndim != 1 or loss_kinds.ndim != 1:
        raise ValueError("flattened E2E semantic end tensors have invalid rank")
    if logits.shape[0] != labels.numel() or labels.shape != loss_kinds.shape:
        raise ValueError("flattened E2E semantic end tensors differ in length")
    mask = (loss_kinds == LOSS_BOUNDARY) & (
        labels == c.TOKEN_END_SEMANTIC
    )
    denominator = mask.sum().float()
    if not bool(mask.any()):
        return LossTerm(logits.sum() * 0.0, denominator)
    selected = logits[mask].float()
    semantic_max = selected[
        :, c.BICODEC_SEMANTIC_OFFSET : c.BICODEC_SEMANTIC_OFFSET
        + c.BICODEC_SEMANTIC_SIZE
    ].max(dim=1).values
    end_logits = selected[:, c.TOKEN_END_SEMANTIC]
    violations = F.relu(semantic_max + float(margin) - end_logits)
    return LossTerm(violations.sum(), denominator)


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


def flattened_teacher_kl(
    logits: torch.Tensor,
    batch: Mapping[str, object],
    *,
    cache_kind_id: int,
    original_seq_length: int,
) -> LossTerm:
    required = (
        "teacher_batch",
        "teacher_positions",
        "teacher_cache_kind",
        "teacher_indices",
        "teacher_probabilities",
    )
    if any(name not in batch for name in required):
        return _zero(logits)
    cache_kind = batch["teacher_cache_kind"]
    if not isinstance(cache_kind, torch.Tensor):
        raise TypeError("flattened teacher cache kind is not a tensor")
    active = cache_kind.long() == int(cache_kind_id)
    if not bool(active.any()):
        return _zero(logits)
    teacher_batch = batch["teacher_batch"]
    teacher_positions = batch["teacher_positions"]
    indices = batch["teacher_indices"]
    probabilities = batch["teacher_probabilities"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (teacher_batch, teacher_positions, indices, probabilities)
    ):
        raise TypeError("flattened teacher posterior sidecars are malformed")
    flat = (
        teacher_batch.long()[active] * int(original_seq_length)
        + teacher_positions.long()[active]
    )
    selected_indices = indices.long()[active]
    teacher = probabilities.float()[active]
    if selected_indices.shape != teacher.shape or selected_indices.ndim != 2:
        raise ValueError("flattened teacher top-k tensors differ")
    if torch.any(flat < 0) or torch.any(flat >= logits.shape[0]):
        raise ValueError("flattened teacher position exceeds student logits")
    if torch.any(selected_indices < 0) or torch.any(
        selected_indices >= logits.shape[1]
    ):
        raise ValueError("flattened teacher token exceeds student vocabulary")
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    student_log = F.log_softmax(logits.index_select(0, flat).float(), dim=-1)
    student_topk = student_log.gather(1, selected_indices)
    values = (
        teacher * (teacher.clamp_min(1e-8).log() - student_topk)
    ).sum(dim=-1)
    return LossTerm(values.sum(), values.new_tensor(float(values.numel())))


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


def flattened_commit_consistency_kl(
    logits: torch.Tensor,
    batch: Mapping[str, object],
    *,
    original_seq_length: int,
    temperature: float = 1.0,
) -> LossTerm:
    if temperature <= 0:
        raise ValueError("commit consistency temperature must be positive")
    required = (
        "commit_batch",
        "commit_previous_positions",
        "commit_current_positions",
    )
    if any(name not in batch for name in required):
        return _zero(logits)
    commit_batch = batch["commit_batch"]
    previous_positions = batch["commit_previous_positions"]
    current_positions = batch["commit_current_positions"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (commit_batch, previous_positions, current_positions)
    ):
        raise TypeError("flattened commit sidecars are malformed")
    if not (
        commit_batch.ndim
        == previous_positions.ndim
        == current_positions.ndim
        == 1
        and commit_batch.shape
        == previous_positions.shape
        == current_positions.shape
    ):
        raise ValueError("flattened commit sidecars differ in length")
    if not commit_batch.numel():
        return _zero(logits)
    previous_flat = (
        commit_batch.long() * int(original_seq_length)
        + previous_positions.long()
    )
    current_flat = (
        commit_batch.long() * int(original_seq_length)
        + current_positions.long()
    )
    if (
        torch.any(previous_flat < 0)
        or torch.any(current_flat < 0)
        or torch.any(previous_flat >= logits.shape[0])
        or torch.any(current_flat >= logits.shape[0])
    ):
        raise ValueError("flattened commit position exceeds student logits")
    teacher = F.softmax(
        logits.index_select(0, previous_flat).detach().float() / temperature,
        dim=-1,
    )
    student = F.log_softmax(
        logits.index_select(0, current_flat).float() / temperature,
        dim=-1,
    )
    values = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
    values = values * (temperature**2)
    return LossTerm(values.sum(), values.new_tensor(float(values.numel())))


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


def flattened_e2e_objective(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    batch: Mapping[str, object],
    original_seq_length: int,
    semantic_end_logit_margin: float = 0.0,
) -> Mapping[str, LossTerm]:
    token_terms = flattened_token_ce_terms(logits, labels, loss_kinds)
    v1_asr_kl = flattened_teacher_kl(
        logits,
        batch,
        cache_kind_id=0,
        original_seq_length=original_seq_length,
    )
    phase3_kl = flattened_teacher_kl(
        logits,
        batch,
        cache_kind_id=1,
        original_seq_length=original_seq_length,
    )
    commit_consistency = flattened_commit_consistency_kl(
        logits,
        batch,
        original_seq_length=original_seq_length,
    )
    # No genuine cross-fragment speaker embedding sidecar exists yet.  The
    # launch entrypoint therefore requires this term's configured weight to be
    # zero and records that fact instead of inventing supervision.
    terms = {
        "asr_ce": token_terms["asr_ce"],
        "mt_ce": token_terms["mt_ce"],
        "semantic_ce": token_terms["semantic_ce"],
        "replay_ce": token_terms["replay_ce"],
        "v1_asr_kl": v1_asr_kl,
        "phase3_kl": phase3_kl,
        "commit_consistency": commit_consistency,
        "boundary_ce": token_terms["boundary_ce"],
        "eos_ce": token_terms["eos_ce"],
        "content_end_ce": token_terms["content_end_ce"],
        "semantic_end_ce": token_terms["semantic_end_ce"],
        "semantic_end_margin": flattened_semantic_end_margin_term(
            logits,
            labels,
            loss_kinds,
            margin=semantic_end_logit_margin,
        ),
        "speaker_continuity": _zero(logits),
    }
    if tuple(terms) != E2E_TERM_NAMES:
        raise AssertionError("flattened E2E term order changed")
    return terms


def distributed_e2e_objective(
    terms: Mapping[str, LossTerm],
    *,
    weights: E2ELossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a globally normalized scalar with DDP-correct local gradients."""

    weights = weights or E2ELossWeights()
    if tuple(terms) != E2E_TERM_NAMES:
        raise ValueError("distributed E2E objective term order changed")
    numerators = torch.stack([terms[name].numerator for name in E2E_TERM_NAMES])
    denominators = torch.stack(
        [terms[name].denominator.to(numerators.dtype) for name in E2E_TERM_NAMES]
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
        world_size * numerators / global_denominators.clamp_min(1.0),
        numerators * 0.0,
    )
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1.0),
        global_numerators * 0.0,
    )
    index = {name: position for position, name in enumerate(E2E_TERM_NAMES)}
    boundary_indices = [index["boundary_ce"], index["eos_ce"]]
    boundary_active = active[boundary_indices].to(numerators.dtype)
    boundary_eos = (
        local_means[boundary_indices] * boundary_active
    ).sum() / boundary_active.sum().clamp_min(1.0)
    weighted = {
        "asr_ce": local_means[index["asr_ce"]] * weights.asr_ce,
        "mt_ce": local_means[index["mt_ce"]] * weights.mt_ce,
        "semantic_ce": local_means[index["semantic_ce"]] * weights.semantic_ce,
        "replay_ce": local_means[index["replay_ce"]] * weights.replay_ce,
        "v1_asr_kl": local_means[index["v1_asr_kl"]] * weights.v1_asr_kl,
        "phase3_kl": local_means[index["phase3_kl"]] * weights.phase3_kl,
        "commit_consistency": local_means[index["commit_consistency"]]
        * weights.commit_consistency,
        "boundary_eos": boundary_eos * weights.boundary_eos,
        "content_end_ce": local_means[index["content_end_ce"]]
        * weights.content_end_ce,
        "semantic_end_ce": local_means[index["semantic_end_ce"]]
        * weights.semantic_end_ce,
        "semantic_end_margin": local_means[index["semantic_end_margin"]]
        * weights.semantic_end_margin,
        "speaker_continuity": local_means[index["speaker_continuity"]]
        * weights.speaker_continuity,
    }
    if tuple(weighted) != E2E_WEIGHTED_NAMES:
        raise AssertionError("distributed E2E weighted term order changed")
    total = torch.stack(list(weighted.values())).sum()
    metrics: dict[str, torch.Tensor] = {
        f"loss/{name}": global_means[position]
        for position, name in enumerate(E2E_TERM_NAMES)
    }
    metrics.update(
        {
            f"denominator/{name}": global_denominators[position]
            for position, name in enumerate(E2E_TERM_NAMES)
        }
    )
    global_boundary_eos = (
        global_means[boundary_indices] * boundary_active
    ).sum() / boundary_active.sum().clamp_min(1.0)
    metrics["loss/boundary_eos"] = global_boundary_eos
    for name in E2E_WEIGHTED_NAMES:
        if name == "boundary_eos":
            metrics[f"weighted/{name}"] = (
                global_boundary_eos * weights.boundary_eos
            )
        else:
            metrics[f"weighted/{name}"] = (
                global_means[index[name]] * float(getattr(weights, name))
            )
    return total, metrics


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
    terms["content_end_ce"] = _zero(token_nll)
    terms["semantic_end_ce"] = _zero(token_nll)
    terms["semantic_end_margin"] = _zero(token_nll)
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
        "content_end_ce": terms["content_end_ce"].loss
        * weights.content_end_ce,
        "semantic_end_ce": terms["semantic_end_ce"].loss
        * weights.semantic_end_ce,
        "semantic_end_margin": terms["semantic_end_margin"].loss
        * weights.semantic_end_margin,
        "speaker_continuity": terms["speaker_continuity"].loss
        * weights.speaker_continuity,
    }
    total = torch.stack(list(weighted.values())).sum()
    return E2EObjectiveOutput(total=total, terms=terms, weighted=weighted)


__all__ = [
    "E2E_TERM_NAMES",
    "E2E_WEIGHTED_NAMES",
    "E2ELossWeights",
    "E2EObjectiveOutput",
    "LogitConsistencyPair",
    "LossTerm",
    "SpeakerContinuityPair",
    "commit_consistency_kl",
    "commit_pairs_from_full_logits",
    "compute_e2e_objective",
    "distributed_e2e_objective",
    "flattened_commit_consistency_kl",
    "flattened_e2e_objective",
    "flattened_semantic_end_margin_term",
    "flattened_teacher_kl",
    "flattened_token_ce_terms",
    "speaker_continuity_loss",
    "token_ce_terms",
    "token_nll_from_logits",
    "topk_teacher_kl",
]
