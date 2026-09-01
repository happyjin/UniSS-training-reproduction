#!/usr/bin/env python3
"""The three terms the bias sweep says are the fix, and none of the ones it says are not.

Measured in `reports/uniss_phase3_e2e_speak_decision_v1/family_logit_probe/
BIAS_SWEEP_ANALYSIS.zh-CN.md`:

* A +3 logit bias on the continue-after-fragment decision alone takes WRITE_MT
  per event 0.168 -> 0.863, semantic coverage 0.666 -> 0.997 and `natural_eos`
  0.50 -> 1.00, with WRITE_ASR unchanged at 0.842.  The probe measured that
  decision's median gap at -2.88, and delta 1 / 2 / 3 reproduce that threshold
  exactly (no change / slight / jump).  So the requirement is about 3 logits at
  one position.
* Its only side effect is over-generation, text length ratio 1.03 -> 2.25 with a
  44.45 worst case.  `content_end_ce` masks exactly `TOKEN_END_CONTENT` and has
  been 0.0 in every run in the project's history, so the model has never been
  taught to close a text fragment.
* The repetition term measured at delta 4 cuts the worst-case length ratio from
  44.45 to 4.07, a 10.9x reduction.  It works; it was simply never exercised
  because the model never wrote enough to repeat.

Deliberately absent: the `speak_decision` term from the previous experiment.  It
targeted the *first* WRITE/WAIT decision of an event, which already leads by a
median 28.58 logits, and its half-weight WAIT class lifted `wait_logit`
globally, moving the one decision that matters from -2.88 to -3.75.

Nothing in `uniss_phase3_v4_e2e_simuls2st_pilot15_v1` is modified.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn import functional as F

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training import objective as base
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
    E2E_TERM_NAMES,
    E2E_WEIGHTED_NAMES,
    LossTerm,
)
from experiments.uniss_phase3_e2e_speak_decision_v1.training.objective_ext import (
    LOGSUMEXP_ROW_CHUNK,
    _chunked_logsumexp,
    repetition_penalty_term,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
)
from training import constants_uniss as c

EXTRA_TERM_NAMES = ("continue_after_fragment", "content_end_margin", "repetition_penalty")
EXTRA_WEIGHTED_NAMES = ("continue_after_fragment", "content_end_margin", "repetition_penalty")
EXTENDED_TERM_NAMES = E2E_TERM_NAMES + EXTRA_TERM_NAMES
EXTENDED_WEIGHTED_NAMES = E2E_WEIGHTED_NAMES + EXTRA_WEIGHTED_NAMES

# A decision row opens an event when no fragment has been written since the last
# event terminator.  These are the tokens `run_event` breaks on, plus the family
# tokens that mark a fragment as having started.
EVENT_TERMINATORS = (c.TOKEN_WAIT_READ, c.TOKEN_START_GLM, c.TOKEN_EOS)
FAMILY_TOKENS = (c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS)


def continue_after_fragment_mask(
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    original_seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WRITE / WAIT rows that are *not* the first decision of their event.

    Walking the label stream: a family token means a fragment has started, and
    an event terminator resets the count.  A WRITE_GENERATE or WAIT_READ row
    with at least one fragment behind it in the current event is the decision
    the probe measured at -2.88.  The first decision of an event leads by +28.58
    and is deliberately left unsupervised.

    Two details the tests pin down.  The reset is *exclusive*: WAIT_READ is both
    an event terminator and a decision token, so a reset that included its own
    row would make every WAIT row invisible.  And the scan runs per sequence,
    not down the flattened batch, so one sequence's trailing state cannot leak
    into the next one's opening decision.
    """

    if labels.ndim != 1 or labels.shape != loss_kinds.shape:
        raise ValueError("labels and loss_kinds must be matching 1-D tensors")
    if original_seq_length <= 0 or labels.numel() % original_seq_length:
        raise ValueError("labels must divide evenly into sequences")
    rows = labels.numel() // original_seq_length
    grid = labels.reshape(rows, original_seq_length)

    starts_fragment = torch.zeros_like(grid, dtype=torch.bool)
    for token in FAMILY_TOKENS:
        starts_fragment |= grid == token
    resets = torch.zeros_like(grid, dtype=torch.bool)
    for token in EVENT_TERMINATORS:
        resets |= grid == token

    # Fragments seen strictly before each position, within its own sequence.
    order = torch.cumsum(starts_fragment.long(), dim=1) - starts_fragment.long()
    marked = torch.where(resets, order, torch.zeros_like(order))
    running = torch.cummax(marked, dim=1).values
    baseline = torch.zeros_like(running)
    baseline[:, 1:] = running[:, :-1]
    after_fragment = ((order - baseline) >= 1).reshape(-1)

    boundary = loss_kinds == LOSS_BOUNDARY
    write = (labels == c.TOKEN_WRITE_GENERATE) & boundary & after_fragment
    wait = (labels == c.TOKEN_WAIT_READ) & boundary & after_fragment
    return write, wait


def continue_after_fragment_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    margin: float,
    original_seq_length: int,
) -> LossTerm:
    """One margin on the WRITE-minus-WAIT score, both classes in a single term.

    The previous experiment returned the two classes as independent terms, which
    handed the minority WAIT class 2.28x the per-row weight and pushed the score
    the wrong way.  Here both classes share one denominator, so a row's weight is
    a row's weight.
    """

    if logits.ndim != 2 or labels.ndim != 1 or loss_kinds.ndim != 1:
        raise ValueError("expected flattened logits with 1-D labels and kinds")
    if logits.shape[0] != labels.numel():
        raise ValueError("logits rows must match labels")
    write, wait = continue_after_fragment_mask(
        labels, loss_kinds, original_seq_length=original_seq_length
    )
    selected = write | wait
    denominator = selected.sum().float()
    if not bool(selected.any()):
        return LossTerm(logits.sum() * 0.0, denominator)
    rows = logits[selected].float()
    score = rows[:, c.TOKEN_WRITE_GENERATE] - rows[:, c.TOKEN_WAIT_READ]
    # WRITE rows want score >= +margin, WAIT rows want score <= -margin.
    sign = torch.where(write[selected], 1.0, -1.0).to(score.dtype)
    losses = F.softplus(margin - sign * score)
    return LossTerm(losses.sum(), denominator)


def content_end_margin_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    margin: float,
) -> LossTerm:
    """Make END_CONTENT dominate at gold text-fragment ends, not merely be likely.

    `content_end_ce` (weight 0.0 in every run to date) is the cross-entropy form.
    A margin against the strongest competitor is what transfers off-manifold:
    at inference the model is closing a fragment it generated itself, and a
    calibrated dominance survives that shift where a likelihood does not.  This
    mirrors `semantic_end_margin`, which the epoch23 run validated on the
    semantic side.
    """

    if logits.ndim != 2 or labels.ndim != 1 or loss_kinds.ndim != 1:
        raise ValueError("expected flattened logits with 1-D labels and kinds")
    mask = (loss_kinds == LOSS_BOUNDARY) & (labels == c.TOKEN_END_CONTENT)
    denominator = mask.sum().float()
    if not bool(mask.any()):
        return LossTerm(logits.sum() * 0.0, denominator)
    rows = logits[mask].float()
    target = rows[:, c.TOKEN_END_CONTENT]
    competitor = rows.clone()
    competitor[:, c.TOKEN_END_CONTENT] = float("-inf")
    best_other = competitor.max(dim=-1).values
    losses = F.softplus(margin - (target - best_other))
    return LossTerm(losses.sum(), denominator)


def flattened_with_continue_end(
    *,
    continue_after_fragment_logit_margin: float = 0.0,
    content_end_logit_margin: float = 0.0,
    repetition_window: int = 8,
    **kwargs,
) -> Mapping[str, LossTerm]:
    terms = dict(base.flattened_e2e_objective(**kwargs))
    if tuple(terms) != E2E_TERM_NAMES:
        raise AssertionError("established E2E term order changed")
    terms["continue_after_fragment"] = continue_after_fragment_term(
        kwargs["logits"],
        kwargs["labels"],
        kwargs["loss_kinds"],
        margin=continue_after_fragment_logit_margin,
        original_seq_length=int(kwargs["original_seq_length"]),
    )
    terms["content_end_margin"] = content_end_margin_term(
        kwargs["logits"],
        kwargs["labels"],
        kwargs["loss_kinds"],
        margin=content_end_logit_margin,
    )
    terms["repetition_penalty"] = repetition_penalty_term(
        kwargs["logits"],
        kwargs["labels"],
        kwargs["loss_kinds"],
        original_seq_length=int(kwargs["original_seq_length"]),
        window=int(repetition_window),
    )
    if tuple(terms) != EXTENDED_TERM_NAMES:
        raise AssertionError("extended E2E term order changed")
    return terms


def _global_mean(
    term: LossTerm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local mean, global mean and global denominator, matching the base pattern."""

    numerator = term.numerator
    denominator = term.denominator.to(numerator.dtype)
    global_numerator = numerator.detach().clone()
    global_denominator = denominator.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_numerator)
        dist.all_reduce(global_denominator)
        world_size = dist.get_world_size()
    active = global_denominator > 0
    local_mean = torch.where(
        active,
        world_size * numerator / global_denominator.clamp_min(1.0),
        numerator * 0.0,
    )
    global_mean = torch.where(
        active,
        global_numerator / global_denominator.clamp_min(1.0),
        global_numerator * 0.0,
    )
    return local_mean, global_mean, global_denominator


def distributed_with_continue_end(
    terms: Mapping[str, LossTerm],
    *,
    weights=None,
    continue_after_fragment: float = 0.0,
    content_end_margin: float = 0.0,
    repetition_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Delegate the established terms, then add the three new weighted terms.

    Each term carries a single weight over a single denominator.  The previous
    experiment split one decision into two per-class terms, which handed the
    minority class 2.28x the per-row weight; here the WRITE and WAIT rows are
    already 0.98:1 on gold data and share one denominator, so no class
    re-weighting is needed or applied.
    """

    if tuple(terms) != EXTENDED_TERM_NAMES:
        raise ValueError("extended E2E objective term order changed")
    for name, value in (
        ("continue_after_fragment", continue_after_fragment),
        ("content_end_margin", content_end_margin),
        ("repetition_penalty", repetition_penalty),
    ):
        if value < 0:
            raise ValueError(f"{name} weight must be non-negative")
    core = {name: terms[name] for name in E2E_TERM_NAMES}
    total, metrics = base.distributed_e2e_objective(core, weights=weights)
    metrics = dict(metrics)

    emitted = []
    for name, weight in (
        ("continue_after_fragment", continue_after_fragment),
        ("content_end_margin", content_end_margin),
        ("repetition_penalty", repetition_penalty),
    ):
        local, global_mean, denominator = _global_mean(terms[name])
        total = total + local * float(weight)
        emitted.append((name, global_mean, denominator, local * float(weight)))

    # The trainer asserts metric order twice; the contract groups every loss/*
    # before every denominator/* before every weighted/*.
    for name, value, _, _ in emitted:
        metrics[f"loss/{name}"] = value
    for name, _, denominator, _ in emitted:
        metrics[f"denominator/{name}"] = denominator
    for name, _, _, weighted in emitted:
        metrics[f"weighted/{name}"] = weighted
    return total, metrics


def extended_objective_metric_names(established: Sequence[str]) -> tuple[str, ...]:
    """Rebuild the trainer's metric contract with the new names appended."""

    return (
        *established,
        *(f"loss/{name}" for name in EXTRA_TERM_NAMES),
        *(f"denominator/{name}" for name in EXTRA_TERM_NAMES),
        *(f"weighted/{name}" for name in EXTRA_WEIGHTED_NAMES),
    )



__all__ = [
    "EXTENDED_TERM_NAMES",
    "distributed_with_continue_end",
    "extended_objective_metric_names",
    "EXTENDED_WEIGHTED_NAMES",
    "EXTRA_TERM_NAMES",
    "EXTRA_WEIGHTED_NAMES",
    "content_end_margin_term",
    "continue_after_fragment_mask",
    "continue_after_fragment_term",
    "flattened_with_continue_end",
]
