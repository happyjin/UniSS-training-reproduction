#!/usr/bin/env python3
"""Two loss terms the E2E objective does not have: speak decision and repetition.

Measured motivation, from `streaming_s2st_metrics_v1/`:

* Over 95 S2S events the model recognises on 82% but translates on 16.8% and
  speaks on 15.8%.  Low coverage, over-generation, 1240-1640 ms gaps and
  `natural_eos` frozen at 0.50 across three coverage epochs are one root cause
  seen four ways.  S0.1 then showed the decision cannot be supplied at
  inference: the three policies land at 0.168 (starved), 0.958 and 1.000 (both
  repetition loops), with nothing in between.
* Raising `boundary_ce` cannot substitute for this term.  Both decisions *are*
  in that bucket (`_mark_fragment` marks WRITE_GENERATE and WAIT_READ alike as
  `LOSS_BOUNDARY`), but so are `END_CONTENT`, `END_SEMANTIC`, the language and
  speed tokens: an undifferentiated cross-entropy over all of them has no
  margin, no class balancing, and dilutes the decision among tokens that are
  not decisions.
* Every intervention that raises the speak rate triggers repetition.  On
  `emilia_zh_0004122419` the session text length ratio goes 1.70 -> 15.40 with
  output like "in a state of being in a state of being ...", and no existing
  term penalises repetition.

Both terms are computable from `labels` and `loss_kinds` alone, so no data
rebuild is needed.  Nothing in `uniss_phase3_v4_e2e_simuls2st_pilot15_v1` is
modified: this module wraps its two objective entry points and adds three named
terms after the established ones.
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
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_MT,
    LOSS_SEMANTIC,
)
from training import constants_uniss as c


EXTRA_TERM_NAMES = (
    "speak_decision_write",
    "speak_decision_wait",
    "repetition_penalty",
)
EXTRA_WEIGHTED_NAMES = ("speak_decision", "repetition_penalty")
EXTENDED_TERM_NAMES = (*E2E_TERM_NAMES, *EXTRA_TERM_NAMES)
EXTENDED_WEIGHTED_NAMES = (*E2E_WEIGHTED_NAMES, *EXTRA_WEIGHTED_NAMES)

# Repetition is penalised where it was observed: the generated text and speech.
REPETITION_KINDS = (LOSS_MT, LOSS_SEMANTIC)
DEFAULT_REPETITION_WINDOW = 8


LOGSUMEXP_ROW_CHUNK = 1024


def _chunked_logsumexp(
    logits: torch.Tensor, scored: torch.Tensor
) -> torch.Tensor:
    """Per-row log-normaliser for the scored rows, without a full softmax.

    Materialising ``logits.float().softmax(-1)`` costs rows x vocabulary x 4
    bytes, which is 26 GB at this geometry and is what OOMed the first launch.
    Chunking bounds the temporary to ``LOGSUMEXP_ROW_CHUNK`` rows.
    """

    output = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    index = scored.nonzero(as_tuple=False).reshape(-1)
    for start in range(0, int(index.numel()), LOGSUMEXP_ROW_CHUNK):
        piece = index[start : start + LOGSUMEXP_ROW_CHUNK]
        output[piece] = torch.logsumexp(logits[piece].float(), dim=-1)
    return output


def speak_decision_masks(
    labels: torch.Tensor, loss_kinds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows that decided to speak, and rows that decided to wait.

    ``_mark_fragment`` (``task_samples.py:246-259``) labels **both**
    ``TOKEN_WRITE_GENERATE`` and ``TOKEN_WAIT_READ`` as ``LOSS_BOUNDARY``, not
    as the fragment's content kind, so both decisions live in the same bucket.
    A first version required the WRITE rows to carry LOSS_ASR / LOSS_MT /
    LOSS_SEMANTIC; that never matched, the WRITE class stayed empty for every
    interleaved batch, and the term degenerated into a one-sided push toward
    WAIT -- the opposite of its purpose.  ``test_masks_match_the_real_packing``
    pins the convention against ``_mark_fragment`` itself.
    """

    boundary = loss_kinds == LOSS_BOUNDARY
    write = (labels == c.TOKEN_WRITE_GENERATE) & boundary
    wait = (labels == c.TOKEN_WAIT_READ) & boundary
    return write, wait


def speak_decision_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    margin: float,
) -> tuple[LossTerm, LossTerm]:
    """Calibrate WRITE_GENERATE against WAIT_READ, one class at a time.

    The restricted binary score is ``z = wait_logit - write_logit``.  WAIT rows
    minimise ``softplus(margin - z)`` and WRITE rows ``softplus(margin + z)``.
    Softplus rather than relu because a hinge stops producing gradient once the
    margin is met, and the two classes are returned as independent
    :class:`LossTerm` objects so each receives exactly half the configured
    weight regardless of the observed five-to-one imbalance between WAIT at
    0.863 per event and WRITE_MT at 0.168.
    """

    if margin < 0:
        raise ValueError("speak decision margin must be non-negative")
    if logits.ndim != 2 or labels.ndim != 1 or loss_kinds.ndim != 1:
        raise ValueError("flattened speak decision tensors have invalid rank")
    if logits.shape[0] != labels.numel() or labels.shape != loss_kinds.shape:
        raise ValueError("flattened speak decision tensors differ in length")
    write, wait = speak_decision_masks(labels, loss_kinds)

    def class_term(mask: torch.Tensor, *, target_write: bool) -> LossTerm:
        denominator = mask.sum().float()
        if not bool(mask.any()):
            return LossTerm(logits.sum() * 0.0, denominator)
        rows = logits[mask].float()
        score = rows[:, c.TOKEN_WAIT_READ] - rows[:, c.TOKEN_WRITE_GENERATE]
        losses = F.softplus(
            float(margin) + score if target_write else float(margin) - score
        )
        return LossTerm(losses.sum(), denominator)

    return class_term(write, target_write=True), class_term(wait, target_write=False)


def repetition_penalty_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_kinds: torch.Tensor,
    *,
    original_seq_length: int,
    window: int = DEFAULT_REPETITION_WINDOW,
) -> LossTerm:
    """Penalise probability placed on tokens already emitted in this fragment.

    For every supervised generation row the penalty is the probability mass the
    model assigns to labels that appear in the preceding ``window`` positions of
    the *same contiguous run of the same loss kind*, excluding the gold next
    token.  Restricting to one run keeps the window inside a single fragment,
    which is where ``of of`` / ``new new`` / ``in a state of being in a state of
    being`` was observed, and avoids reaching across packed samples.

    The value is a probability in [0, 1], so it is bounded and cannot dominate
    the cross-entropy the way an unbounded log-ratio penalty would.
    """

    if window < 1:
        raise ValueError("repetition window must be positive")
    width = int(original_seq_length)
    if width <= 0 or labels.numel() % width != 0:
        raise ValueError("repetition penalty sequence geometry differs")
    rows = labels.numel() // width
    labels_2d = labels.reshape(rows, width)
    kinds_2d = loss_kinds.reshape(rows, width)

    scored = torch.zeros_like(loss_kinds, dtype=torch.bool)
    for kind in REPETITION_KINDS:
        scored |= loss_kinds == kind
    if not bool(scored.any()):
        return LossTerm(logits.sum() * 0.0, scored.sum().float())

    # A full-vocabulary softmax over every position is 18000 x 2 rows by a
    # 180k vocabulary in float32, which is 26 GB and OOMs a 139 GB card at
    # iteration four.  Only p(specific token | row) is ever needed, so the
    # normaliser is computed in row chunks and the probabilities are gathered.
    log_normaliser = _chunked_logsumexp(logits, scored)
    flat_index = torch.arange(labels.numel(), device=logits.device).reshape(rows, width)
    penalty = torch.zeros(
        labels.numel(), device=logits.device, dtype=log_normaliser.dtype
    )
    counted = torch.zeros_like(penalty, dtype=torch.bool)

    for offset in range(1, int(window) + 1):
        if offset >= width:
            break
        # Position p looks back at p - offset; both must be supervised rows of
        # the same kind, which keeps the comparison inside one fragment.
        current = flat_index[:, offset:].reshape(-1)
        previous = flat_index[:, : width - offset].reshape(-1)
        same_run = (
            scored[current]
            & scored[previous]
            & (loss_kinds[current] == loss_kinds[previous])
        )
        if not bool(same_run.any()):
            continue
        rows_current = current[same_run]
        earlier_labels = labels[previous[same_run]].long()
        gold = labels[rows_current].long()
        # A legitimate repeat of the gold token is not a defect.
        contributes = earlier_labels != gold
        if not bool(contributes.any()):
            continue
        target_rows = rows_current[contributes]
        gathered = logits[target_rows, earlier_labels[contributes]].float()
        penalty.index_add_(
            0,
            target_rows,
            torch.exp(gathered - log_normaliser[target_rows]),
        )
        counted[target_rows] = True

    denominator = counted.sum().float()
    if not bool(counted.any()):
        return LossTerm(logits.sum() * 0.0, denominator)
    return LossTerm(penalty[counted].sum(), denominator)


def flattened_with_speak_decision(
    *,
    speak_decision_logit_margin: float = 0.0,
    repetition_window: int = DEFAULT_REPETITION_WINDOW,
    **kwargs,
) -> Mapping[str, LossTerm]:
    """The established terms, then the two this experiment adds."""

    terms = dict(base.flattened_e2e_objective(**kwargs))
    if tuple(terms) != E2E_TERM_NAMES:
        raise AssertionError("established E2E term order changed")
    write, wait = speak_decision_terms(
        kwargs["logits"],
        kwargs["labels"],
        kwargs["loss_kinds"],
        margin=speak_decision_logit_margin,
    )
    terms["speak_decision_write"] = write
    terms["speak_decision_wait"] = wait
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


def distributed_with_speak_decision(
    terms: Mapping[str, LossTerm],
    *,
    weights=None,
    speak_decision: float = 0.0,
    repetition_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Delegate the established terms, then add the two new weighted terms."""

    if tuple(terms) != EXTENDED_TERM_NAMES:
        raise ValueError("extended E2E objective term order changed")
    if speak_decision < 0 or repetition_penalty < 0:
        raise ValueError("extension weights must be non-negative")
    core = {name: terms[name] for name in E2E_TERM_NAMES}
    total, metrics = base.distributed_e2e_objective(core, weights=weights)
    metrics = dict(metrics)

    write_local, write_global, write_den = _global_mean(terms["speak_decision_write"])
    wait_local, wait_global, wait_den = _global_mean(terms["speak_decision_wait"])
    rep_local, rep_global, rep_den = _global_mean(terms["repetition_penalty"])
    # Half the weight to each class, exactly as semantic_boundary_binary does,
    # so the five-to-one WAIT/WRITE imbalance cannot decide the gradient.
    speak_local = 0.5 * (write_local + wait_local)
    speak_global = 0.5 * (write_global + wait_global)

    total = total + speak_local * float(speak_decision)
    total = total + rep_local * float(repetition_penalty)

    # The trainer asserts metric order twice, and the contract groups every
    # loss/* before every denominator/* -- see extended_objective_metric_names.
    emitted = (
        ("speak_decision_write", write_global, write_den),
        ("speak_decision_wait", wait_global, wait_den),
        ("repetition_penalty", rep_global, rep_den),
    )
    for name, value, _ in emitted:
        metrics[f"loss/{name}"] = value
    for name, _, denominator in emitted:
        metrics[f"denominator/{name}"] = denominator
    metrics["loss/speak_decision"] = speak_global
    metrics["weighted/speak_decision"] = speak_local * float(speak_decision)
    metrics["weighted/repetition_penalty"] = rep_local * float(repetition_penalty)
    return total, metrics


def extended_objective_metric_names(established: Sequence[str]) -> tuple[str, ...]:
    """Rebuild the trainer's metric contract with the new names appended."""

    return (
        *established,
        *(f"loss/{name}" for name in EXTRA_TERM_NAMES),
        *(f"denominator/{name}" for name in EXTRA_TERM_NAMES),
        "loss/speak_decision",
        *(f"weighted/{name}" for name in EXTRA_WEIGHTED_NAMES),
    )


__all__ = [
    "DEFAULT_REPETITION_WINDOW",
    "EXTENDED_TERM_NAMES",
    "EXTENDED_WEIGHTED_NAMES",
    "EXTRA_TERM_NAMES",
    "EXTRA_WEIGHTED_NAMES",
    "distributed_with_speak_decision",
    "extended_objective_metric_names",
    "flattened_with_speak_decision",
    "repetition_penalty_term",
    "speak_decision_masks",
    "speak_decision_terms",
]
