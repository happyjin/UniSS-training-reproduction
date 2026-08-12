"""Length-preserving model-prefix rollout for Megatron DAgger training.

The dense trajectory format is structurally immutable: changing a WAIT into a
WRITE would insert a variable-size payload and invalidate every following THD
offset.  Generalize14 therefore rolls in the two variable payloads that cause
the observed exposure failure while preserving grammar and length:

* Qwen text tokens are replaced by Qwen's constrained base-vocabulary choices;
* BiCodec semantic tokens are replaced by the causal microblock head choices.

Labels remain oracle tokens.  The differentiable pass consequently learns to
recover from states induced by its own previous predictions, which is the
DAgger oracle-correction step.  Natural action heads are then trained on hidden
states downstream of those model prefixes without corrupting the packed
action/payload grammar itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from training import constants_uniss as c


@dataclass(frozen=True)
class PrefixProbe:
    """Predicted next-token labels and positions legal for roll-in."""

    labels: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class PrefixSchedule:
    """Number of model roll-in rounds and per-round replacement probability."""

    rounds: int
    probability: float


def prefix_schedule(progress: float) -> PrefixSchedule:
    """Warm up on oracle prefixes, then approach two-round model rollouts."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.10:
        return PrefixSchedule(0, 0.0)
    if progress < 0.30:
        fraction = (progress - 0.10) / 0.20
        return PrefixSchedule(1, 0.10 + 0.15 * fraction)
    if progress < 0.60:
        fraction = (progress - 0.30) / 0.30
        return PrefixSchedule(2, 0.25 + 0.20 * fraction)
    return PrefixSchedule(2, 0.50)


def _semantic_block_geometry(
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return block contexts, flattened target positions and block lengths."""

    active = (token_roles == ROLE_SEMANTIC) & (loss_mask > 0)
    previous = torch.zeros_like(active)
    previous[1:] = active[:-1]
    following = torch.zeros_like(active)
    following[:-1] = active[1:]
    starts = (active & ~previous).nonzero(as_tuple=False).flatten()
    ends = (active & ~following).nonzero(as_tuple=False).flatten()
    if not starts.numel():
        return None
    if starts.numel() != ends.numel():
        raise ValueError("semantic spans have unmatched boundaries")

    contexts: list[int] = []
    positions: list[int] = []
    lengths: list[int] = []
    for start_tensor, end_tensor in zip(starts, ends):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        span_length = end - start + 1
        for offset in range(0, span_length, block_size):
            length = min(block_size, span_length - offset)
            contexts.append(start + offset)
            lengths.append(length)
            positions.extend(range(start + offset, start + offset + length))
    return (
        torch.tensor(contexts, dtype=torch.long, device=token_roles.device),
        torch.tensor(positions, dtype=torch.long, device=token_roles.device),
        torch.tensor(lengths, dtype=torch.long, device=token_roles.device),
    )


def probe_runtime_prefix_labels(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
    word_embedding_weight: torch.Tensor,
    semantic_microblock_head,
    *,
    text_chunk_size: int = 1024,
) -> PrefixProbe:
    """Predict the exact text and microblock labels used by runtime.

    Full-vocabulary logits are intentionally avoided in this no-gradient
    probe.  Only active text rows are projected against the base Qwen
    vocabulary, and semantic rows use the same causal microblock head as PCM
    inference.
    """

    if hidden.ndim != 2:
        raise ValueError("prefix probe hidden states must be [tokens,hidden]")
    values = (labels, token_roles, loss_mask)
    if any(value.ndim != 1 for value in values):
        raise ValueError("prefix probe labels, roles and mask must be flattened")
    if not all(value.numel() == hidden.shape[0] for value in values):
        raise ValueError("prefix probe tensor lengths differ")
    if text_chunk_size <= 0:
        raise ValueError("text probe chunk size must be positive")

    predicted = labels.detach().clone()
    eligible = torch.zeros_like(labels, dtype=torch.bool)
    active = loss_mask > 0
    text_positions = ((token_roles == ROLE_TEXT) & active).nonzero(
        as_tuple=False
    ).flatten()
    if text_positions.numel():
        classes = word_embedding_weight[: c.QWEN_BASE_VOCAB_END + 1]
        choices: list[torch.Tensor] = []
        for start in range(0, text_positions.numel(), text_chunk_size):
            positions = text_positions[start : start + text_chunk_size]
            logits = F.linear(hidden[positions], classes)
            choices.append(logits.float().argmax(dim=-1))
        predicted[text_positions] = torch.cat(choices)
        eligible[text_positions] = True

    geometry = _semantic_block_geometry(
        token_roles,
        loss_mask,
        block_size=int(semantic_microblock_head.block_size),
    )
    if geometry is not None:
        context_positions, semantic_positions, lengths = geometry
        logits = semantic_microblock_head.content_logits(
            hidden[context_positions], word_embedding_weight, teacher_targets=None
        )
        choices = logits.float().argmax(dim=-1) + c.BICODEC_SEMANTIC_OFFSET
        selected: list[torch.Tensor] = []
        for row, length in enumerate(lengths.tolist()):
            selected.append(choices[row, : int(length)])
        semantic_choices = torch.cat(selected)
        if semantic_choices.numel() != semantic_positions.numel():
            raise AssertionError("semantic probe geometry changed token count")
        predicted[semantic_positions] = semantic_choices
        eligible[semantic_positions] = True

    return PrefixProbe(predicted, eligible)


def apply_prefix_predictions(
    tokens: torch.Tensor,
    predicted_labels: torch.Tensor,
    eligible_labels: torch.Tensor,
    *,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift predicted labels into input positions without changing length."""

    if tokens.ndim != 2 or predicted_labels.shape != tokens.shape:
        raise ValueError("prefix replacement tensors must share [batch,seq]")
    if eligible_labels.shape != tokens.shape:
        raise ValueError("prefix eligibility shape differs")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("prefix replacement probability must be in [0,1]")

    updated = tokens.clone()
    corrupted = torch.zeros_like(tokens, dtype=torch.bool)
    if probability <= 0.0 or tokens.shape[1] <= 1:
        return updated, corrupted
    candidates = eligible_labels[:, :-1].bool()
    if probability < 1.0:
        candidates = candidates & (
            torch.rand(candidates.shape, device=tokens.device) < float(probability)
        )
    replacements = predicted_labels[:, :-1]
    destination = updated[:, 1:]
    changed = candidates & (destination != replacements)
    destination[changed] = replacements[changed]
    corrupted[:, 1:] = changed
    return updated, corrupted


def expand_recovery_mask(
    corrupted_inputs: torch.Tensor, *, horizon: int = 8
) -> torch.Tensor:
    """Supervise several oracle continuations after every model-prefix error."""

    if corrupted_inputs.ndim != 2:
        raise ValueError("corrupted input mask must be [batch,seq]")
    if horizon <= 0:
        raise ValueError("recovery horizon must be positive")
    recovery = torch.zeros_like(corrupted_inputs, dtype=torch.bool)
    sequence = corrupted_inputs.shape[1]
    for shift in range(horizon):
        if shift >= sequence:
            break
        recovery[:, shift:] |= corrupted_inputs[:, : sequence - shift]
    return recovery


__all__ = [
    "PrefixProbe",
    "PrefixSchedule",
    "apply_prefix_predictions",
    "expand_recovery_mask",
    "prefix_schedule",
    "probe_runtime_prefix_labels",
]
