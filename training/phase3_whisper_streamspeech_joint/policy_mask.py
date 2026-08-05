"""Detached CTC-count policy and decoder-only Phase3 attention masks."""

from __future__ import annotations

import torch


def ctc_expected_increments(probabilities: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Expected CTC label-count increments from StreamSpeech's approximation."""

    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B,T,V]")
    if not 0 <= blank_id < probabilities.shape[-1]:
        raise ValueError("blank_id is out of range")
    nonblank = probabilities.clone()
    nonblank[..., blank_id] = 0
    previous = torch.zeros_like(nonblank)
    previous[:, 1:] = nonblank[:, :-1]
    repeated = (nonblank * previous).sum(dim=-1)
    increments = 1.0 - probabilities[..., blank_id] - repeated
    return increments.clamp_min(0.0)


def ctc_expected_counts(logits: torch.Tensor, blank_id: int) -> torch.Tensor:
    probabilities = logits.detach().float().softmax(dim=-1)
    return ctc_expected_increments(probabilities, blank_id).cumsum(dim=-1)


@torch.no_grad()
def build_g_from_ctc_logits(
    asr_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    asr_blank_id: int,
    target_blank_id: int,
    target_lengths: torch.Tensor,
    encoder_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return zero-based source boundaries ``g`` with shape ``[B,Ymax]``.

    Boundaries are restricted to frames where rounded ASR expected count gains
    a token. The earliest such frame whose target expected count supports token
    ``i`` is selected. Unsupported tail tokens fall back to the last valid
    encoder frame and can be tracked separately by the caller.
    """

    if asr_logits.shape[:2] != target_logits.shape[:2]:
        raise ValueError("ASR and target CTC time geometry must match")
    batch, time = asr_logits.shape[:2]
    if target_lengths.shape != (batch,) or encoder_lengths.shape != (batch,):
        raise ValueError("length tensors must have shape [B]")
    if bool((encoder_lengths <= 0).any()) or bool((encoder_lengths > time).any()):
        raise ValueError("invalid encoder lengths")
    asr_count = ctc_expected_counts(asr_logits, asr_blank_id)
    target_count = ctc_expected_counts(target_logits, target_blank_id)
    max_target = int(target_lengths.max().item()) if batch else 0
    boundaries = torch.zeros(batch, max_target, dtype=torch.long, device=asr_logits.device)
    for row in range(batch):
        valid_time = int(encoder_lengths[row].item())
        rounded_asr = torch.floor(asr_count[row, :valid_time] + 0.5)
        previous = torch.cat([rounded_asr.new_zeros(1), rounded_asr[:-1]])
        source_events = rounded_asr > previous
        candidate_frames = torch.nonzero(source_events, as_tuple=False).flatten()
        if not len(candidate_frames):
            candidate_frames = torch.tensor([valid_time - 1], device=asr_logits.device)
        for target_index in range(int(target_lengths[row].item())):
            supported = target_count[row, candidate_frames] >= float(target_index + 1)
            if bool(supported.any()):
                boundaries[row, target_index] = candidate_frames[torch.nonzero(supported)[0, 0]]
            else:
                boundaries[row, target_index] = valid_time - 1
    return boundaries


def phase3_block_attention_allowed(
    *,
    prefix_length: int,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Build ``[B,L,L]`` causal visibility with per-target source boundaries.

    Layout is ``[fixed prefix][source hidden][target text]``. Prefix/source
    queries remain ordinarily causal. A target query sees all fixed-prefix
    keys, source keys through ``g(i)``, and earlier/current target keys.
    """

    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    if source_lengths.ndim != 1 or target_lengths.ndim != 1:
        raise ValueError("length tensors must be rank 1")
    if source_lengths.shape != target_lengths.shape:
        raise ValueError("source/target batch sizes differ")
    batch = len(source_lengths)
    max_source = int(source_lengths.max().item()) if batch else 0
    max_target = int(target_lengths.max().item()) if batch else 0
    if g.shape != (batch, max_target):
        raise ValueError(f"g must have shape {(batch, max_target)}, got {tuple(g.shape)}")
    total = prefix_length + max_source + max_target
    positions = torch.arange(total, device=source_lengths.device)
    causal = positions[None, :] <= positions[:, None]
    allowed = causal.unsqueeze(0).expand(batch, -1, -1).clone()
    for row in range(batch):
        source_length = int(source_lengths[row].item())
        target_length = int(target_lengths[row].item())
        source_start = prefix_length
        target_start = prefix_length + max_source
        valid = torch.zeros(total, dtype=torch.bool, device=allowed.device)
        valid[:prefix_length] = True
        valid[source_start : source_start + source_length] = True
        valid[target_start : target_start + target_length] = True
        allowed[row] &= valid[:, None] & valid[None, :]
        for target_index in range(target_length):
            query = target_start + target_index
            visible_source_end = min(int(g[row, target_index].item()) + 1, source_length)
            allowed[row, query, source_start : source_start + source_length] = False
            allowed[row, query, source_start : source_start + visible_source_end] = True
    return allowed
