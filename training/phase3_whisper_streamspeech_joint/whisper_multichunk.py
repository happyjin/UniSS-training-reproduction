"""Chunk-causal masks for the unchanged Phase3 WhisperVQ encoder."""

from __future__ import annotations

import random

import torch

from .config import MultiChunkConfig


def choose_chunk_ms(config: MultiChunkConfig, *, seed: int, sample_index: int) -> int | None:
    """Choose a reproducible chunk without depending on rank-local RNG state."""

    generator = random.Random((int(seed) << 32) ^ int(sample_index))
    return config.chunk_ms[generator.randrange(len(config.chunk_ms))]


def chunk_causal_allowed(
    valid_frames: torch.Tensor,
    *,
    sequence_length: int,
    chunk_frames: int | None,
    right_context_frames: int,
) -> torch.Tensor:
    """Return a boolean ``[B,T,T]`` key-visibility mask.

    A query sees all history plus the remainder of its current chunk and the
    configured right context. Offline mode sees every valid key. Padded query
    rows are kept false so downstream code cannot accidentally train on them.
    """

    if valid_frames.ndim != 1:
        raise ValueError("valid_frames must be rank 1")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if right_context_frames < 0:
        raise ValueError("right_context_frames must be non-negative")
    if chunk_frames is not None and chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive or None")

    device = valid_frames.device
    positions = torch.arange(sequence_length, device=device)
    query_valid = positions[None, :] < valid_frames[:, None]
    key_valid = query_valid
    if chunk_frames is None:
        temporal = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
    else:
        chunk_end = (
            torch.div(positions, chunk_frames, rounding_mode="floor") + 1
        ) * chunk_frames - 1
        temporal = positions[None, :] <= (chunk_end[:, None] + right_context_frames)
    return temporal.unsqueeze(0) & query_valid[:, :, None] & key_valid[:, None, :]


def additive_attention_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert ``[B,Q,K]`` visibility to additive ``[B,1,Q,K]`` form."""

    if allowed.ndim != 3 or allowed.dtype != torch.bool:
        raise ValueError("allowed must be a boolean [B,Q,K] tensor")
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(1)
