"""Λ-shaped KV cache helpers (InfiniSST-style window) for Step 3.

Keeps the system/prompt prefix intact and retains only the trailing ``window``
tokens of the growing decode cache. Full InfiniSST also strips RoPE before
storage and reapplies after concat; this module starts with the length-window
half so Stage10's append-only ``past_key_values`` can be bounded without editing
shipping adapters. RoPE-correct storage is a follow-up behind the same API.
"""

from __future__ import annotations

from typing import Any

import torch


def cache_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    first = past_key_values[0][0]
    return int(first.shape[-2])


def _cat_prefix_tail(tensor: torch.Tensor, system_tokens: int, start_tail: int) -> torch.Tensor:
    return torch.cat((tensor[..., :system_tokens, :], tensor[..., start_tail:, :]), dim=-2)


def prune_to_lambda(
    past_key_values: Any,
    *,
    system_tokens: int,
    window: int,
) -> Any:
    """Return a new past_key_values truncated to ``system + last window``.

    If the cache is already within budget, returns it unchanged.
    """

    if past_key_values is None:
        return None
    if system_tokens < 0 or window < 0:
        raise ValueError("system_tokens and window must be non-negative")
    length = cache_length(past_key_values)
    budget = system_tokens + window
    if length <= budget:
        return past_key_values
    keep_tail = max(0, length - system_tokens)
    tail = min(window, keep_tail)
    if system_tokens > 0 and tail > 0:
        start_tail = length - tail
        if start_tail > system_tokens:
            return tuple(
                (
                    _cat_prefix_tail(key, system_tokens, start_tail),
                    _cat_prefix_tail(value, system_tokens, start_tail),
                )
                for key, value in past_key_values
            )
    return tuple(
        (key[..., -budget:, :], value[..., -budget:, :]) for key, value in past_key_values
    )
