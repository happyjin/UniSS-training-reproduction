"""Λ-shaped KV cache helpers (InfiniSST-style window) for Step 3.

Keeps the system/prompt prefix intact and retains only the trailing ``window``
tokens of the growing decode cache. Full InfiniSST also strips RoPE before
storage and reapplies after concat; this module starts with the length-window
half so Stage10's append-only ``past_key_values`` can be bounded without editing
shipping adapters. RoPE-correct storage is a follow-up behind the same API.

Transformers >=4.53 returns ``DynamicCache`` from Qwen2; pruning must preserve
that type (legacy tuples raise ``ValueError`` in ``modeling_qwen2``).
"""

from __future__ import annotations

from typing import Any

import torch


def _is_hf_cache(past_key_values: Any) -> bool:
    return hasattr(past_key_values, "to_legacy_cache") and hasattr(
        past_key_values, "get_seq_length"
    )


def cache_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    first = past_key_values[0][0]
    return int(first.shape[-2])


def _cat_prefix_tail(tensor: torch.Tensor, system_tokens: int, start_tail: int) -> torch.Tensor:
    return torch.cat((tensor[..., :system_tokens, :], tensor[..., start_tail:, :]), dim=-2)


def _prune_legacy(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    system_tokens: int,
    window: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
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


def _to_hf_cache(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...], template: Any) -> Any:
    from_legacy = getattr(type(template), "from_legacy_cache", None)
    if callable(from_legacy):
        return from_legacy(legacy)
    # Fallback import for DynamicCache if template class lost the classmethod.
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(legacy)


def prune_to_lambda(
    past_key_values: Any,
    *,
    system_tokens: int,
    window: int,
) -> Any:
    """Return past_key_values truncated to ``system + last window``.

    Preserves HF ``Cache`` objects. If already within budget, returns unchanged.
    """

    if past_key_values is None:
        return None
    if system_tokens < 0 or window < 0:
        raise ValueError("system_tokens and window must be non-negative")
    length = cache_length(past_key_values)
    budget = system_tokens + window
    if length <= budget:
        return past_key_values

    if _is_hf_cache(past_key_values):
        legacy = past_key_values.to_legacy_cache()
        pruned = _prune_legacy(legacy, system_tokens=system_tokens, window=window)
        return _to_hf_cache(pruned, past_key_values)

    return _prune_legacy(
        past_key_values, system_tokens=system_tokens, window=window
    )
