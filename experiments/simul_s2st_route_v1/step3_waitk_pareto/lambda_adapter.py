"""Non-invasive Λ-window wrapper around Stage10 ``CachedMicroWriteAdapter``.

Does not edit Stage10/11 sources. After each cache mutation, prunes
``past_key_values`` to ``system_tokens + window`` using ``prune_to_lambda``.
"""

from __future__ import annotations

from typing import Any

import torch

from experiments.simul_s2st_route_v1.step3_waitk_pareto.lambda_kv_cache import (
    cache_length,
    prune_to_lambda,
)


class LambdaWindowAdapter:
    def __init__(self, adapter: Any, *, window: int) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        self.inner = adapter
        self.window = int(window)
        # Prompt tokens written in CachedMicroWriteAdapter.__post_init__.
        self.system_tokens = int(adapter._cache_length())

    def _prune(self) -> None:
        if self.window <= 0:
            return
        cache = self.inner.cache
        if cache is None:
            return
        pruned = prune_to_lambda(
            cache, system_tokens=self.system_tokens, window=self.window
        )
        self.inner.cache = pruned
        self.inner.cache_tokens = cache_length(pruned)

    def append_source(self, embeddings: torch.Tensor) -> None:
        self.inner.append_source(embeddings)
        self._prune()

    def commit_wait(self) -> None:
        self.inner.commit_wait()
        self._prune()

    def generate_write(self):
        write = self.inner.generate_write()
        self._prune()
        return write

    def __getattr__(self, name: str):
        return getattr(self.inner, name)
