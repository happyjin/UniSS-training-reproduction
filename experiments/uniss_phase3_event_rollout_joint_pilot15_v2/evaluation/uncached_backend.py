"""No-KV-cache Qwen backend for fixed15 runtime parity auditing."""

from __future__ import annotations

from typing import Any

import torch

from web_demo.runtime_parity_streaming_v2.hf_backend import HuggingFaceKVBackend


class UncachedHuggingFaceBackend(HuggingFaceKVBackend):
    """Recompute the complete committed prompt on every append.

    The regular backend uses Hugging Face's dynamic KV cache.  This backend
    preserves the identical session grammar and frontend residuals but stores
    the committed input embeddings and forwards the full history with
    ``use_cache=False``.  Comparing its actions/text/semantic/EOS trace against
    the dynamic-cache backend therefore audits real cached/uncached parity.
    """

    def __init__(self, *args, use_static_cache: bool = False, **kwargs) -> None:
        if use_static_cache:
            raise ValueError("uncached backend cannot also use a static KV cache")
        super().__init__(*args, use_static_cache=False, **kwargs)
        self._committed_embeddings: torch.Tensor | None = None

    def _forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any,
        capture_last_hidden: bool = False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one model input representation is required")
        if past_key_values is not None:
            raise RuntimeError("uncached session unexpectedly supplied past_key_values")
        appended = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        assert appended is not None
        self._committed_embeddings = (
            appended
            if self._committed_embeddings is None
            else torch.cat((self._committed_embeddings, appended), dim=1)
        )
        return self.model(
            inputs_embeds=self._committed_embeddings,
            use_cache=False,
            output_hidden_states=capture_last_hidden,
            return_dict=True,
        )


__all__ = ["UncachedHuggingFaceBackend"]
