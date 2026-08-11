"""Hugging Face Qwen backend for the training-identical persistent session."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.session import KVAppendResult


class HuggingFaceKVBackend:
    """Append tokens or one source delta to a single evolving Qwen cache.

    Dense training applies the frontend adapter independently to each tick's
    newly visible source-code delta.  This runtime intentionally does the same:
    adapter state is reset at every ``append_source_codes`` call.  A future
    continuous adapter is legal only after both the packer/training objective
    and this backend are changed together.
    """

    def __init__(self, model, objective, *, device: str | torch.device) -> None:
        self.model = model.eval()
        self.objective = objective.eval()
        self.device = torch.device(device)
        self.embedding = self.model.get_input_embeddings()

    @torch.inference_mode()
    def append_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        past_key_values: Any,
        capture_last_hidden: bool = False,
    ) -> KVAppendResult:
        values = [int(value) for value in token_ids]
        if not values:
            raise ValueError("cannot append an empty token block")
        ids = torch.tensor([values], dtype=torch.long, device=self.device)
        output = self.model(
            input_ids=ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=capture_last_hidden,
            return_dict=True,
        )
        last_hidden = (
            output.hidden_states[-1][:, -1].detach()
            if capture_last_hidden
            else None
        )
        return KVAppendResult(
            past_key_values=output.past_key_values,
            logits=output.logits[:, -1].detach(),
            last_hidden=last_hidden,
        )

    @torch.inference_mode()
    def append_source_codes(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values: Any,
    ) -> KVAppendResult:
        codes = [int(value) for value in source_codes]
        canonical = [int(value) for value in canonical_token_ids]
        expected = list(c.encode_glm_semantic(codes))
        if not codes or canonical != expected:
            raise ValueError("source codes and canonical GLM token IDs disagree")
        code_tensor = torch.tensor(
            [codes], dtype=torch.long, device=self.device
        )
        token_tensor = torch.tensor(
            [canonical], dtype=torch.long, device=self.device
        )
        # Exact dense-training behavior: a fresh causal adapter evaluation for
        # this tick's delta, not a hidden continuous state carried across ticks.
        adapted = self.objective.frontend_adapter(
            self.objective.codebook(code_tensor)
        )
        residual = self.objective.frontend_projection(adapted)
        embeddings = self.embedding(token_tensor)
        embeddings = embeddings + residual.to(embeddings.dtype)
        output = self.model(
            inputs_embeds=embeddings,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return KVAppendResult(
            past_key_values=output.past_key_values,
            logits=output.logits[:, -1].detach(),
        )


__all__ = ["HuggingFaceKVBackend"]
