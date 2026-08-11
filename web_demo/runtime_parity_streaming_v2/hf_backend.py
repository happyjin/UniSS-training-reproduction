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

    def __init__(
        self,
        model,
        objective,
        *,
        device: str | torch.device,
        fuse_ticks: bool = False,
        use_static_cache: bool = False,
        maximum_cache_tokens: int = 32_768,
    ) -> None:
        self.model = model.eval()
        self.objective = objective.eval()
        self.device = torch.device(device)
        self.embedding = self.model.get_input_embeddings()
        self.fuse_ticks = bool(fuse_ticks)
        self.use_static_cache = bool(use_static_cache)
        self.maximum_cache_tokens = int(maximum_cache_tokens)
        self._static_cache = None
        self._next_cache_position = 0

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
        length = int(
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )
        kwargs: dict[str, Any] = {}
        if self.use_static_cache:
            from transformers import StaticCache

            if self._static_cache is None:
                dtype = next(self.model.parameters()).dtype
                self._static_cache = StaticCache(
                    config=self.model.config,
                    max_batch_size=1,
                    max_cache_len=self.maximum_cache_tokens,
                    device=self.device,
                    dtype=dtype,
                )
            if past_key_values is not None and past_key_values is not self._static_cache:
                raise RuntimeError("persistent session supplied a foreign static cache")
            past_key_values = self._static_cache
            kwargs["cache_position"] = torch.arange(
                self._next_cache_position,
                self._next_cache_position + length,
                dtype=torch.long,
                device=self.device,
            )
        output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=capture_last_hidden,
            return_dict=True,
            **kwargs,
        )
        if self.use_static_cache:
            self._next_cache_position += length
        return output

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
        output = self._forward(
            input_ids=ids,
            past_key_values=past_key_values,
            capture_last_hidden=capture_last_hidden,
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
        output = self._forward(
            inputs_embeds=embeddings,
            past_key_values=past_key_values,
        )
        return KVAppendResult(
            past_key_values=output.past_key_values,
            logits=output.logits[:, -1].detach(),
        )

    @torch.inference_mode()
    def append_tick(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values: Any,
    ) -> KVAppendResult:
        """Append START_GLM, source delta and END_GLM in one exact forward.

        Eval-mode causal attention makes this numerically equivalent to three
        separate cache appends, while avoiding two Python/model dispatches per
        160 ms source tick.  The frontend adapter still resets once per tick,
        exactly as it does in dense training.
        """

        if not self.fuse_ticks:
            raise RuntimeError("fused tick append was not enabled")
        codes = [int(value) for value in source_codes]
        canonical = [int(value) for value in canonical_token_ids]
        if canonical != list(c.encode_glm_semantic(codes)):
            raise ValueError("source codes and canonical GLM token IDs disagree")
        boundary_ids = torch.tensor(
            [[c.TOKEN_START_GLM, c.TOKEN_END_GLM]],
            dtype=torch.long,
            device=self.device,
        )
        boundary_embeddings = self.embedding(boundary_ids)
        pieces = [boundary_embeddings[:, :1]]
        if codes:
            code_tensor = torch.tensor([codes], dtype=torch.long, device=self.device)
            token_tensor = torch.tensor(
                [canonical], dtype=torch.long, device=self.device
            )
            adapted = self.objective.frontend_adapter(
                self.objective.codebook(code_tensor)
            )
            residual = self.objective.frontend_projection(adapted)
            source_embeddings = self.embedding(token_tensor)
            pieces.append(source_embeddings + residual.to(source_embeddings.dtype))
        pieces.append(boundary_embeddings[:, 1:])
        output = self._forward(
            inputs_embeds=torch.cat(pieces, dim=1),
            past_key_values=past_key_values,
            capture_last_hidden=True,
        )
        return KVAppendResult(
            past_key_values=output.past_key_values,
            logits=output.logits[:, -1].detach(),
            last_hidden=output.hidden_states[-1][:, -1].detach(),
        )


__all__ = ["HuggingFaceKVBackend"]
