"""Natural streaming inference with one semantic-block model dispatch per WRITE.

The v8 runtime first appended all predicted BiCodec semantic tokens and then
appended ``END_SEMANTIC`` in a second Qwen call.  Causal eval-mode inference
does not require that dispatch boundary: appending the same tokens as one
contiguous block leaves the canonical transcript and continuation position
unchanged.  This module keeps the v8 model and all natural decisions intact
while removing one model dispatch from every WRITE.
"""

from __future__ import annotations

from typing import Sequence

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.hf_backend import HuggingFaceKVBackend
from web_demo.runtime_parity_streaming_v2.inference import (
    GeneratedWrite,
    _decode_text_choice,
)
from web_demo.runtime_parity_streaming_v2.session import SessionPhase
from web_demo.runtime_parity_streaming_v5.inference import (
    ParallelSemanticPromptSession,
)


class FusedSemanticPromptSession(ParallelSemanticPromptSession):
    """Commit semantic codes and their natural END marker in one forward."""

    def commit_semantic_block(self, values: Sequence[int]):
        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        codes = self._validate_semantic_codes(values)
        if not codes:
            raise ValueError("WRITE must contain at least one semantic code")
        result = self._append_token_ids(
            (*c.encode_bicodec_semantic(codes), c.TOKEN_END_SEMANTIC)
        )
        assert self._pending is not None
        self._pending.semantic_codes.extend(codes)
        return self._finish_tick("WRITE", result.logits)


class FusedSemanticRuntimeGenerator:
    """Keep every v8 prediction unchanged and fuse only its final KV append."""

    def __init__(
        self,
        model,
        tokenizer,
        objective,
        *,
        target_lang: str,
        speaker_global: Sequence[int],
        device: str | torch.device,
        maximum_text_tokens: int = 16,
        maximum_semantic_tokens: int = 80,
        fuse_ticks: bool = False,
        use_static_cache: bool = False,
        maximum_cache_tokens: int = 32_768,
    ) -> None:
        # The trained natural-length posterior, not this runtime argument,
        # determines block length.  Preserve the interface used by the strict
        # evaluator without introducing an oracle or forced cap.
        del maximum_semantic_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.objective = objective
        self.backend = HuggingFaceKVBackend(
            model,
            objective,
            device=device,
            fuse_ticks=fuse_ticks,
            use_static_cache=use_static_cache,
            maximum_cache_tokens=maximum_cache_tokens,
        )
        self.session = FusedSemanticPromptSession(
            self.backend,
            target_lang=target_lang,
            speaker_global=speaker_global,
        )
        self.maximum_text_tokens = int(maximum_text_tokens)
        self.text_ids: list[int] = []
        self.semantic_codes: list[int] = []

    def action_probability(self, last_hidden: torch.Tensor) -> float:
        logits = self.objective.action_head(last_hidden)
        return float(torch.softmax(logits.float(), dim=-1)[0, 1])

    def generate_write(self) -> GeneratedWrite:
        result = self.session.begin_write()
        text_ids: list[int] = []
        logits = result.logits
        for _ in range(self.maximum_text_tokens):
            token = _decode_text_choice(logits)
            if token == c.TOKEN_END_CONTENT:
                break
            text_ids.append(token)
            logits = self.session.append_text_ids((token,)).logits

        semantic_start = self.session.end_text_with_hidden()
        if semantic_start.last_hidden is None:
            raise RuntimeError("v9 backend did not return START_SEMANTIC hidden state")
        embedding_weight = self.model.get_input_embeddings().weight
        semantic, natural_end = self.objective.semantic_block_head.decode(
            semantic_start.last_hidden, embedding_weight
        )
        if not natural_end:
            raise RuntimeError(
                "parallel semantic head did not naturally select END_SEMANTIC"
            )
        if not semantic:
            raise RuntimeError(
                "parallel semantic head selected END before playable audio"
            )

        self.session.commit_semantic_block(semantic)
        self.text_ids.extend(text_ids)
        self.semantic_codes.extend(semantic)
        return GeneratedWrite(tuple(text_ids), tuple(semantic))

    @property
    def text(self) -> str:
        return self.tokenizer.decode(
            self.text_ids, skip_special_tokens=True
        ).strip()


__all__ = [
    "FusedSemanticPromptSession",
    "FusedSemanticRuntimeGenerator",
]
