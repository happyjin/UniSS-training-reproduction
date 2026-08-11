"""Natural-action inference with one parallel semantic prediction per WRITE."""

from __future__ import annotations

from typing import Sequence

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.hf_backend import HuggingFaceKVBackend
from web_demo.runtime_parity_streaming_v2.inference import (
    GeneratedWrite,
    _decode_text_choice,
)
from web_demo.runtime_parity_streaming_v2.session import (
    PersistentPromptSession,
    SessionPhase,
)


class ParallelSemanticPromptSession(PersistentPromptSession):
    """Expose the exact ``START_SEMANTIC`` hidden state for the v5 head."""

    def end_text_with_hidden(self):
        self._require_phase(SessionPhase.WRITE_TEXT)
        result = self._append_token_ids(
            (c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC),
            capture_last_hidden=True,
        )
        self._phase = SessionPhase.WRITE_SEMANTIC
        return result


class ParallelSemanticRuntimeGenerator:
    """Keep natural v4 policy/text/EOS and replace only serial semantic AR."""

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
        self.session = ParallelSemanticPromptSession(
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
            raise RuntimeError("v5 backend did not return START_SEMANTIC hidden state")
        embedding_weight = self.model.get_input_embeddings().weight
        semantic, natural_end = self.objective.semantic_block_head.decode(
            semantic_start.last_hidden, embedding_weight
        )
        if not natural_end:
            raise RuntimeError("parallel semantic head did not naturally select END_SEMANTIC")
        if not semantic:
            raise RuntimeError("parallel semantic head selected END before playable audio")
        self.session.append_semantic_codes(semantic)
        self.session.finish_write()
        self.text_ids.extend(text_ids)
        self.semantic_codes.extend(semantic)
        return GeneratedWrite(tuple(text_ids), tuple(semantic))

    @property
    def text(self) -> str:
        return self.tokenizer.decode(
            self.text_ids, skip_special_tokens=True
        ).strip()


__all__ = [
    "ParallelSemanticPromptSession",
    "ParallelSemanticRuntimeGenerator",
]

