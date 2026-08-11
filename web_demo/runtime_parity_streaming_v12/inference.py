"""Persistent-KV inference for the v12 causal semantic microblock head."""

from __future__ import annotations

import math
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


class MicroblockPromptSession(ParallelSemanticPromptSession):
    """Append intermediate microblocks and expose the resulting Qwen hidden."""

    def append_semantic_microblock_with_hidden(self, values: Sequence[int]):
        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        codes = self._validate_semantic_codes(values)
        if not codes:
            raise ValueError("semantic microblock cannot be empty")
        result = self._append_token_ids(
            c.encode_bicodec_semantic(codes), capture_last_hidden=True
        )
        assert self._pending is not None
        self._pending.semantic_codes.extend(codes)
        return result

    def commit_final_semantic_microblock(self, values: Sequence[int]):
        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        codes = self._validate_semantic_codes(values)
        if not codes:
            raise ValueError("final semantic microblock cannot be empty")
        result = self._append_token_ids(
            (*c.encode_bicodec_semantic(codes), c.TOKEN_END_SEMANTIC)
        )
        assert self._pending is not None
        self._pending.semantic_codes.extend(codes)
        return self._finish_tick("WRITE", result.logits)


class MicroblockRuntimeGenerator:
    """Decode natural 1..4-unit blocks until the head selects END."""

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
        self.session = MicroblockPromptSession(
            self.backend,
            target_lang=target_lang,
            speaker_global=speaker_global,
        )
        self.maximum_text_tokens = int(maximum_text_tokens)
        block_size = int(objective.semantic_microblock_head.block_size)
        if maximum_semantic_tokens <= 0:
            raise ValueError("semantic safety ceiling must be positive")
        self.maximum_microblocks = int(math.ceil(maximum_semantic_tokens / block_size))
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
            raise RuntimeError("v12 backend did not return START_SEMANTIC hidden state")
        context = semantic_start.last_hidden
        embedding_weight = self.model.get_input_embeddings().weight
        semantic: list[int] = []
        naturally_ended = False
        for _ in range(self.maximum_microblocks):
            block, should_continue = self.objective.semantic_microblock_head.decode(
                context, embedding_weight
            )
            if not block:
                raise RuntimeError("natural semantic microblock was empty")
            semantic.extend(block)
            if should_continue:
                appended = self.session.append_semantic_microblock_with_hidden(block)
                if appended.last_hidden is None:
                    raise RuntimeError(
                        "v12 backend did not return intermediate semantic hidden state"
                    )
                context = appended.last_hidden
                continue
            self.session.commit_final_semantic_microblock(block)
            naturally_ended = True
            break
        if not naturally_ended:
            raise RuntimeError(
                "semantic microblock safety ceiling reached before natural END"
            )

        self.text_ids.extend(text_ids)
        self.semantic_codes.extend(semantic)
        return GeneratedWrite(tuple(text_ids), tuple(semantic))

    @property
    def text(self) -> str:
        return self.tokenizer.decode(
            self.text_ids, skip_special_tokens=True
        ).strip()


__all__ = ["MicroblockPromptSession", "MicroblockRuntimeGenerator"]
