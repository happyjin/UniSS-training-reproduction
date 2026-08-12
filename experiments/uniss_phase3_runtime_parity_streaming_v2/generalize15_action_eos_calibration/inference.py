"""Strict runtime using the learned Generalize15 continuation head."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.hf_backend import HuggingFaceKVBackend
from web_demo.runtime_parity_streaming_v2.inference import _decode_text_choice
from web_demo.runtime_parity_streaming_v2.session import SessionPhase
from web_demo.runtime_parity_streaming_v12.inference import MicroblockPromptSession


def continuation_vocab_logits(
    reference_logits: torch.Tensor, pair_logits: torch.Tensor
) -> torch.Tensor:
    """Map learned CONTINUE/EOS logits into the existing constrained decoder."""

    value = torch.full_like(reference_logits, torch.finfo(reference_logits.dtype).min)
    pair = pair_logits.reshape(-1, 2)
    flat = value.reshape(-1, value.shape[-1])
    if flat.shape[0] != pair.shape[0]:
        raise ValueError("continuation logit batch geometry differs")
    flat[:, c.TOKEN_START_GLM] = pair[:, 0].to(flat.dtype)
    flat[:, c.TOKEN_EOS] = pair[:, 1].to(flat.dtype)
    return value


class CalibratedContinuationPromptSession(MicroblockPromptSession):
    def __init__(self, *args, continuation_head, **kwargs) -> None:
        self.continuation_head = continuation_head
        super().__init__(*args, **kwargs)

    def _calibrated_logits(self, result):
        if result.last_hidden is None:
            raise RuntimeError("continuation head requires captured final hidden state")
        pair = self.continuation_head(result.last_hidden)
        return continuation_vocab_logits(result.logits, pair)

    def commit_wait(self):
        self._require_phase(SessionPhase.ACTION_PENDING)
        result = self._append_token_ids(
            (c.TOKEN_WAIT_READ,), capture_last_hidden=True
        )
        return self._finish_tick("WAIT", self._calibrated_logits(result))

    def commit_final_semantic_microblock(self, values: Sequence[int]):
        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        codes = self._validate_semantic_codes(values)
        if not codes:
            raise ValueError("final semantic microblock cannot be empty")
        result = self._append_token_ids(
            (*c.encode_bicodec_semantic(codes), c.TOKEN_END_SEMANTIC),
            capture_last_hidden=True,
        )
        assert self._pending is not None
        self._pending.semantic_codes.extend(codes)
        return self._finish_tick("WRITE", self._calibrated_logits(result))


class CalibratedMicroblockRuntimeGenerator:
    """V12 microblock generation with v15 action and continuation heads."""

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
        self.session = CalibratedContinuationPromptSession(
            self.backend,
            target_lang=target_lang,
            speaker_global=speaker_global,
            continuation_head=objective.continuation_head,
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

    def generate_write(self):
        from web_demo.runtime_parity_streaming_v2.inference import GeneratedWrite

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
            raise RuntimeError("v15 backend did not return semantic-start hidden")
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
                    raise RuntimeError("v15 intermediate block returned no hidden")
                context = appended.last_hidden
                continue
            self.session.commit_final_semantic_microblock(block)
            naturally_ended = True
            break
        if not naturally_ended:
            raise RuntimeError("semantic safety ceiling reached before natural END")
        self.text_ids.extend(text_ids)
        self.semantic_codes.extend(semantic)
        return GeneratedWrite(tuple(text_ids), tuple(semantic))

    @property
    def text(self) -> str:
        return self.tokenizer.decode(
            self.text_ids, skip_special_tokens=True
        ).strip()


__all__ = [
    "CalibratedContinuationPromptSession",
    "CalibratedMicroblockRuntimeGenerator",
    "continuation_vocab_logits",
]

