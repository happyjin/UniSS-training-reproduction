"""Append-only Qwen KV-cache adapter for continuous B1 source chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from evaluation.simultaneous_streaming.stage4_streaming_generate import (
    parse_write_tokens,
)
from training import constants_uniss as c


STOP_IDS = {
    c.TOKEN_END_SEMANTIC,
    c.TOKEN_WAIT_READ,
    c.TOKEN_WRITE_GENERATE,
    c.TOKEN_EOS,
}


def maximum_identical_run(values: Sequence[int]) -> int:
    best = current = 0
    previous = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: Sequence[int], penalty: float
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0 or not token_ids:
        return logits
    output = logits.clone()
    for token in set(int(value) for value in token_ids):
        if not 0 <= token < output.shape[-1]:
            continue
        value = output[..., token]
        output[..., token] = torch.where(value < 0, value * penalty, value / penalty)
    return output


def block_collapsed_semantic(logits: torch.Tensor, generated: Sequence[int]) -> torch.Tensor:
    values = [int(value) for value in generated]
    try:
        start = len(values) - 1 - values[::-1].index(c.TOKEN_START_SEMANTIC)
    except ValueError:
        return logits
    semantic = [
        value
        for value in values[start + 1 :]
        if c.BICODEC_SEMANTIC_OFFSET <= value <= c.BICODEC_SEMANTIC_SPAN.last_id
    ]
    if not semantic:
        return logits
    output = logits.clone()
    last = semantic[-1]
    run = 0
    for value in reversed(semantic):
        if value != last:
            break
        run += 1
    if run >= 6:
        output[..., last] = float("-inf")
    tail = semantic[-24:]
    if len(tail) == 24 and len(set(tail)) <= 2:
        for value in set(tail):
            output[..., value] = float("-inf")
    return output


@dataclass
class CachedWrite:
    text: str
    text_ids: list[int]
    semantic_values: list[int]
    raw_token_ids: list[int]
    structurally_valid: bool
    first_token_seconds: float
    total_seconds: float
    cache_tokens_before: int
    cache_tokens_after: int
    semantic_unique_ratio: float
    semantic_max_identical_run: int


@dataclass
class CachedMicroWriteAdapter:
    model: object
    tokenizer: object
    device: torch.device
    target_language: str
    speaker_tokens: Sequence[int]
    max_write_tokens: int = 700
    repetition_penalty: float = 1.1
    cache: object | None = field(default=None, init=False, repr=False)
    cache_tokens: int = field(default=0, init=False)
    source_tokens: int = field(default=0, init=False)
    token_history: list[int] = field(default_factory=list, init=False)
    generated_text_ids: list[int] = field(default_factory=list, init=False)
    writes: list[CachedWrite] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        values = [int(value) for value in self.speaker_tokens]
        if len(values) != 32:
            raise ValueError("speaker_tokens must contain exactly 32 values")
        if self.max_write_tokens <= 0:
            raise ValueError("max_write_tokens must be positive")
        prompt = [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(self.target_language),
            c.speed_token_id(1.0),
            *c.wrap_global_tokens(values),
        ]
        self._forward_ids(prompt)

    def _cache_length(self) -> int:
        if self.cache is None:
            return 0
        getter = getattr(self.cache, "get_seq_length", None)
        if getter is not None:
            return int(getter())
        try:
            return int(self.cache[0][0].shape[-2])
        except (TypeError, IndexError, AttributeError):
            return self.cache_tokens

    @torch.inference_mode()
    def _forward_ids(self, ids: Sequence[int]) -> torch.Tensor:
        values = [int(value) for value in ids]
        if not values:
            raise ValueError("cannot cache an empty ID segment")
        tensor = torch.tensor([values], dtype=torch.long, device=self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(
                input_ids=tensor,
                past_key_values=self.cache,
                use_cache=True,
            )
        self.cache = output.past_key_values
        self.cache_tokens += len(values)
        self.token_history.extend(values)
        return output.logits[:, -1].float()

    @torch.inference_mode()
    def _forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[1] == 0:
            raise ValueError("source embeddings must be [time,hidden] or [1,time,hidden]")
        values = embeddings.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(
                inputs_embeds=values,
                past_key_values=self.cache,
                use_cache=True,
            )
        self.cache = output.past_key_values
        count = int(values.shape[1])
        self.cache_tokens += count
        self.source_tokens += count
        return output.logits[:, -1].float()

    def append_source(self, embeddings: torch.Tensor) -> None:
        self._forward_ids([c.TOKEN_START_GLM])
        self._forward_embeddings(embeddings)
        self._forward_ids([c.TOKEN_END_GLM])

    def commit_wait(self) -> None:
        self._forward_ids([c.TOKEN_WAIT_READ])

    def generate_write(self) -> CachedWrite:
        before = self._cache_length()
        started = time.perf_counter()
        logits = self._forward_ids([c.TOKEN_WRITE_GENERATE])
        first_token_seconds = 0.0
        generated: list[int] = []
        for index in range(self.max_write_tokens):
            logical = logits[:, : c.VOCAB_SIZE]
            logical = apply_repetition_penalty(
                logical,
                [*self.generated_text_ids, *generated],
                self.repetition_penalty,
            )
            logical = block_collapsed_semantic(logical, generated)
            token = int(logical.argmax(dim=-1)[0])
            if index == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                first_token_seconds = time.perf_counter() - started
            generated.append(token)
            logits = self._forward_ids([token])
            if token in STOP_IDS:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        parsed = parse_write_tokens(generated, self.tokenizer)
        text_ids = [int(value) for value in parsed["text_ids"]]
        semantic = [int(value) for value in parsed["semantic_values"]]
        structural = bool(
            parsed["has_content_start"]
            and parsed["has_content_end"]
            and parsed["has_semantic_start"]
            and parsed["has_semantic_end"]
            and int(parsed["invalid_semantic_tokens"]) == 0
        )
        if text_ids:
            self.generated_text_ids.extend(text_ids)
        write = CachedWrite(
            text=str(parsed["text"]),
            text_ids=text_ids,
            semantic_values=semantic,
            raw_token_ids=generated,
            structurally_valid=structural,
            first_token_seconds=first_token_seconds,
            total_seconds=time.perf_counter() - started,
            cache_tokens_before=before,
            cache_tokens_after=self._cache_length(),
            semantic_unique_ratio=len(set(semantic)) / max(1, len(semantic)),
            semantic_max_identical_run=maximum_identical_run(semantic),
        )
        self.writes.append(write)
        return write

    @property
    def translation(self) -> str:
        return self.tokenizer.decode(
            self.generated_text_ids, skip_special_tokens=False
        ).strip()
