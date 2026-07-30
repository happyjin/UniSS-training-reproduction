"""Append-only Transformers adapter matching the Stage4 free-running format."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Sequence

import torch

from evaluation.simultaneous_streaming.stage4_streaming_generate import (
    normalized_write_tail,
    parse_write_tokens,
)
from training import constants_uniss as c
from uniss.streaming.controller import WriteResult
from uniss.streaming.policy import PolicyDecision


@dataclass
class AdapterAction:
    action: str
    raw_token_id: int
    forced_reason: str | None
    seconds: float


@dataclass
class AdapterWrite:
    text: str
    text_ids: list[int]
    semantic_values: list[int]
    raw_token_ids: list[int]
    normalized_token_ids: list[int]
    structurally_valid: bool
    seconds: float


@dataclass
class QwenLiveAdapter:
    model: object
    tokenizer: object
    device: torch.device
    target_language: str
    speaker_tokens: Sequence[int]
    max_write_tokens: int = 700
    max_model_len: int = 32768
    training_context_limit: int = 18000
    repetition_penalty: float = 1.1
    prompt_ids: list[int] = field(init=False)
    generated_text_ids: list[int] = field(default_factory=list)
    forced_actions: int = 0
    structural_recoveries: int = 0
    max_prompt_tokens: int = 0
    training_context_exceeded: bool = False
    last_action: AdapterAction | None = None
    last_write: AdapterWrite | None = None

    def __post_init__(self) -> None:
        if len(self.speaker_tokens) != 32:
            raise ValueError("speaker_tokens must contain exactly 32 values")
        self.prompt_ids = [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(self.target_language),
            c.speed_token_id(1.0),
            *c.wrap_global_tokens([int(value) for value in self.speaker_tokens]),
        ]
        self._account_prompt()

    def _account_prompt(self) -> None:
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(self.prompt_ids))
        self.training_context_exceeded |= len(self.prompt_ids) > self.training_context_limit
        if len(self.prompt_ids) >= self.max_model_len:
            raise ValueError(
                f"streaming prompt length {len(self.prompt_ids)} reached "
                f"max_model_len {self.max_model_len}"
            )

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor([self.prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def append_source(self, glm_tokens: Sequence[int]) -> None:
        values = [int(value) for value in glm_tokens]
        if not values:
            return
        self.prompt_ids.extend(
            [c.TOKEN_START_GLM, *c.encode_glm_semantic(values), c.TOKEN_END_GLM]
        )
        self._account_prompt()

    def choose_action(self, eligible: bool = True, is_final: bool = False) -> PolicyDecision:
        del eligible  # The audited Stage4 free-running path does not oracle-gate actions.
        input_ids, attention_mask = self._inputs()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logical_logits = output.logits[0, -1, : c.VOCAB_SIZE].float()
            raw_action = int(torch.argmax(logical_logits).item())
        forced_reason: str | None = None
        if raw_action not in {c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE}:
            forced_reason = "invalid_action"
            action_token = c.TOKEN_WRITE_GENERATE if is_final else c.TOKEN_WAIT_READ
        elif is_final and raw_action == c.TOKEN_WAIT_READ:
            forced_reason = "final_flush"
            action_token = c.TOKEN_WRITE_GENERATE
        else:
            action_token = raw_action
        if forced_reason is not None:
            self.forced_actions += 1
        self.prompt_ids.append(action_token)
        self._account_prompt()
        decision = (
            PolicyDecision.WRITE
            if action_token == c.TOKEN_WRITE_GENERATE
            else PolicyDecision.WAIT
        )
        self.last_action = AdapterAction(
            action=decision.value,
            raw_token_id=raw_action,
            forced_reason=forced_reason,
            seconds=time.perf_counter() - started,
        )
        return decision

    def commit_wait(self) -> None:
        if self.last_action is None or self.last_action.action != "wait":
            raise RuntimeError("commit_wait called without a WAIT decision")

    def generate_write(self, is_final: bool = False) -> WriteResult:
        del is_final
        if self.last_action is None or self.last_action.action != "write":
            raise RuntimeError("generate_write called without a WRITE decision")
        input_ids, attention_mask = self._inputs()
        model_vocab_size = int(getattr(self.model.config, "vocab_size"))
        suppressed_dummy_ids = list(range(c.VOCAB_SIZE, model_vocab_size))
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_write_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=c.TOKEN_PAD,
                eos_token_id=[
                    c.TOKEN_END_SEMANTIC,
                    c.TOKEN_WAIT_READ,
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_EOS,
                ],
                suppress_tokens=suppressed_dummy_ids,
            )
        raw_tail = [int(value) for value in generated[0, input_ids.shape[1] :].tolist()]
        parsed = parse_write_tokens(raw_tail, self.tokenizer)
        normalized = normalized_write_tail(parsed, self.target_language)
        structurally_valid = bool(
            parsed["has_content_start"]
            and parsed["has_content_end"]
            and parsed["has_semantic_start"]
            and parsed["has_semantic_end"]
            and int(parsed["invalid_semantic_tokens"]) == 0
        )
        if raw_tail != normalized:
            self.structural_recoveries += 1
        self.prompt_ids.extend(normalized)
        self._account_prompt()
        text_ids = [int(value) for value in parsed["text_ids"]]
        semantic_values = [int(value) for value in parsed["semantic_values"]]
        self.generated_text_ids.extend(text_ids)
        self.last_write = AdapterWrite(
            text=str(parsed["text"]),
            text_ids=text_ids,
            semantic_values=semantic_values,
            raw_token_ids=raw_tail,
            normalized_token_ids=normalized,
            structurally_valid=structurally_valid,
            seconds=time.perf_counter() - started,
        )
        return WriteResult(text_ids, semantic_values)

    @property
    def translation(self) -> str:
        return self.tokenizer.decode(
            self.generated_text_ids, skip_special_tokens=False
        ).strip()


def fake_output(logits: torch.Tensor) -> SimpleNamespace:
    """Small public helper used by unit tests without loading a real model."""

    return SimpleNamespace(logits=logits)
