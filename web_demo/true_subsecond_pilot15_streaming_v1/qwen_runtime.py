"""Append-only Qwen KV runtime for true-subsecond action and micro-WRITE."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model.chunk_causal_whispervq import (
    CausalAdapterState,
)
from training import constants_uniss as c


@dataclass(frozen=True)
class PolicyObservation:
    write_probability: float
    wait_probability: float
    support_bucket: int
    support_probabilities: tuple[float, ...]
    source_hidden: torch.Tensor
    source_cache: object


@dataclass(frozen=True)
class MicroWrite:
    text_ids: tuple[int, ...]
    text: str
    safe_probabilities: tuple[float, ...]
    semantic_ids: tuple[int, ...]
    forced_anticipation: bool


class IncrementalQwenRuntime:
    def __init__(
        self,
        model,
        tokenizer,
        objective,
        *,
        target_lang: str,
        speaker_global: Sequence[int],
        device: torch.device,
        safe_threshold: float = 0.5,
        seed: int = 20260811,
    ) -> None:
        values = tuple(int(value) for value in speaker_global)
        if len(values) != 32:
            raise ValueError("speaker_global must contain exactly 32 tokens")
        self.model = model
        self.tokenizer = tokenizer
        self.objective = objective
        self.target_lang = c.normalize_language(target_lang)
        self.speaker_global = values
        self.device = device
        self.safe_threshold = float(safe_threshold)
        self.seed = int(seed)
        self.frontend_state: CausalAdapterState | None = None
        self.source_cache = None
        self.source_codes = 0
        self.committed_text_ids: list[int] = []
        self._initialize_source_cache()

    @staticmethod
    def _clone_cache(value):
        return copy.deepcopy(value)

    def _initialize_source_cache(self) -> None:
        header = [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(self.target_lang),
            c.speed_token_id(1.0),
            *c.wrap_global_tokens(self.speaker_global),
            c.TOKEN_START_GLM,
        ]
        ids = torch.tensor([header], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=ids, use_cache=True, return_dict=True)
        self.source_cache = output.past_key_values

    def append_source_codes(self, values: Sequence[int]) -> None:
        codes = [int(value) for value in values]
        if not codes:
            return
        if any(not 0 <= value < c.GLM_SEMANTIC_SIZE for value in codes):
            raise ValueError("source WhisperVQ code is outside the 16k codebook")
        code_tensor = torch.tensor([codes], dtype=torch.long, device=self.device)
        token_ids = code_tensor + c.GLM_SEMANTIC_OFFSET
        with torch.inference_mode():
            hidden, self.frontend_state = self.objective.frontend_adapter.forward_chunk(
                self.objective.codebook(code_tensor), self.frontend_state
            )
            residual = self.objective.frontend_projection(hidden)
            embeddings = self.model.get_input_embeddings()(token_ids) + residual.to(
                self.model.get_input_embeddings().weight.dtype
            )
            output = self.model(
                inputs_embeds=embeddings,
                past_key_values=self.source_cache,
                use_cache=True,
                return_dict=True,
            )
        self.source_cache = output.past_key_values
        self.source_codes += len(codes)

    def observe_policy(self) -> PolicyObservation:
        if self.source_codes <= 0:
            raise RuntimeError("policy cannot run before the first source code")
        branch = self._clone_cache(self.source_cache)
        end = torch.tensor([[c.TOKEN_END_GLM]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=end,
                past_key_values=branch,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            source_hidden = output.hidden_states[-1][:, -1]
            action_probability = F.softmax(
                self.objective.action_head(source_hidden).float(), dim=-1
            )[0]
            support_probability = F.softmax(
                self.objective.support_head(source_hidden).float(), dim=-1
            )[0]
        return PolicyObservation(
            write_probability=float(action_probability[1]),
            wait_probability=float(action_probability[0]),
            support_bucket=int(support_probability.argmax()),
            support_probabilities=tuple(float(value) for value in support_probability),
            source_hidden=source_hidden.detach(),
            source_cache=output.past_key_values,
        )

    def _prefill(self, cache, values: Sequence[int]):
        ids = torch.tensor([list(values)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        return output.logits[0, -1].float(), output.past_key_values

    @staticmethod
    def _text_next(logits: torch.Tensor) -> int:
        text_value, text_index = logits[: c.QWEN_BASE_VOCAB_END + 1].max(dim=0)
        end_value = logits[c.TOKEN_END_CONTENT]
        if end_value >= text_value:
            return c.TOKEN_END_CONTENT
        return int(text_index)

    def _candidate_text(self, observation: PolicyObservation, maximum: int) -> list[int]:
        cache = self._clone_cache(observation.source_cache)
        logits, cache = self._prefill(
            cache,
            [
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.target_lang),
                c.speed_token_id(1.0),
                c.TOKEN_START_CONTENT,
            ],
        )
        result: list[int] = []
        for _ in range(maximum):
            token = self._text_next(logits)
            if token == c.TOKEN_END_CONTENT:
                break
            result.append(token)
            logits, cache = self._prefill(cache, [token])
        return result

    def _safe_prefix(
        self, observation: PolicyObservation, candidates: Sequence[int]
    ) -> tuple[list[int], tuple[float, ...]]:
        if not candidates:
            return [], ()
        ids = torch.tensor([list(candidates)], dtype=torch.long, device=self.device)
        target_hidden = self.model.get_input_embeddings()(ids)
        with torch.inference_mode():
            probabilities = torch.sigmoid(
                self.objective.safe_commit_head(
                    observation.source_hidden, target_hidden
                ).float()
            )[0]
        count = 0
        for probability in probabilities:
            if float(probability) < self.safe_threshold:
                break
            count += 1
        return list(candidates[:count]), tuple(float(value) for value in probabilities)

    def _semantic_block(
        self,
        observation: PolicyObservation,
        text_ids: Sequence[int],
        *,
        block_tokens: int,
    ) -> list[int]:
        cache = self._clone_cache(observation.source_cache)
        logits, cache = self._prefill(
            cache,
            [
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.target_lang),
                c.speed_token_id(1.0),
                c.TOKEN_START_CONTENT,
                *[int(value) for value in text_ids],
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_SEMANTIC,
            ],
        )
        result: list[int] = []
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + len(self.committed_text_ids)
        )
        for _ in range(block_tokens):
            semantic_logits = logits[
                c.BICODEC_SEMANTIC_OFFSET : c.BICODEC_SEMANTIC_OFFSET
                + c.BICODEC_SEMANTIC_SIZE
            ]
            probability = F.softmax(semantic_logits / 0.7, dim=-1)
            token = int(torch.multinomial(probability, 1, generator=generator))
            result.append(token)
            logits, cache = self._prefill(cache, [c.BICODEC_SEMANTIC_OFFSET + token])
        return result

    def micro_write(
        self,
        observation: PolicyObservation,
        *,
        maximum_text_tokens: int,
        semantic_block_tokens: int,
        forced: bool,
    ) -> MicroWrite:
        if maximum_text_tokens <= 0 or semantic_block_tokens <= 0:
            raise ValueError("micro-WRITE token budgets must be positive")
        candidates = self._candidate_text(observation, maximum_text_tokens)
        accepted, probabilities = self._safe_prefix(observation, candidates)
        anticipation = False
        if forced and not accepted and candidates:
            accepted = candidates[:1]
            anticipation = True
        semantic = (
            self._semantic_block(
                observation, accepted, block_tokens=semantic_block_tokens
            )
            if accepted
            else []
        )
        self.committed_text_ids.extend(accepted)
        return MicroWrite(
            text_ids=tuple(accepted),
            text=self.tokenizer.decode(accepted, skip_special_tokens=True).strip(),
            safe_probabilities=probabilities,
            semantic_ids=tuple(semantic),
            forced_anticipation=anticipation,
        )

    @property
    def committed_text(self) -> str:
        return self.tokenizer.decode(
            self.committed_text_ids, skip_special_tokens=True
        ).strip()
