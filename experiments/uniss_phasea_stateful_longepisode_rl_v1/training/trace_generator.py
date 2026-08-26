"""Autoregressive generator that records policy log probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from training import constants_uniss as c


@dataclass(frozen=True)
class GenerationTrace:
    family: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    stop_reached: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["prompt_ids"] = list(self.prompt_ids)
        value["generated_ids"] = list(self.generated_ids)
        value["old_log_probs"] = list(self.old_log_probs)
        return value


def family_for_prompt(prompt_ids: Sequence[int]) -> str:
    values = {int(value) for value in prompt_ids}
    if c.TOKEN_TASK_ASR in values:
        return "asr"
    if c.TOKEN_TASK_T2T_TRANSLATION in values or c.TOKEN_TASK_S2T_TRANSLATION in values:
        return "mt"
    if c.TOKEN_TASK_TTS in values:
        return "tts"
    raise ValueError("unknown generation family")


def apply_repetition_penalty(logits: torch.Tensor, history: Sequence[int], penalty: float) -> None:
    if float(penalty) == 1.0 or not history:
        return
    indices = torch.tensor(sorted(set(int(value) for value in history)), device=logits.device)
    selected = logits.index_select(0, indices)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits.index_copy_(0, indices, selected)


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> int:
    if float(temperature) <= 0:
        return int(logits.argmax())
    values, indices = torch.sort(logits / float(temperature), descending=True)
    probabilities = torch.softmax(values, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative > float(top_p)
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    values[remove] = -torch.inf
    probabilities = torch.softmax(values, dim=-1)
    choice = int(torch.multinomial(probabilities, 1, generator=generator))
    return int(indices[choice])


class TraceGenerator:
    """Callable replacement for the historical generation helper."""

    def __init__(self, *, policy_temperature: float = 0.7, policy_top_p: float = 0.9):
        self.policy_temperature = float(policy_temperature)
        self.policy_top_p = float(policy_top_p)
        self.traces: list[GenerationTrace] = []

    @torch.inference_mode()
    def __call__(
        self,
        model,
        tokenizer,
        *,
        prompt_ids: Sequence[int],
        speech_embeddings: torch.Tensor | None,
        stop_ids: set[int],
        maximum: int,
        seed: int,
        temperature: float = 0.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        family = family_for_prompt(prompt_ids)
        if family != "asr":
            temperature = self.policy_temperature
            top_p = self.policy_top_p
        device = next(model.parameters()).device
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        embeddings = model.get_input_embeddings()(ids)
        if speech_embeddings is not None:
            positions = [
                index
                for index, token in enumerate(prompt_ids)
                if c.GLM_SEMANTIC_OFFSET <= int(token) <= c.GLM_SEMANTIC_SPAN.last_id
            ]
            if len(positions) != len(speech_embeddings):
                raise ValueError("speech/prompt geometry differs")
            embeddings.index_copy_(
                0,
                torch.tensor(positions, dtype=torch.long, device=device),
                speech_embeddings.to(embeddings.dtype),
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs_embeds=embeddings.unsqueeze(0), use_cache=True)
        cache = output.past_key_values
        logits = output.logits[0, -1].float()
        generated: list[int] = []
        log_probs: list[float] = []
        generator = torch.Generator(device=device).manual_seed(int(seed))
        for _ in range(int(maximum)):
            logical = logits[: len(tokenizer)].clone()
            apply_repetition_penalty(logical, generated, repetition_penalty)
            token = sample_token(
                logical,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            log_probs.append(float(torch.log_softmax(logical, dim=-1)[token]))
            generated.append(token)
            if token in stop_ids:
                break
            step = torch.tensor([[token]], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=step, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[0, -1].float()
        if family != "asr":
            self.traces.append(
                GenerationTrace(
                    family=family,
                    prompt_ids=tuple(int(value) for value in prompt_ids),
                    generated_ids=tuple(generated),
                    old_log_probs=tuple(log_probs),
                    stop_reached=bool(generated and generated[-1] in stop_ids),
                )
            )
        return generated


__all__ = ["GenerationTrace", "TraceGenerator", "family_for_prompt"]

