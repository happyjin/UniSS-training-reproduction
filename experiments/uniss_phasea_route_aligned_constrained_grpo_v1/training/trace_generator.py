"""Autoregressive sampler that records ASR, MT, and TTS policy traces."""

from __future__ import annotations

from typing import Sequence

import torch

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.trace_generator import (
    GenerationTrace,
    apply_repetition_penalty,
    family_for_prompt,
    sample_token,
)
from training import constants_uniss as c


class RouteAlignedTraceGenerator:
    def __init__(
        self,
        *,
        asr_temperature: float = 0.30,
        policy_temperature: float = 0.70,
        policy_top_p: float = 0.90,
    ) -> None:
        self.asr_temperature = float(asr_temperature)
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
        temperature = (
            self.asr_temperature if family == "asr" else self.policy_temperature
        )
        top_p = self.policy_top_p
        device = next(model.parameters()).device
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        embeddings = model.get_input_embeddings()(ids)
        if speech_embeddings is not None:
            positions = [
                index
                for index, token in enumerate(prompt_ids)
                if c.GLM_SEMANTIC_OFFSET
                <= int(token)
                <= c.GLM_SEMANTIC_SPAN.last_id
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


__all__ = ["RouteAlignedTraceGenerator"]

