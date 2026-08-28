"""Routed MT/TTS traces plus an explicit sampled WAIT/WRITE action trace."""

from __future__ import annotations

from typing import Sequence

import torch

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.trace_generator import (
    GenerationTrace,
    TraceGenerator,
)
from training import constants_uniss as c


class EventTraceGenerator:
    """Keep Phase-A ASR frozen while routing policy LoRA through control/MT/TTS."""

    def __init__(
        self,
        controller,
        *,
        policy_temperature: float = 0.70,
        policy_top_p: float = 0.90,
        action_temperature: float = 0.80,
    ) -> None:
        self.controller = controller
        self.generator = TraceGenerator(
            policy_temperature=policy_temperature, policy_top_p=policy_top_p
        )
        self.action_temperature = float(action_temperature)
        self.current_event = -1
        self.tagged_traces: list[dict[str, object]] = []

    def set_event(self, event_index: int) -> None:
        self.current_event = int(event_index)

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
        values = {int(value) for value in prompt_ids}
        is_asr = c.TOKEN_TASK_ASR in values
        before = len(self.generator.traces)
        with self.controller.route(not is_asr):
            generated = self.generator(
                model,
                tokenizer,
                prompt_ids=prompt_ids,
                speech_embeddings=speech_embeddings,
                stop_ids=stop_ids,
                maximum=maximum,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        for trace in self.generator.traces[before:]:
            value = trace.to_dict()
            value["event_index"] = self.current_event
            self.tagged_traces.append(value)
        return generated

    @torch.inference_mode()
    def decide(
        self,
        model,
        tokenizer,
        *,
        speech_embeddings: torch.Tensor,
        target_lang: str,
        speaker_global: Sequence[int],
        seed: int,
    ) -> str:
        prompt = [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(target_lang),
            c.speed_token_id(1.0),
            *c.wrap_global_tokens([int(value) for value in speaker_global]),
            c.TOKEN_START_GLM,
            *([c.glm_semantic_id(0)] * len(speech_embeddings)),
            c.TOKEN_END_GLM,
        ]
        device = next(model.parameters()).device
        ids = torch.tensor(prompt, dtype=torch.long, device=device)
        embeddings = model.get_input_embeddings()(ids)
        positions = [
            index
            for index, token in enumerate(prompt)
            if c.GLM_SEMANTIC_OFFSET <= int(token) <= c.GLM_SEMANTIC_SPAN.last_id
        ]
        if len(positions) != len(speech_embeddings):
            raise ValueError("control speech/prompt geometry differs")
        embeddings.index_copy_(
            0,
            torch.tensor(positions, dtype=torch.long, device=device),
            speech_embeddings.to(embeddings.dtype),
        )
        with self.controller.route(True), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs_embeds=embeddings.unsqueeze(0), use_cache=False)
        pair_ids = torch.tensor(
            [c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE], dtype=torch.long, device=device
        )
        pair_logits = output.logits[0, -1].float().index_select(0, pair_ids)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        probabilities = torch.softmax(pair_logits / self.action_temperature, dim=-1)
        choice = int(torch.multinomial(probabilities, 1, generator=generator))
        token = int(pair_ids[choice])
        old_log_prob = float(torch.log_softmax(pair_logits, dim=-1)[choice])
        trace = GenerationTrace(
            family="control",
            prompt_ids=tuple(prompt),
            generated_ids=(token,),
            old_log_probs=(old_log_prob,),
            stop_reached=True,
        ).to_dict()
        trace["event_index"] = self.current_event
        self.tagged_traces.append(trace)
        return "WAIT" if token == c.TOKEN_WAIT_READ else "WRITE"


__all__ = ["EventTraceGenerator"]
