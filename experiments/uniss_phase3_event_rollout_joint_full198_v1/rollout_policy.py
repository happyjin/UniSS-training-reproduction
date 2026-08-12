"""Exact grammar-constrained event rollout driven by the current model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    GeneratedTick,
    OracleSession,
    RolloutTrace,
    build_rollout_trace,
)
from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.inference import _decode_text_choice
from web_demo.runtime_parity_streaming_v12.inference import MicroblockPromptSession


@dataclass(frozen=True)
class RolloutSchedule:
    fraction: float
    maximum_sessions: int


def rollout_schedule(progress: float) -> RolloutSchedule:
    """One-run curriculum: 0% warm-up, then 5%→40% exact event roll-in."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.05:
        return RolloutSchedule(0.0, 0)
    if progress < 0.20:
        fraction = 0.05 + (progress - 0.05) / 0.15 * 0.10
    elif progress < 0.60:
        fraction = 0.15 + (progress - 0.20) / 0.40 * 0.15
    else:
        fraction = 0.30 + (progress - 0.60) / 0.40 * 0.10
    return RolloutSchedule(fraction, 1)


def _continuation_choice(continuation_logits: torch.Tensor) -> bool:
    values = continuation_logits.reshape(-1).float()
    return bool(values[c.TOKEN_EOS] >= values[c.TOKEN_START_GLM])


@torch.inference_mode()
def rollout_session(
    session: OracleSession,
    backend,
    objective,
    embedding_weight: torch.Tensor,
    *,
    maximum_text_tokens: int = 16,
    maximum_semantic_tokens: int = 80,
) -> RolloutTrace:
    prompt = MicroblockPromptSession(
        backend,
        target_lang=session.target_lang,
        speaker_global=session.speaker_global,
    )
    generated: list[GeneratedTick] = []
    block_size = int(objective.semantic_microblock_head.block_size)
    maximum_blocks = (maximum_semantic_tokens + block_size - 1) // block_size
    for event in session.events:
        observation = prompt.begin_tick(event.source_codes)
        action_logits = objective.action_head(observation.last_hidden)
        action = "WRITE" if int(action_logits.float().argmax(dim=-1)[0]) == 1 else "WAIT"
        text_ids: list[int] = []
        semantic_codes: list[int] = []
        natural_end = True
        if action == "WAIT":
            committed = prompt.commit_wait()
        else:
            result = prompt.begin_write()
            logits = result.logits
            for _ in range(maximum_text_tokens):
                token = _decode_text_choice(logits)
                if token == c.TOKEN_END_CONTENT:
                    break
                text_ids.append(token)
                logits = prompt.append_text_ids((token,)).logits
            semantic_start = prompt.end_text_with_hidden()
            context = semantic_start.last_hidden
            if context is None:
                raise RuntimeError("semantic START returned no hidden state")
            natural_end = False
            for block_index in range(maximum_blocks):
                block, should_continue = objective.semantic_microblock_head.decode(
                    context, embedding_weight
                )
                semantic_codes.extend(block)
                if should_continue and block_index + 1 < maximum_blocks:
                    appended = prompt.append_semantic_microblock_with_hidden(block)
                    if appended.last_hidden is None:
                        raise RuntimeError("semantic microblock returned no hidden state")
                    context = appended.last_hidden
                else:
                    committed = prompt.commit_final_semantic_microblock(block)
                    natural_end = not should_continue
                    break
        choose_eos = bool(event.source_finished) and _continuation_choice(
            committed.continuation_logits
        )
        generated.append(
            GeneratedTick(
                action,
                tuple(text_ids),
                tuple(semantic_codes),
                natural_semantic_end=natural_end,
                choose_eos=choose_eos,
            )
        )
        if choose_eos:
            prompt.finish_session()
            break
    return build_rollout_trace(session, generated)


__all__ = ["RolloutSchedule", "rollout_schedule", "rollout_session"]
