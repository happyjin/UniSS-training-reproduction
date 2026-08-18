"""Exact persistent-KV roll-in with Phase3-native bounded AR semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.event_rollout import (
    GeneratedTick,
    OracleSession,
    RolloutTrace,
    build_rollout_trace,
    generated_tick_matches_oracle,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.inference import (
    CalibratedContinuationPromptSession,
    continuation_vocab_logits,
)
from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.inference import (
    _decode_semantic_choice,
    _decode_text_choice,
)
from web_demo.runtime_parity_streaming_v2.session import SessionPhase


@dataclass(frozen=True)
class RolloutSchedule:
    fraction: float
    maximum_sessions: int


def rollout_schedule(progress: float) -> RolloutSchedule:
    """Warm up clean SFT, then expose 10%→35% of trajectory batches."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress < 0.05:
        return RolloutSchedule(0.0, 0)
    if progress < 0.25:
        fraction = 0.10 + (progress - 0.05) / 0.20 * 0.10
    elif progress < 0.70:
        fraction = 0.20 + (progress - 0.25) / 0.45 * 0.10
    else:
        fraction = 0.30 + (progress - 0.70) / 0.30 * 0.05
    return RolloutSchedule(fraction, 1)


def _head_input(module, value: torch.Tensor) -> torch.Tensor:
    parameter = next(module.parameters())
    return value.to(dtype=parameter.dtype)


class _ARContinuationPromptSession(CalibratedContinuationPromptSession):
    """Standard token-by-token WRITE grammar with learned event EOS."""

    def _calibrated_logits(self, result):
        if result.last_hidden is None:
            raise RuntimeError("continuation head requires final hidden state")
        pair = self.continuation_head(
            _head_input(self.continuation_head, result.last_hidden)
        )
        return continuation_vocab_logits(result.logits, pair)

    def finish_write(self):
        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        assert self._pending is not None
        if not self._pending.semantic_codes:
            raise RuntimeError("WRITE must commit at least one semantic code")
        result = self._append_token_ids(
            (c.TOKEN_END_SEMANTIC,), capture_last_hidden=True
        )
        return self._finish_tick("WRITE", self._calibrated_logits(result))


def _continuation_choice(logits: torch.Tensor) -> bool:
    values = logits.reshape(-1).float()
    return bool(values[c.TOKEN_EOS] >= values[c.TOKEN_START_GLM])


@torch.inference_mode()
def rollout_session(
    session: OracleSession,
    backend,
    objective,
    *,
    maximum_text_tokens: int = 16,
    maximum_semantic_tokens: int = 24,
) -> RolloutTrace:
    """Generate exact events and stop after the first complete divergence.

    A generated WRITE is always fully appended to the persistent cache before
    divergence is inspected.  Recovery can therefore use both the correction
    state immediately before a wrong choice and the real corrupted state after
    the generated payload.
    """

    if maximum_text_tokens <= 0 or maximum_semantic_tokens <= 0:
        raise ValueError("runtime safety ceilings must be positive")
    if maximum_semantic_tokens > 24:
        raise ValueError("v3 semantic ceiling must not exceed raw WRITE maximum 24")
    prompt = _ARContinuationPromptSession(
        backend,
        target_lang=session.target_lang,
        speaker_global=session.speaker_global,
        continuation_head=objective.continuation_head,
    )
    generated: list[GeneratedTick] = []
    for event in session.events:
        observation = prompt.begin_tick(event.source_codes)
        if observation.last_hidden is None:
            raise RuntimeError("action observation returned no hidden state")
        action_logits = objective.action_head(
            _head_input(objective.action_head, observation.last_hidden)
        )
        action = "WRITE" if int(action_logits.float().argmax(dim=-1)[0]) == 1 else "WAIT"
        text_ids: list[int] = []
        semantic_codes: list[int] = []
        natural_text_end = True
        natural_semantic_end = True
        if action == "WAIT":
            committed = prompt.commit_wait()
        else:
            result = prompt.begin_write()
            logits = result.logits
            natural_text_end = False
            for _ in range(maximum_text_tokens):
                token = _decode_text_choice(logits)
                if token == c.TOKEN_END_CONTENT:
                    natural_text_end = True
                    break
                text_ids.append(token)
                logits = prompt.append_text_ids((token,)).logits
            semantic_start = prompt.end_text()
            logits = semantic_start.logits
            natural_semantic_end = False
            for _ in range(maximum_semantic_tokens):
                token = _decode_semantic_choice(
                    logits, allow_end=bool(semantic_codes)
                )
                if token == c.TOKEN_END_SEMANTIC:
                    natural_semantic_end = True
                    break
                semantic_codes.append(token)
                logits = prompt.append_semantic_codes((token,)).logits
            if not semantic_codes:
                # The constrained decoder cannot naturally select END before
                # one unit.  Keep a structurally invalid generated tick rather
                # than inventing a semantic unit or accepting empty PCM.
                raise RuntimeError("natural WRITE produced no semantic unit")
            committed = prompt.finish_write()
        choose_eos = bool(event.source_finished) and _continuation_choice(
            committed.continuation_logits
        )
        tick = GeneratedTick(
            action,
            tuple(text_ids),
            tuple(semantic_codes),
            natural_text_end=natural_text_end,
            natural_semantic_end=natural_semantic_end,
            choose_eos=choose_eos,
        )
        generated.append(tick)
        if choose_eos:
            prompt.finish_session()
            break
        if not generated_tick_matches_oracle(event, tick):
            break
    return build_rollout_trace(session, generated)


__all__ = ["RolloutSchedule", "rollout_schedule", "rollout_session"]
