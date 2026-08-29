"""Credit actual, non-empty COMMIT transactions in the v4 cascade.

``policy_action`` is a sampled control token.  It is deliberately not treated
as a write by itself: a v4 write matters only if the runtime executed it and a
healthy TTS segment was emitted.  This prevents the old empty-WRITE shortcut.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence


def softplus(value: float) -> float:
    if value > 30.0:
        return value
    return math.log1p(math.exp(value))


def nearest_gold_action(
    source_end_ms: int,
    mapped_events: Sequence[dict[str, object]],
    *,
    tolerance_ms: int = 480,
) -> str | None:
    eligible = [
        event
        for event in mapped_events
        if not bool(event.get("boundary_masked", False))
        and abs(int(event["global_chunk_end_ms"]) - int(source_end_ms)) <= tolerance_ms
    ]
    if not eligible:
        return None
    best = min(
        eligible,
        key=lambda event: abs(int(event["global_chunk_end_ms"]) - int(source_end_ms)),
    )
    return str(best["natural_action_target"])


def local_event_rewards(
    events: Sequence[dict[str, object]],
    mapped_events: Sequence[dict[str, object]],
) -> list[float]:
    rewards: list[float] = []
    last_audio_ms: int | None = None
    for event in events:
        source_ms = int(event["source_end_ms"])
        sampled_action = str(event.get("policy_action", "WAIT"))
        executed_action = str(event.get("executed_action", "WAIT"))
        emitted = [
            item for item in event.get("tts_emissions", [])
            if bool(item.get("acknowledged", False))
        ]
        actual_commit = executed_action == "WRITE" and bool(emitted)
        actionable = bool(str(event.get("mt_new_commit", "")).strip())
        gold = nearest_gold_action(source_ms, mapped_events)
        reward = 0.0
        # Gold actions supervise the control decision only when a real target
        # delta exists.  A sampled WRITE when no delta exists is executed as a
        # harmless WAIT by the runtime and must not receive contradictory
        # negative credit.
        if gold is not None and actionable:
            reward += 0.20 if sampled_action == gold else -0.20
            if sampled_action == "WAIT" and gold == "WRITE":
                reward -= 0.50
            if sampled_action == "WRITE" and gold == "READ":
                reward -= 0.35
        coverage = dict(event.get("coverage", {}))
        target_delta = float(coverage.get("target_coverage_delta", 0.0))
        spoken_delta = float(coverage.get("spoken_target_coverage_delta", 0.0))
        language_leak = float(coverage.get("language_leak", 0.0))
        # MT can progress on every observation, but control gets substantial
        # credit only for the portion that is actually spoken by this commit.
        reward += 0.50 * target_delta + 7.0 * spoken_delta
        reward -= 0.50 * language_leak
        # Before useful spoken coverage has accumulated, coverage—not speed—
        # is the objective.  This avoids rewarding a very early fragment that
        # leaves the remainder of a long episode silent.
        enough_spoken_coverage = float(
            coverage.get("spoken_target_coverage", 0.0)
        ) >= 0.20
        if actual_commit and enough_spoken_coverage:
            if last_audio_ms is None:
                reward -= 0.08 * math.log1p(source_ms / 1_000.0)
            else:
                reward -= 0.12 * softplus((source_ms - last_audio_ms - 4_000.0) / 1_000.0)
            last_audio_ms = source_ms
        elif executed_action == "WRITE" and actionable and not actual_commit:
            # A failed actionable commit is harmful.  The same penalty is not
            # applied to an unexecutable sampled WRITE.
            reward -= 0.50
        if bool(event.get("deadline_forced_write", False)) and not actual_commit:
            reward -= 0.25
        if bool(event.get("true_source_final", False)):
            terminal_coverage = float(coverage.get("spoken_target_coverage", 0.0))
            reward += 2.0 * terminal_coverage
            if not bool(event.get("flush_complete", False)):
                reward -= 1.0
        rewards.append(reward)
    return rewards


def reward_to_go(values: Sequence[float], gamma: float = 0.97) -> list[float]:
    output = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + float(gamma) * running
        output[index] = running
    return output


def normalize(values: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    if not values:
        return []
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    scale = max(float(epsilon), variance**0.5)
    return [(float(value) - mean) / scale for value in values]


def assign_trace_advantages(candidates: list[dict[str, object]]) -> None:
    """Mutate traces with advantages normalized by family/event across a group."""

    grouped: dict[tuple[str, int], list[tuple[dict[str, object], float]]] = defaultdict(list)
    for candidate in candidates:
        terminal = candidate["reward"]
        events = candidate["result"]["events"]
        local = local_event_rewards(events, candidate["mapped_action_events"])
        control_return = reward_to_go(local)
        for trace in candidate["traces"]:
            family = str(trace["family"])
            event_index = int(trace["event_index"])
            if family == "control" and not bool(trace.get("actionable_commit", True)):
                # This sampled action was necessarily executed as WAIT; it
                # cannot receive credit for an audio outcome it could not
                # influence.
                trace["local_return"] = 0.0
                trace["advantage"] = 0.0
                continue
            if family == "control":
                raw = control_return[event_index] + 0.15 * float(terminal["total"])
            elif family == "asr":
                raw = float(terminal["asr_quality"]) - 2.0 * float(terminal["asr_shortfall"])
            elif family == "mt":
                raw = (
                    float(terminal["mt_quality"])
                    + 1.0 * float(terminal["target_coverage"])
                    + 3.0 * float(terminal["spoken_target_coverage"])
                    - 2.0 * float(terminal["mt_shortfall"])
                    - 1.0 * float(terminal["completeness_shortfall"])
                    - float(terminal["language_penalty"])
                    - float(terminal["repetition_penalty"])
                )
            elif family == "tts":
                raw = (
                    float(terminal["audio_health"])
                    + 3.0 * float(terminal["spoken_target_coverage"])
                    - 0.10 * float(terminal["silence_penalty"])
                    - float(terminal["failure_penalty"])
                    - float(terminal["pending_penalty"])
                )
            else:
                raise ValueError(f"unknown trace family {family}")
            grouped[(family, event_index)].append((trace, raw))
    for rows in grouped.values():
        advantages = normalize([raw for _, raw in rows])
        for (trace, raw), advantage in zip(rows, advantages):
            trace["local_return"] = raw
            trace["advantage"] = advantage


__all__ = [
    "assign_trace_advantages",
    "local_event_rewards",
    "nearest_gold_action",
    "normalize",
    "reward_to_go",
]
