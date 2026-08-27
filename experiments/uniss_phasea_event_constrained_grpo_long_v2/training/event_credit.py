"""Local WAIT/WRITE rewards and family-specific group advantages."""

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
        action = str(event.get("policy_action", "WAIT"))
        gold = nearest_gold_action(source_ms, mapped_events)
        reward = 0.0
        if gold is not None:
            reward += 0.30 if action == gold else -0.30
            if action == "WAIT" and gold == "WRITE":
                reward -= 1.00
            if action == "WRITE" and gold == "READ":
                reward -= 1.50
        emissions = list(event.get("tts_emissions", []))
        healthy = [item for item in emissions if bool(item.get("acknowledged", False))]
        if healthy:
            if last_audio_ms is None:
                reward -= 0.35 * math.log1p(source_ms / 1_000.0)
            else:
                reward -= 0.55 * softplus((source_ms - last_audio_ms - 2_000.0) / 1_000.0)
            reward += 0.40 * len(healthy)
            last_audio_ms = source_ms
        elif action == "WRITE":
            reward -= 0.20
        if bool(event.get("deadline_forced_write", False)):
            reward -= 1.00
        if bool(event.get("true_source_final", False)):
            reward += 0.75 if bool(event.get("flush_complete", False)) else -1.50
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
            if family == "control":
                raw = control_return[event_index] + 0.20 * float(terminal["total"])
            elif family == "asr":
                raw = float(terminal["asr_quality"]) - 2.0 * float(terminal["asr_shortfall"])
            elif family == "mt":
                raw = (
                    float(terminal["mt_quality"])
                    + float(terminal["completeness"])
                    - 2.0 * float(terminal["mt_shortfall"])
                    - 2.0 * float(terminal["completeness_shortfall"])
                )
            elif family == "tts":
                raw = (
                    float(terminal["audio_health"])
                    + float(terminal["completeness"])
                    - 0.25 * float(terminal["silence_penalty"])
                    - float(terminal["failure_penalty"])
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
