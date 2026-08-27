"""Non-saturating episode reward for long-form simultaneous S2ST."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def completeness(value: EpisodeObservation) -> float:
    ratio = float(value.translation_length_ratio)
    length = math.exp(-abs(math.log(max(1e-4, ratio))))
    return min(unit(length), unit(value.spoken_text_fraction))


def continuous_delay_penalty(value_ms: float, scale_ms: float) -> float:
    return math.log1p(max(0.0, float(value_ms)) / float(scale_ms))


@dataclass(frozen=True)
class EventConstrainedReward:
    total: float
    asr_quality: float
    mt_quality: float
    completeness: float
    audio_health: float
    commit_stability: float
    first_write_penalty: float
    silence_penalty: float
    asr_shortfall: float
    mt_shortfall: float
    completeness_shortfall: float
    failure_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_episode(
    observation: EpisodeObservation,
    baseline: EpisodeObservation,
    *,
    asr_retention: float = 0.95,
    mt_retention: float = 0.90,
    completeness_retention: float = 0.90,
) -> EventConstrainedReward:
    asr = unit(observation.asr_teacher_similarity)
    mt = unit(observation.mt_teacher_similarity)
    complete = completeness(observation)
    asr_target = asr_retention * unit(baseline.asr_teacher_similarity)
    mt_target = mt_retention * unit(baseline.mt_teacher_similarity)
    complete_target = completeness_retention * completeness(baseline)
    asr_shortfall = max(0.0, asr_target - asr)
    mt_shortfall = max(0.0, mt_target - mt)
    complete_shortfall = max(0.0, complete_target - complete)
    first_penalty = continuous_delay_penalty(observation.first_write_ms, 1_000.0)
    silence_penalty = continuous_delay_penalty(
        observation.maximum_internal_silence_ms, 2_000.0
    )
    failures = min(
        4.0,
        float(observation.premature_end_count)
        + float(observation.tts_failure_count)
        + 2.0 * unit(observation.invalid_semantic_fraction),
    )
    total = (
        1.0 * asr
        + 1.5 * mt
        + 1.5 * complete
        + 1.0 * unit(observation.healthy_audio_fraction)
        + 0.5 * unit(observation.commit_stability)
        - 0.35 * first_penalty
        - 0.55 * silence_penalty
        - 4.0 * asr_shortfall
        - 5.0 * mt_shortfall
        - 5.0 * complete_shortfall
        - 1.5 * failures
    )
    return EventConstrainedReward(
        total=total,
        asr_quality=asr,
        mt_quality=mt,
        completeness=complete,
        audio_health=unit(observation.healthy_audio_fraction),
        commit_stability=unit(observation.commit_stability),
        first_write_penalty=first_penalty,
        silence_penalty=silence_penalty,
        asr_shortfall=asr_shortfall,
        mt_shortfall=mt_shortfall,
        completeness_shortfall=complete_shortfall,
        failure_penalty=failures,
    )


__all__ = [
    "EventConstrainedReward",
    "completeness",
    "continuous_delay_penalty",
    "score_episode",
]

