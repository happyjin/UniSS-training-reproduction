"""Coverage-first reward for content-gated long-form S2ST rollouts.

The v3 reward only enabled timing credit after a fixed 0.75 coverage target.
That target was unreachable in the observed free-running regime, so group
advantages contained almost no useful control signal.  V4 rewards healthy
*spoken* target progress directly and only introduces a modest timing term
after a candidate has actually spoken some target-aligned content.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.coverage import (
    CoverageAudit,
)


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def continuous_delay_penalty(value_ms: float, scale_ms: float) -> float:
    return math.log1p(max(0.0, float(value_ms)) / float(scale_ms))


@dataclass(frozen=True)
class EventConstrainedReward:
    total: float
    asr_quality: float
    mt_quality: float
    completeness: float
    target_coverage: float
    spoken_target_coverage: float
    language_purity: float
    repetition_fraction: float
    audio_health: float
    commit_stability: float
    first_write_penalty: float
    silence_penalty: float
    asr_shortfall: float
    mt_shortfall: float
    completeness_shortfall: float
    terminal_coverage_penalty: float
    pending_penalty: float
    language_penalty: float
    repetition_penalty: float
    failure_penalty: float
    actionable_spoken_progress: float
    latency_eligible: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_episode(
    observation: EpisodeObservation,
    baseline: EpisodeObservation,
    coverage: CoverageAudit,
    *,
    asr_retention: float = 0.98,
    mt_retention: float = 0.98,
    latency_coverage_floor: float = 0.08,
) -> EventConstrainedReward:
    """Reward real coverage progress before latency.

    ``spoken_target_coverage`` counts only text with a healthy emitted
    waveform.  Generated-but-never-spoken text therefore cannot win a group.
    """
    asr = unit(observation.asr_teacher_similarity)
    mt = unit(observation.mt_teacher_similarity)
    complete = min(
        unit(coverage.spoken_target_coverage), unit(coverage.length_score)
    )
    asr_target = asr_retention * unit(baseline.asr_teacher_similarity)
    mt_target = mt_retention * unit(baseline.mt_teacher_similarity)
    asr_shortfall = max(0.0, asr_target - asr)
    mt_shortfall = max(0.0, mt_target - mt)
    complete_shortfall = 1.0 - complete
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
    terminal_coverage_penalty = 2.0 * complete_shortfall * complete_shortfall
    pending_penalty = min(4.0, float(coverage.eos_pending_items))
    language_penalty = 2.0 * (1.0 - unit(coverage.language_purity))
    repetition_penalty = 1.5 * unit(coverage.repetition_fraction)
    eligible_for_latency = (
        unit(coverage.spoken_target_coverage) >= latency_coverage_floor
        and mt >= 0.75 * mt_target
        and asr >= 0.75 * asr_target
    )
    latency_term = (
        -0.15 * first_penalty - 0.25 * silence_penalty
        if eligible_for_latency
        else 0.0
    )
    total = (
        1.0 * asr
        + 2.0 * mt
        + 3.0 * unit(coverage.target_coverage)
        + 7.0 * unit(coverage.spoken_target_coverage)
        + 1.0 * unit(observation.healthy_audio_fraction)
        + 0.5 * unit(observation.commit_stability)
        + latency_term
        - 3.0 * asr_shortfall
        - 4.0 * mt_shortfall
        - terminal_coverage_penalty
        - 2.0 * pending_penalty
        - language_penalty
        - repetition_penalty
        - 1.5 * failures
    )
    return EventConstrainedReward(
        total=total,
        asr_quality=asr,
        mt_quality=mt,
        completeness=complete,
        target_coverage=unit(coverage.target_coverage),
        spoken_target_coverage=unit(coverage.spoken_target_coverage),
        language_purity=unit(coverage.language_purity),
        repetition_fraction=unit(coverage.repetition_fraction),
        audio_health=unit(observation.healthy_audio_fraction),
        commit_stability=unit(observation.commit_stability),
        first_write_penalty=first_penalty,
        silence_penalty=silence_penalty,
        asr_shortfall=asr_shortfall,
        mt_shortfall=mt_shortfall,
        completeness_shortfall=complete_shortfall,
        terminal_coverage_penalty=terminal_coverage_penalty,
        pending_penalty=pending_penalty,
        language_penalty=language_penalty,
        repetition_penalty=repetition_penalty,
        failure_penalty=failures,
        actionable_spoken_progress=unit(coverage.spoken_target_coverage),
        latency_eligible=float(eligible_for_latency),
    )


__all__ = [
    "EventConstrainedReward",
    "completeness",
    "continuous_delay_penalty",
    "score_episode",
]
