"""Quality-constrained reward for train-seen route-aligned GRPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lower_is_better(value: float, *, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError("bad threshold must exceed good threshold")
    return unit((bad - float(value)) / (bad - good))


def completeness(value: EpisodeObservation) -> float:
    ratio = float(value.translation_length_ratio)
    ratio_score = lower_is_better(abs(ratio - 1.0), good=0.05, bad=0.65)
    return min(ratio_score, unit(value.spoken_text_fraction))


@dataclass(frozen=True)
class ConstrainedReward:
    total: float
    asr_quality: float
    mt_quality: float
    completeness: float
    audio_health: float
    commit_stability: float
    latency: float
    silence: float
    quality_gate: float
    asr_shortfall: float
    mt_shortfall: float
    completeness_shortfall: float
    failure_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_constrained_episode(
    observation: EpisodeObservation,
    baseline: EpisodeObservation,
    *,
    retention: float = 0.98,
) -> ConstrainedReward:
    """Reward latency only after retaining Phase-A quality and completeness."""
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must be in (0, 1]")
    asr = unit(observation.asr_teacher_similarity)
    mt = unit(observation.mt_teacher_similarity)
    complete = completeness(observation)
    base_complete = completeness(baseline)
    asr_target = retention * unit(baseline.asr_teacher_similarity)
    mt_target = retention * unit(baseline.mt_teacher_similarity)
    complete_target = retention * base_complete
    asr_shortfall = max(0.0, asr_target - asr)
    mt_shortfall = max(0.0, mt_target - mt)
    complete_shortfall = max(0.0, complete_target - complete)
    gate_parts = (
        1.0 if asr_target <= 0 else unit(asr / asr_target),
        1.0 if mt_target <= 0 else unit(mt / mt_target),
        1.0 if complete_target <= 0 else unit(complete / complete_target),
        unit(observation.healthy_audio_fraction),
    )
    quality_gate = min(gate_parts)
    latency = lower_is_better(
        observation.first_write_ms, good=640.0, bad=12_000.0
    )
    silence = lower_is_better(
        observation.maximum_internal_silence_ms, good=1_000.0, bad=30_000.0
    )
    failure_penalty = min(
        3.0,
        float(observation.premature_end_count)
        + float(observation.tts_failure_count)
        + 2.0 * unit(observation.invalid_semantic_fraction),
    )
    total = (
        1.50 * asr
        + 2.00 * mt
        + 1.50 * complete
        + 0.50 * unit(observation.healthy_audio_fraction)
        + 0.40 * unit(observation.commit_stability)
        + quality_gate * (0.80 * latency + 0.80 * silence)
        - 4.00 * asr_shortfall
        - 5.00 * mt_shortfall
        - 5.00 * complete_shortfall
        - 2.00 * failure_penalty
    )
    return ConstrainedReward(
        total=total,
        asr_quality=asr,
        mt_quality=mt,
        completeness=complete,
        audio_health=unit(observation.healthy_audio_fraction),
        commit_stability=unit(observation.commit_stability),
        latency=latency,
        silence=silence,
        quality_gate=quality_gate,
        asr_shortfall=asr_shortfall,
        mt_shortfall=mt_shortfall,
        completeness_shortfall=complete_shortfall,
        failure_penalty=failure_penalty,
    )


__all__ = ["ConstrainedReward", "completeness", "score_constrained_episode"]

