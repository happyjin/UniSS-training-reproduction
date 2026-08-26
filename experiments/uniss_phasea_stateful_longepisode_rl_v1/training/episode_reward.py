"""Auditable reward for free-running long-episode simultaneous S2ST rollouts.

Unlike the historical position-wise top-k objective, this reward consumes one
complete free-running episode summary.  It can therefore observe early END,
unspoken committed text, unhealthy audio, semantic continuation, long playback
gaps, first WRITE latency, and quality retention against frozen teachers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lower_is_better(value: float, *, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError("bad threshold must be greater than good threshold")
    return _unit((bad - float(value)) / (bad - good))


@dataclass(frozen=True)
class EpisodeObservation:
    asr_teacher_similarity: float
    mt_teacher_similarity: float
    translation_length_ratio: float
    healthy_audio_fraction: float
    spoken_text_fraction: float
    commit_stability: float
    speaker_similarity: float
    first_write_ms: float
    maximum_internal_silence_ms: float
    premature_end_count: int
    tts_failure_count: int
    invalid_semantic_fraction: float


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    asr_quality: float
    mt_quality: float
    completeness: float
    audio_health: float
    spoken_coverage: float
    commit_stability: float
    speaker_continuity: float
    latency: float
    silence: float
    premature_end_penalty: float
    tts_failure_penalty: float
    invalid_semantic_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_episode(observation: EpisodeObservation) -> RewardBreakdown:
    """Score one complete free-running episode without hard-gate aborts."""

    asr = _unit(observation.asr_teacher_similarity)
    mt = _unit(observation.mt_teacher_similarity)
    # Ratios above one are not rewarded for hallucinated verbosity.  Ratios
    # below 0.75 rapidly expose incomplete translations.
    completeness = _lower_is_better(
        abs(float(observation.translation_length_ratio) - 1.0),
        good=0.05,
        bad=0.65,
    )
    audio = _unit(observation.healthy_audio_fraction)
    spoken = _unit(observation.spoken_text_fraction)
    stability = _unit(observation.commit_stability)
    speaker = _unit(observation.speaker_similarity)
    latency = _lower_is_better(observation.first_write_ms, good=640.0, bad=12_000.0)
    silence = _lower_is_better(
        observation.maximum_internal_silence_ms, good=1_000.0, bad=30_000.0
    )
    early_penalty = min(1.0, max(0, int(observation.premature_end_count)) / 3.0)
    tts_penalty = min(1.0, max(0, int(observation.tts_failure_count)) / 3.0)
    invalid_penalty = _unit(observation.invalid_semantic_fraction)
    total = (
        0.55 * asr
        + 1.40 * mt
        + 0.85 * completeness
        + 0.55 * audio
        + 1.10 * spoken
        + 0.55 * stability
        + 0.35 * speaker
        + 0.20 * latency
        + 0.30 * silence
        - 1.10 * early_penalty
        - 0.80 * tts_penalty
        - 0.80 * invalid_penalty
    )
    return RewardBreakdown(
        total=total,
        asr_quality=asr,
        mt_quality=mt,
        completeness=completeness,
        audio_health=audio,
        spoken_coverage=spoken,
        commit_stability=stability,
        speaker_continuity=speaker,
        latency=latency,
        silence=silence,
        premature_end_penalty=early_penalty,
        tts_failure_penalty=tts_penalty,
        invalid_semantic_penalty=invalid_penalty,
    )


def group_relative_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("group-relative reward requires at least two rollouts")
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = max(float(epsilon), variance**0.5)
    return [(value - mean) / scale for value in values]


def observation_from_mapping(value: Mapping[str, object]) -> EpisodeObservation:
    return EpisodeObservation(**{field: value[field] for field in EpisodeObservation.__dataclass_fields__})  # type: ignore[arg-type]


__all__ = [
    "EpisodeObservation",
    "RewardBreakdown",
    "group_relative_advantages",
    "observation_from_mapping",
    "score_episode",
]

