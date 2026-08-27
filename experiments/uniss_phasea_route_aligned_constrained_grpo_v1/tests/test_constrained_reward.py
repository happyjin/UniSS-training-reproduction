from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.constrained_reward import (
    score_constrained_episode,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)


def observation(**updates) -> EpisodeObservation:
    values = {
        "asr_teacher_similarity": 0.70,
        "mt_teacher_similarity": 0.40,
        "translation_length_ratio": 0.90,
        "healthy_audio_fraction": 1.0,
        "spoken_text_fraction": 1.0,
        "commit_stability": 0.50,
        "speaker_similarity": 1.0,
        "first_write_ms": 8_000.0,
        "maximum_internal_silence_ms": 20_000.0,
        "premature_end_count": 0,
        "tts_failure_count": 0,
        "invalid_semantic_fraction": 0.0,
    }
    values.update(updates)
    return EpisodeObservation(**values)


def test_latency_only_wins_after_quality_retention() -> None:
    baseline = observation()
    safe_fast = score_constrained_episode(
        observation(first_write_ms=1_000.0, maximum_internal_silence_ms=2_000.0),
        baseline,
    )
    incomplete_fast = score_constrained_episode(
        observation(
            mt_teacher_similarity=0.10,
            translation_length_ratio=0.20,
            first_write_ms=640.0,
            maximum_internal_silence_ms=1_000.0,
        ),
        baseline,
    )
    assert safe_fast.quality_gate == 1.0
    assert incomplete_fast.quality_gate < 0.5
    assert safe_fast.total > incomplete_fast.total


def test_asr_shortfall_is_explicitly_penalized() -> None:
    baseline = observation()
    value = score_constrained_episode(
        observation(asr_teacher_similarity=0.20, first_write_ms=640.0), baseline
    )
    assert value.asr_shortfall > 0.0
    assert value.quality_gate < 1.0

