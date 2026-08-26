from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
    group_relative_advantages,
    score_episode,
)


def observation(**updates):
    values = {
        "asr_teacher_similarity": 0.9,
        "mt_teacher_similarity": 0.9,
        "translation_length_ratio": 0.95,
        "healthy_audio_fraction": 1.0,
        "spoken_text_fraction": 1.0,
        "commit_stability": 0.95,
        "speaker_similarity": 0.9,
        "first_write_ms": 1280.0,
        "maximum_internal_silence_ms": 1500.0,
        "premature_end_count": 0,
        "tts_failure_count": 0,
        "invalid_semantic_fraction": 0.0,
    }
    values.update(updates)
    return EpisodeObservation(**values)


def test_complete_healthy_episode_beats_incomplete_silent_episode():
    good = score_episode(observation())
    bad = score_episode(
        observation(
            mt_teacher_similarity=0.2,
            translation_length_ratio=0.2,
            healthy_audio_fraction=0.2,
            spoken_text_fraction=0.1,
            maximum_internal_silence_ms=40_000,
            premature_end_count=4,
            tts_failure_count=4,
        )
    )
    assert good.total > bad.total


def test_reward_prioritizes_completeness_over_small_latency_gain():
    complete = score_episode(observation(first_write_ms=3000, spoken_text_fraction=1.0))
    incomplete = score_episode(
        observation(first_write_ms=640, spoken_text_fraction=0.4, translation_length_ratio=0.4)
    )
    assert complete.total > incomplete.total


def test_group_advantages_are_zero_mean_and_order_preserving():
    values = group_relative_advantages([1.0, 2.0, 4.0, 8.0])
    assert abs(sum(values)) < 1e-9
    assert values == sorted(values)

