from experiments.uniss_phasea_event_constrained_grpo_long_v2.training.event_credit import (
    local_event_rewards,
    reward_to_go,
)
from experiments.uniss_phasea_event_constrained_grpo_long_v2.training.reward import (
    continuous_delay_penalty,
    score_episode,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)


def observation(first=1000.0, silence=2000.0):
    return EpisodeObservation(
        asr_teacher_similarity=0.8,
        mt_teacher_similarity=0.7,
        translation_length_ratio=1.0,
        healthy_audio_fraction=1.0,
        spoken_text_fraction=1.0,
        commit_stability=1.0,
        speaker_similarity=1.0,
        first_write_ms=first,
        maximum_internal_silence_ms=silence,
        premature_end_count=0,
        tts_failure_count=0,
        invalid_semantic_fraction=0.0,
    )


def test_delay_penalty_never_saturates():
    assert continuous_delay_penalty(40_000, 1_000) > continuous_delay_penalty(12_000, 1_000)
    assert score_episode(observation(12_000), observation()).total > score_episode(
        observation(40_000), observation()
    ).total


def test_unnecessary_wait_and_gap_receive_local_penalty():
    events = [
        {"source_end_ms": 1000, "policy_action": "WAIT", "tts_emissions": []},
        {
            "source_end_ms": 6000,
            "policy_action": "WRITE",
            "tts_emissions": [{"acknowledged": True}],
        },
    ]
    mapped = [
        {
            "global_chunk_end_ms": 1000,
            "natural_action_target": "WRITE",
            "boundary_masked": False,
        }
    ]
    rewards = local_event_rewards(events, mapped)
    assert rewards[0] < -1.0
    assert len(reward_to_go(rewards)) == 2

