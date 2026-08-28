from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.event_credit import (
    local_event_rewards,
    reward_to_go,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.reward import (
    continuous_delay_penalty,
    score_episode,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.coverage import (
    audit_episode,
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


def coverage(text="one two three four five six seven eight nine ten"):
    return audit_episode(
        teacher_translation="one two three four five six seven eight nine ten",
        generated_translation=text,
        target_language="eng",
        events=[],
        eos_pending_items=0,
    )


def test_delay_penalty_never_saturates():
    assert continuous_delay_penalty(40_000, 1_000) > continuous_delay_penalty(12_000, 1_000)
    assert score_episode(observation(12_000), observation(), coverage()).total > score_episode(
        observation(40_000), observation(), coverage()
    ).total


def test_incomplete_early_candidate_cannot_win_on_latency_alone():
    full = score_episode(observation(8_000), observation(), coverage())
    partial = score_episode(observation(640), observation(), coverage("one two"))
    assert full.total > partial.total


def test_unnecessary_wait_and_gap_receive_local_penalty():
    events = [
        {"source_end_ms": 1000, "policy_action": "WAIT", "tts_emissions": [], "coverage": {}},
        {
            "source_end_ms": 6000,
            "policy_action": "WRITE",
            "tts_emissions": [{"acknowledged": True}],
            "coverage": {"target_coverage_delta": 0.2, "spoken_target_coverage_delta": 0.2},
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
