from experiments.uniss_phasea_commit_complete_sft_rl_v4.training.event_credit import (
    assign_trace_advantages,
    local_event_rewards,
)
from experiments.uniss_phasea_commit_complete_sft_rl_v4.training.reward import score_episode
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.coverage import audit_episode
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
)


def _observation(first=1_000.0, silence=2_000.0):
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


def test_sampled_empty_write_is_not_rewarded_as_a_commit():
    events = [
        {
            "source_end_ms": 1_000,
            "policy_action": "WRITE",
            "executed_action": "WAIT",
            "mt_new_commit": "",
            "tts_emissions": [],
            "coverage": {},
        },
        {
            "source_end_ms": 2_000,
            "policy_action": "WRITE",
            "executed_action": "WRITE",
            "mt_new_commit": "hello",
            "tts_emissions": [{"acknowledged": True, "text": "hello"}],
            "coverage": {
                "target_coverage_delta": 0.1,
                "spoken_target_coverage_delta": 0.1,
            },
        },
    ]
    rewards = local_event_rewards(events, [])
    assert rewards[0] == 0.0
    assert rewards[1] > 0.5


def test_spoken_coverage_beats_generated_only_coverage():
    reference = "one two three four five six seven eight nine ten"
    generated_only = audit_episode(
        teacher_translation=reference,
        generated_translation=reference,
        target_language="eng",
        events=[],
        eos_pending_items=0,
    )
    spoken_events = [{"tts_emissions": [{"acknowledged": True, "text": reference}]}]
    spoken = audit_episode(
        teacher_translation=reference,
        generated_translation=reference,
        target_language="eng",
        events=spoken_events,
        eos_pending_items=0,
    )
    assert score_episode(_observation(), _observation(), spoken).total > score_episode(
        _observation(), _observation(), generated_only
    ).total


def test_non_actionable_control_trace_receives_no_rl_gradient():
    terminal = {
        "total": 2.0,
        "asr_quality": 0.8,
        "mt_quality": 0.7,
        "target_coverage": 0.3,
        "spoken_target_coverage": 0.3,
        "audio_health": 1.0,
        "asr_shortfall": 0.0,
        "mt_shortfall": 0.0,
        "completeness_shortfall": 0.7,
        "silence_penalty": 1.0,
        "language_penalty": 0.0,
        "repetition_penalty": 0.0,
        "pending_penalty": 0.0,
        "failure_penalty": 0.0,
    }
    candidate = {
        "reward": terminal,
        "mapped_action_events": [],
        "result": {"events": [{"source_end_ms": 320, "coverage": {}}]},
        "traces": [{"family": "control", "event_index": 0, "actionable_commit": False}],
    }
    assign_trace_advantages([candidate])
    assert candidate["traces"][0]["advantage"] == 0.0
