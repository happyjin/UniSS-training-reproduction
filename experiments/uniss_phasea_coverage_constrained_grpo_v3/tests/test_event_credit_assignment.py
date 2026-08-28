from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.event_credit import (
    assign_trace_advantages,
)


def terminal(total):
    return {
        "total": total,
        "asr_quality": 0.8,
        "mt_quality": 0.7,
        "completeness": 0.8,
        "target_coverage": 0.8,
        "spoken_target_coverage": 0.8,
        "audio_health": 1.0,
        "asr_shortfall": 0.0,
        "mt_shortfall": 0.0,
        "completeness_shortfall": 0.0,
        "silence_penalty": 1.0,
        "language_penalty": 0.0,
        "repetition_penalty": 0.0,
        "pending_penalty": 0.0,
        "failure_penalty": 0.0,
    }


def candidate(action, total):
    return {
        "reward": terminal(total),
        "mapped_action_events": [
            {
                "global_chunk_end_ms": 1000,
                "natural_action_target": "WRITE",
                "boundary_masked": False,
            }
        ],
        "result": {
            "events": [
                {"source_end_ms": 1000, "policy_action": action, "tts_emissions": [], "coverage": {}}
            ]
        },
        "traces": [{"family": "control", "event_index": 0}],
    }


def test_control_advantage_is_normalized_within_same_event():
    candidates = [candidate("WRITE", 1.0), candidate("WAIT", 0.0)]
    assign_trace_advantages(candidates)
    assert candidates[0]["traces"][0]["advantage"] > 0
    assert candidates[1]["traces"][0]["advantage"] < 0
