from experiments.uniss_phasea_coverage_constrained_grpo_v3.evaluation.write_report import (
    best_per_episode,
    summarize,
    validate_rollout,
)


def candidate(episode, group, reward):
    return {
        "group_index": group,
        "reward": {"total": reward},
        "observation": {
            "asr_teacher_similarity": 0.8,
            "mt_teacher_similarity": 0.7,
            "translation_length_ratio": 0.6,
            "spoken_text_fraction": 1.0,
            "first_write_ms": 640.0,
            "maximum_internal_silence_ms": 320.0,
            "healthy_audio_fraction": 1.0,
        },
        "result": {
            "rtf": 2.0,
            "audio_writes": 3,
            "tts_pending_unspoken_items": 0,
            "tts_failures": 0,
            "translation_audio_to_source_duration_ratio": 0.5,
        },
    }


def test_strict_geometry_and_best_of_four():
    summaries = []
    for episode in range(64):
        summaries.append(
            {
                "episode_id": f"episode_{episode}",
                "direction": "cmn->eng" if episode % 2 == 0 else "eng->cmn",
                "source_audio": "source.wav",
                "candidates": [candidate(episode, group, float(group)) for group in range(4)],
            }
        )
    payload = {"status": "complete", "episodes": 64, "group_size": 4, "summaries": summaries}
    rows = validate_rollout(payload)
    best = best_per_episode(rows)
    assert len(rows) == 256
    assert len(best) == 64
    assert {row["group_index"] for row in best} == {3}
    assert summarize(best)["reward"]["mean"] == 3.0
    assert len([row for row in rows if row["direction"] == "cmn->eng"]) == 128
    assert len([row for row in best if row["direction"] == "eng->cmn"]) == 32
