from experiments.uniss_phasea_rl_trainlong_eval_v1.evaluation.extract_rollout_group import (
    extract,
)


def test_extract_preserves_protocol_order_and_normalizes_sample_id(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    rollout = {
        "summaries": [
            {
                "episode_id": "episode_1",
                "candidates": [
                    {
                        "group_index": 0,
                        "result": {"sample_id": "episode_1_g0", "source_audio": str(audio)},
                    },
                    {
                        "group_index": 1,
                        "result": {"sample_id": "episode_1_g1", "source_audio": str(audio)},
                    },
                ],
            }
        ]
    }
    protocol = {
        "records": [
            {"episode_id": "episode_1", "source_audio": str(audio)}
        ]
    }
    rows = extract(rollout, protocol, 0)
    assert rows[0]["sample_id"] == "episode_1"
    assert rows[0]["rollout_candidate_sample_id"] == "episode_1_g0"
    assert rows[0]["pre_rl_candidate_group"] == 0
