from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.pack_trajectories import (
    pack_samples,
    trajectory_sample,
)


def test_generated_token_log_probs_align_with_response_labels():
    sample = trajectory_sample(
        {
            "prompt_ids": [1, 2, 3],
            "generated_ids": [4, 5],
            "old_log_probs": [-1.0, -2.0],
            "advantage": 0.5,
            "episode_id": "e",
            "group_index": 0,
            "trace_index": 0,
            "family": "mt",
        }
    )
    assert sample["tokens"] == [1, 2, 3, 4]
    assert sample["labels"] == [2, 3, 4, 5]
    assert sample["response_mask"] == [0.0, 0.0, 1.0, 1.0]
    assert sample["old_log_probs"] == [0.0, 0.0, -1.0, -2.0]


def test_packer_preserves_sample_boundaries_and_padding():
    first = {
        "tokens": [1, 2], "labels": [2, 3], "response_mask": [1.0, 1.0],
        "old_log_probs": [-1.0, -1.0], "advantages": [1.0, 1.0],
        "replay_mask": [0.0, 0.0], "family_ids": [1, 1], "identity": "a",
    }
    rows = list(pack_samples([first, first], 8))
    assert len(rows) == 1
    assert rows[0]["sample_boundaries"] == [[0, 2], [2, 4]]
    assert rows[0]["used_tokens"] == 4
    assert len(rows[0]["tokens"]) == 8

