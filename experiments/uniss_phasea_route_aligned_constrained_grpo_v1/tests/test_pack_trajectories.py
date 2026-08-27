from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.pack_trajectories import (
    trajectory_sample,
)


def test_asr_trajectory_is_trainable() -> None:
    row = {
        "family": "asr",
        "prompt_ids": [1, 2, 3],
        "generated_ids": [4, 5],
        "old_log_probs": [-0.1, -0.2],
        "advantage": 1.25,
        "episode_id": "episode",
        "group_index": 0,
        "trace_index": 0,
    }
    sample = trajectory_sample(row)
    assert sample["family_ids"] == [1, 1, 1, 1]
    assert sum(sample["response_mask"]) == 2.0
    assert sample["advantages"][-2:] == [1.25, 1.25]

