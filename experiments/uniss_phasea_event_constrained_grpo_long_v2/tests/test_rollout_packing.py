from experiments.uniss_phasea_event_constrained_grpo_long_v2.training.pack_rollouts import (
    trajectory_sample,
)
from training import constants_uniss as c


def test_control_trajectory_has_one_local_advantage():
    row = {
        "family": "control",
        "prompt_ids": [c.TOKEN_TASK_STREAMING_S2ST, c.TOKEN_END_GLM],
        "generated_ids": [c.TOKEN_WRITE_GENERATE],
        "old_log_probs": [-0.2],
        "advantage": 1.25,
        "episode_id": "episode",
        "group_index": 0,
        "event_index": 3,
        "trace_index": 4,
    }
    sample = trajectory_sample(row)
    assert sample is not None
    position = sample["response_mask"].index(1.0)
    assert sample["advantages"][position] == 1.25
    assert sample["labels"][position] == c.TOKEN_WRITE_GENERATE


def test_asr_trace_is_excluded_to_preserve_phase_a():
    assert trajectory_sample({"family": "asr"}) is None
