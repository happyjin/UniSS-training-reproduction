from experiments.uniss_phasea_commit_complete_sft_rl_v4.data.build_commit_events import relabel


def row(delta, safe=True):
    return {
        "target_text_delta_ids": delta,
        "safe_commit_mask": [safe],
        "natural_action_target": "WRITE",
    }


def test_only_safe_nonempty_teacher_delta_is_commit():
    value = relabel(row([1, 2]), minimum_delta_tokens=2)
    assert value["natural_action_target"] == "WRITE"
    assert value["commit_target"] == "COMMIT"
    assert value["commit_delta_tokens"] == 2


def test_short_or_unsafe_delta_becomes_wait():
    assert relabel(row([1]), minimum_delta_tokens=2)["commit_target"] == "WAIT"
    assert relabel(row([1, 2], safe=False), minimum_delta_tokens=2)["commit_target"] == "WAIT"
