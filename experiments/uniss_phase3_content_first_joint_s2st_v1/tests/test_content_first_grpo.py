from experiments.uniss_phase3_content_first_joint_s2st_v1.training.pretrain_content_first_grpo import (
    CONTENT_PREFIXES,
    POLICY_PREFIX,
    _base_key,
)


def test_content_first_checkpoint_namespaces_are_disjoint() -> None:
    assert POLICY_PREFIX == "quality_grpo_lora."
    assert all(not POLICY_PREFIX.startswith(prefix) for prefix in CONTENT_PREFIXES)
    assert _base_key("true_subsecond_lora.branches.x/shard_0_8") == (
        "true_subsecond_lora.branches.x"
    )

