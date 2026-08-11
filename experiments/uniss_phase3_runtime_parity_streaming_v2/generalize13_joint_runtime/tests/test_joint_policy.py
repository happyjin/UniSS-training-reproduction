from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize13_joint_runtime.pretrain_generalize13 import (
    V13_WEIGHTS,
    is_generalize13_trainable_parameter,
)


def test_joint_scope_updates_runtime_routes_but_not_phase3_base() -> None:
    assert is_generalize13_trainable_parameter(
        "module.true_subsecond_lora.branches.layer.lora_a"
    )
    assert is_generalize13_trainable_parameter(
        "module.true_subsecond_objective.action_head.network.1.weight"
    )
    assert is_generalize13_trainable_parameter(
        "module.true_subsecond_objective.semantic_microblock_head.transition.1.weight"
    )
    assert not is_generalize13_trainable_parameter(
        "module.true_subsecond_objective.frontend_projection.weight"
    )
    assert not is_generalize13_trainable_parameter(
        "module.decoder.layers.0.self_attention.linear_qkv.weight"
    )


def test_text_action_semantic_and_replay_all_have_optimization_mass() -> None:
    assert V13_WEIGHTS["phase3_replay"] > 0
    assert V13_WEIGHTS["runtime_text_content"] > V13_WEIGHTS["interleaved_trajectory"]
    assert V13_WEIGHTS["runtime_critical_boundary"] > 0
    assert V13_WEIGHTS["runtime_action"] > 0
    assert V13_WEIGHTS["microblock_semantic_content"] > 0
    assert V13_WEIGHTS["microblock_continue"] > 0
