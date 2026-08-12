from __future__ import annotations

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.training.pretrain_event_rollout_megatron import (
    is_event_rollout_v2_trainable_parameter,
)


def test_v2_repairs_both_causal_frontend_modules() -> None:
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_objective.frontend_projection.weight"
    )
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_objective.frontend_adapter.layers.0.in_proj.weight"
    )


def test_v2_retains_existing_runtime_scope() -> None:
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_lora.branches.decoder__layers__0__self_attention__linear_qkv.lora_a"
    )
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_objective.action_head.network.3.weight"
    )
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_objective.semantic_microblock_head.continue_head.3.weight"
    )
    assert is_event_rollout_v2_trainable_parameter(
        "true_subsecond_objective.continuation_head.weight"
    )


def test_v2_keeps_phase3_base_frozen() -> None:
    assert not is_event_rollout_v2_trainable_parameter("decoder.layers.0.self_attention.linear_qkv.weight")
    assert not is_event_rollout_v2_trainable_parameter("embedding.word_embeddings.weight")
