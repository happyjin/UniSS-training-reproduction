from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    V12_WEIGHTS,
    is_generalize12_trainable_parameter,
)


def test_only_isolated_microblock_head_is_trainable() -> None:
    prefix = "module.true_subsecond_objective.semantic_microblock_head."
    assert is_generalize12_trainable_parameter(prefix + "slot_embeddings")
    assert not is_generalize12_trainable_parameter(
        "module.true_subsecond_objective.semantic_block_head.output_projection.weight"
    )
    assert not is_generalize12_trainable_parameter("module.true_subsecond_lora.x")


def test_only_microblock_losses_have_optimization_weight() -> None:
    assert V12_WEIGHTS["microblock_semantic_content"] == 1.0
    assert V12_WEIGHTS["microblock_final_length"] == 0.5
    assert V12_WEIGHTS["microblock_continue"] == 1.0
    assert all(
        weight == 0.0
        for name, weight in V12_WEIGHTS.items()
        if not name.startswith("microblock_")
    )
