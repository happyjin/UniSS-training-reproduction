from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize10.pretrain_generalize10 import (
    is_generalize10_trainable_parameter,
)


def test_only_natural_semantic_head_is_trainable() -> None:
    prefix = "module.true_subsecond_objective.semantic_block_head."
    assert is_generalize10_trainable_parameter(prefix + "output_projection.weight")
    assert is_generalize10_trainable_parameter(prefix + "length_head.3.weight")
    assert is_generalize10_trainable_parameter(prefix + "slot_embeddings.weight")
    assert not is_generalize10_trainable_parameter(
        "module.true_subsecond_objective.action_head.weight"
    )
    assert not is_generalize10_trainable_parameter(
        "module.true_subsecond_lora.decoder.layers.0.self_attention.linear_qkv.lora_A"
    )
