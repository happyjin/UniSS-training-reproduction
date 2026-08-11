from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit8.pretrain_overfit8 import (
    is_v8_trainable_parameter,
)


def test_only_natural_length_parameters_are_trainable() -> None:
    prefix = "module.true_subsecond_objective.semantic_block_head."
    assert is_v8_trainable_parameter(prefix + "length_head.3.weight")
    assert not is_v8_trainable_parameter(prefix + "output_projection.weight")
    assert not is_v8_trainable_parameter(prefix + "slot_embeddings.weight")
    assert not is_v8_trainable_parameter("module.true_subsecond_lora.decoder.layers.0")
