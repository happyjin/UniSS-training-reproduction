from collections import OrderedDict

import torch

import experiments.uniss_phase3_runtime_parity_streaming_v2.generalize11_full15.pretrain_generalize11 as generalize11


def test_only_natural_semantic_head_is_trainable() -> None:
    prefix = "module.true_subsecond_objective.semantic_block_head."
    assert generalize11.is_generalize11_trainable_parameter(
        prefix + "output_projection.weight"
    )
    assert generalize11.is_generalize11_trainable_parameter(prefix + "length_head.3.weight")
    assert generalize11.is_generalize11_trainable_parameter(prefix + "slot_embeddings.weight")
    assert not generalize11.is_generalize11_trainable_parameter(
        "module.true_subsecond_objective.action_head.weight"
    )
    assert not generalize11.is_generalize11_trainable_parameter(
        "module.true_subsecond_lora.decoder.layers.0.self_attention.linear_qkv.lora_A"
    )


def test_replay_fraction_matches_frozen_manifest_policy() -> None:
    assert generalize11.REPLAY_FRACTION == 0.01


def test_reducer_reports_actual_replay_fraction(monkeypatch) -> None:
    total = torch.tensor(3.0)

    def original(output, *, progress):
        assert output == "output"
        assert progress == 0.5
        return total, OrderedDict(curriculum_replay_fraction=torch.tensor(0.10))

    monkeypatch.setattr(generalize11, "_ORIGINAL_DISTRIBUTED_OBJECTIVE", original)
    value, metrics = generalize11.distributed_generalize11_objective(
        "output", progress=0.5
    )
    assert value is total
    assert torch.isclose(
        metrics["curriculum_replay_fraction"],
        torch.tensor(generalize11.REPLAY_FRACTION),
    )
