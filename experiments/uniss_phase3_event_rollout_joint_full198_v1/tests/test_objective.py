from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
    ROLLOUT_DIAGNOSTIC_NAMES,
    ROLLOUT_METRIC_NAMES,
    ROLLOUT_TERM_NAMES,
    continuation_positions_and_targets,
    distributed_event_rollout_objective,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.pretrain_event_rollout_megatron import (
    _unwrap_training_model,
    validate_phase3_handoff_key_sets,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    ObjectiveOutput,
)
from training import constants_uniss as c


def test_recovery_continuation_uses_explicit_variable_positions() -> None:
    hidden = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    labels = torch.zeros(10, dtype=torch.long)
    batch = {
        "event_rollout_recovery": True,
        "original_seq_length": torch.tensor(5),
        "continuation_batch": torch.tensor([0, 1]),
        "continuation_position": torch.tensor([3, 2]),
        "continuation_target": torch.tensor([0, 1]),
    }
    selected, targets = continuation_positions_and_targets(hidden, labels, batch)
    assert torch.equal(selected, hidden[torch.tensor([3, 7])])
    assert torch.equal(targets, torch.tensor([0, 1]))


def test_clean_continuation_uses_start_glm_and_eos_labels() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    labels = torch.tensor([1, c.TOKEN_START_GLM, 2, c.TOKEN_EOS, 3, 4])
    selected, targets = continuation_positions_and_targets(hidden, labels, {})
    assert torch.equal(selected, hidden[torch.tensor([1, 3])])
    assert torch.equal(targets, torch.tensor([0, 1]))


def test_distributed_objective_schema_is_fixed() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    terms = OrderedDict(
        (name, LossTerm(parameter * 0.0 + index + 1, torch.tensor(1.0)))
        for index, name in enumerate(ROLLOUT_TERM_NAMES)
    )
    diagnostics = OrderedDict(
        (name, torch.tensor(float(index)))
        for index, name in enumerate(ROLLOUT_DIAGNOSTIC_NAMES)
    )
    total, metrics = distributed_event_rollout_objective(
        ObjectiveOutput(terms, diagnostics), progress=0.5
    )
    assert total.requires_grad
    assert tuple(metrics) == ROLLOUT_METRIC_NAMES
    assert 0.0 < float(metrics["curriculum_event_rollout_fraction"]) < 0.4


def test_event_objective_registers_continuation_head() -> None:
    objective = EventRolloutJointObjective(
        hidden_size=8,
        codebook_weight=torch.zeros(16_384, 1_280),
        adapter_layers=1,
        adapter_expansion=1,
    )
    assert tuple(objective.continuation_head.weight.shape) == (2, 8)
    assert all(
        getattr(parameter, "uniss_lr_new_heads", False)
        for parameter in objective.continuation_head.parameters()
    )


def test_phase3_handoff_allows_only_isolated_new_modules() -> None:
    result = validate_phase3_handoff_key_sets(
        {
            "embedding.word_embeddings.weight",
            "decoder.layers.self_attention.linear_qkv.weight",
            "decoder.final_layernorm._extra_state/shard_0_1",
            "optimizer.state.exp_avg.embedding.word_embeddings.weight",
        },
        {
            "embedding.word_embeddings.weight",
            "decoder.layers.self_attention.linear_qkv.weight",
            "decoder.final_layernorm._extra_state",
            "true_subsecond_lora.branches.layer.lora_a",
            "true_subsecond_objective.continuation_head.weight",
        },
    )
    assert result["native_checkpoint_keys"] == 3
    assert result["allowed_new_keys"] == 2


def test_phase3_handoff_rejects_missing_native_or_foreign_keys() -> None:
    with pytest.raises(RuntimeError, match="missing_native"):
        validate_phase3_handoff_key_sets(
            {"embedding.word_embeddings.weight", "decoder.final_layernorm.weight"},
            {"embedding.word_embeddings.weight"},
        )
    with pytest.raises(RuntimeError, match="illegal_new"):
        validate_phase3_handoff_key_sets(
            {"embedding.word_embeddings.weight"},
            {
                "embedding.word_embeddings.weight",
                "foreign_module.weight",
                "true_subsecond_objective.action_head.weight",
            },
        )


def test_sharded_checkpoint_layer_keys_compare_in_canonical_namespace() -> None:
    class Shard:
        def __init__(self, key):
            self.key = key

    current = {
        "decoder.layers.0.self_attention.linear_qkv.weight": Shard(
            "decoder.layers.self_attention.linear_qkv.weight"
        ),
        "true_subsecond_objective.action_head.weight": Shard(
            "true_subsecond_objective.action_head.weight"
        ),
    }
    canonical = {getattr(value, "key", key) for key, value in current.items()}
    result = validate_phase3_handoff_key_sets(
        {"decoder.layers.self_attention.linear_qkv.weight"}, canonical
    )
    assert result["native_checkpoint_keys"] == 1


def test_unwrap_training_model_accepts_single_local_chunk(monkeypatch) -> None:
    marker = object()
    monkeypatch.setattr(
        "megatron.core.utils.unwrap_model", lambda model: [marker]
    )
    assert _unwrap_training_model(object()) is marker


def test_unwrap_training_model_rejects_multiple_local_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        "megatron.core.utils.unwrap_model", lambda model: [object(), object()]
    )
    with pytest.raises(ValueError, match="exactly one local model chunk"):
        _unwrap_training_model(object())
