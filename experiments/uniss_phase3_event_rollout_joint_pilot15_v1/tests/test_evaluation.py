from __future__ import annotations

import torch

from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.evaluation.model_loader import (
    validate_objective_state,
)


def _objective() -> EventRolloutJointObjective:
    return EventRolloutJointObjective(
        hidden_size=8,
        codebook_weight=torch.randn(16, 1280),
        adapter_layers=1,
        adapter_kernel_size=3,
        adapter_expansion=1,
        adapter_dropout=0.0,
        kd_temperature=1.5,
        action_write_weight=1.0,
        safe_positive_alpha=0.5,
    )


def test_event_rollout_objective_export_contract() -> None:
    source = _objective()
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name != "codebook.weight"
    }
    audit = validate_objective_state(_objective(), state)
    assert audit["continuation_head_shape"] == [2, 8]
    assert audit["microblock_size"] == 4


def test_event_rollout_objective_export_rejects_missing_runtime_head() -> None:
    source = _objective()
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name not in {"codebook.weight", "continuation_head.weight"}
    }
    try:
        validate_objective_state(_objective(), state)
    except ValueError as exc:
        assert "state mismatch" in str(exc)
    else:
        raise AssertionError("missing continuation head was accepted")

