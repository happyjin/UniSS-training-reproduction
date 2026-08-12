from __future__ import annotations

import torch
from torch import nn

from experiments.uniss_phase3_event_rollout_joint_full198_v1.rollout_policy import (
    _head_input,
    rollout_schedule,
)


def test_rollout_schedule_is_one_run_curriculum() -> None:
    assert rollout_schedule(0.0).fraction == 0.0
    assert 0.049 <= rollout_schedule(0.05).fraction <= 0.051
    assert 0.29 <= rollout_schedule(0.60).fraction <= 0.31
    assert rollout_schedule(1.0).fraction == 0.4
    assert rollout_schedule(1.0).maximum_sessions == 1


def test_rollout_new_head_input_matches_parameter_dtype() -> None:
    head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2)).float()
    value = torch.ones(1, 4, dtype=torch.bfloat16)
    converted = _head_input(head, value)
    assert converted.dtype == torch.float32
    assert head(converted).dtype == torch.float32
