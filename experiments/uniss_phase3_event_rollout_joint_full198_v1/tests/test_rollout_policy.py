from __future__ import annotations

from experiments.uniss_phase3_event_rollout_joint_full198_v1.rollout_policy import (
    rollout_schedule,
)


def test_rollout_schedule_is_one_run_curriculum() -> None:
    assert rollout_schedule(0.0).fraction == 0.0
    assert 0.049 <= rollout_schedule(0.05).fraction <= 0.051
    assert 0.29 <= rollout_schedule(0.60).fraction <= 0.31
    assert rollout_schedule(1.0).fraction == 0.4
    assert rollout_schedule(1.0).maximum_sessions == 1

