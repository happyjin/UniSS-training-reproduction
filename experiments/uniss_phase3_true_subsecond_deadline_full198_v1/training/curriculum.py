"""Progress-based single-epoch curriculum; loss modules are never replaced."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumPoint:
    replay_fraction: float
    trajectory_fraction: float
    deadline_weight: float
    frontend_lr_multiplier: float

    def __post_init__(self) -> None:
        if abs(self.replay_fraction + self.trajectory_fraction - 1.0) > 1e-8:
            raise ValueError("replay and trajectory fractions must sum to one")
        if not 0.0 <= self.deadline_weight <= 0.30:
            raise ValueError("deadline weight is outside the frozen range")
        if self.frontend_lr_multiplier <= 0.0:
            raise ValueError("frontend LR multiplier must be positive")


def point_for_progress(progress: float) -> CurriculumPoint:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0,1]")
    if progress < 0.083:
        local = progress / 0.083
        return CurriculumPoint(0.45, 0.55, 0.10 * local, 0.25 + 0.75 * local)
    if progress < 0.333:
        local = (progress - 0.083) / (0.333 - 0.083)
        return CurriculumPoint(0.40, 0.60, 0.10 + 0.20 * local, 1.0)
    if progress < 0.75:
        return CurriculumPoint(0.35, 0.65, 0.30, 1.0)
    return CurriculumPoint(0.40, 0.60, 0.30, 0.5)


def point_for_iteration(iteration: int, train_iters: int) -> CurriculumPoint:
    if train_iters <= 0 or not 0 <= iteration <= train_iters:
        raise ValueError("invalid iteration/train_iters")
    return point_for_progress(iteration / train_iters)
