"""Packed-epoch geometry shared by preprocessing and launchers."""

from __future__ import annotations

import math
from dataclasses import dataclass


CURRICULUM_PHASES = (
    (0.083, 0.45),
    (0.333, 0.40),
    (0.750, 0.35),
    (1.000, 0.40),
)


def curriculum_group_counts(group_count: int) -> tuple[int, int]:
    """Return deterministic replay/trajectory group counts for one schedule."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    boundaries = tuple(min(group_count, round(end * group_count)) for end, _ in CURRICULUM_PHASES)
    replay = 0
    start = 0
    for (_, replay_fraction), end in zip(CURRICULUM_PHASES, boundaries):
        replay += int(max(0, end - start) * replay_fraction)
        start = end
    return replay, group_count - replay


@dataclass(frozen=True)
class PackedEpochGeometry:
    packed_count: int
    global_batch_size: int = 128

    def __post_init__(self) -> None:
        if self.packed_count <= 0 or self.global_batch_size <= 0:
            raise ValueError("packed_count and global_batch_size must be positive")

    @property
    def train_iters(self) -> int:
        return math.ceil(self.packed_count / self.global_batch_size)

    @property
    def padded_positions(self) -> int:
        return self.train_iters * self.global_batch_size - self.packed_count

    @property
    def warmup_iters(self) -> int:
        return min(1000, max(200, math.ceil(0.025 * self.train_iters)))

    def iteration_for_progress(self, fraction: float) -> int:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0,1]")
        return min(self.train_iters, math.ceil(self.train_iters * fraction))

    @property
    def curriculum_boundaries(self) -> tuple[int, int, int, int]:
        return (
            self.iteration_for_progress(0.083),
            self.iteration_for_progress(0.333),
            self.iteration_for_progress(0.75),
            self.train_iters,
        )


@dataclass(frozen=True)
class JointPackedEpochGeometry:
    """One exact replay+trajectory epoch with homogeneous DP microbatches.

    Replay and trajectory records cannot share one rank-local microbatch because
    their collation sidecars differ.  Each source therefore needs padding to a
    complete DP microbatch before the combined schedule is padded to a complete
    optimizer global batch.  The padding is deterministic and never hides an
    unconsumed source record.
    """

    replay_count: int
    trajectory_count: int
    data_parallel_microbatch: int
    global_batch_size: int = 128

    def __post_init__(self) -> None:
        values = (
            self.replay_count,
            self.trajectory_count,
            self.data_parallel_microbatch,
            self.global_batch_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError("joint packed-epoch geometry must be positive")
        if self.global_batch_size % self.data_parallel_microbatch:
            raise ValueError("global batch must be divisible by the DP microbatch")

    def _grouped(self, count: int) -> int:
        group = self.data_parallel_microbatch
        return math.ceil(count / group) * group

    @property
    def required_replay_groups(self) -> int:
        return math.ceil(self.replay_count / self.data_parallel_microbatch)

    @property
    def required_trajectory_groups(self) -> int:
        return math.ceil(self.trajectory_count / self.data_parallel_microbatch)

    @property
    def minimum_curriculum_groups(self) -> int:
        """Smallest schedule that meets curriculum ratios and full coverage."""

        required_total = self.required_replay_groups + self.required_trajectory_groups

        def covers(value: int) -> bool:
            replay, trajectory = curriculum_group_counts(value)
            return replay >= self.required_replay_groups and trajectory >= self.required_trajectory_groups

        high = max(1, required_total)
        while not covers(high):
            high *= 2
        low = required_total
        while low < high:
            middle = (low + high) // 2
            if covers(middle):
                high = middle
            else:
                low = middle + 1
        return low

    @property
    def groups_per_global_batch(self) -> int:
        return self.global_batch_size // self.data_parallel_microbatch

    @property
    def schedule_groups(self) -> int:
        width = self.groups_per_global_batch
        return math.ceil(self.minimum_curriculum_groups / width) * width

    @property
    def replay_scheduled(self) -> int:
        replay, _ = curriculum_group_counts(self.schedule_groups)
        return replay * self.data_parallel_microbatch

    @property
    def trajectory_scheduled(self) -> int:
        _, trajectory = curriculum_group_counts(self.schedule_groups)
        return trajectory * self.data_parallel_microbatch

    @property
    def minimum_schedule_count(self) -> int:
        return self.minimum_curriculum_groups * self.data_parallel_microbatch

    @property
    def train_iters(self) -> int:
        return self.schedule_groups // self.groups_per_global_batch

    @property
    def schedule_count(self) -> int:
        return self.schedule_groups * self.data_parallel_microbatch

    @property
    def replay_padding(self) -> int:
        return self.replay_scheduled - self.replay_count

    @property
    def trajectory_padding(self) -> int:
        return self.trajectory_scheduled - self.trajectory_count

    @property
    def global_batch_padding(self) -> int:
        return self.schedule_count - self.minimum_schedule_count

    @property
    def warmup_iters(self) -> int:
        return min(1000, max(200, math.ceil(0.025 * self.train_iters)))
