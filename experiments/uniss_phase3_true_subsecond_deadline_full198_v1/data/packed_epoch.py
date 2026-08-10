"""Packed-epoch geometry shared by preprocessing and launchers."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
    def replay_scheduled(self) -> int:
        return self._grouped(self.replay_count)

    @property
    def trajectory_scheduled(self) -> int:
        return self._grouped(self.trajectory_count)

    @property
    def minimum_schedule_count(self) -> int:
        return self.replay_scheduled + self.trajectory_scheduled

    @property
    def train_iters(self) -> int:
        return math.ceil(self.minimum_schedule_count / self.global_batch_size)

    @property
    def schedule_count(self) -> int:
        return self.train_iters * self.global_batch_size

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
