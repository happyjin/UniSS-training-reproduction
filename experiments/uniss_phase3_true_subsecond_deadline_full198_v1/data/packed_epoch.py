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
