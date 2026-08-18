"""Restart-exact global-step schedule for the five E2E task families."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_STREAMING_ASR,
    TASK_FAMILIES,
)


EARLY_WEIGHTS = {
    FAMILY_STREAMING_ASR: 0.40,
    FAMILY_INCREMENTAL_MT: 0.00,
    FAMILY_INTERLEAVED: 0.20,
    FAMILY_PHASE3_QUALITY: 0.24,
    FAMILY_PHASE3_PERFORMANCE: 0.16,
}
MID_WEIGHTS = {
    FAMILY_STREAMING_ASR: 0.325,
    FAMILY_INCREMENTAL_MT: 0.10,
    FAMILY_INTERLEAVED: 0.25,
    FAMILY_PHASE3_QUALITY: 0.195,
    FAMILY_PHASE3_PERFORMANCE: 0.13,
}
STEADY_WEIGHTS = {
    FAMILY_STREAMING_ASR: 0.25,
    FAMILY_INCREMENTAL_MT: 0.20,
    FAMILY_INTERLEAVED: 0.30,
    FAMILY_PHASE3_QUALITY: 0.15,
    FAMILY_PHASE3_PERFORMANCE: 0.10,
}
PHASES = (
    (0.00, 0.10, EARLY_WEIGHTS),
    (0.10, 0.35, MID_WEIGHTS),
    (0.35, 1.00, STEADY_WEIGHTS),
)


def _stable_seed(seed: int, *values: object) -> int:
    digest = hashlib.blake2b(
        ":".join([str(seed), *(str(value) for value in values)]).encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _largest_remainder(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0 or set(weights) != set(TASK_FAMILIES):
        raise ValueError("invalid E2E family allocation geometry")
    weight_sum = sum(float(value) for value in weights.values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise ValueError("E2E family weights do not sum to one")
    exact = {name: total * float(weights[name]) for name in TASK_FAMILIES}
    output = {name: math.floor(exact[name]) for name in TASK_FAMILIES}
    remaining = total - sum(output.values())
    order = sorted(
        TASK_FAMILIES,
        key=lambda name: (exact[name] - output[name], -TASK_FAMILIES.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        output[name] += 1
    return output


def family_blocks(total_blocks: int, *, seed: int) -> tuple[str, ...]:
    if total_blocks <= 0:
        raise ValueError("E2E schedule must contain global blocks")
    boundaries = [0, round(0.10 * total_blocks), round(0.35 * total_blocks), total_blocks]
    blocks: list[str] = []
    for phase_index, (_, _, weights) in enumerate(PHASES):
        phase_blocks = boundaries[phase_index + 1] - boundaries[phase_index]
        counts = _largest_remainder(phase_blocks, weights)
        phase = [
            family
            for family in TASK_FAMILIES
            for _ in range(counts[family])
        ]
        if phase:
            generator = torch.Generator().manual_seed(
                _stable_seed(seed, "phase", phase_index)
            )
            order = torch.randperm(len(phase), generator=generator).tolist()
            phase = [phase[index] for index in order]
        blocks.extend(phase)
    if len(blocks) != total_blocks:
        raise AssertionError("E2E family block allocation did not close")
    return tuple(blocks)


def required_total_blocks(
    interleaved_records: int,
    *,
    global_batch_size: int,
    coverage_epochs: int,
    seed: int,
) -> int:
    if interleaved_records <= 0 or global_batch_size <= 0 or coverage_epochs <= 0:
        raise ValueError("invalid E2E coverage geometry")
    required = math.ceil(
        coverage_epochs * interleaved_records / global_batch_size
    )
    integrated = 0.10 * 0.20 + 0.25 * 0.25 + 0.65 * 0.30
    total = max(1, math.ceil(required / integrated))
    while family_blocks(total, seed=seed).count(FAMILY_INTERLEAVED) < required:
        total += 1
    return total


@dataclass(frozen=True)
class FamilyScheduledIndex:
    global_block: int
    family: str
    source_index: int
    source_cycle: int


class FiveFamilyGlobalSchedule(Dataset):
    """All samples in one optimizer global batch come from one task family."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        coverage_epochs: int,
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
    ) -> None:
        if set(datasets) != set(TASK_FAMILIES) or any(
            len(datasets[name]) <= 0 for name in TASK_FAMILIES
        ):
            raise ValueError("E2E schedule requires five non-empty family datasets")
        if global_batch_size <= 0 or data_parallel_group_size <= 0:
            raise ValueError("E2E schedule batch geometry must be positive")
        if global_batch_size % data_parallel_group_size:
            raise ValueError("global batch must be divisible by the DP group")
        self.datasets = dict(datasets)
        self.coverage_epochs = int(coverage_epochs)
        self.global_batch_size = int(global_batch_size)
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.shuffle_seed = int(shuffle_seed)
        self.total_blocks = required_total_blocks(
            len(self.datasets[FAMILY_INTERLEAVED]),
            global_batch_size=self.global_batch_size,
            coverage_epochs=self.coverage_epochs,
            seed=self.shuffle_seed,
        )
        self.blocks = family_blocks(self.total_blocks, seed=self.shuffle_seed)
        self.total_samples = self.total_blocks * self.global_batch_size
        self.synchronize_task_family = True
        self.split = "train"
        self.family_block_cursors: dict[str, list[int]] = {
            name: [] for name in TASK_FAMILIES
        }
        counts = {name: 0 for name in TASK_FAMILIES}
        for family in self.blocks:
            self.family_block_cursors[family].append(counts[family])
            counts[family] += 1
        self.family_block_counts = counts
        self._block_family_ordinals: list[int] = []
        running = {name: 0 for name in TASK_FAMILIES}
        for family in self.blocks:
            self._block_family_ordinals.append(running[family])
            running[family] += 1
        self._permutations: dict[tuple[str, int], torch.Tensor] = {}
        for family in TASK_FAMILIES:
            samples = self.family_block_counts[family] * self.global_batch_size
            cycles = math.ceil(samples / len(self.datasets[family]))
            for cycle in range(cycles):
                self._permutations[(family, cycle)] = torch.randperm(
                    len(self.datasets[family]),
                    generator=torch.Generator().manual_seed(
                        _stable_seed(self.shuffle_seed, family, cycle)
                    ),
                )

    def __len__(self) -> int:
        return self.total_samples

    def scheduled_index(self, index: int) -> FamilyScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        global_block, lane = divmod(index, self.global_batch_size)
        family = self.blocks[global_block]
        family_flat = (
            self._block_family_ordinals[global_block] * self.global_batch_size
            + lane
        )
        dataset_length = len(self.datasets[family])
        cycle, position = divmod(family_flat, dataset_length)
        source = int(self._permutations[(family, cycle)][position])
        return FamilyScheduledIndex(global_block, family, source, cycle)

    def __getitem__(self, index: int):
        scheduled = self.scheduled_index(index)
        value = dict(self.datasets[scheduled.family][scheduled.source_index])
        if value.get("family") != scheduled.family:
            raise ValueError("scheduled E2E family disagrees with source dataset")
        return value


class FiveFamilyCoverageSampler:
    """Expose the schedule without applying a second Megatron shuffle."""

    def __init__(
        self,
        dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        data_sharding: bool,
    ) -> None:
        del data_sharding
        if not getattr(dataset, "synchronize_task_family", False):
            raise ValueError("dataset does not expose synchronized task families")
        self.dataset = dataset
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        self.micro_batch_size = int(micro_batch_size)
        self.global_microbatch_size = self.data_parallel_size * self.micro_batch_size
        if self.global_microbatch_size != dataset.data_parallel_group_size:
            raise ValueError("dataset DP group differs from Megatron geometry")
        self.active_total_samples = min(int(total_samples), len(dataset))
        self.active_total_samples -= (
            self.active_total_samples % dataset.global_batch_size
        )
        self.consumed_samples = int(consumed_samples)
        if self.consumed_samples % dataset.global_batch_size:
            raise ValueError("resume must end on an optimizer global-batch boundary")

    def __len__(self) -> int:
        return self.active_total_samples

    def __iter__(self):
        start_group = self.consumed_samples // self.global_microbatch_size
        total_groups = self.active_total_samples // self.global_microbatch_size
        for group in range(start_group, total_groups):
            group_start = group * self.global_microbatch_size
            rank_start = (
                group_start + self.data_parallel_rank * self.micro_batch_size
            )
            self.consumed_samples += self.global_microbatch_size
            yield list(range(rank_start, rank_start + self.micro_batch_size))


__all__ = [
    "EARLY_WEIGHTS",
    "FamilyScheduledIndex",
    "FiveFamilyCoverageSampler",
    "FiveFamilyGlobalSchedule",
    "MID_WEIGHTS",
    "STEADY_WEIGHTS",
    "family_blocks",
    "required_total_blocks",
]
