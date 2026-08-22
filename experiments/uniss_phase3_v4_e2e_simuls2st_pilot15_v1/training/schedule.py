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
        from megatron.core.datasets.utils import Split

        self.split = Split.train
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


class FiveFamilyValidationSchedule(Dataset):
    """Deterministic validation blocks that never mix task families."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        total_samples: int,
        global_batch_size: int,
        data_parallel_group_size: int,
        shuffle_seed: int,
    ) -> None:
        if set(datasets) != set(TASK_FAMILIES) or any(
            len(datasets[name]) <= 0 for name in TASK_FAMILIES
        ):
            raise ValueError("E2E validation requires five non-empty family datasets")
        if total_samples <= 0 or global_batch_size <= 0 or data_parallel_group_size <= 0:
            raise ValueError("invalid E2E validation geometry")
        if global_batch_size % data_parallel_group_size:
            raise ValueError("validation global batch must be divisible by the DP group")
        if total_samples % global_batch_size:
            raise ValueError("validation samples must end on a global-batch boundary")
        self.datasets = dict(datasets)
        self.total_samples = int(total_samples)
        self.global_batch_size = int(global_batch_size)
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.shuffle_seed = int(shuffle_seed)
        self.total_blocks = self.total_samples // self.global_batch_size
        self.blocks = family_blocks(self.total_blocks, seed=self.shuffle_seed)
        self.synchronize_task_family = True
        # Megatron compares this attribute with the Split enum by identity.
        # A plain string silently selects the train batch-size path for eval.
        from megatron.core.datasets.utils import Split

        self.split = Split.valid
        running = {name: 0 for name in TASK_FAMILIES}
        self._block_family_ordinals: list[int] = []
        for family in self.blocks:
            self._block_family_ordinals.append(running[family])
            running[family] += 1

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
        source = family_flat % len(self.datasets[family])
        cycle = family_flat // len(self.datasets[family])
        return FamilyScheduledIndex(global_block, family, source, cycle)

    def __getitem__(self, index: int):
        scheduled = self.scheduled_index(index)
        value = dict(self.datasets[scheduled.family][scheduled.source_index])
        if value.get("family") != scheduled.family:
            raise ValueError("scheduled E2E validation family disagrees with source")
        return value


class FiveFamilySchedulePrefix(Dataset):
    """Global-batch-aligned prefix used only by bounded smoke runs."""

    def __init__(self, dataset: Dataset, total_samples: int) -> None:
        global_batch_size = int(getattr(dataset, "global_batch_size", 0))
        if (
            total_samples <= 0
            or total_samples > len(dataset)
            or global_batch_size <= 0
            or total_samples % global_batch_size
        ):
            raise ValueError("invalid E2E smoke schedule prefix")
        self.dataset = dataset
        self.total_samples = int(total_samples)
        self.global_batch_size = global_batch_size
        self.data_parallel_group_size = int(dataset.data_parallel_group_size)
        self.total_blocks = self.total_samples // self.global_batch_size
        source_blocks = tuple(getattr(dataset, "blocks"))
        self.blocks = source_blocks[: self.total_blocks]
        self.family_block_counts = {
            family: self.blocks.count(family) for family in TASK_FAMILIES
        }
        self.synchronize_task_family = True
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset[index]


class FiveFamilyPhaseStratifiedCanary(Dataset):
    """Bounded canary spanning the formal early, mid and steady phases.

    A plain prefix of the formal schedule cannot exercise incremental MT because
    that family intentionally has zero weight during the first 10% of training.
    This view selects complete global-family blocks from all three formal phases
    while preserving the phase order and the formal within-phase family quotas.
    The underlying shuffled source indices are reused unchanged.
    """

    _PHASE_WIDTHS = (0.10, 0.25, 0.65)

    def __init__(self, dataset: Dataset, total_samples: int) -> None:
        global_batch_size = int(getattr(dataset, "global_batch_size", 0))
        source_blocks = tuple(getattr(dataset, "blocks", ()))
        if (
            total_samples <= 0
            or global_batch_size <= 0
            or total_samples % global_batch_size
            or not source_blocks
        ):
            raise ValueError("invalid phase-stratified E2E canary geometry")
        total_blocks = total_samples // global_batch_size
        if total_blocks < 10 or total_blocks > 100:
            raise ValueError(
                "phase-stratified E2E canary requires 10--100 global blocks"
            )

        phase_exact = [total_blocks * width for width in self._PHASE_WIDTHS]
        phase_counts = [math.floor(value) for value in phase_exact]
        remaining = total_blocks - sum(phase_counts)
        phase_order = sorted(
            range(len(PHASES)),
            key=lambda index: (phase_exact[index] - phase_counts[index], -index),
            reverse=True,
        )
        for index in phase_order[:remaining]:
            phase_counts[index] += 1
        if any(value <= 0 for value in phase_counts):
            raise ValueError("phase-stratified E2E canary omitted a formal phase")

        full_boundaries = [
            0,
            round(0.10 * len(source_blocks)),
            round(0.35 * len(source_blocks)),
            len(source_blocks),
        ]
        selected: list[int] = []
        phase_family_counts: list[dict[str, int]] = []
        phase_source_blocks: list[tuple[int, ...]] = []
        for phase_index, (_, _, weights) in enumerate(PHASES):
            family_counts = _largest_remainder(phase_counts[phase_index], weights)
            candidates = {
                family: [
                    block
                    for block in range(
                        full_boundaries[phase_index],
                        full_boundaries[phase_index + 1],
                    )
                    if source_blocks[block] == family
                ]
                for family in TASK_FAMILIES
            }
            phase_selected: list[int] = []
            for family in TASK_FAMILIES:
                needed = family_counts[family]
                available = candidates[family]
                if needed > len(available):
                    raise ValueError(
                        f"formal phase {phase_index} has too few {family} blocks"
                    )
                # Midpoint sampling spans the full formal phase instead of
                # collapsing each family to another short local prefix.
                phase_selected.extend(
                    available[math.floor((slot + 0.5) * len(available) / needed)]
                    for slot in range(needed)
                )
            phase_selected.sort()
            selected.extend(phase_selected)
            phase_family_counts.append(family_counts)
            phase_source_blocks.append(tuple(phase_selected))

        self.dataset = dataset
        self.total_samples = int(total_samples)
        self.global_batch_size = global_batch_size
        self.data_parallel_group_size = int(dataset.data_parallel_group_size)
        self.total_blocks = total_blocks
        self.source_block_indices = tuple(selected)
        self.blocks = tuple(source_blocks[index] for index in selected)
        self.family_block_counts = {
            family: self.blocks.count(family) for family in TASK_FAMILIES
        }
        if any(self.family_block_counts[family] <= 0 for family in TASK_FAMILIES):
            raise ValueError("phase-stratified canary must contain all five families")
        self.phase_block_counts = tuple(phase_counts)
        self.phase_family_block_counts = tuple(phase_family_counts)
        self.phase_source_block_indices = tuple(phase_source_blocks)
        self.synchronize_task_family = True
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block, lane = divmod(index, self.global_batch_size)
        source = self.source_block_indices[block] * self.global_batch_size + lane
        value = self.dataset[source]
        if value.get("family") != self.blocks[block]:
            raise ValueError("phase-stratified E2E canary family mismatch")
        return value


class FiveFamilySingleBlock(Dataset):
    """One explicit family block for an isolated one-update GPU canary."""

    def __init__(self, dataset: Dataset, family: str) -> None:
        if family not in TASK_FAMILIES:
            raise ValueError("unknown E2E canary family")
        blocks = tuple(getattr(dataset, "blocks", ()))
        if family not in blocks:
            raise ValueError("E2E canary family has no scheduled block")
        self.dataset = dataset
        self.family = family
        self.global_batch_size = int(getattr(dataset, "global_batch_size", 0))
        self.data_parallel_group_size = int(
            getattr(dataset, "data_parallel_group_size", 0)
        )
        if self.global_batch_size <= 0 or self.data_parallel_group_size <= 0:
            raise ValueError("invalid E2E canary schedule geometry")
        self.source_block = blocks.index(family)
        self.total_blocks = 1
        self.blocks = (family,)
        self.family_block_counts = {
            name: int(name == family) for name in TASK_FAMILIES
        }
        self.synchronize_task_family = True
        self.split = getattr(dataset, "split", None)

    def __len__(self) -> int:
        return self.global_batch_size

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        value = self.dataset[self.source_block * self.global_batch_size + index]
        if value.get("family") != self.family:
            raise ValueError("E2E canary block escaped its requested family")
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
    "FiveFamilyPhaseStratifiedCanary",
    "FiveFamilySchedulePrefix",
    "FiveFamilySingleBlock",
    "FiveFamilyValidationSchedule",
    "MID_WEIGHTS",
    "STEADY_WEIGHTS",
    "family_blocks",
    "required_total_blocks",
]
