from __future__ import annotations

from collections import Counter

import pytest
from torch.utils.data import Dataset

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.schedule import (
    EARLY_WEIGHTS,
    MID_WEIGHTS,
    STEADY_WEIGHTS,
    FiveFamilyCoverageSampler,
    FiveFamilyGlobalSchedule,
    FiveFamilyPhaseStratifiedCanary,
    FiveFamilySchedulePrefix,
    FiveFamilySingleBlock,
    FiveFamilyValidationSchedule,
    family_blocks,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INTERLEAVED,
    TASK_FAMILIES,
)


class _FamilyDataset(Dataset):
    def __init__(self, family: str, length: int) -> None:
        self.family = family
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {"family": self.family, "source_index": index}


def _datasets(interleaved: int = 100):
    return {
        family: _FamilyDataset(
            family,
            interleaved if family == FAMILY_INTERLEAVED else interleaved + 7,
        )
        for family in TASK_FAMILIES
    }


def test_schedules_expose_megatron_split_enums() -> None:
    from megatron.core.datasets.utils import Split

    datasets = _datasets(41)
    train = FiveFamilyGlobalSchedule(
        datasets,
        coverage_epochs=3,
        global_batch_size=16,
        data_parallel_group_size=8,
        shuffle_seed=19,
    )
    valid = FiveFamilyValidationSchedule(
        datasets,
        total_samples=80,
        global_batch_size=8,
        data_parallel_group_size=8,
        shuffle_seed=20,
    )
    prefix = FiveFamilySchedulePrefix(train, 16)
    assert train.split is Split.train
    assert valid.split is Split.valid
    assert prefix.split is Split.train


def test_family_block_curriculum_has_exact_phase_quotas() -> None:
    blocks = family_blocks(400, seed=1234)
    phases = (
        (blocks[:40], EARLY_WEIGHTS),
        (blocks[40:140], MID_WEIGHTS),
        (blocks[140:], STEADY_WEIGHTS),
    )
    for values, weights in phases:
        counts = Counter(values)
        for family in TASK_FAMILIES:
            assert abs(counts[family] - len(values) * weights[family]) <= 1.0


def test_five_family_schedule_synchronizes_each_global_batch_and_covers_e2e() -> None:
    schedule = FiveFamilyGlobalSchedule(
        _datasets(100),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=17,
    )
    for block in range(schedule.total_blocks):
        families = {
            schedule.scheduled_index(block * 8 + lane).family for lane in range(8)
        }
        assert families == {schedule.blocks[block]}
    interleaved = [
        schedule.scheduled_index(index).source_index
        for index in range(len(schedule))
        if schedule.scheduled_index(index).family == FAMILY_INTERLEAVED
    ]
    assert len(interleaved) >= 300
    assert all(
        len(set(interleaved[start : start + 100])) == 100
        for start in range(0, 300, 100)
    )
    value = schedule[0]
    assert value["family"] == schedule.scheduled_index(0).family


def test_coverage_sampler_preserves_rank_sync_and_requires_update_boundary() -> None:
    schedule = FiveFamilyGlobalSchedule(
        _datasets(32),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=19,
    )
    samplers = [
        FiveFamilyCoverageSampler(
            schedule,
            total_samples=len(schedule),
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=rank,
            data_parallel_size=2,
            data_sharding=True,
        )
        for rank in range(2)
    ]
    iterators = [iter(value) for value in samplers]
    for _ in range(4):
        indices = [next(value)[0] for value in iterators]
        families = {schedule.scheduled_index(index).family for index in indices}
        assert len(families) == 1
    with pytest.raises(ValueError, match="global-batch boundary"):
        FiveFamilyCoverageSampler(
            schedule,
            total_samples=len(schedule),
            consumed_samples=2,
            micro_batch_size=1,
            data_parallel_rank=0,
            data_parallel_size=2,
            data_sharding=True,
        )


def test_validation_schedule_is_exact_length_and_family_synchronized() -> None:
    schedule = FiveFamilyValidationSchedule(
        _datasets(13),
        total_samples=80,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=23,
    )
    assert len(schedule) == 80
    for block in range(10):
        values = [
            schedule.scheduled_index(block * 8 + lane) for lane in range(8)
        ]
        assert len({value.family for value in values}) == 1
        assert all(
            schedule[index]["family"] == values[index % 8].family
            for index in range(block * 8, (block + 1) * 8)
        )


def test_smoke_prefix_keeps_complete_global_family_blocks() -> None:
    full = FiveFamilyGlobalSchedule(
        _datasets(32),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=29,
    )
    prefix = FiveFamilySchedulePrefix(full, 16)
    assert len(prefix) == 16
    assert prefix.total_blocks == 2
    for block in range(2):
        assert {
            prefix[index]["family"]
            for index in range(block * 8, (block + 1) * 8)
        } == {full.blocks[block]}


def test_phase_stratified_canary_spans_phases_and_all_families() -> None:
    full = FiveFamilyGlobalSchedule(
        _datasets(100),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=29,
    )
    canary = FiveFamilyPhaseStratifiedCanary(full, 10 * 8)
    assert len(canary) == 80
    assert canary.total_blocks == 10
    assert canary.phase_block_counts == (1, 3, 6)
    assert set(canary.blocks) == set(TASK_FAMILIES)
    boundaries = (0, round(0.10 * full.total_blocks), round(0.35 * full.total_blocks), full.total_blocks)
    for phase, indices in enumerate(canary.phase_source_block_indices):
        assert indices
        assert all(boundaries[phase] <= value < boundaries[phase + 1] for value in indices)
    for block in range(canary.total_blocks):
        assert {
            canary[index]["family"]
            for index in range(block * 8, (block + 1) * 8)
        } == {canary.blocks[block]}


def test_phase_stratified_canary_rejects_non_learning_geometry() -> None:
    full = FiveFamilyGlobalSchedule(
        _datasets(100),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=31,
    )
    with pytest.raises(ValueError, match="10--100"):
        FiveFamilyPhaseStratifiedCanary(full, 9 * 8)


def test_single_family_canary_selects_one_complete_requested_block() -> None:
    full = FiveFamilyGlobalSchedule(
        _datasets(32),
        coverage_epochs=3,
        global_batch_size=8,
        data_parallel_group_size=2,
        shuffle_seed=31,
    )
    for family in TASK_FAMILIES:
        canary = FiveFamilySingleBlock(full, family)
        assert len(canary) == 8
        assert canary.blocks == (family,)
        assert {canary[index]["family"] for index in range(len(canary))} == {
            family
        }
