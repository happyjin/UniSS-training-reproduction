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
