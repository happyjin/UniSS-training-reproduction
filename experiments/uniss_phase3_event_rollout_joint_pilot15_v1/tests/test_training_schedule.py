from __future__ import annotations

from torch.utils.data import Dataset

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    CoverageEpochSampler,
    ThreeEpochGlobalShuffleSchedule,
)


class _Toy(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return index


def _schedule(seed: int = 20260812):
    return ThreeEpochGlobalShuffleSchedule(
        _Toy(37),
        _Toy(21),
        coverage_epochs=2,
        data_parallel_group_size=16,
        global_batch_size=128,
        shuffle_seed=seed,
        target_replay_fraction=0.35,
    )


def test_every_epoch_covers_global_pack_ids_before_tail_repetition() -> None:
    schedule = _schedule()
    for epoch in range(2):
        start = epoch * schedule.epoch_samples
        trajectory = [
            schedule.scheduled_index(start + index).source_index
            for index in range(schedule.epoch_samples)
            if schedule.scheduled_index(start + index).sample_kind == "trajectory"
        ]
        first_position = {value: trajectory.index(value) for value in set(trajectory)}
        assert set(first_position) == set(range(37))
        first_repeat = next(
            (index for index, value in enumerate(trajectory) if trajectory.index(value) != index),
            len(trajectory),
        )
        assert first_repeat >= 37


def test_epochs_are_independent_and_restart_exact() -> None:
    first = _schedule()
    repeated = _schedule()
    signatures = []
    for epoch in range(2):
        start = epoch * first.epoch_samples
        signature = [first.scheduled_index(start + index) for index in range(64)]
        signatures.append(signature)
        assert signature == [repeated.scheduled_index(start + index) for index in range(64)]
    assert signatures[0] != signatures[1]


def test_megatron_resume_consumes_the_exact_next_dp_lane() -> None:
    schedule = _schedule()
    sampler = CoverageEpochSampler(
        schedule,
        total_samples=len(schedule),
        consumed_samples=128,
        micro_batch_size=2,
        data_parallel_rank=3,
        data_parallel_size=8,
        data_sharding=False,
    )
    assert next(iter(sampler)) == [134, 135]

