from __future__ import annotations

import unittest

from torch.utils.data import Dataset

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    CoverageEpochSampler,
    ThreeEpochGlobalShuffleSchedule,
)


class _Toy(Dataset):
    def __init__(self, kind: str, length: int) -> None:
        self.kind = kind
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {"sample_kind": self.kind, "index": index}


class ThreeEpochShuffleTest(unittest.TestCase):
    def _schedule(self, seed: int = 1234) -> ThreeEpochGlobalShuffleSchedule:
        return ThreeEpochGlobalShuffleSchedule(
            _Toy("trajectory", 29),
            _Toy("replay", 17),
            coverage_epochs=3,
            data_parallel_group_size=4,
            global_batch_size=16,
            shuffle_seed=seed,
            target_replay_fraction=0.35,
        )

    def test_each_epoch_covers_both_sources(self) -> None:
        schedule = self._schedule()
        for epoch in range(3):
            start = epoch * schedule.epoch_samples
            indices = [
                schedule.scheduled_index(start + offset)
                for offset in range(schedule.epoch_samples)
            ]
            trajectory = {
                value.source_index
                for value in indices
                if value.sample_kind == "trajectory"
            }
            replay = {
                value.source_index
                for value in indices
                if value.sample_kind == "replay"
            }
            self.assertEqual(trajectory, set(range(29)))
            self.assertEqual(replay, set(range(17)))

    def test_every_dp_group_is_homogeneous(self) -> None:
        schedule = self._schedule()
        for start in range(0, len(schedule), schedule.data_parallel_group_size):
            kinds = {
                schedule.scheduled_index(start + lane).sample_kind
                for lane in range(schedule.data_parallel_group_size)
            }
            self.assertEqual(len(kinds), 1)

    def test_epochs_have_independent_permutations_and_restart_is_exact(self) -> None:
        first = self._schedule()
        repeated = self._schedule()
        signatures = []
        for epoch in range(3):
            start = epoch * first.epoch_samples
            signature = [
                (
                    first.scheduled_index(start + index).sample_kind,
                    first.scheduled_index(start + index).source_index,
                )
                for index in range(min(32, first.epoch_samples))
            ]
            signatures.append(signature)
            self.assertEqual(
                signature,
                [
                    (
                        repeated.scheduled_index(start + index).sample_kind,
                        repeated.scheduled_index(start + index).source_index,
                    )
                    for index in range(min(32, repeated.epoch_samples))
                ],
            )
        self.assertEqual(len({tuple(value) for value in signatures}), 3)

    def test_seed_changes_global_order(self) -> None:
        first = self._schedule(1234)
        changed = self._schedule(1235)
        self.assertNotEqual(
            [first.scheduled_index(index) for index in range(32)],
            [changed.scheduled_index(index) for index in range(32)],
        )

    def test_shuffle_is_over_individual_pack_ids_not_contiguous_blocks(self) -> None:
        schedule = self._schedule()
        for epoch in range(3):
            start = epoch * schedule.epoch_samples
            source_ids = [
                schedule.scheduled_index(start + index).source_index
                for index in range(schedule.epoch_samples)
                if schedule.scheduled_index(start + index).sample_kind == "trajectory"
            ][: schedule.data_parallel_group_size]
            self.assertEqual(len(source_ids), schedule.data_parallel_group_size)
            self.assertNotEqual(
                source_ids,
                list(range(source_ids[0], source_ids[0] + len(source_ids))),
            )

    def test_sampler_resume_starts_at_exact_consumed_group(self) -> None:
        schedule = self._schedule()
        sampler = CoverageEpochSampler(
            schedule,
            total_samples=len(schedule),
            consumed_samples=16,
            micro_batch_size=2,
            data_parallel_rank=1,
            data_parallel_size=2,
            data_sharding=False,
        )
        first = next(iter(sampler))
        self.assertEqual(first, [18, 19])


if __name__ == "__main__":
    unittest.main()
