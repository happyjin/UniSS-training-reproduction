from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.packed_epoch import (
    JointPackedEpochGeometry,
    PackedEpochGeometry,
    curriculum_group_counts,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.curriculum import (
    point_for_iteration,
    point_for_progress,
)


class CurriculumEpochTest(unittest.TestCase):
    def test_phase3_v4_is_one_packed_epoch(self) -> None:
        geometry = PackedEpochGeometry(1_161_587, 128)
        self.assertEqual(geometry.train_iters, 9_075)
        self.assertEqual(geometry.padded_positions, 13)
        self.assertEqual(geometry.warmup_iters, 227)

    def test_joint_geometry_accounts_for_homogeneous_dp_padding(self) -> None:
        geometry = JointPackedEpochGeometry(
            replay_count=23,
            trajectory_count=31,
            data_parallel_microbatch=16,
            global_batch_size=128,
        )
        replay_groups, trajectory_groups = curriculum_group_counts(geometry.schedule_groups)
        self.assertGreaterEqual(replay_groups, 2)
        self.assertGreaterEqual(trajectory_groups, 2)
        self.assertEqual(geometry.schedule_count % 128, 0)
        self.assertEqual(geometry.train_iters * 128, geometry.schedule_count)
        self.assertGreaterEqual(geometry.replay_scheduled, 23)
        self.assertGreaterEqual(geometry.trajectory_scheduled, 31)
        self.assertEqual(geometry.warmup_iters, 200)

    def test_replay_heavy_sources_repeat_trajectory_to_match_curriculum(self) -> None:
        geometry = JointPackedEpochGeometry(
            replay_count=1_161_587,
            trajectory_count=180_000,
            data_parallel_microbatch=16,
            global_batch_size=128,
        )
        self.assertGreater(geometry.trajectory_scheduled, geometry.trajectory_count)
        fraction = geometry.replay_scheduled / geometry.schedule_count
        self.assertGreater(fraction, 0.37)
        self.assertLess(fraction, 0.40)

    def test_curriculum_keeps_both_task_families(self) -> None:
        for progress in (0.0, 0.083, 0.333, 0.75, 1.0):
            point = point_for_progress(progress)
            self.assertGreater(point.replay_fraction, 0.0)
            self.assertGreater(point.trajectory_fraction, 0.0)

    def test_iteration_schedule_scales_with_epoch(self) -> None:
        early = point_for_iteration(100, 10_000)
        late = point_for_iteration(9_000, 10_000)
        self.assertLess(early.deadline_weight, late.deadline_weight)
        self.assertGreater(early.frontend_lr_multiplier, 0.25)
        self.assertEqual(late.frontend_lr_multiplier, 0.5)


if __name__ == "__main__":
    unittest.main()
