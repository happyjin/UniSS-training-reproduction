from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_schedule import (
    choose_times,
    plans_for_row,
)


class ScheduleTest(unittest.TestCase):
    def test_time_choice_is_deterministic_and_bounded(self) -> None:
        self.assertEqual(choose_times("sample", 3200), choose_times("sample", 3200))
        early, later = choose_times("sample", 3200)
        self.assertIn(early, {320, 480, 640, 800})
        self.assertGreaterEqual(later, 960)
        self.assertLessEqual(later, 3200)

    def test_every_row_produces_early_and_middle_late(self) -> None:
        early, later = plans_for_row(
            2,
            7,
            {
                "id": "abc",
                "source_glm": list(range(20)),
                "source_bicodec": list(range(80)),
                "target_bicodec": list(range(64)),
                "src_lang": "eng",
                "tgt_lang": "cmn",
            },
        )
        self.assertEqual(early.trajectory_kind, "early")
        self.assertEqual(later.trajectory_kind, "middle_late")
        self.assertEqual(early.source_duration_ms, 1600)
        self.assertEqual(early.shard, 2)
        self.assertEqual(early.row_index, 7)

    def test_short_utterance_is_still_valid(self) -> None:
        early, later = plans_for_row(
            0,
            0,
            {
                "id": "short",
                "source_glm": [1],
                "source_bicodec": list(range(16)),
                "target_bicodec": list(range(16)),
                "src_lang": "cmn",
                "tgt_lang": "eng",
            },
        )
        self.assertEqual(early.chunk_end_ms, 320)
        self.assertEqual(later.chunk_end_ms, 320)


if __name__ == "__main__":
    unittest.main()
