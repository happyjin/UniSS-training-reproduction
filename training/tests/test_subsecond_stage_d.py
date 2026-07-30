from __future__ import annotations

import unittest

from training.simul_uniss.subsecond_v1.prepare_stage_d import micro_split_schedule


class StageDMicroWriteTest(unittest.TestCase):
    def test_semantic_is_preserved_and_bounded(self) -> None:
        semantic = list(range(37))
        schedule = {
            "id": "x",
            "events": [
                {
                    "action": "write",
                    "source_glm": [1, 2],
                    "source_is_final": True,
                    "target_text_ids": [10, 11, 12, 13],
                    "target_semantic": semantic,
                    "target_semantic_start": 5,
                }
            ],
        }
        result = micro_split_schedule(schedule)
        events = result["events"]
        self.assertEqual([value for event in events for value in event["target_semantic"]], semantic)
        self.assertTrue(all(8 <= len(event["target_semantic"]) <= 16 for event in events))
        self.assertEqual(events[0]["source_glm"], [1, 2])
        self.assertTrue(all(not event["source_glm"] for event in events[1:]))
        self.assertTrue(events[-1]["source_is_final"])
        self.assertTrue(all(not event["source_is_final"] for event in events[:-1]))


if __name__ == "__main__":
    unittest.main()
