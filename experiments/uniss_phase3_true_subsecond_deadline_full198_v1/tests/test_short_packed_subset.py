from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_short_packed_subset import (
    shorten_record,
)
from training import constants_uniss as c


class ShortPackedSubsetTest(unittest.TestCase):
    def _record(self, trajectory: bool) -> dict:
        value = {
            "tokens": list(range(12)),
            "labels": list(range(12)),
            "loss_mask": [1] * 12,
            "position_ids": list(range(12)),
            "sample_boundaries": [[0, 3], [3, 7], [7, 11]],
            "tasks": ["a", "b", "c"],
            "source_ids": ["0", "1", "2"],
        }
        if trajectory:
            value["token_roles"] = [1] * 12
            value["trajectory_sidecars"] = [{"id": index} for index in range(3)]
        return value

    def test_replay_keeps_only_complete_samples(self) -> None:
        value = shorten_record(self._record(False), kind="replay", seq_length=8)
        self.assertEqual(value["sample_boundaries"], [[0, 3], [3, 7]])
        self.assertEqual(value["tokens"], list(range(7)) + [c.TOKEN_PAD])
        self.assertEqual(value["loss_mask"], [1] * 7 + [0])
        self.assertEqual(value["source_ids"], ["0", "1"])

    def test_trajectory_sidecars_and_roles_follow_boundaries(self) -> None:
        value = shorten_record(self._record(True), kind="trajectory", seq_length=8)
        self.assertEqual(len(value["trajectory_sidecars"]), 2)
        self.assertEqual(value["token_roles"], [1] * 7 + [0])


if __name__ == "__main__":
    unittest.main()
