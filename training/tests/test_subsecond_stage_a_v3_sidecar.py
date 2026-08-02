from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v2.prepare_stage_a_v3_sidecar import (
    nearest_codebook_topk,
    partition_range,
)


class StageAV3SidecarTest(unittest.TestCase):
    def test_partition_range_is_contiguous_and_complete(self) -> None:
        parts = [
            partition_range(
                103,
                start_index=3,
                limit_records=97,
                rank=rank,
                world_size=8,
            )
            for rank in range(8)
        ]
        self.assertEqual(parts[0][0], 3)
        self.assertEqual(parts[-1][1], 100)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(parts, parts[1:])))

    def test_topk_codebook_keeps_nearest_target_first(self) -> None:
        codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        hidden = torch.tensor([[0.9, 0.0], [2.8, 0.0]])
        ids, distances = nearest_codebook_topk(hidden, codebook, topk=2)
        self.assertEqual(ids[:, 0].tolist(), [1, 2])
        self.assertTrue(bool((distances[:, 0] <= distances[:, 1]).all()))


if __name__ == "__main__":
    unittest.main()
