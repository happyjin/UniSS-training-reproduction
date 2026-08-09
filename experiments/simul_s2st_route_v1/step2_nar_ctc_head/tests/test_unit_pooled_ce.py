from __future__ import annotations

import unittest

import torch

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.pretrain_nar_ctc_megatron import (
    unit_pooled_ce,
)


class UnitPooledCeTests(unittest.TestCase):
    def test_perfect_pool_near_zero(self) -> None:
        # 4 frames, 2 units: pool halves → unit0 then unit1.
        logits = torch.full((1, 4, 4), -20.0)  # blank_id=3
        logits[0, 0:2, 1] = 20.0
        logits[0, 2:4, 2] = 20.0
        units = torch.tensor([[1, 2]])
        loss = unit_pooled_ce(
            logits,
            units,
            frame_lengths=torch.tensor([4]),
            unit_lengths=torch.tensor([2]),
            blank_id=3,
        )
        self.assertLess(float(loss), 1e-3)

    def test_ignores_empty(self) -> None:
        logits = torch.zeros(1, 3, 5)
        loss = unit_pooled_ce(
            logits,
            torch.zeros(1, 1, dtype=torch.long),
            frame_lengths=torch.tensor([0]),
            unit_lengths=torch.tensor([0]),
            blank_id=4,
        )
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
