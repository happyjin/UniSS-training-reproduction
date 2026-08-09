from __future__ import annotations

import unittest

import torch

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.pretrain_nar_ctc_megatron import (
    guided_duration_ce,
)


class GuidedDurationCeTests(unittest.TestCase):
    def test_perfect_logits_near_zero(self) -> None:
        # 4 frames, 2 units -> frames map to [u0,u0,u1,u1]
        logits = torch.full((1, 4, 5), -20.0)
        logits[0, 0, 1] = 20.0
        logits[0, 1, 1] = 20.0
        logits[0, 2, 2] = 20.0
        logits[0, 3, 2] = 20.0
        units = torch.tensor([[1, 2, 0, 0]])
        loss = guided_duration_ce(
            logits,
            units,
            frame_lengths=torch.tensor([4]),
            unit_lengths=torch.tensor([2]),
        )
        self.assertLess(float(loss), 1e-3)

    def test_ignores_padded_frames(self) -> None:
        logits = torch.zeros(1, 6, 4)
        logits[0, 5, 3] = 50.0  # padded junk
        units = torch.tensor([[1, 2]])
        loss_a = guided_duration_ce(
            logits,
            units,
            frame_lengths=torch.tensor([4]),
            unit_lengths=torch.tensor([2]),
        )
        logits_b = logits.clone()
        logits_b[0, 5, 3] = -50.0
        loss_b = guided_duration_ce(
            logits_b,
            units,
            frame_lengths=torch.tensor([4]),
            unit_lengths=torch.tensor([2]),
        )
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)


if __name__ == "__main__":
    unittest.main()
