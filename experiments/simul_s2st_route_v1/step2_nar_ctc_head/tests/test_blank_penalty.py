from __future__ import annotations

import unittest

import torch

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.pretrain_nar_ctc_megatron import (
    blank_probability_penalty,
)


class BlankPenaltyTests(unittest.TestCase):
    def test_uniform_logits_near_one_over_vocab(self) -> None:
        logits = torch.zeros(2, 5, 10)
        lengths = torch.tensor([5, 3])
        value = blank_probability_penalty(logits, lengths, blank_id=9)
        self.assertAlmostEqual(float(value), 0.1, places=5)

    def test_masks_padded_frames(self) -> None:
        logits = torch.zeros(1, 4, 3)
        logits[0, 3, 2] = 50.0  # padded frame strongly blank
        lengths = torch.tensor([3])
        value = blank_probability_penalty(logits, lengths, blank_id=2)
        self.assertLess(float(value), 0.4)


if __name__ == "__main__":
    unittest.main()
