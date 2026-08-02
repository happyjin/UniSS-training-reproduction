from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v2.train_stage_b_v2 import codebook_ce_margin


class StageBV2LossTest(unittest.TestCase):
    def test_codebook_ce_and_margin_reward_correct_cell(self) -> None:
        codebook = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        good = torch.tensor([[[1.9, 0.1]]], requires_grad=True)
        bad = torch.tensor([[[0.1, 1.9]]], requires_grad=True)
        target = torch.tensor([[1]])
        mask = torch.tensor([[True]])
        good_ce, good_margin, good_exact, good_top5 = codebook_ce_margin(
            good,
            target,
            mask,
            codebook,
            temperature=0.1,
            margin=0.1,
            chunk_size=2,
        )
        bad_ce, bad_margin, bad_exact, _ = codebook_ce_margin(
            bad,
            target,
            mask,
            codebook,
            temperature=0.1,
            margin=0.1,
            chunk_size=2,
        )
        self.assertLess(float(good_ce), float(bad_ce))
        self.assertLess(float(good_margin), float(bad_margin))
        self.assertEqual(float(good_exact), 1.0)
        self.assertEqual(float(bad_exact), 0.0)
        self.assertEqual(float(good_top5), 1.0)
        (good_ce + good_margin).backward()
        self.assertIsNotNone(good.grad)


if __name__ == "__main__":
    unittest.main()
