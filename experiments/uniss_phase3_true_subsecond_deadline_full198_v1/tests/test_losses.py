from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    deadline_survival_loss,
    focal_binary_loss,
    symmetric_topk_kl,
)


class LossTest(unittest.TestCase):
    def test_deadline_loss_rewards_earlier_write(self) -> None:
        mask = torch.ones(1, 4, dtype=torch.bool)
        deadline = torch.tensor([[True, True, True, False]])
        low = deadline_survival_loss(torch.full((1, 4), -6.0), mask, deadline)
        high = deadline_survival_loss(torch.tensor([[6.0, -6.0, -6.0, -6.0]]), mask, deadline)
        self.assertLess(float(high), float(low))

    def test_focal_loss_respects_mask(self) -> None:
        logits = torch.tensor([[0.0, 100.0]])
        targets = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[True, False]])
        masked = focal_binary_loss(logits, targets, mask)
        reference = focal_binary_loss(logits[:, :1], targets[:, :1], mask[:, :1])
        torch.testing.assert_close(masked, reference)

    def test_symmetric_kl_is_zero_for_identical_logits(self) -> None:
        logits = torch.randn(2, 3, 7)
        mask = torch.ones(2, 3, dtype=torch.bool)
        value = symmetric_topk_kl(logits, logits.clone(), mask)
        self.assertLess(abs(float(value)), 1e-7)


if __name__ == "__main__":
    unittest.main()
