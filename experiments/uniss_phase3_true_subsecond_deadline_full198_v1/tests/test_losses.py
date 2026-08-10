from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    deadline_survival_loss,
    focal_binary_loss,
    grouped_deadline_survival_term,
    restricted_symmetric_topk_term,
    symmetric_topk_kl,
    token_cross_entropy_term,
    topk_teacher_kl_term,
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

    def test_normalized_token_ce_respects_role_weights(self) -> None:
        logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
        labels = torch.tensor([[0, 0]])
        first = token_cross_entropy_term(logits, labels, torch.tensor([[1.0, 0.0]]))
        second = token_cross_entropy_term(logits, labels, torch.tensor([[0.0, 1.0]]))
        self.assertLess(float(first.mean), float(second.mean))

    def test_cached_topk_kl_and_symmetric_stability(self) -> None:
        logits = torch.tensor([[[5.0, 1.0, -2.0]]])
        indices = torch.tensor([[[0, 1]]])
        probabilities = torch.tensor([[[0.98, 0.02]]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        aligned = topk_teacher_kl_term(logits, indices, probabilities, mask)
        symmetric = restricted_symmetric_topk_term(logits, indices, probabilities, mask)
        self.assertTrue(torch.isfinite(aligned.mean))
        self.assertLess(float(symmetric.mean), 0.05)

    def test_grouped_deadline_uses_multiple_ticks(self) -> None:
        metadata = {
            "sample_group": torch.tensor([7, 7]),
            "chunk_end_ms": torch.tensor([320, 640]),
            "soft_deadline_ms": torch.tensor([640, 640]),
            "hard_deadline_ms": torch.tensor([800, 800]),
        }
        late = grouped_deadline_survival_term(
            torch.tensor([[-6.0, 6.0], [6.0, -6.0]]), **metadata
        )
        never = grouped_deadline_survival_term(
            torch.tensor([[6.0, -6.0], [6.0, -6.0]]), **metadata
        )
        self.assertLess(float(late.mean), float(never.mean))


if __name__ == "__main__":
    unittest.main()
