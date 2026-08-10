from __future__ import annotations

import torch
import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron import (
    audit_gradient,
)


class GradientAuditTest(unittest.TestCase):
    def test_gradient_audit_accepts_finite_values(self) -> None:
        gradient = torch.tensor([0.0, -1.25, 3.5])
        self.assertIs(
            audit_gradient("weight", "uniss_lr_new_heads", gradient), gradient
        )

    def test_gradient_audit_rejects_nonfinite_values(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                FloatingPointError, "non-finite gradient for weight"
            ):
                audit_gradient(
                    "weight", "uniss_lr_new_heads", torch.tensor([1.0, value])
                )

    def test_zero_placeholder_does_not_backpropagate_through_zero_rms(self) -> None:
        value = torch.zeros(4, requires_grad=True)
        unstable_rms = value.square().mean().sqrt()
        stable_anchor = value.sum() * 0.0
        stable_anchor.backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(unstable_rms.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
