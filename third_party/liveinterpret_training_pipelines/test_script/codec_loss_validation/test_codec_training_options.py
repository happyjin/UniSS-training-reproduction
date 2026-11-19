#!/usr/bin/env python3
"""CPU unit tests for the isolated codec loss proposal."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "custom" / "qwen_omnispeak"))

from codec_training_options import (
    CodecTrainingOptions,
    corrupt_codec_history,
    eos_training_metrics,
    weighted_codec_cross_entropy,
)
from modeling_qwen_omnispeak import (  # noqa: E402
    _corrupt_codec_history as production_corrupt_codec_history,
    _weighted_codec_cross_entropy as production_weighted_codec_cross_entropy,
)


class CodecTrainingOptionsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.eos_id = 5
        self.labels = torch.tensor([[-100, 1, 2, 3, self.eos_id]])
        self.logits = torch.randn(1, 5, 6, dtype=torch.float64)

    def _gradient(self, weight: float) -> torch.Tensor:
        logits = self.logits.clone().requires_grad_(True)
        loss = weighted_codec_cross_entropy(
            logits, self.labels, eos_id=self.eos_id, eos_weight=weight)
        return torch.autograd.grad(loss, logits)[0]

    def test_default_config_preserves_current_behavior(self) -> None:
        options = CodecTrainingOptions()
        self.assertEqual(options.codec_eos_loss_weight, 1.0)
        self.assertEqual(options.codec_history_corruption_prob, 0.0)

        proposed = weighted_codec_cross_entropy(
            self.logits, self.labels, eos_id=self.eos_id, eos_weight=1.0)
        current = F.cross_entropy(
            self.logits.reshape(-1, self.logits.size(-1)),
            self.labels.reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(proposed, current, rtol=0, atol=0)

    def test_eos_weight_scales_only_eos_gradient(self) -> None:
        baseline = self._gradient(1.0)
        weighted = self._gradient(10.0)
        eos_position = self.labels.eq(self.eos_id)
        regular_positions = self.labels.ne(-100) & self.labels.ne(self.eos_id)
        torch.testing.assert_close(weighted[eos_position], baseline[eos_position] * 10)
        torch.testing.assert_close(weighted[regular_positions], baseline[regular_positions])

    def test_production_loss_matches_validated_prototype(self) -> None:
        expected = weighted_codec_cross_entropy(
            self.logits, self.labels, eos_id=self.eos_id, eos_weight=10.0)
        actual = production_weighted_codec_cross_entropy(
            self.logits, self.labels, eos_id=self.eos_id, eos_weight=10.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_corruption_is_exact_noop(self) -> None:
        ids = torch.tensor([[2149, 2151, 0, 1, 2]])
        corrupted, mask = corrupt_codec_history(
            ids, self.labels, probability=0.0, acoustic_vocab_size=2048)
        torch.testing.assert_close(corrupted, ids)
        self.assertFalse(bool(mask.any()))

    def test_corruption_changes_only_eligible_inputs_and_not_labels(self) -> None:
        ids = torch.tensor([[2149, 2151, 0, 1, 2]])
        labels_before = self.labels.clone()
        corrupted, mask = corrupt_codec_history(
            ids, self.labels, probability=1.0, acoustic_vocab_size=2048, seed=3)
        expected_mask = torch.tensor([[False, False, True, True, True]])
        torch.testing.assert_close(mask, expected_mask)
        self.assertTrue(bool(corrupted[mask].ne(ids[mask]).all()))
        torch.testing.assert_close(corrupted[~mask], ids[~mask])
        torch.testing.assert_close(self.labels, labels_before)

    def test_five_percent_corruption_converges_on_large_history(self) -> None:
        ids = torch.arange(100_000).remainder(2048).unsqueeze(0)
        labels = ids.clone()
        _, mask = corrupt_codec_history(
            ids, labels, probability=0.05, acoustic_vocab_size=2048, seed=17)
        self.assertAlmostEqual(float(mask.float().mean().item()), 0.05, delta=0.002)

    def test_production_corruption_preserves_prefix_and_labels(self) -> None:
        ids = torch.tensor([[2149, 2151, 0, 1, 2]])
        labels_before = self.labels.clone()
        torch.manual_seed(3)
        corrupted = production_corrupt_codec_history(
            ids,
            self.labels,
            probability=1.0,
            acoustic_vocab_size=2048,
        )
        torch.testing.assert_close(corrupted[:, :2], ids[:, :2])
        self.assertTrue(bool(corrupted[:, 2:].ne(ids[:, 2:]).all()))
        torch.testing.assert_close(self.labels, labels_before)

    def test_two_metrics(self) -> None:
        labels = torch.tensor([[1, 2, self.eos_id], [1, 2, self.eos_id]])
        predictions = torch.tensor([[1, self.eos_id, self.eos_id], [1, 2, 0]])
        logits = torch.full((2, 3, 6), -10.0)
        logits.scatter_(-1, predictions.unsqueeze(-1), 10.0)
        metrics = eos_training_metrics(logits, labels, eos_id=self.eos_id)
        self.assertEqual(metrics["eos_boundary_top1_acc"], 0.5)
        self.assertEqual(metrics["premature_eos_top1_rate"], 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
