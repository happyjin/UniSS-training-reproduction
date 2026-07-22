from __future__ import annotations

import unittest

import torch

from training.simul_uniss.prefix_reencode_baseline import evaluate_prefixes
from training.simul_uniss.reconstruct_unist_audio import safe_name


class PrefixBaselineTests(unittest.TestCase):
    def test_safe_name(self) -> None:
        self.assertEqual(safe_name("a/b c"), "a_b_c")

    def test_prefix_metrics_detect_revision(self) -> None:
        candidates = iter([[1, 2, 3], [1, 2, 4, 5], [1, 2, 4, 5, 6]])

        def tokenize(_prefix, _sample_rate):
            return next(candidates)

        waveform = torch.zeros(1, 3 * 160)
        metrics = evaluate_prefixes(waveform, 1000, tokenize, chunk_ms=160, holdback_tokens=0)
        self.assertGreater(metrics["prefix_revision_rate"], 0.0)
        self.assertEqual(metrics["committed_rollback_events"], 0)
        self.assertEqual(metrics["committed_length"], 5)


if __name__ == "__main__":
    unittest.main()
