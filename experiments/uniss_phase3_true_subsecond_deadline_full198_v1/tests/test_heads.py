from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model import (
    ActionHead,
    SafeCommitHead,
    SupportOrdinalHead,
)


class HeadTest(unittest.TestCase):
    def test_shapes(self) -> None:
        source = torch.randn(4, 32)
        target = torch.randn(4, 9, 32)
        self.assertEqual(SupportOrdinalHead(32)(source).shape, (4, 5))
        self.assertEqual(ActionHead(32)(source).shape, (4, 2))
        self.assertEqual(SafeCommitHead(32)(source, target).shape, (4, 9))


if __name__ == "__main__":
    unittest.main()
