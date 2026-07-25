from __future__ import annotations

import unittest

import torch
from torch import nn

from training.simul_uniss.distributed import DistributedContext


class SimulDistributedTests(unittest.TestCase):
    def test_single_process_context_is_noop(self) -> None:
        context = DistributedContext.initialize("cpu")
        self.assertFalse(context.enabled)
        self.assertTrue(context.is_main)
        self.assertEqual(context.world_size, 1)
        model = nn.Linear(2, 2)
        self.assertIs(context.wrap(model), model)
        self.assertIs(context.unwrap(model), model)
        self.assertEqual(context.reduce_sums([1.0, 2.0]), [1.0, 2.0])
        context.close()

    def test_reduce_sums_preserves_float_values(self) -> None:
        context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))
        self.assertEqual(context.reduce_sums([0.5, 3.25]), [0.5, 3.25])


if __name__ == "__main__":
    unittest.main()
