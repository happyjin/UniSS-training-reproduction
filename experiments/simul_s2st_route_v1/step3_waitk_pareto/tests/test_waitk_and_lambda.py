from __future__ import annotations

import unittest

import torch

from experiments.simul_s2st_route_v1.step3_waitk_pareto.lambda_kv_cache import (
    cache_length,
    prune_to_lambda,
)
from experiments.simul_s2st_route_v1.step3_waitk_pareto.waitk_policy import (
    StabilityWaitKPolicy,
)


class WaitKTests(unittest.TestCase):
    def test_writes_after_k_stable_tokens(self) -> None:
        policy = StabilityWaitKPolicy(k=3, threshold=0.5)
        self.assertEqual(policy.observe([0.1, 0.2]).action, "WAIT")
        self.assertEqual(policy.observe([0.1, 0.2, 0.9, 0.8]).action, "WAIT")
        decision = policy.observe([0.1, 0.2, 0.9, 0.8, 0.7])
        self.assertEqual(decision.action, "WRITE")
        self.assertEqual(decision.stable_count, 3)


class LambdaCacheTests(unittest.TestCase):
    def test_prunes_to_system_plus_window(self) -> None:
        layers = tuple(
            (
                torch.randn(1, 2, 20, 4),
                torch.randn(1, 2, 20, 4),
            )
            for _ in range(2)
        )
        pruned = prune_to_lambda(layers, system_tokens=4, window=6)
        self.assertEqual(cache_length(pruned), 10)
        # Prefix preserved.
        self.assertTrue(torch.equal(pruned[0][0][..., :4, :], layers[0][0][..., :4, :]))
        self.assertTrue(torch.equal(pruned[0][0][..., 4:, :], layers[0][0][..., -6:, :]))


if __name__ == "__main__":
    unittest.main()
