from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model import (
    ChunkCausalWhisperVQAdapter,
)


class NoFutureLeakageTest(unittest.TestCase):
    def test_future_perturbation_does_not_change_prefix(self) -> None:
        torch.manual_seed(7)
        model = ChunkCausalWhisperVQAdapter(
            hidden_size=16, layers=3, kernel_size=5, expansion=2
        ).eval()
        first = torch.randn(2, 20, 16)
        second = first.clone()
        second[:, 12:] = torch.randn_like(second[:, 12:]) * 9.0
        with torch.no_grad():
            first_output = model(first)
            second_output = model(second)
        torch.testing.assert_close(first_output[:, :12], second_output[:, :12])


if __name__ == "__main__":
    unittest.main()
