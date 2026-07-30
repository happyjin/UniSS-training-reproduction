from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v1.stage_c import (
    BayesianSourceSafeCommitGate,
    StageCGateConfig,
    stage_c_losses,
)


class StageCBayesianGateTest(unittest.TestCase):
    def test_posterior_is_prior_times_likelihood_ratio_in_log_space(self) -> None:
        gate = BayesianSourceSafeCommitGate(StageCGateConfig())
        context = torch.rand(5, 4)
        evidence = torch.rand(5, 8)
        output = gate(context, evidence)
        expected = output["prior_logit"] + (
            output["class_log_likelihood"][:, 1]
            - output["class_log_likelihood"][:, 0]
        )
        torch.testing.assert_close(output["posterior_logit"], expected)

    def test_loss_updates_gate_only(self) -> None:
        gate = BayesianSourceSafeCommitGate(StageCGateConfig())
        context = torch.rand(8, 4)
        evidence = torch.rand(8, 8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], dtype=torch.float32)
        loss = stage_c_losses(gate, context, evidence, labels)["total"]
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in gate.parameters()))


if __name__ == "__main__":
    unittest.main()
