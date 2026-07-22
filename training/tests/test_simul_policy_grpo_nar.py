from __future__ import annotations

import unittest

import torch

from training.simul_uniss.nar_semantic import NARSemanticGenerator, nar_losses
from training.simul_uniss.policy_grpo import ActionPolicy, grpo_loss, rollout_rewards


class PolicyGrpoNarTests(unittest.TestCase):
    def test_rollout_reward_penalizes_premature_write(self) -> None:
        actions = torch.tensor([[0, 1]])
        labels = torch.tensor([0])
        features = torch.tensor([[0.2, 0.0, 0.0, 0.1, 0.0]])
        rewards = rollout_rewards(actions, labels, features)
        self.assertGreater(float(rewards[0, 0]), float(rewards[0, 1]))

    def test_grpo_loss_is_finite(self) -> None:
        torch.manual_seed(2)
        policy = ActionPolicy(16)
        reference = ActionPolicy(16)
        reference.load_state_dict(policy.state_dict())
        features = torch.rand(4, 5)
        labels = torch.tensor([0, 1, 0, 1])
        loss, metrics = grpo_loss(policy, reference, features, labels, group_size=4, kl_beta=0.02)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reward_mean", metrics)

    def test_nar_shapes_and_loss(self) -> None:
        model = NARSemanticGenerator(hidden_size=32, num_layers=1, num_heads=4, max_output_tokens=12)
        batch = {
            "text": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "text_lengths": torch.tensor([3, 2]),
            "semantic": torch.tensor([1, 2, 3, 4, 5]),
            "semantic_lengths": torch.tensor([3, 2]),
        }
        outputs = model(batch["text"], batch["text_lengths"])
        self.assertEqual(outputs["logits"].shape, (2, 12, 8193))
        losses = nar_losses(model, batch)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
