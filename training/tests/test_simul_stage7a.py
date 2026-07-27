from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION
from training.simul_uniss.stage7a.data import (
    batch_action_samples,
    iter_action_samples_once,
    parse_action_sample,
)
from training.simul_uniss.stage7a.fixed_policy import fixed_wait_k_actions
from training.simul_uniss.stage7a.policy import (
    ActionHead,
    grpo_action_loss,
    rollout_rewards,
)
from training.simul_uniss.stage7a.train import learning_rate_factor


def sample_item(sample_id: str = "sample") -> dict[str, object]:
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "id": sample_id,
        "task": "simul_action",
        "input_ids": [
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_START_GLM,
            c.GLM_SEMANTIC_OFFSET,
            c.TOKEN_END_GLM,
            c.TOKEN_WAIT_READ,
            c.TOKEN_START_GLM,
            c.GLM_SEMANTIC_OFFSET + 1,
            c.TOKEN_END_GLM,
            c.TOKEN_WRITE_GENERATE,
        ],
        "token_weights": [0.0] * 9,
    }


class Stage7ADataTest(unittest.TestCase):
    def test_parse_action_positions_and_final_flag(self):
        sample = parse_action_sample(sample_item(), max_sequence_length=32)
        self.assertEqual(sample.prediction_positions, [3, 7])
        self.assertEqual(sample.labels, [0, 1])
        self.assertEqual(sample.event_fractions, [0.5, 1.0])
        self.assertEqual(sample.final_flags, [False, True])

    def test_rank_partition_and_dynamic_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            rows = [sample_item(f"sample-{index}") for index in range(5)]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            rank_one = list(
                iter_action_samples_once(
                    path, rank=1, world_size=2, max_sequence_length=32
                )
            )
            self.assertEqual(
                [sample.sample_id for sample in rank_one], ["sample-1", "sample-3"]
            )
            batches = list(
                batch_action_samples(rank_one, max_batch_tokens=32, max_batch_size=8)
            )
            self.assertEqual(len(batches), 1)
            batch = batches[0]
            self.assertEqual(batch.samples, 2)
            self.assertEqual(batch.events, 4)
            self.assertEqual(batch.actual_tokens, 18)

    def test_overlong_sample_can_be_rejected(self):
        with self.assertRaises(OverflowError):
            parse_action_sample(sample_item(), max_sequence_length=4)


class Stage7APolicyTest(unittest.TestCase):
    def test_head_initializes_from_exact_lm_rows(self):
        lm_head = nn.Linear(3, c.TOKEN_WRITE_GENERATE + 2, bias=False)
        with torch.no_grad():
            lm_head.weight[c.TOKEN_WAIT_READ].copy_(torch.tensor([1.0, 2.0, 3.0]))
            lm_head.weight[c.TOKEN_WRITE_GENERATE].copy_(torch.tensor([4.0, 5.0, 6.0]))
        head = ActionHead.from_lm_head(lm_head)
        self.assertTrue(
            torch.equal(
                head.projection.weight,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            )
        )

    def test_reward_prefers_correct_safe_trajectory(self):
        labels = torch.tensor([0, 1, 1])
        sample_ids = torch.tensor([0, 0, 0])
        fractions = torch.tensor([1 / 3, 2 / 3, 1.0])
        final = torch.tensor([False, False, True])
        actions = torch.tensor(
            [
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
            ]
        )
        rewards, _components = rollout_rewards(
            actions,
            labels,
            sample_ids,
            fractions,
            final,
            sample_count=1,
        )
        self.assertGreater(float(rewards[0, 0]), float(rewards[0, 1]))
        self.assertGreater(float(rewards[0, 0]), float(rewards[0, 2]))

    def test_grpo_loss_is_finite_and_differentiable(self):
        logits = torch.tensor(
            [[2.0, -1.0], [-0.2, 0.3], [-1.0, 2.0]], requires_grad=True
        )
        reference = logits.detach().clone()
        labels = torch.tensor([0, 1, 1])
        sample_ids = torch.tensor([0, 0, 0])
        fractions = torch.tensor([1 / 3, 2 / 3, 1.0])
        final = torch.tensor([False, False, True])
        torch.manual_seed(7)
        loss, metrics = grpo_action_loss(
            logits,
            reference,
            labels,
            sample_ids,
            fractions,
            final,
            sample_count=1,
            group_size=4,
            kl_beta=0.02,
            sft_weight=0.2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reward_mean", metrics)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_learning_rate_schedule(self):
        self.assertAlmostEqual(
            learning_rate_factor(0, warmup_steps=10, total_steps=100), 0.1
        )

    def test_fixed_wait_k_respects_capacity_and_final_flush(self):
        events = [
            {"target_ctc_count_proxy": 0, "source_is_final": False},
            {"target_ctc_count_proxy": 2, "source_is_final": False},
            {"target_ctc_count_proxy": 2, "source_is_final": False},
            {"target_ctc_count_proxy": 4, "source_is_final": True},
        ]
        self.assertEqual(
            fixed_wait_k_actions(events, wait_k=2),
            ["wait", "write", "wait", "write"],
        )
        self.assertAlmostEqual(
            learning_rate_factor(99, warmup_steps=10, total_steps=100),
            0.0,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
