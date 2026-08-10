from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_OBSERVED,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    TERM_NAMES,
    TrueSubsecondObjective,
    distributed_weighted_objective,
)


class JointModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.objective = TrueSubsecondObjective(
            hidden_size=16,
            codebook_weight=torch.randn(64, 1280),
            adapter_layers=1,
            adapter_kernel_size=3,
            adapter_expansion=1,
        )

    def test_zero_projection_preserves_decoder_input(self) -> None:
        decoder = torch.randn(12, 1, 16)
        batch = {
            "frontend_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "frontend_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "frontend_positions": torch.tensor([[1, 2, 0], [4, 0, 0]]),
            "action_batch": torch.tensor([0, 0]),
        }
        corrected, rms = self.objective.inject_frontend_residual(
            decoder, batch, original_seq_length=12
        )
        torch.testing.assert_close(corrected, decoder)
        self.assertEqual(float(rms), 0.0)

    def test_replay_uses_only_phase3_term_but_anchors_new_modules(self) -> None:
        logits = torch.randn(1, 4, 20, requires_grad=True)
        output = self.objective.replay(
            logits,
            torch.tensor([[1, 2, 3, 4]]),
            torch.ones(1, 4),
        )
        self.assertEqual(tuple(output.terms), TERM_NAMES)
        self.assertGreater(float(output.terms["phase3_replay"].mean), 0)
        self.assertEqual(float(output.terms["deadline_survival"].mean), 0)
        total, metrics = distributed_weighted_objective(output, progress=0.5)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIn("curriculum_deadline_weight", metrics)

    def test_trajectory_computes_all_supervised_heads(self) -> None:
        sequence, vocab = 8, 40
        hidden = torch.randn(sequence, 16, requires_grad=True)
        logits = torch.randn(sequence, vocab, requires_grad=True)
        labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        roles = torch.tensor(
            [
                ROLE_OBSERVED,
                ROLE_ACTION,
                ROLE_TEXT,
                ROLE_SEMANTIC,
                ROLE_SEMANTIC,
                ROLE_BOUNDARY,
                ROLE_OBSERVED,
                ROLE_OBSERVED,
            ]
        )
        teacher_indices = torch.tensor(
            [[[[3, 4], [5, 6]], [[3, 4], [5, 6]], [[3, 4], [5, 6]], [[3, 4], [5, 6]]]]
        )
        teacher_probabilities = torch.full_like(teacher_indices, 0.5, dtype=torch.float32)
        batch = {
            "original_seq_length": torch.tensor(sequence),
            "action_batch": torch.tensor([0]),
            "action_position": torch.tensor([1]),
            "support_bucket": torch.tensor([1]),
            "natural_action": torch.tensor([1]),
            "deadline_action": torch.tensor([1]),
            "deadline_forced": torch.tensor([False]),
            "sample_group": torch.tensor([11]),
            "chunk_end_ms": torch.tensor([640]),
            "soft_deadline_ms": torch.tensor([640]),
            "hard_deadline_ms": torch.tensor([800]),
            "translation_ids": torch.tensor([[3, 4]]),
            "translation_mask": torch.tensor([[True, True]]),
            "safe_commit_targets": torch.tensor([[1.0, 0.0]]),
            "teacher_indices": teacher_indices,
            "teacher_probabilities": teacher_probabilities,
            "teacher_mask": torch.ones(1, 4, 2, dtype=torch.bool),
            "kd_batch": torch.tensor([0]),
            "kd_position": torch.tensor([2]),
            "kd_annotation": torch.tensor([0]),
            "kd_target_index": torch.tensor([0]),
        }
        output = self.objective.trajectory(
            hidden,
            logits,
            labels,
            torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            roles,
            torch.randn(vocab, 16),
            batch,
            frontend_residual_rms=logits.sum() * 0.0,
        )
        for name in (
            "interleaved_trajectory",
            "real_prefix_kd",
            "support_ordinal",
            "token_safe_commit",
            "deadline_survival",
            "prefix_stability",
            "ar_semantic_microblock",
            "boundary_continuity",
        ):
            self.assertTrue(torch.isfinite(output.terms[name].mean), name)
        total, _ = distributed_weighted_objective(output, progress=0.5)
        total.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
