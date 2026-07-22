from __future__ import annotations

import unittest

import torch

from training.simul_uniss.streaming_student import StreamingTokenStudent, collate_student_batch


class StreamingStudentTests(unittest.TestCase):
    def test_output_lengths(self) -> None:
        lengths = StreamingTokenStudent.output_lengths(torch.tensor([1, 4, 5, 8]))
        self.assertEqual(lengths.tolist(), [1, 1, 2, 2])

    def test_model_shapes_and_causal_prefix(self) -> None:
        torch.manual_seed(1)
        model = StreamingTokenStudent(65, hidden_size=32, num_layers=1, num_heads=4, dropout=0.0)
        model.eval()
        full = torch.arange(32).unsqueeze(0) % 8192
        prefix = full[:, :20]
        with torch.no_grad():
            full_out = model(full, torch.tensor([32]))["hidden"]
            prefix_out = model(prefix, torch.tensor([20]))["hidden"]
        self.assertEqual(full_out.shape, (1, 8, 32))
        self.assertTrue(torch.allclose(full_out[:, :5], prefix_out, atol=1e-5, rtol=1e-5))

    def test_collate_keeps_ctc_targets_concatenated(self) -> None:
        batch = [
            {
                "source_bicodec": torch.tensor([1, 2, 3]),
                "teacher_glm": torch.tensor([4]),
                "source_policy": torch.tensor([5, 6]),
                "target_policy": torch.tensor([7]),
            },
            {
                "source_bicodec": torch.tensor([8, 9]),
                "teacher_glm": torch.tensor([10]),
                "source_policy": torch.tensor([11]),
                "target_policy": torch.tensor([12, 13]),
            },
        ]
        result = collate_student_batch(batch)
        self.assertEqual(result["source_bicodec"].shape, (2, 3))
        self.assertEqual(result["source_policy"].tolist(), [5, 6, 11])
        self.assertEqual(result["target_policy_lengths"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
