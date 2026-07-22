from __future__ import annotations

import unittest

import torch

from training.simul_uniss.audio_streaming_student import AudioStreamingStudent, audio_student_losses


class AudioStudentTests(unittest.TestCase):
    def test_lengths_and_shapes(self) -> None:
        model = AudioStreamingStudent(65, hidden_size=32, num_layers=1, num_heads=4)
        waveform = torch.randn(2, 3200)
        lengths = torch.tensor([3200, 2400])
        outputs = model(waveform, lengths)
        self.assertEqual(outputs["hidden"].shape[0], 2)
        self.assertEqual(outputs["teacher_glm_logits"].shape[-1], 16385)
        self.assertTrue(torch.all(outputs["output_lengths"] > 0))

    def test_losses_are_finite(self) -> None:
        model = AudioStreamingStudent(65, hidden_size=32, num_layers=1, num_heads=4)
        batch = {
            "waveform": torch.randn(1, 3200),
            "waveform_lengths": torch.tensor([3200]),
            "teacher_glm": torch.tensor([1, 2]),
            "teacher_glm_lengths": torch.tensor([2]),
            "source_policy": torch.tensor([1, 2]),
            "source_policy_lengths": torch.tensor([2]),
            "target_policy": torch.tensor([3]),
            "target_policy_lengths": torch.tensor([1]),
        }
        losses = audio_student_losses(model, batch)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
