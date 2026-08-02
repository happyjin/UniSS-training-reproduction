from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
    StreamingWhisperVQTeacher,
    chunk_right_attention_mask,
)


class StreamingWhisperVQTeacherTest(unittest.TestCase):
    def test_chunk_mask_allows_history_current_chunk_and_right_context(self) -> None:
        mask = chunk_right_attention_mask(
            torch.ones(1, 8, dtype=torch.long),
            chunk_frames=4,
            right_context_frames=2,
            dtype=torch.float32,
        )[0, 0]
        allowed = mask == 0
        self.assertTrue(bool(allowed[0, :6].all()))
        self.assertFalse(bool(allowed[0, 6:].any()))
        self.assertTrue(bool(allowed[3, :6].all()))
        self.assertTrue(bool(allowed[4, :8].all()))

    def test_chunk_mask_respects_padding(self) -> None:
        attention = torch.tensor([[1, 1, 1, 0]])
        mask = chunk_right_attention_mask(
            attention,
            chunk_frames=2,
            right_context_frames=1,
            dtype=torch.float32,
        )[0, 0]
        self.assertTrue(bool((mask[:, 3] < 0).all()))

    def test_in_memory_tensor_defaults_to_16khz(self) -> None:
        waveform, sample_rate = StreamingWhisperVQTeacher._audio_tuple(
            torch.zeros(1, 1600)
        )
        self.assertEqual(waveform.shape, (1, 1600))
        self.assertEqual(sample_rate, 16_000)


if __name__ == "__main__":
    unittest.main()
