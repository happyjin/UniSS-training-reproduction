from __future__ import annotations

import random
import unittest

import torch

from uniss.speech_tokenizer.glm4.modeling_whisper import (
    _sample_codebook_restart_vectors,
)


class WhisperVQRestartTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260806)
        self.hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    def test_zero_updates_returns_empty_tensor_with_matching_shape(self) -> None:
        selected = _sample_codebook_restart_vectors(self.hidden, 0)
        self.assertEqual(tuple(selected.shape), (0, 3))
        self.assertEqual(selected.dtype, self.hidden.dtype)

    def test_samples_without_replacement_when_candidates_are_sufficient(self) -> None:
        selected = _sample_codebook_restart_vectors(self.hidden, 4)
        self.assertEqual(tuple(selected.shape), (4, 3))
        self.assertEqual(len(torch.unique(selected[:, 0])), 4)

    def test_samples_with_replacement_for_short_streaming_chunk(self) -> None:
        selected = _sample_codebook_restart_vectors(self.hidden[:2], 7)
        self.assertEqual(tuple(selected.shape), (7, 3))
        valid_rows = {tuple(row.tolist()) for row in self.hidden[:2]}
        self.assertTrue(all(tuple(row.tolist()) in valid_rows for row in selected))

    def test_nonzero_updates_require_at_least_one_valid_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "no valid encoder vectors"):
            _sample_codebook_restart_vectors(self.hidden[:0], 1)


if __name__ == "__main__":
    unittest.main()
