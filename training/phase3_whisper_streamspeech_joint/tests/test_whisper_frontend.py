from __future__ import annotations

import unittest

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from training.phase3_whisper_streamspeech_joint.whisper_frontend import (
    TrainableMultiChunkWhisperVQ,
    WhisperGPUFeatureExtractor,
)


MODEL = "pretrained_models/UniSS/glm4_tokenizer"


class WhisperFrontendTest(unittest.TestCase):
    def test_gradient_checkpointing_callable_is_installed_on_bare_encoder(self) -> None:
        frontend = TrainableMultiChunkWhisperVQ.__new__(TrainableMultiChunkWhisperVQ)
        torch.nn.Module.__init__(frontend)
        frontend.encoder = torch.nn.Linear(2, 2)

        frontend.configure_gradient_checkpointing(True)

        self.assertTrue(frontend.encoder.gradient_checkpointing)
        self.assertTrue(callable(frontend.encoder._gradient_checkpointing_func))

    def test_torch_features_match_transformers_reference(self) -> None:
        waveform = torch.linspace(-0.2, 0.2, 16_000).unsqueeze(0)
        frontend = WhisperGPUFeatureExtractor(MODEL)
        actual, mask = frontend(
            waveform,
            torch.tensor([16_000]),
            pad_to_multiple_of=1280,
        )
        reference = WhisperFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
        expected = reference(
            [waveform[0].numpy().astype(np.float32)],
            sampling_rate=16_000,
            padding="longest",
            pad_to_multiple_of=1280,
            return_attention_mask=True,
            return_tensors="pt",
        )
        torch.testing.assert_close(actual, expected.input_features, atol=2e-5, rtol=2e-5)
        self.assertEqual(mask.sum(dim=1).tolist(), [100])


if __name__ == "__main__":
    unittest.main()
