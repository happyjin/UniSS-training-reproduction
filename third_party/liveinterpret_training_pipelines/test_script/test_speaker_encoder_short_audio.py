#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'custom'))

from qwen_omnispeak.speaker_encoder import (  # noqa: E402
    Qwen3TTSSpeakerEmbedder,
    _min_waveform_samples,
)


class ShortAudioSpeakerEncoderTest(unittest.TestCase):
    def make_embedder(self):
        embedder = Qwen3TTSSpeakerEmbedder(enc_dim=8)
        seen = {}

        def fake_mel(wav, **kwargs):
            seen['samples'] = int(wav.shape[-1])
            return torch.zeros(1, 128, 20)

        class FakeEncoder:
            def __call__(self, mels):
                return torch.ones(1, 8)

        embedder._mel_fn = fake_mel
        embedder._encoder = FakeEncoder()
        embedder._ensure_loaded = lambda device, dtype: None
        return embedder, seen

    def test_short_waveform_is_replicate_padded(self):
        embedder, seen = self.make_embedder()
        result = embedder.extract(
            torch.tensor([0.25, -0.5]), 24000,
            device=torch.device('cpu'), dtype=torch.float32)
        self.assertGreaterEqual(seen['samples'], _min_waveform_samples())
        self.assertEqual(tuple(result.shape), (8,))

    def test_empty_waveform_returns_zero_vector(self):
        embedder, seen = self.make_embedder()
        result = embedder.extract(
            torch.empty(0), 24000,
            device=torch.device('cpu'), dtype=torch.float32)
        self.assertNotIn('samples', seen)
        self.assertTrue(torch.equal(result, torch.zeros(8)))


if __name__ == '__main__':
    unittest.main()
