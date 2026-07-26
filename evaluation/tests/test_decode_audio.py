import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.decode_audio import batched, decode_token_batch


class _FakeBiCodec:
    def batch_decode(self, items):
        return {
            "indices": [item["index"] for item in items],
            "wavs": [np.zeros(len(item["semantic_tokens"]) * 320, dtype=np.float32) for item in items],
        }


class _FakeTokenizer:
    def __init__(self):
        self.bicodec = _FakeBiCodec()

    def save_audio(self, wave, path, sample_rate=16000):
        sf.write(path, wave, sample_rate)


class DecodeAudioBatchTest(unittest.TestCase):
    def test_batched(self):
        self.assertEqual(list(batched(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_batch_decode_saves_trimmed_audio_and_tracks_missing_semantics(self):
        tokenizer = _FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = decode_token_batch(
                speech_tokenizer=tokenizer,
                items=[
                    {
                        "index": 0,
                        "global_values": [1] * 32,
                        "semantic_values": [2, 3],
                        "output_path": root / "a.wav",
                    },
                    {
                        "index": 1,
                        "global_values": [1] * 32,
                        "semantic_values": [2, 3, 4],
                        "output_path": root / "b.wav",
                    },
                    {
                        "index": 2,
                        "global_values": [1] * 32,
                        "semantic_values": [],
                        "output_path": root / "missing.wav",
                    },
                ],
                device=torch.device("cpu"),
            )
            self.assertEqual(sf.info(root / "a.wav").frames, 640)
            self.assertEqual(sf.info(root / "b.wav").frames, 960)
            self.assertFalse((root / "missing.wav").exists())
        self.assertEqual(results[2], (None, "no_semantic_tokens"))


if __name__ == "__main__":
    unittest.main()
