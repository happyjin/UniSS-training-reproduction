import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from web_demo.streaming_s2st_r2_v1.engine.prefix_frontend import CumulativePrefixFrontend
from web_demo.streaming_s2st_r2_v1.engine.streaming_pipeline import StreamingDemoEngine


class StreamingPipelineHelpersTest(unittest.TestCase):
    def test_speaker_extraction_flattens_batched_bicodec_tokens(self):
        class FakeBiCodec:
            def encode_wav_to_tokens(self, _path):
                return torch.arange(40, dtype=torch.long).reshape(1, 40)

        class FakeSpeechTokenizer:
            bicodec = FakeBiCodec()

        with tempfile.TemporaryDirectory() as directory:
            frontend = CumulativePrefixFrontend(FakeSpeechTokenizer())
            values = frontend.extract_speaker_tokens(
                np.zeros(1600, dtype=np.float32), Path(directory) / "speaker.wav"
            )
            self.assertEqual(values, list(range(32)))

    def test_timeline_preserves_wait_gaps_and_serializes_overlaps(self):
        first = np.ones(1600, dtype=np.float32) * 0.1
        second = np.ones(1600, dtype=np.float32) * 0.2
        timeline = StreamingDemoEngine._timeline_audio(
            [(500.0, first), (550.0, second)]
        )
        self.assertEqual(len(timeline), 8000 + 3200)
        self.assertTrue(np.allclose(timeline[:8000], 0.0))
        self.assertTrue(np.allclose(timeline[8000:9600], 0.1))
        self.assertTrue(np.allclose(timeline[9600:], 0.2))

    def test_stereo_export_keeps_source_left_and_target_right(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            StreamingDemoEngine._write_stereo(
                np.ones(100, dtype=np.float32) * 0.1,
                np.ones(200, dtype=np.float32) * 0.2,
                path,
            )
            audio, sample_rate = sf.read(path, always_2d=True)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(audio.shape, (200, 2))
            self.assertAlmostEqual(float(audio[:100, 0].mean()), 0.1, places=3)
            self.assertAlmostEqual(float(audio[:, 1].mean()), 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
