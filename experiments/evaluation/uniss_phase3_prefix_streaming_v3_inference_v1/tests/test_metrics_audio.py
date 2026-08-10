from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.metrics import latency_metrics
from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.streaming_engine import (
    timeline_audio,
    write_stereo,
)


class MetricsAudioTest(unittest.TestCase):
    def test_latency_and_stereo_channel_contract(self) -> None:
        metrics = latency_metrics([500.0, 1000.0], 2000.0)
        self.assertAlmostEqual(metrics["ap"], 0.375)
        source = np.ones(320, dtype=np.float32) * 0.25
        target = timeline_audio([(10.0, np.ones(160, dtype=np.float32) * -0.5)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            write_stereo(source, target, path)
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
        self.assertEqual(rate, 16000)
        self.assertEqual(audio.shape[1], 2)
        self.assertGreater(float(audio[:, 0].max()), 0.2)
        self.assertLess(float(audio[:, 1].min()), -0.4)


if __name__ == "__main__":
    unittest.main()

