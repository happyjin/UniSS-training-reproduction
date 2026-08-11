from __future__ import annotations

import unittest

import numpy as np

from web_demo.true_subsecond_pilot15_streaming_v1.engine import (
    SAMPLE_RATE,
    speech_active,
    stereo_waveform,
    timeline_audio,
)


class EngineHelpersTest(unittest.TestCase):
    def test_timeline_is_monotonic_and_stereo_keeps_source_left(self) -> None:
        first = np.ones(1600, dtype=np.float32)
        second = np.ones(1600, dtype=np.float32) * 2
        timeline = timeline_audio(((100, first), (150, second)))
        self.assertEqual(len(timeline), 4800)
        source = np.linspace(-1, 1, 3200, dtype=np.float32)
        stereo = stereo_waveform(source, timeline)
        np.testing.assert_allclose(stereo[: len(source), 0], source)
        np.testing.assert_allclose(stereo[:, 1], timeline)

    def test_vad_is_simple_and_deterministic(self) -> None:
        self.assertFalse(speech_active(np.zeros(SAMPLE_RATE // 10, dtype=np.float32)))
        self.assertTrue(
            speech_active(np.ones(SAMPLE_RATE // 10, dtype=np.float32) * 0.01)
        )


if __name__ == "__main__":
    unittest.main()
