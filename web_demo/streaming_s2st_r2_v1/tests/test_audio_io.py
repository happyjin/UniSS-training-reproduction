import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from web_demo.streaming_s2st_r2_v1.audio_io import (
    AudioValidationError,
    SAMPLE_RATE,
    normalize_uploaded_audio,
    resample_mono,
    write_aligned_stereo,
)


class StreamingAudioIoTest(unittest.TestCase):
    def test_resample_stereo_and_reject_non_finite(self):
        stereo = np.stack([np.ones(800), np.zeros(800)], axis=1).astype(np.float32)
        values = resample_mono(stereo, 8000)
        self.assertEqual(len(values), 1600)
        self.assertAlmostEqual(float(values.mean()), 0.5, places=2)
        with self.assertRaises(AudioValidationError):
            resample_mono(np.array([np.nan], dtype=np.float32), SAMPLE_RATE)

    def test_normalize_and_aligned_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            normalized = root / "normalized.wav"
            sf.write(source, np.linspace(-0.1, 0.1, 8000, dtype=np.float32), 8000)
            metadata = normalize_uploaded_audio(
                source,
                normalized,
                max_upload_bytes=2_000_000,
                min_audio_seconds=0.5,
                max_audio_seconds=2.0,
            )
            self.assertEqual(metadata["sample_rate"], SAMPLE_RATE)
            audio, sr = sf.read(normalized, always_2d=False)
            self.assertEqual(sr, SAMPLE_RATE)
            aligned = write_aligned_stereo(
                audio,
                np.ones(1600, dtype=np.float32) * 0.1,
                root / "aligned.wav",
                translation_offset_ms=500,
            )
            stereo, sr = sf.read(aligned, always_2d=True)
            self.assertEqual(sr, SAMPLE_RATE)
            self.assertTrue(np.allclose(stereo[:8000, 1], 0.0))
            self.assertGreater(float(np.abs(stereo[8000:, 1]).mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
