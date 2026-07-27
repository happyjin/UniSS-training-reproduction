from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from web_demo.offline_s2st_phase3_v1.audio_io import (
    SAMPLE_RATE,
    AudioValidationError,
    cleanup_expired,
    create_request_directory,
    normalize_uploaded_audio,
    split_on_silence,
    stitch_audio,
)


class AudioIOTest(unittest.TestCase):
    def test_normalize_stereo_resampled_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            destination = root / "normalized.wav"
            samples = np.stack(
                [np.sin(np.linspace(0, 40, 8_000)), np.sin(np.linspace(0, 20, 8_000))],
                axis=1,
            ).astype(np.float32)
            sf.write(source, samples, 8_000)
            metadata = normalize_uploaded_audio(
                source,
                destination,
                max_upload_bytes=1_000_000,
                min_audio_seconds=0.2,
                max_audio_seconds=2.0,
            )
            info = sf.info(destination)
            self.assertEqual(info.samplerate, SAMPLE_RATE)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(metadata["duration_seconds"], 1.0, places=2)

    def test_rejects_too_short_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "short.wav"
            sf.write(source, np.zeros(100, dtype=np.float32), SAMPLE_RATE)
            with self.assertRaises(AudioValidationError):
                normalize_uploaded_audio(
                    source,
                    Path(directory) / "out.wav",
                    max_upload_bytes=100_000,
                    min_audio_seconds=0.25,
                    max_audio_seconds=2.0,
                )

    def test_split_and_stitch_preserve_order(self):
        first = np.ones(SAMPLE_RATE * 2, dtype=np.float32) * 0.1
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        second = np.ones(SAMPLE_RATE * 2, dtype=np.float32) * -0.1
        waveform = np.concatenate([first, silence, second])
        chunks = split_on_silence(waveform, max_chunk_seconds=2.5)
        self.assertGreaterEqual(len(chunks), 2)
        stitched = stitch_audio(chunks, silence_seconds=0.1)
        self.assertGreater(stitched.size, first.size + second.size)

    def test_expired_request_cleanup_is_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = create_request_directory(root)
            old = time.time() - 7200
            request.touch()
            request.chmod(0o755)
            import os

            os.utime(request, (old, old))
            self.assertEqual(cleanup_expired(root, ttl_hours=1), 1)
            self.assertFalse(request.exists())


if __name__ == "__main__":
    unittest.main()
