from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluation.cvss_t.canonicalize import (
    CANONICAL_SAMPLE_RATE,
    build_direction_rows,
    convert_audio,
    is_valid_canonical,
)


class CvssCanonicalizeTest(unittest.TestCase):
    def test_convert_audio_resamples_and_downmixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            output_path = root / "output.wav"
            sample_rate = 48_000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            waveform = np.stack(
                [np.sin(2 * np.pi * 220 * time), np.sin(2 * np.pi * 440 * time)],
                axis=1,
            )
            sf.write(input_path, waveform, sample_rate, subtype="PCM_16")
            result = convert_audio(input_path, output_path, resume=False)
            self.assertTrue(is_valid_canonical(output_path))
            self.assertEqual(result["sample_rate"], CANONICAL_SAMPLE_RATE)
            self.assertEqual(result["channels"], 1)
            self.assertAlmostEqual(float(result["duration_seconds"]), 1.0, places=4)
            reused = convert_audio(input_path, output_path, resume=True)
            self.assertTrue(reused["reused"])
            self.assertEqual(result["sha256"], reused["sha256"])

    def test_direction_rows_mark_synthetic_side(self) -> None:
        pairs = [
            {
                "id": "sample.mp3",
                "source_zh_audio_path": "/canonical/source.wav",
                "target_en_audio_path": "/canonical/target.wav",
                "source_zh_raw_audio_path": "/raw/source.mp3",
                "target_en_raw_audio_path": "/raw/target.wav",
                "source_zh_text": "你好",
                "target_en_text": "hello",
            }
        ]
        zh_en, en_zh = build_direction_rows(pairs)
        self.assertFalse(zh_en[0]["synthetic_source"])
        self.assertTrue(zh_en[0]["synthetic_reference"])
        self.assertTrue(en_zh[0]["synthetic_source"])
        self.assertFalse(en_zh[0]["synthetic_reference"])
        self.assertEqual(en_zh[0]["translation_ref"], "你好")


if __name__ == "__main__":
    unittest.main()
