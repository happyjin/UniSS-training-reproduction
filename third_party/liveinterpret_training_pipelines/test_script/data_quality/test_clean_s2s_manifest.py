"""CPU tests for the production S2S manifest cleaning rules."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset_maker.clean_s2s_manifest import (
    DEFAULT_CONFIG,
    QualityScores,
    _load_config,
    _quality_reasons,
    evaluate_audio_rules,
    evaluate_static_rules,
)


def _codes(frames: int, *, collapsed: bool = False) -> list[list[int]]:
    if collapsed:
        return np.zeros((frames, 16), dtype=np.int32).tolist()
    values = np.arange(frames * 16, dtype=np.int32).reshape(frames, 16)
    return (values % 2048).tolist()


def _row(**overrides):
    result = {
        "record_id": "test:1",
        "source_duration_ms": 4000,
        "target_duration_ms": 4000,
        "target_codec_codes": _codes(50),
        "target_text": "This is a normal translated sentence.",
        "target_lang": "en",
        "target_audio": "unused.wav",
        "audios": ["source.wav"],
        "source_lang": "zh",
    }
    result.update(overrides)
    return result


class CleanS2SManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _load_config(DEFAULT_CONFIG)

    def test_quality_thresholds_and_missing_policy(self):
        reasons, _ = _quality_reasons(QualityScores(3.0, 0.7), self.config, "ok")
        self.assertEqual(reasons, [])
        reasons, _ = _quality_reasons(QualityScores(2.4, 0.3), self.config, "low")
        self.assertEqual(reasons, ["low_utmos", "low_ecapa_similarity"])
        reasons, _ = _quality_reasons(None, self.config, "missing")
        self.assertEqual(reasons, ["missing_mos_or_ecapa"])

    def test_01_codec_silence_collapse(self):
        row = _row(
            target_duration_ms=8000,
            target_codec_codes=_codes(100, collapsed=True),
            target_text="This sentence is long enough to avoid rule three.",
        )
        reasons, _ = evaluate_static_rules(row, self.config)
        self.assertIn("01_shorter_silence_collapse", reasons)

    def test_03_short_text_long_audio(self):
        row = _row(target_duration_ms=8000, target_codec_codes=_codes(100), target_text="Too short")
        reasons, _ = evaluate_static_rules(row, self.config)
        self.assertIn("03_very_short_text_long_audio", reasons)

    def test_04_and_05_extreme_duration_ratio(self):
        longer, _ = evaluate_static_rules(_row(
            source_duration_ms=1000, target_duration_ms=6000,
            target_codec_codes=_codes(75)), self.config)
        shorter, _ = evaluate_static_rules(_row(
            source_duration_ms=5000, target_duration_ms=800,
            target_codec_codes=_codes(10)), self.config)
        self.assertIn("04_target_much_longer_than_source", longer)
        self.assertIn("05_target_much_shorter_than_source", shorter)

    def test_previous_30s_cap(self):
        reasons, _ = evaluate_static_rules(_row(
            target_duration_ms=29_000, target_codec_codes=_codes(363),
            target_text="This is deliberately long enough for the short text rule not to fire."),
            self.config)
        self.assertIn("previous_30s_generation_cap", reasons)

    def test_02_trailing_silence_decodes_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tail.wav"
            sample_rate = 12_000
            audible = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
            waveform = np.concatenate([audible, np.zeros(sample_rate * 2)]).astype(np.float32)
            sf.write(path, waveform, sample_rate, subtype="FLOAT")
            reasons, metrics = evaluate_audio_rules(
                _row(target_audio=str(path)), self.config, None)
        self.assertIn("02_long_trailing_silence", reasons)
        self.assertAlmostEqual(metrics["decoded_trailing_low_energy_seconds"], 2.0, places=2)

    def test_normal_row_passes_static_rules(self):
        reasons, metrics = evaluate_static_rules(_row(), self.config)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["codec_groups"], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
