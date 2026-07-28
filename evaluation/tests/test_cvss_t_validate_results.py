from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluation.cvss_t.validate_results import validate_rows


class CvssValidateResultsTest(unittest.TestCase):
    def make_audio(self, path: Path) -> None:
        sf.write(path, np.zeros(160, dtype=np.float32), 16000)

    def test_accepts_complete_bidirectional_mode_set_for_one_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            reference = root / "reference.wav"
            generated_q = root / "generated_q.wav"
            generated_p = root / "generated_p.wav"
            for path in (source, reference, generated_q, generated_p):
                self.make_audio(path)
            rows = [
                {
                    "id": "sample",
                    "mode": mode,
                    "src_lang": "cmn",
                    "tgt_lang": "eng",
                    "source_audio_path": str(source),
                    "reference_audio_path": str(reference),
                    "audio_path": str(generated),
                    "source_audio_duration_seconds": 0.01,
                    "reference_audio_duration_seconds": 0.01,
                    "synthetic_source": False,
                    "synthetic_reference": True,
                    "error": None,
                }
                for mode, generated in (("quality", generated_q), ("performance", generated_p))
            ]
            report = validate_rows(
                rows,
                input_path=root / "results.jsonl",
                expected_pairs=1,
                expected_direction="cmn->eng",
                modes=("quality", "performance"),
                allow_generated_failures=False,
            )
        self.assertTrue(report["valid"])
        self.assertEqual(report["row_count"], 2)

    def test_rejects_generated_audio_that_reuses_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            reference = root / "reference.wav"
            for path in (source, reference):
                self.make_audio(path)
            row = {
                "id": "sample",
                "mode": "quality",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "source_audio_path": str(source),
                "reference_audio_path": str(reference),
                "audio_path": str(reference),
                "source_audio_duration_seconds": 0.01,
                "reference_audio_duration_seconds": 0.01,
                "synthetic_source": True,
                "synthetic_reference": False,
                "error": None,
            }
            with self.assertRaisesRegex(ValueError, "generated_audio_reuses_official_waveform"):
                validate_rows(
                    [row],
                    input_path=root / "results.jsonl",
                    expected_pairs=1,
                    expected_direction="eng->cmn",
                    modes=("quality",),
                    allow_generated_failures=False,
                )


if __name__ == "__main__":
    unittest.main()
