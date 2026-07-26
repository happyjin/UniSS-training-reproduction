import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluation import asr_transcribe, autopcp_metrics, utmos_metrics


class MetricAdaptersTest(unittest.TestCase):
    def test_asr_backend_routing(self):
        self.assertEqual(asr_transcribe.target_asr_backend("eng"), "whisper-large-v3")
        self.assertEqual(asr_transcribe.target_asr_backend("cmn"), "paraformer-zh")

    def test_whisper_audio_loader_uses_soundfile_and_mixes_to_mono(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            stereo = np.column_stack(
                [np.full(160, 0.25, dtype=np.float32), np.full(160, 0.75, dtype=np.float32)]
            )
            sf.write(path, stereo, 16000, subtype="FLOAT")
            audio = asr_transcribe.load_audio_array(path, expected_sample_rate=16000)
        self.assertEqual(audio.shape, (160,))
        np.testing.assert_allclose(audio, 0.5)

    def test_whisper_audio_loader_rejects_wrong_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_rate.wav"
            sf.write(path, np.zeros(80, dtype=np.float32), 8000)
            with self.assertRaisesRegex(ValueError, "Expected 16000 Hz"):
                asr_transcribe.load_audio_array(path, expected_sample_rate=16000)

    def test_score_aggregation(self):
        rows = [
            {"mode": "quality", "src_lang": "eng", "tgt_lang": "cmn", "utmos_score": 3.0},
            {"mode": "quality", "src_lang": "eng", "tgt_lang": "cmn", "utmos_score": 4.0},
        ]
        report = utmos_metrics.aggregate_scores(rows)
        self.assertEqual(report["groups"]["quality:eng->cmn"]["mean"], 3.5)
        autopcp_rows = [
            {"mode": "performance", "src_lang": "cmn", "tgt_lang": "eng", "autopcp_score": 2.0},
            {"mode": "performance", "src_lang": "cmn", "tgt_lang": "eng", "autopcp_score": 4.0},
        ]
        report = autopcp_metrics.aggregate_scores(autopcp_rows)
        self.assertEqual(report["groups"]["performance:cmn->eng"]["mean"], 3.0)

    def test_autopcp_defaults_to_single_audio_reader(self):
        args = autopcp_metrics.parse_args(
            ["--input", "results.jsonl", "--output-dir", "metrics", "--comparator-path", "model"]
        )
        self.assertEqual(args.num_process, 1)


if __name__ == "__main__":
    unittest.main()
