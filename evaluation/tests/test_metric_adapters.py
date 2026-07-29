import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from evaluation import asr_transcribe, autopcp_metrics, utmos_metrics


class MetricAdaptersTest(unittest.TestCase):
    def test_asr_backend_routing(self):
        self.assertEqual(asr_transcribe.target_asr_backend("eng"), "whisper-large-v3")
        self.assertEqual(asr_transcribe.target_asr_backend("cmn"), "paraformer-zh")

    def test_asr_duration_sort_puts_unknown_lengths_last(self):
        rows = [
            {"audio_duration_seconds": 3.0},
            {},
            {"audio_duration_seconds": 1.5},
            {"audio_duration_seconds": 0},
        ]
        ordered = sorted(rows, key=asr_transcribe.audio_duration_sort_key)
        self.assertEqual([row.get("audio_duration_seconds") for row in ordered], [1.5, 3.0, None, 0])

    def test_whisper_duration_buckets_do_not_mix_preprocess_schemas(self):
        self.assertEqual(
            asr_transcribe.whisper_duration_bucket(
                {"audio_duration_seconds": 30.0}, max_duration_seconds=30.0
            ),
            "short",
        )
        self.assertEqual(
            asr_transcribe.whisper_duration_bucket(
                {"audio_duration_seconds": 30.01}, max_duration_seconds=30.0
            ),
            "long",
        )
        self.assertEqual(
            asr_transcribe.whisper_duration_bucket({}, max_duration_seconds=30.0),
            "unknown",
        )
        self.assertNotIn("return_timestamps", asr_transcribe.whisper_call_options("short"))
        self.assertTrue(asr_transcribe.whisper_call_options("long")["return_timestamps"])
        self.assertTrue(asr_transcribe.whisper_call_options("unknown")["return_timestamps"])

    def test_whisper_very_short_audio_is_not_batched(self):
        rows = [
            {"id": "very-short", "audio_duration_seconds": 0.4},
            {"id": "short-boundary", "audio_duration_seconds": 2.0},
            {"id": "normal-1", "audio_duration_seconds": 2.1},
            {"id": "normal-2", "audio_duration_seconds": 2.2},
            {"id": "normal-3", "audio_duration_seconds": 2.3},
        ]
        batches = list(asr_transcribe.whisper_batches(rows, batch_size=8))
        self.assertEqual([[row["id"] for row in batch] for batch in batches], [
            ["very-short"],
            ["short-boundary"],
            ["normal-1", "normal-2", "normal-3"],
        ])

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

    def test_whisper_attention_mask_is_enabled_for_batched_inference(self):
        recognizer = SimpleNamespace(
            feature_extractor=SimpleNamespace(return_attention_mask=False)
        )
        asr_transcribe.configure_whisper_attention_mask(recognizer)
        self.assertTrue(recognizer.feature_extractor.return_attention_mask)

    def test_whisper_attention_mask_requires_feature_extractor(self):
        with self.assertRaisesRegex(TypeError, "feature_extractor"):
            asr_transcribe.configure_whisper_attention_mask(SimpleNamespace())

    def test_whisper_length_guard_rejects_decoder_limit_hallucination(self):
        row = {"audio_duration_seconds": 4.0}
        self.assertTrue(
            asr_transcribe.whisper_transcript_is_suspicious(
                row, "word " * 200
            )
        )
        self.assertFalse(
            asr_transcribe.whisper_transcript_is_suspicious(
                row, "a normal short transcript"
            )
        )

    def test_whisper_length_guard_skips_unknown_duration(self):
        self.assertFalse(
            asr_transcribe.whisper_transcript_is_suspicious({}, "word " * 200)
        )

    def test_whisper_suspicious_batch_result_can_retry_as_single_item(self):
        class FakeRecognizer:
            def __call__(self, inputs, *, batch_size, **kwargs):
                self.inputs = inputs
                self.batch_size = batch_size
                self.kwargs = kwargs
                return [{"text": "a short corrected transcript"}]

        recognizer = FakeRecognizer()
        text, rejected_reason = asr_transcribe.retry_suspicious_whisper_transcript(
            recognizer,
            row={"id": "sample", "mode": "quality", "audio_duration_seconds": 0.3},
            audio=np.zeros(160, dtype=np.float32),
            call_options={"generate_kwargs": {"language": "english"}},
        )
        self.assertEqual(text, "a short corrected transcript")
        self.assertIsNone(rejected_reason)
        self.assertEqual(recognizer.batch_size, 1)

    def test_whisper_single_item_retry_marks_persistent_hallucination(self):
        class HallucinatingRecognizer:
            def __call__(self, inputs, *, batch_size, **kwargs):
                return [{"text": "word " * 200}]

        text, rejected_reason = asr_transcribe.retry_suspicious_whisper_transcript(
            HallucinatingRecognizer(),
            row={"id": "sample", "mode": "quality", "audio_duration_seconds": 0.3},
            audio=np.zeros(160, dtype=np.float32),
            call_options={},
        )
        self.assertEqual(text, "")
        self.assertEqual(rejected_reason, asr_transcribe.WHISPER_REJECTED_REASON)

    def test_asr_target_language_filter_argument(self):
        args = asr_transcribe.parse_args(
            [
                "--input",
                "results.jsonl",
                "--output",
                "asr.jsonl",
                "--target-language",
                "eng",
            ]
        )
        self.assertEqual(args.target_language, ["eng"])

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
