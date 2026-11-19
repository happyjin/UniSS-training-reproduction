import unittest
from types import SimpleNamespace

import numpy as np
import torch

from inference.s2s_tf_diagnostics import (
    build_reply_add_prefix,
    merge_sample_diagnostics,
    normalize_target_codes,
    prepare_gt_thinker_inputs,
    waveform_metrics,
)


class TargetCodeContractTest(unittest.TestCase):
    def test_normalizes_time_major_and_group_major_codes(self):
        time_major = np.arange(3 * 16).reshape(3, 16)
        group_major = time_major.T

        np.testing.assert_array_equal(
            normalize_target_codes(time_major.tolist(), expected_groups=16),
            time_major,
        )
        np.testing.assert_array_equal(
            normalize_target_codes(group_major.tolist(), expected_groups=16),
            time_major,
        )

    def test_rejects_invalid_group_count(self):
        with self.assertRaisesRegex(ValueError, "16 codec groups"):
            normalize_target_codes(np.zeros((4, 15), dtype=np.int64), expected_groups=16)


class WaveformMetricContractTest(unittest.TestCase):
    def test_reports_trailing_silence_in_fixed_windows(self):
        waveform = np.concatenate([
            np.full(500, 0.1, dtype=np.float32),
            np.zeros(500, dtype=np.float32),
        ])

        metrics = waveform_metrics(waveform, 1000)

        self.assertEqual(metrics["duration_ms"], 1000.0)
        self.assertEqual(metrics["trailing_silence_s"], 0.5)
        self.assertEqual(metrics["silent_window_pct"], 50.0)
        self.assertAlmostEqual(metrics["peak"], 0.1, places=5)


class SummaryMergeContractTest(unittest.TestCase):
    def test_preserves_ar_fields_when_adding_diagnostics(self):
        existing = {
            "sample_id": "val_7",
            "generated_audio": "/old/generated.wav",
            "codec_frames": 384,
        }
        diagnostic = {
            "target_audio": "/new/target.wav",
            "teacher_forced_audio": "/new/teacher_forced.wav",
            "tf_eos_correct": True,
        }

        merged = merge_sample_diagnostics(existing, diagnostic)

        self.assertEqual(merged["generated_audio"], "/old/generated.wav")
        self.assertEqual(merged["codec_frames"], 384)
        self.assertEqual(merged["target_audio"], "/new/target.wav")
        self.assertIs(merged["tf_eos_correct"], True)
        self.assertIsNot(merged, existing)


class TeacherForcingInputContractTest(unittest.TestCase):
    def test_reply_add_prefix_includes_language_speaker_pad_and_bos(self):
        config = SimpleNamespace(
            codec_bos_id=19,
            codec_eos_id=20,
            codec_pad_id=18,
            codec_think_id=24,
            codec_nothink_id=25,
            codec_think_bos_id=26,
            codec_think_eos_id=27,
            codec_language_id={"english": 31},
            talker_config=SimpleNamespace(
                num_code_groups=15,
                code_predictor_vocab_size=2048,
            ),
        )

        prefix, speaker_slot = build_reply_add_prefix(
            config, "en", with_speaker=True)

        self.assertEqual(prefix, [24, 26, 31, 27, 18, 18, 19])
        self.assertEqual(speaker_slot, 4)

    def test_gt_reply_is_appended_and_only_reply_positions_are_labeled(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_text = text
                self.last_add_special_tokens = add_special_tokens
                return [10, 11]

        processor = SimpleNamespace(tokenizer=FakeTokenizer())
        prompt_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "input_features": torch.tensor([[0.5]]),
        }

        thinker_inputs, labels, reply_ids = prepare_gt_thinker_inputs(
            processor, prompt_inputs, "translated text", torch.device("cpu"))

        self.assertEqual(thinker_inputs["input_ids"].tolist(), [[1, 2, 3, 10, 11]])
        self.assertEqual(thinker_inputs["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 10, 11]])
        self.assertEqual(reply_ids, [10, 11])
        self.assertEqual(thinker_inputs["input_features"].tolist(), [[0.5]])
        self.assertEqual(processor.tokenizer.last_text, "translated text<|im_end|>")
        self.assertIs(processor.tokenizer.last_add_special_tokens, False)


if __name__ == "__main__":
    unittest.main()
