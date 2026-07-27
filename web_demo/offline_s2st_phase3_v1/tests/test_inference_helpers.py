from __future__ import annotations

import unittest

from web_demo.offline_s2st_phase3_v1.config import DemoConfig
from web_demo.offline_s2st_phase3_v1.inference_engine import (
    maximum_identical_run,
    quality_output_warnings,
    target_language_tag,
)


class InferenceHelpersTest(unittest.TestCase):
    def test_direction_mapping(self):
        self.assertEqual(target_language_tag("中文 → 英文"), "<|eng|>")
        self.assertEqual(target_language_tag("英文 → 中文"), "<|cmn|>")
        with self.assertRaises(ValueError):
            target_language_tag("auto")

    def test_maximum_identical_run(self):
        self.assertEqual(maximum_identical_run([]), 0)
        self.assertEqual(maximum_identical_run([1, 1, 2, 2, 2, 1]), 3)

    def test_quality_warnings_require_all_three_outputs(self):
        complete = {
            "generated_transcription": "你好",
            "generated_translation": "hello",
            "has_semantic_start": True,
            "has_semantic_end": True,
            "has_eos": True,
            "semantic_values": [1, 2],
        }
        self.assertEqual(quality_output_warnings(complete), [])
        incomplete = {**complete, "generated_transcription": "", "semantic_values": []}
        warnings = quality_output_warnings(incomplete)
        self.assertTrue(any("transcription" in warning for warning in warnings))
        self.assertTrue(any("semantic" in warning for warning in warnings))

    def test_frozen_config_rejects_non_quality(self):
        with self.assertRaises(ValueError):
            DemoConfig(mode="performance", task_name="Performance").validate()


if __name__ == "__main__":
    unittest.main()
