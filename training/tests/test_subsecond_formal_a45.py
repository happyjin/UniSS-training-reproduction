from __future__ import annotations

import unittest

from training.simul_uniss.subsecond_v2.prepare_a45 import build_a45_record


class FormalStageAA45Test(unittest.TestCase):
    def test_reconstructed_teacher_and_released_glm_are_audited_separately(self) -> None:
        item = {
            "id": "sample",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "hello world",
            "translation": "你好世界",
            "source_duration_ms": 1000,
            "source_glm": [1, 2, 3, 4],
            "quality_flags": [],
        }
        value = build_a45_record(
            item,
            target_audio="target.flac",
            target_duration_ms=800,
            source_alignment=[
                {"text": "hello", "start_ms": 0, "end_ms": 400},
                {"text": "world", "start_ms": 450, "end_ms": 900},
            ],
            target_alignment=[
                {"text": "你", "start_ms": 0, "end_ms": 180},
                {"text": "好", "start_ms": 180, "end_ms": 360},
                {"text": "世", "start_ms": 400, "end_ms": 580},
                {"text": "界", "start_ms": 580, "end_ms": 760},
            ],
            teacher_source_glm=[1, 9, 3, 4],
            minimum_alignment_coverage=0.85,
        )
        self.assertTrue(value["formal_a45_pass"])
        self.assertEqual(value["stage_b_supervision_field"], "teacher_source_glm")
        self.assertEqual(value["released_source_glm"], [1, 2, 3, 4])
        self.assertEqual(value["teacher_source_glm"], [1, 9, 3, 4])
        self.assertEqual(value["released_vs_reconstructed_teacher_agreement"], 0.75)
        self.assertIn("released_vs_reconstructed_teacher_domain_mismatch", value["quality_flags"])


if __name__ == "__main__":
    unittest.main()
