from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from evaluation.cvss_t.merge_tokenized import validate_rows
from evaluation.cvss_t.tokenize import build_direction_records, split_bicodec


class CvssTokenizeTest(unittest.TestCase):
    def test_split_bicodec_requires_global_and_semantic(self) -> None:
        global_values, semantic = split_bicodec(list(range(40)), field_name="fixture")
        self.assertEqual(global_values, list(range(32)))
        self.assertEqual(semantic, list(range(32, 40)))
        with self.assertRaises(ValueError):
            split_bicodec(list(range(32)), field_name="fixture")

    def test_build_direction_records_uses_source_voice_tokens(self) -> None:
        pair = {
            "id": "sample.mp3",
            "source_zh_audio_path": "/canonical/zh.wav",
            "target_en_audio_path": "/canonical/en.wav",
            "source_zh_text": "你好",
            "target_en_text": "hello",
            "source_zh_duration_seconds": 2.0,
            "target_en_duration_seconds": 1.5,
            "source_zh_audio_sha256": "zh-hash",
            "target_en_audio_sha256": "en-hash",
        }
        zh_global = list(range(32))
        en_global = list(range(100, 132))
        zh_en, en_zh = build_direction_records(
            pair,
            pair_index=0,
            zh_glm=[1, 2],
            zh_bicodec=[*zh_global, 10, 11],
            en_glm=[3, 4],
            en_bicodec=[*en_global, 20, 21],
            tokenizer_model="fixture",
        )
        self.assertEqual(zh_en["bicodec_global"], zh_global)
        self.assertEqual(zh_en["target_bicodec"], [20, 21])
        self.assertEqual(en_zh["bicodec_global"], en_global)
        self.assertEqual(en_zh["target_bicodec"], [10, 11])
        self.assertFalse(zh_en["synthetic_source"])
        self.assertTrue(en_zh["synthetic_source"])

    def test_validate_rows_accepts_complete_eval_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "audio.wav"
            audio_path.touch()
            row = {
                "id": "sample.mp3",
                "pair_index": 0,
                "transcription": "hello",
                "translation": "你好",
                "source_glm": [1, 2],
                "source_bicodec": [3, 4],
                "target_bicodec": [5, 6],
                "bicodec_global": list(range(32)),
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "source_audio_path": str(audio_path),
                "reference_audio_path": str(audio_path),
            }
            validate_rows([row], expected_pairs=1, direction="fixture")


if __name__ == "__main__":
    unittest.main()
