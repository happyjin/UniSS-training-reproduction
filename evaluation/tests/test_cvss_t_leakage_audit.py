from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.cvss_t.leakage_audit import audit_shard, build_references


class CvssLeakageAuditTest(unittest.TestCase):
    def test_detects_normalized_text_and_id_overlap(self) -> None:
        references = build_references(
            [
                {
                    "id": "common_voice_zh-CN_1.mp3",
                    "source_zh_text": "你好，世界！",
                    "target_en_text": "Hello, World's!",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "id": "common_voice_zh-CN_1.mp3",
                            "dataset_name": "fixture",
                            "src_lang": "cmn",
                            "tgt_lang": "eng",
                            "transcription": "你好世界",
                            "translation": "hello world's",
                        },
                        {
                            "id": "unrelated",
                            "dataset_name": "fixture",
                            "src_lang": "eng",
                            "tgt_lang": "cmn",
                            "transcription": "different",
                            "translation": "不同",
                        },
                    ]
                ),
                path,
            )
            result = audit_shard(str(path), references, max_examples=10)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["matched_record_count"], 1)
        self.assertEqual(result["id_match_count"], 1)
        self.assertEqual(result["text_match_counts"]["transcription:cmn"], 1)
        self.assertEqual(result["text_match_counts"]["translation:eng"], 1)


if __name__ == "__main__":
    unittest.main()
