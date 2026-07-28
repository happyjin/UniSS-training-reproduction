from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.cvss_t.records import cvss_record_map


class CvssRecordsTest(unittest.TestCase):
    def test_loader_preserves_official_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            reference = root / "reference.wav"
            source.touch()
            reference.touch()
            parquet_path = root / "data.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "id": "sample",
                            "transcription": "hello",
                            "translation": "你好",
                            "source_glm": [1],
                            "source_bicodec": [2],
                            "target_bicodec": [3],
                            "bicodec_global": list(range(32)),
                            "src_lang": "eng",
                            "tgt_lang": "cmn",
                            "source_audio_path": str(source),
                            "reference_audio_path": str(reference),
                            "synthetic_source": True,
                            "synthetic_reference": False,
                        }
                    ]
                ),
                parquet_path,
            )
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"id": "sample", "parquet_path": str(parquet_path), "row_index": 0}) + "\n",
                encoding="utf-8",
            )
            records = cvss_record_map(manifest_path)
        self.assertEqual(records["sample"]["source_audio_path"], str(source))
        self.assertEqual(records["sample"]["reference_audio_path"], str(reference))
        self.assertTrue(records["sample"]["synthetic_source"])


if __name__ == "__main__":
    unittest.main()
