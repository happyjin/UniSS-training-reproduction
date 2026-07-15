import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training import prepare_unist_s2st as prep


def fake_text_encoder(text: str) -> list[int]:
    return [2000 + ord(ch) for ch in text]


class PrepareUniSTS2STTest(unittest.TestCase):
    def _write_parquet(self, path: Path) -> None:
        table = pa.table(
            {
                "id": ["row-1", "row-2"],
                "transcription": ["hi", "ok"],
                "translation": ["你好", "好的"],
                "source_glm": [[0, 1], [2, 3]],
                "source_bicodec": [[0, 1, 2], [3, 4, 5]],
                "target_glm": [[10], [11]],
                "target_bicodec": [[6, 7], [8, 9]],
                "bicodec_global": [list(range(32)), list(range(31, -1, -1))],
                "dataset_name": ["unit", "unit"],
                "src_lang": ["eng", "eng"],
                "tgt_lang": ["cmn", "cmn"],
                "split": ["dev", "dev"],
                "source_glm_len": [2, 2],
                "target_glm_len": [1, 1],
                "source_bicodec_len": [3, 3],
                "target_bicodec_len": [2, 2],
                "duration_ratio": [1.0, 1.1],
            }
        )
        pq.write_table(table, path)

    def test_phase2_and_phase3_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "sample.parquet"
            self._write_parquet(parquet_path)

            paths = prep.expand_input_paths([str(parquet_path)])
            records = list(prep.iter_unist_records(paths, limit_records=1))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], "row-1")

            phase2 = list(
                prep.convert_records_to_samples(records, fake_text_encoder, phase="phase2")
            )
            self.assertEqual([sample["task"] for sample in phase2], ["quality", "performance", "direct_s2st"])
            self.assertEqual(phase2[0]["phase"], "phase2")
            self.assertGreater(phase2[0]["prompt_length"], 0)
            self.assertGreater(phase2[0]["target_length"], 0)

            records = list(prep.iter_unist_records(paths, limit_records=1))
            phase3 = list(
                prep.convert_records_to_samples(records, fake_text_encoder, phase="phase3")
            )
            self.assertEqual([sample["task"] for sample in phase3], ["quality", "performance"])

    def test_write_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "sample.parquet"
            output_path = Path(tmp) / "out" / "samples.jsonl"
            self._write_parquet(parquet_path)
            records = prep.iter_unist_records([parquet_path], limit_records=1)
            samples = prep.convert_records_to_samples(records, fake_text_encoder, phase="phase2")
            counts = prep.write_jsonl(samples, output_path)
            self.assertEqual(counts["total"], 3)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            first = json.loads(lines[0])
            self.assertEqual(first["task"], "quality")
            self.assertIn("quality_semantic", first["segment_spans"])

    def test_real_unist_small_split_smoke(self):
        path = Path("data/raw/UniST/clean_dev-00000.parquet")
        if not path.exists():
            self.skipTest("UniST clean_dev parquet is not downloaded")
        records = list(prep.iter_unist_records([path], limit_records=1))
        self.assertEqual(len(records), 1)
        samples = list(prep.convert_records_to_samples(records, fake_text_encoder, "phase2"))
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["task"], "quality")


if __name__ == "__main__":
    unittest.main()
