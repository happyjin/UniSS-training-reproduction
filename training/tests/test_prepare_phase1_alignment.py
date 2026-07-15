import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training import prepare_phase1_alignment as prep


def fake_text_encoder(text: str) -> list[int]:
    return [3000 + ord(ch) for ch in text]


class PreparePhase1AlignmentTest(unittest.TestCase):
    def _write_parquet(self, path: Path, include_source_bicodec: bool = True) -> None:
        columns = {
            "id": ["row-1", "row-2"],
            "transcription": ["hello", "test"],
            "translation": ["你好", "测试"],
            "source_glm": [[0, 1], [2, 3]],
            "bicodec_global": [list(range(32)), list(range(31, -1, -1))],
            "dataset_name": ["unit", "unit"],
            "src_lang": ["eng", "eng"],
            "tgt_lang": ["cmn", "cmn"],
            "split": ["train", "train"],
            "duration_ratio": [1.0, 0.9],
        }
        if include_source_bicodec:
            columns["source_bicodec"] = [[4, 5], [6, 7]]
        pq.write_table(pa.table(columns), path)

    def test_convert_phase1_tasks_from_unist_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "sample.parquet"
            self._write_parquet(parquet_path)

            records = list(prep.iter_alignment_records([parquet_path], limit_records=1))
            self.assertEqual(records[0]["id"], "row-1")

            samples = list(prep.convert_records_to_samples(records, fake_text_encoder))
            self.assertEqual([sample["task"] for sample in samples], ["asr", "s2tt", "tts"])
            self.assertEqual({sample["phase"] for sample in samples}, {"phase1"})
            self.assertGreater(samples[0]["prompt_length"], 0)
            self.assertGreater(samples[0]["target_length"], 0)

    def test_selected_tasks_can_skip_tts_when_source_bicodec_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "sample.parquet"
            self._write_parquet(parquet_path, include_source_bicodec=False)
            records = list(prep.iter_alignment_records([parquet_path], limit_records=1))

            samples = list(prep.convert_records_to_samples(records, fake_text_encoder, tasks=["asr", "s2tt"]))
            self.assertEqual([sample["task"] for sample in samples], ["asr", "s2tt"])

            with self.assertRaises(KeyError):
                list(prep.convert_records_to_samples(records, fake_text_encoder, tasks=["tts"]))

    def test_write_jsonl_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "sample.parquet"
            output_path = Path(tmp) / "phase1.jsonl"
            self._write_parquet(parquet_path)
            records = prep.iter_alignment_records([parquet_path], limit_records=1)
            samples = prep.convert_records_to_samples(records, fake_text_encoder, tasks=["asr"])

            counts = prep.write_jsonl(samples, output_path)
            self.assertEqual(counts["asr"], 1)
            first = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(first["task"], "asr")
            self.assertEqual(first["phase"], "phase1")

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "training/prepare_phase1_alignment.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--tasks", result.stdout)


if __name__ == "__main__":
    unittest.main()
