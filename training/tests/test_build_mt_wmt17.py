import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training import build_mt_wmt17 as mt


def fake_text_encoder(text: str) -> list[int]:
    return [4000 + ord(ch) for ch in text]


class BuildMtWmt17Test(unittest.TestCase):
    def test_iter_parallel_text_filters_empty_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "train.en"
            tgt = Path(tmp) / "train.zh"
            src.write_text("hello\n\nworld\n", encoding="utf-8")
            tgt.write_text("你好\n空\n世界\n", encoding="utf-8")

            records = list(mt.iter_parallel_text(src, tgt, "eng", "cmn"))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["source_text"], "hello")
            self.assertEqual(records[0]["target_text"], "你好")
            self.assertEqual(records[1]["id"], "wmt17/3")

            tgt.write_text("你好\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(mt.iter_parallel_text(src, tgt, "eng", "cmn"))

    def test_jsonl_pairs_and_mt_sample_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "pair-1",
                        "src_lang": "eng",
                        "tgt_lang": "cmn",
                        "source_text": "hello world",
                        "target_text": "你好 世界",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            records = list(mt.iter_jsonl_pairs(path))
            samples = list(mt.convert_records_to_samples(records, fake_text_encoder))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["task"], "mt")
            self.assertEqual(samples[0]["phase"], "phase1")
            self.assertEqual(samples[0]["id"], "pair-1")

    def test_max_sample_tokens_filter(self):
        records = [
            {
                "id": "long",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "source_text": "a" * 20,
                "target_text": "b" * 20,
            }
        ]
        samples = list(mt.convert_records_to_samples(records, fake_text_encoder, max_sample_tokens=5))
        self.assertEqual(samples, [])

    def test_write_jsonl_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mt.jsonl"
            records = [
                {
                    "id": "pair-1",
                    "src_lang": "eng",
                    "tgt_lang": "cmn",
                    "source_text": "hi",
                    "target_text": "你好",
                }
            ]
            counts = mt.write_jsonl(mt.convert_records_to_samples(records, fake_text_encoder), output)
            self.assertEqual(counts["mt"], 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["task"], "mt")

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "training/build_mt_wmt17.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--source-text", result.stdout)
        self.assertIn("--input-jsonl", result.stdout)


if __name__ == "__main__":
    unittest.main()
