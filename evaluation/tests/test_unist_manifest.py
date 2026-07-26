import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation import unist_manifest
from evaluation.io_utils import iter_jsonl


class UniSTManifestTest(unittest.TestCase):
    def test_create_deterministic_stratified_manifests(self):
        rows = []
        for index in range(12):
            src_lang, tgt_lang = ("eng", "cmn") if index % 2 == 0 else ("cmn", "eng")
            rows.append(
                {
                    "id": f"sample-{index}",
                    "dataset_name": f"dataset-{index % 3}",
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "transcription": f"source {index}",
                    "translation": f"target {index}",
                    "source_glm": [1, 2],
                    "source_bicodec": [3, 4, 5],
                    "target_bicodec": [6, 7],
                    "bicodec_global": list(range(32)),
                    "duration_ratio": 1.0,
                }
            )
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_path = root / "dev.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet_path)
            first = root / "first"
            second = root / "second"
            summary = unist_manifest.create_manifests(
                parquet_path,
                first,
                split_name="dev",
                seed=17,
                smoke_count=3,
                listen_count=6,
                repo_root=root,
            )
            unist_manifest.create_manifests(
                parquet_path,
                second,
                split_name="dev",
                seed=17,
                smoke_count=3,
                listen_count=6,
                repo_root=root,
            )
            smoke = list(iter_jsonl(first / "unist_dev_smoke_3.jsonl"))
            smoke_again = list(iter_jsonl(second / "unist_dev_smoke_3.jsonl"))

        self.assertEqual(summary["all"]["count"], 12)
        self.assertEqual(smoke, smoke_again)
        self.assertEqual(len(smoke), 3)
        self.assertEqual({row["src_lang"] for row in smoke}, {"eng", "cmn"})
        self.assertTrue(all(row["bicodec_global_length"] == 32 for row in smoke))


if __name__ == "__main__":
    unittest.main()
