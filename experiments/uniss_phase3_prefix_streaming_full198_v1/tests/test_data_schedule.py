from __future__ import annotations

import bisect
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.uniss_phase3_prefix_streaming_full198_v1.data import (
    Full198CurriculumDataset,
    REQUIRED_COLUMNS,
    UniSTDevDataset,
)


class ValidationScheduleTest(unittest.TestCase):
    def test_formal_block_size_balances_each_global_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = [
                {
                    "file": str(root / "unused.parquet"),
                    "rows": 256,
                    "eng": 128,
                    "cmn": 128,
                    "eng_index": str(root / "unused.eng.npy"),
                    "cmn_index": str(root / "unused.cmn.npy"),
                }
                for _ in range(198)
            ]
            metadata = {
                "schema_version": "uniss_phase3_prefix_streaming_direction_index_v3",
                "shards": shards,
                "rows": 198 * 256,
                "eng": 198 * 128,
                "cmn": 198 * 128,
            }
            index = root / "index.json"
            index.write_text(json.dumps(metadata), encoding="utf-8")
            dataset = Full198CurriculumDataset(index, root, block_size=64)
            directions = []
            for sample_index in range(128):
                block_index = bisect.bisect_right(dataset.ends, sample_index)
                directions.append(dataset.blocks[block_index].direction)
            self.assertEqual(directions.count("eng"), 64)
            self.assertEqual(directions.count("cmn"), 64)

    def test_sorted_validation_rows_are_direction_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.parquet"
            rows = []
            for index, language in enumerate(["cmn"] * 8 + ["eng"] * 4):
                rows.append(
                    {
                        "id": str(index),
                        "transcription": "a",
                        "translation": "b",
                        "source_glm": [1],
                        "target_bicodec": [2, 3],
                        "bicodec_global": [3] * 32,
                        "src_lang": language,
                        "tgt_lang": "eng" if language == "cmn" else "cmn",
                    }
                )
            table = pa.Table.from_pylist(rows).select(REQUIRED_COLUMNS)
            pq.write_table(table, path)
            dataset = UniSTDevDataset(path, directory, limit=8)
            directions = [
                dataset.table.slice(int(row), 1).column("src_lang")[0].as_py()
                for row in dataset.row_indices
            ]
            self.assertEqual(directions, ["eng", "cmn"] * 4)


if __name__ == "__main__":
    unittest.main()
