from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_dev_direction_index import (
    build as build_dev_index,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_schedule import (
    build as build_schedule,
)


class DevValidationDataTest(unittest.TestCase):
    def test_direction_stratified_parts_cover_dev_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dev-00000.parquet"
            rows = 12
            table = pa.Table.from_pydict(
                {
                    "id": [f"sample-{index}" for index in range(rows)],
                    "transcription": ["source"] * rows,
                    "translation": ["target"] * rows,
                    "source_glm": [list(range(20))] * rows,
                    "source_bicodec": [list(range(80))] * rows,
                    "target_bicodec": [list(range(32))] * rows,
                    "bicodec_global": [list(range(32))] * rows,
                    "src_lang": ["eng"] * 6 + ["cmn"] * 6,
                    "tgt_lang": ["cmn"] * 6 + ["eng"] * 6,
                }
            )
            pq.write_table(table, source)
            index_root = root / "index"
            index = build_dev_index(source, index_root, 3)
            self.assertEqual(index["accepted"], rows)
            self.assertEqual([part["eng"] for part in index["shards"]], [2, 2, 2])
            self.assertEqual([part["cmn"] for part in index["shards"]], [2, 2, 2])
            represented = []
            for part in range(3):
                for lang in ("eng", "cmn"):
                    represented.extend(
                        np.load(index_root / f"part-{part:03d}.{lang}.npy").tolist()
                    )
            self.assertEqual(sorted(represented), list(range(rows)))

            plan_root = root / "plan"
            summary = build_schedule(
                index_root / "index.json",
                plan_root,
                1,
                shard_count=3,
                index_template="part-{shard:03d}.{lang}.npy",
            )
            self.assertEqual(summary["accepted_rows"], rows)
            self.assertEqual(summary["trajectory_count"], rows * 2)
            first = json.loads(
                (plan_root / "part-000" / "trajectory_plan.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn(first["src_lang"], ("eng", "cmn"))
            self.assertIn(first["trajectory_kind"], ("early", "middle_late"))


if __name__ == "__main__":
    unittest.main()
