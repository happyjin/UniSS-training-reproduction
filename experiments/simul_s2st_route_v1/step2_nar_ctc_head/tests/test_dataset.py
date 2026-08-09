#!/usr/bin/env python3
"""Dataset filter tests for Step 2 NAR CTC."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.dataset import NarCtcJointDataset
from training.simul_uniss.jsonl_index import write_index


def write_manifest(directory: Path, rows: list[dict]) -> Path:
    path = directory / "joint.jsonl"
    offsets = []
    with path.open("wb") as handle:
        for row in rows:
            offsets.append(handle.tell())
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
    write_index(path, offsets)
    return path


class DatasetTest(unittest.TestCase):
    def test_filters_duration_degenerate_and_empty(self) -> None:
        rows = [
            {  # keep
                "id": "ok",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "translation": "hello",
                "source_duration_ms": 4000,
                "source_glm": [1, 2, 3],
                "target_bicodec": [4, 5, 6, 7],
                "target_qwen_ids": [8, 9],
                "bicodec_global": [0] * 32,
            },
            {  # too short
                "id": "short",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "translation": "x",
                "source_duration_ms": 100,
                "source_glm": [1],
                "target_bicodec": [1],
                "target_qwen_ids": [1],
                "bicodec_global": [0] * 32,
            },
            {  # degenerate: 3000 units vs 1 text token
                "id": "degen",
                "src_lang": "cmn",
                "tgt_lang": "eng",
                "translation": "y",
                "source_duration_ms": 4000,
                "source_glm": [1],
                "target_bicodec": [1] * 300,
                "target_qwen_ids": [1],
                "bicodec_global": [0] * 32,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            # Raise degenerate limit high enough that the third row still trips
            # on ratio 100 (300 required / 1 text).
            path = write_manifest(Path(directory), rows)
            # Patch offset schema if load_index expects a different schema.
            dataset = NarCtcJointDataset(
                path,
                min_audio_seconds=0.4,
                max_audio_seconds=12.0,
                degenerate_ratio_limit=100.0,
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            meta = json.loads(item["record_json"])
            self.assertEqual(meta["id"], "ok")
            self.assertEqual(int(item["target_bicodec_length"]), 4)

    def test_max_samples_caps_selection(self) -> None:
        rows = [
            {
                "id": f"r{index}",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "translation": "t",
                "source_duration_ms": 4000,
                "source_glm": [1],
                "target_bicodec": [1, 2, 3],
                "target_qwen_ids": [1, 2],
                "bicodec_global": [0] * 32,
            }
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(Path(directory), rows)
            dataset = NarCtcJointDataset(path, max_samples=3)
            self.assertEqual(len(dataset), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
