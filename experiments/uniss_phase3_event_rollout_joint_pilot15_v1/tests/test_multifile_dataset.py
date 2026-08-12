from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_multifile_manifest import (
    build,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MultiFilePackIndex,
)
from training.simul_uniss.jsonl_index import write_index


class MultiFileDatasetTest(unittest.TestCase):
    def _parts(self, root: Path, counts: tuple[int, ...]) -> Path:
        parts = root / "parts"
        for index, count in enumerate(counts):
            directory = parts / f"part-{index:03d}"
            directory.mkdir(parents=True)
            packed = directory / "packed.jsonl"
            offsets: list[int] = []
            with packed.open("w", encoding="utf-8") as handle:
                for record in range(count):
                    offsets.append(handle.tell())
                    handle.write(json.dumps({"part": index, "record": record}) + "\n")
            index_meta = write_index(packed, offsets)
            marker = {
                "schema_version": "uniss_dense_aligned_streaming_pack_part_v3",
                "status": "complete",
                "seq_length": 18000,
                "counts": {
                    "packed_records": count,
                    "sessions": count,
                    "annotations": count,
                },
                "index": index_meta,
                "packing_efficiency": 1.0,
            }
            (directory / "PACK_COMPLETE.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
        return parts

    def test_prefix_sum_namespace_covers_every_part_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = self._parts(root, (2, 3, 1))
            manifest = root / "manifest.json"
            build(
                parts_root=parts,
                output=manifest,
                split="train",
                expected_parts=3,
                records_per_part=None,
            )
            index = MultiFilePackIndex(manifest, expected_split="train")
            self.assertEqual(len(index), 6)
            self.assertEqual(
                [index.resolve(value) for value in range(6)],
                [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
            )
            self.assertEqual(index.resolve(-1), (2, 0))
            with self.assertRaises(IndexError):
                index.resolve(6)

    def test_manifest_rejects_missing_part(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = self._parts(root, (1, 1))
            with self.assertRaises(ValueError):
                build(
                    parts_root=parts,
                    output=root / "manifest.json",
                    split="valid",
                    expected_parts=3,
                    records_per_part=None,
                )

    def test_read_only_prefix_view_avoids_copying_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = self._parts(root, (2, 3, 4))
            manifest = root / "manifest.json"
            build(
                parts_root=parts,
                output=manifest,
                split="train",
                expected_parts=3,
                records_per_part=1,
            )
            index = MultiFilePackIndex(manifest, expected_split="train")
            self.assertEqual(len(index), 3)
            self.assertEqual(
                [index.resolve(value) for value in range(3)],
                [(0, 0), (1, 0), (2, 0)],
            )


if __name__ == "__main__":
    unittest.main()
