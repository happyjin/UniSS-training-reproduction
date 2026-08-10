from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    assemble,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_trajectory_cache import (
    PACK_PART_SCHEMA,
)


class TrajectoryAssemblyTest(unittest.TestCase):
    def test_assembly_preserves_order_and_builds_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = root / "parts"
            lines = []
            for shard, values in enumerate(((0, 1), (2,))):
                part = parts / f"part-{shard:03d}"
                part.mkdir(parents=True)
                output = part / "packed_trajectory.jsonl"
                encoded = "".join(json.dumps({"value": value}) + "\n" for value in values)
                output.write_text(encoded, encoding="utf-8")
                lines.extend(encoded.splitlines(keepends=True))
                marker = {
                    "schema_version": PACK_PART_SCHEMA,
                    "seq_length": 18000,
                    "output": {
                        "path": str(output.resolve()),
                        "size_bytes": output.stat().st_size,
                        "mtime_ns": output.stat().st_mtime_ns,
                    },
                    "packed_records": len(values),
                    "trajectory_samples": 2 * len(values),
                    "deadline_forced": shard,
                    "supervised_tokens": 4.0 * len(values),
                }
                (part / "PACK_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")

            output = root / "packed.jsonl"
            offsets = root / "packed.offsets.u64"
            marker_path = root / "ASSEMBLY_COMPLETE.json"
            marker = assemble(
                parts,
                output,
                offsets,
                marker_path,
                shard_count=2,
                seq_length=18000,
            )
            self.assertEqual(output.read_text().splitlines(), [line.strip() for line in lines])
            self.assertEqual(
                np.fromfile(offsets, dtype="<u8").tolist(),
                [0, len(lines[0].encode()), len((lines[0] + lines[1]).encode())],
            )
            self.assertEqual(marker["packed_records"], 3)
            self.assertEqual(marker["trajectory_samples"], 6)
            self.assertEqual(int(output.with_suffix(".jsonl.count").read_text()), 3)


if __name__ == "__main__":
    unittest.main()
