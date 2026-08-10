from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset import (
    build_subset,
)


class OffsetSubsetTest(unittest.TestCase):
    def test_trajectory_subset_keeps_source_fingerprint_and_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = root / "packed.jsonl"
            packed.write_bytes(b"{}\n" * 10)
            source = root / "source.u64"
            np.arange(0, 30, 3, dtype="<u8").tofile(source)
            stat = packed.stat()
            source.with_suffix(".u64.json").write_text(
                json.dumps(
                    {
                        "schema_version": OFFSET_SCHEMA,
                        "source": {
                            "path": str(packed.resolve()),
                            "size_bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        },
                        "records": 10,
                    }
                )
            )
            output = root / "subset.u64"
            metadata = build_subset(
                kind="trajectory",
                packed=packed,
                source_offsets=source,
                output_offsets=output,
                records=4,
            )
            self.assertEqual(np.fromfile(output, dtype="<u8").tolist(), [0, 9, 18, 27])
            self.assertEqual(metadata["records"], 4)
            self.assertEqual(metadata["subset"]["last_source_index"], 9)

    def test_replay_subset_is_explicitly_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = root / "packed.jsonl"
            packed.write_bytes(b"{}\n" * 5)
            source = root / "source.u64"
            np.arange(0, 15, 3, dtype="<u8").tofile(source)
            stat = packed.stat()
            source.with_suffix(".u64.json").write_text(
                json.dumps(
                    {
                        "schema_version": "uniss_phase3_replay_offsets_v1",
                        "source": str(packed.resolve()),
                        "source_size_bytes": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "records": 5,
                        "complete": True,
                    }
                )
            )
            output = root / "subset.u64"
            metadata = build_subset(
                kind="replay",
                packed=packed,
                source_offsets=source,
                output_offsets=output,
                records=3,
            )
            self.assertEqual(np.fromfile(output, dtype="<u8").tolist(), [0, 6, 12])
            self.assertFalse(metadata["complete"])
            self.assertEqual(metadata["records"], 3)


if __name__ == "__main__":
    unittest.main()
