from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.dataset import (
    DeterministicReplayTrajectorySchedule,
    IndexedTrajectoryDataset,
    collate_trajectory,
)


ROOT = Path(__file__).resolve().parents[3]
SMOKE_PACK = (
    ROOT
    / "data/processed/uniss_phase3_true_subsecond_deadline_full198_v1/smoke/trajectory_pack_soft_kd_v1/packed_trajectory.jsonl"
)


def _indexed_smoke(root: Path) -> tuple[Path, Path]:
    packed = root / "packed.jsonl"
    offsets = root / "packed.offsets.u64"
    source = SMOKE_PACK.read_bytes()
    packed.write_bytes(source)
    byte_offsets = []
    offset = 0
    for line in source.splitlines(keepends=True):
        if line.strip():
            byte_offsets.append(offset)
        offset += len(line)
    import numpy as np

    np.asarray(byte_offsets, dtype="<u8").tofile(offsets)
    stat = packed.stat()
    metadata = {
        "schema_version": OFFSET_SCHEMA,
        "source": {
            "path": str(packed.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "offsets": {},
        "dtype": "uint64-little-endian",
        "records": len(byte_offsets),
    }
    offsets.with_suffix(offsets.suffix + ".json").write_text(json.dumps(metadata))
    return packed, offsets


class TrainingDatasetTest(unittest.TestCase):
    def test_real_smoke_cache_resolves_and_collates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            packed, offsets = _indexed_smoke(Path(directory))
            dataset = IndexedTrajectoryDataset(packed, offsets, seq_length=18_000)
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["sample_kind"], "trajectory")
            self.assertEqual(len(item["annotations"]), 4)
            batch = collate_trajectory([item])
            self.assertEqual(tuple(batch["tokens"].shape), (1, 18_000))
            self.assertEqual(len(batch["action_position"]), 4)
            self.assertGreater(int(batch["frontend_code"].numel()), 0)
            self.assertEqual(int(batch["teacher_indices"].shape[1]), 4)
            self.assertEqual(int(batch["teacher_indices"].shape[-1]), 32)
            self.assertGreater(int(batch["kd_position"].numel()), 0)
            self.assertEqual(
                int(batch["kd_position"].numel()),
                int(batch["kd_target_index"].numel()),
            )

    def test_schedule_keeps_each_dp_group_homogeneous(self) -> None:
        class Fake:
            def __init__(self, kind: str, length: int):
                self.kind, self.length = kind, length

            def __len__(self):
                return self.length

            def __getitem__(self, index):
                return {"sample_kind": self.kind, "index": index}

        schedule = DeterministicReplayTrajectorySchedule(
            Fake("trajectory", 31),
            Fake("replay", 23),
            total_samples=160,
            data_parallel_group_size=16,
        )
        self.assertEqual(len(schedule), 160)
        for start in range(0, len(schedule), 16):
            kinds = {schedule[index]["sample_kind"] for index in range(start, start + 16)}
            self.assertEqual(len(kinds), 1)
        self.assertGreater(schedule.replay_groups, 0)
        self.assertGreater(schedule.trajectory_groups, 0)


if __name__ == "__main__":
    unittest.main()
