"""Random access over many immutable trajectory-pack parts.

The historical dense-aligned data were assembled into a 51 GB monolith, but
the authoritative pack parts and their uint64 indexes already exist.  This
module gives those parts one global pack-ID namespace without copying them.
Every global ID resolves to exactly one complete 18k pack; the existing
event-rollout dataset then preserves every session and its internal event
order.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from torch.utils.data import Dataset

from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    IndexedEventRolloutTrajectoryDataset,
)
from training.simul_uniss.jsonl_index import load_index


MANIFEST_SCHEMA = "uniss_multifile_trajectory_manifest_v1"
T = TypeVar("T")


@dataclass(frozen=True)
class PackPart:
    part_id: str
    packed: Path
    offsets: Path
    marker: Path
    records: int
    global_start: int
    global_end: int

    def __post_init__(self) -> None:
        if not self.part_id:
            raise ValueError("pack part ID is empty")
        if self.records <= 0:
            raise ValueError(f"pack part {self.part_id} has no records")
        if self.global_start < 0 or self.global_end != self.global_start + self.records:
            raise ValueError(f"pack part {self.part_id} has invalid global bounds")
        for path in (self.packed, self.offsets, self.marker):
            if not path.is_file():
                raise FileNotFoundError(path)


class MultiFilePackIndex:
    """Validated prefix-sum mapping from global pack IDs to local part IDs."""

    def __init__(self, manifest: str | Path, *, expected_split: str | None = None) -> None:
        self.manifest = Path(manifest).resolve()
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("unexpected multi-file trajectory manifest schema")
        if value.get("status") != "complete":
            raise ValueError("multi-file trajectory manifest is incomplete")
        split = str(value.get("split", ""))
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"trajectory split {split!r} differs from expected {expected_split!r}"
            )
        if int(value.get("seq_length", -1)) != 18_000:
            raise ValueError("formal trajectory parts must use seq_length=18000")
        raw_parts = value.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError("multi-file trajectory manifest contains no parts")

        parts: list[PackPart] = []
        cursor = 0
        seen_ids: set[str] = set()
        for raw in raw_parts:
            item = dict(raw)
            part_id = str(item["part_id"])
            if part_id in seen_ids:
                raise ValueError(f"duplicate pack part ID: {part_id}")
            seen_ids.add(part_id)
            packed = self._resolve(str(item["packed"]))
            offsets = self._resolve(str(item["offsets"]))
            marker = self._resolve(str(item["marker"]))
            records = int(item["records"])
            start = int(item["global_start"])
            end = int(item["global_end"])
            if start != cursor:
                raise ValueError(
                    f"pack part {part_id} starts at {start}, expected contiguous {cursor}"
                )
            part = PackPart(part_id, packed, offsets, marker, records, start, end)
            offsets_values = load_index(packed)
            if offsets_values is None or len(offsets_values) != records:
                raise ValueError(
                    f"pack part {part_id} index has "
                    f"{0 if offsets_values is None else len(offsets_values)} records, "
                    f"expected {records}"
                )
            if Path(str(item["offsets"])).name != offsets.name:
                raise ValueError(f"pack part {part_id} offset path is not canonical")
            parts.append(part)
            cursor = end

        if cursor != int(value.get("total_records", -1)):
            raise ValueError("multi-file prefix sums differ from total_records")
        if len(parts) != int(value.get("part_count", -1)):
            raise ValueError("multi-file part count differs from manifest")
        self.split = split
        self.seq_length = int(value["seq_length"])
        self.parts = tuple(parts)
        self._ends = tuple(part.global_end for part in parts)
        self.total_records = cursor

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.manifest.parent / path
        return path.resolve()

    def __len__(self) -> int:
        return self.total_records

    def resolve(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part_index = bisect.bisect_right(self._ends, index)
        part = self.parts[part_index]
        return part_index, index - part.global_start


class MultiFileIndexedDataset(Dataset[T], Generic[T]):
    """Generic global namespace backed by one indexed dataset per part."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        factory: Callable[[Path], Dataset[T]],
        expected_split: str | None = None,
    ) -> None:
        self.index = MultiFilePackIndex(manifest, expected_split=expected_split)
        self.datasets = tuple(factory(part.packed) for part in self.index.parts)
        for part, dataset in zip(self.index.parts, self.datasets):
            if len(dataset) != part.records:
                raise ValueError(
                    f"dataset length for {part.part_id} differs from frozen manifest"
                )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> T:
        part_index, local_index = self.index.resolve(index)
        return self.datasets[part_index][local_index]


class MultiFileIndexedEventRolloutTrajectoryDataset(MultiFileIndexedDataset[dict[str, object]]):
    """Exact-runtime trajectory records addressed by one global pack ID."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        seq_length: int,
        expected_split: str | None = None,
    ) -> None:
        if int(seq_length) != 18_000:
            raise ValueError("formal pilot15 event-rollout requires seq_length=18000")
        super().__init__(
            manifest,
            factory=lambda path: IndexedEventRolloutTrajectoryDataset(
                path, seq_length=seq_length
            ),
            expected_split=expected_split,
        )


__all__ = [
    "MANIFEST_SCHEMA",
    "MultiFileIndexedDataset",
    "MultiFileIndexedEventRolloutTrajectoryDataset",
    "MultiFilePackIndex",
    "PackPart",
]

