"""Stage A v2 dataset adapter for immutable same-prefix teacher bundles."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training import (
    dataset as v1,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.build_teacher_cache import (
    CACHE_SCHEMA,
    sha256,
)


_V1_COLLATE_STAGE_A = v1.collate_stage_a


class BundleLRU:
    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("teacher bundle LRU capacity must be positive")
        self.capacity = int(capacity)
        self._values: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def load(self, path: Path) -> dict[str, np.ndarray]:
        path = path.resolve()
        cached = self._values.pop(path, None)
        if cached is None:
            with np.load(path, allow_pickle=False) as value:
                cached = {name: value[name].copy() for name in value.files}
            schema = str(cached["bundle_schema"].reshape(-1)[0])
            if schema != CACHE_SCHEMA:
                raise ValueError(f"unexpected teacher bundle schema: {schema}")
        self._values[path] = cached
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return cached


class TeacherCache:
    def __init__(
        self,
        manifest: str | Path,
        *,
        packs: str | Path,
        expected_packs: int,
        lru_capacity: int = 16,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        audit_path = self.manifest.parent / "TEACHER_CACHE_AUDIT.json"
        self.audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if self.audit.get("schema_version") != CACHE_SCHEMA or self.audit.get("status") != "complete":
            raise ValueError("same-prefix teacher cache audit did not pass")
        if Path(str(self.audit["packs"])).resolve() != Path(packs).resolve():
            raise ValueError("teacher cache belongs to a different Stage A pack file")
        if Path(str(self.audit.get("output", ""))).resolve() != self.manifest:
            raise ValueError("teacher cache audit names a different manifest")
        if self.audit.get("output_sha256") != sha256(self.manifest):
            raise ValueError("teacher cache manifest digest differs from its audit")
        if int(self.audit["total_packs"]) != int(expected_packs):
            raise ValueError("teacher cache pack coverage is incomplete")
        if self.audit.get("speaker_source") != "stage_a_pack_prompt":
            raise ValueError("teacher cache speaker did not come from the Stage A pack")
        if int(self.audit.get("topk", 0)) != 32:
            raise ValueError("Stage A v2 requires a top-32 teacher cache")
        if float(self.audit.get("temperature", 0.0)) != 1.5:
            raise ValueError("Stage A v2 requires teacher temperature 1.5")
        if not bool(self.audit.get("require_reference_in_topk")):
            raise ValueError("teacher cache did not require reference support")
        if float(self.audit.get("reference_anchor", 0.0)) != 0.5:
            raise ValueError("Stage A v2 requires a 0.5 teacher reference anchor")
        if not Path(str(self.audit.get("model", ""))).is_dir():
            raise ValueError("teacher cache Phase3 model path is unavailable")
        self.rows: dict[tuple[int, int], dict[str, object]] = {}
        with self.manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["pack_index"]), int(row["acoustic_index"]))
                if key in self.rows:
                    raise ValueError(f"duplicate teacher cache key: {key}")
                self.rows[key] = row
        if len(self.rows) != int(self.audit["records"]):
            raise ValueError("teacher cache record count differs from its audit")
        self.lru_capacity = int(lru_capacity)
        self._lru: BundleLRU | None = None

    @property
    def lru(self) -> BundleLRU:
        if self._lru is None:
            self._lru = BundleLRU(self.lru_capacity)
        return self._lru

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_lru"] = None
        return value

    def load(self, pack_index: int, acoustic_index: int) -> dict[str, object]:
        key = (int(pack_index), int(acoustic_index))
        row = self.rows.get(key)
        if row is None:
            raise KeyError(f"missing same-prefix teacher cache row: {key}")
        bundle = self.lru.load(Path(str(row["bundle_path"])))
        prefix = f"row_{int(row['bundle_row'])}"
        result = dict(row)
        result.update(
            {
                "teacher_positions": torch.from_numpy(
                    bundle[f"{prefix}_positions"].astype(np.int64, copy=False)
                ),
                "teacher_labels": torch.from_numpy(
                    bundle[f"{prefix}_labels"].astype(np.int64, copy=False)
                ),
                "teacher_indices": torch.from_numpy(
                    bundle[f"{prefix}_indices"].astype(np.int64, copy=False)
                ),
                "teacher_probabilities": torch.from_numpy(
                    bundle[f"{prefix}_probabilities"].astype(np.float32, copy=False)
                ),
            }
        )
        return result


class IndexedStageAPackDataset(v1.IndexedStageAPackDataset):
    def __init__(
        self,
        path: str | Path,
        *,
        seq_length: int,
        max_acoustics_per_pack: int,
        teacher_cache_manifest: str | Path,
        teacher_lru_capacity: int = 16,
        load_audio: bool = True,
    ) -> None:
        super().__init__(
            path,
            seq_length=seq_length,
            max_acoustics_per_pack=max_acoustics_per_pack,
            load_audio=load_audio,
        )
        self.teacher_cache = TeacherCache(
            teacher_cache_manifest,
            packs=self.path,
            expected_packs=len(self),
            lru_capacity=teacher_lru_capacity,
        )

    def get_for_epoch(self, index: int, epoch: int) -> dict[str, object]:
        result = super().get_for_epoch(index, epoch)
        raw = self._read(index)
        selected = v1.rotated_acoustic_indices(
            len(raw.get("acoustics", [])),
            self.max_acoustics_per_pack,
            epoch,
            index,
        )
        acoustics = result["acoustics"]
        if not isinstance(acoustics, list) or len(acoustics) != len(selected):
            raise ValueError("teacher cache acoustic selection differs from Stage A")
        for acoustic, acoustic_index in zip(acoustics, selected):
            teacher = self.teacher_cache.load(index, acoustic_index)
            if str(teacher["sample_id"]) != str(acoustic["sample_id"]):
                raise ValueError("teacher cache sample ID differs")
            if str(teacher["task"]) != str(acoustic["task"]):
                raise ValueError("teacher cache task differs")
            positions = teacher["teacher_positions"]
            labels = teacher["teacher_labels"]
            indices = teacher["teacher_indices"]
            probabilities = teacher["teacher_probabilities"]
            if not all(isinstance(value, torch.Tensor) for value in (positions, labels, indices, probabilities)):
                raise TypeError("teacher cache tensors are missing")
            if positions.numel():
                if int(positions.min()) < 0 or int(positions.max()) >= self.seq_length:
                    raise ValueError("teacher position exceeds Stage A pack")
                if not bool((result["loss_mask"][positions] > 0).all()):
                    raise ValueError("teacher cache selected an inactive Stage A token")
                if not torch.equal(result["labels"][positions].long(), labels.long()):
                    raise ValueError("teacher cache reference labels differ from Stage A")
            acoustic.update(teacher)
        return result


def collate_stage_a(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    result = _V1_COLLATE_STAGE_A(batch)
    teacher_batch: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    widths: set[int] = set()
    for batch_row, value in enumerate(batch):
        for acoustic in value["acoustics"]:  # type: ignore[index]
            current_positions = acoustic["teacher_positions"]
            current_labels = acoustic["teacher_labels"]
            current_indices = acoustic["teacher_indices"]
            current_probabilities = acoustic["teacher_probabilities"]
            if not all(
                isinstance(current, torch.Tensor)
                for current in (
                    current_positions,
                    current_labels,
                    current_indices,
                    current_probabilities,
                )
            ):
                raise TypeError("Stage A teacher acoustic tensors are missing")
            if current_indices.ndim != 2 or current_probabilities.shape != current_indices.shape:
                raise ValueError("Stage A teacher top-k geometry differs")
            if len(current_positions) != len(current_indices) or len(current_labels) != len(current_indices):
                raise ValueError("Stage A teacher position geometry differs")
            widths.add(int(current_indices.shape[1]))
            if not len(current_positions):
                continue
            teacher_batch.append(torch.full_like(current_positions, batch_row))
            positions.append(current_positions)
            labels.append(current_labels)
            indices.append(current_indices)
            probabilities.append(current_probabilities)
    if len(widths) != 1 or not positions:
        raise ValueError("same-prefix teacher denominator is zero or top-k width differs")
    teacher_indices = torch.cat(indices)
    result.update(
        {
            "teacher_batch": torch.cat(teacher_batch),
            "teacher_positions": torch.cat(positions),
            "teacher_reference_labels": torch.cat(labels),
            "teacher_indices": teacher_indices,
            "teacher_probabilities": torch.cat(probabilities),
            "teacher_mask": torch.ones_like(teacher_indices, dtype=torch.bool),
        }
    )
    return result


class ThreeEpochStageASchedule(v1.ThreeEpochStageASchedule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_stage_a


class PaddedStageAValidationDataset(v1.PaddedStageAValidationDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_stage_a


__all__ = [
    "BundleLRU",
    "IndexedStageAPackDataset",
    "PaddedStageAValidationDataset",
    "TeacherCache",
    "ThreeEpochStageASchedule",
    "collate_stage_a",
]
