"""Random-access 18k task-family datasets and Megatron-safe collation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools import (
    BUILD_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TopKTeacherCacheReader,
    resolve_teacher_bindings,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    PACKED_TASK_SCHEMA,
    validate_packed_task,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    TASK_FAMILIES,
)
from training.simul_uniss.dataset import boundaries_to_cu_seqlens
from training.simul_uniss.jsonl_index import load_index


AudioLoader = Callable[[Path], tuple[torch.Tensor, int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_audio_loader(path: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    return torchaudio.load(str(path))


def _tensor(
    value: Mapping[str, object], key: str, length: int, dtype: torch.dtype
) -> torch.Tensor:
    raw = value.get(key)
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"E2E packed {key} must contain {length} values")
    return torch.tensor(raw, dtype=dtype)


def packed_task_to_runtime_item(
    value: Mapping[str, object],
    *,
    seq_length: int,
    load_audio: bool,
    audio_loader: AudioLoader,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("E2E packed task must be a JSON object")
    validate_packed_task(value, seq_length=seq_length)
    if value.get("family") not in TASK_FAMILIES:
        raise ValueError("E2E packed task has an unknown family")
    boundaries = value.get("sample_boundaries")
    if not isinstance(boundaries, list):
        raise TypeError("E2E packed sample boundaries are malformed")
    cu_seqlens, max_seqlen = boundaries_to_cu_seqlens(
        boundaries, seq_length
    )
    acoustic_rows: list[dict[str, object]] = []
    raw_acoustic = value.get("acoustic_rows")
    if not isinstance(raw_acoustic, list):
        raise TypeError("E2E packed acoustic rows are malformed")
    for raw in raw_acoustic:
        if not isinstance(raw, dict):
            raise TypeError("E2E packed acoustic row is not an object")
        row = dict(raw)
        positions = torch.tensor(row["packed_positions"], dtype=torch.long)
        source_indices = torch.tensor(row["source_indices"], dtype=torch.long)
        source_glm_length = int(row["source_glm_length"])
        if (
            len(positions) != source_glm_length
            or len(source_indices) != source_glm_length
            or source_indices.tolist() != list(range(source_glm_length))
        ):
            raise ValueError("E2E runtime acoustic row does not cover source GLM")
        row["packed_positions"] = positions
        row["source_indices"] = source_indices
        if load_audio:
            path = Path(str(row["source_audio"]))
            waveform, sample_rate = audio_loader(path)
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if waveform.ndim == 2 and waveform.shape[0] == 1:
                waveform = waveform[0]
            if waveform.ndim != 1 or int(sample_rate) != 16_000:
                raise ValueError("E2E source audio must be mono 16 kHz")
            if not torch.isfinite(waveform).all() or waveform.numel() <= 0:
                raise ValueError("E2E source audio is empty or contains NaN/Inf")
            row["waveform"] = waveform.contiguous()
            row["waveform_length"] = int(waveform.numel())
        acoustic_rows.append(row)
    raw_bindings = value.get("teacher_bindings")
    if not isinstance(raw_bindings, list) or any(
        not isinstance(binding, dict) for binding in raw_bindings
    ):
        raise TypeError("E2E packed teacher bindings are malformed")
    return {
        "family": str(value["family"]),
        "tokens": _tensor(value, "tokens", seq_length, torch.long),
        "labels": _tensor(value, "labels", seq_length, torch.long),
        "loss_kinds": _tensor(value, "loss_kinds", seq_length, torch.long),
        "loss_mask": _tensor(value, "loss_mask", seq_length, torch.float32),
        "position_ids": _tensor(value, "position_ids", seq_length, torch.long),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "sample_boundaries": [tuple(int(item) for item in row) for row in boundaries],
        "source_ids": [str(item) for item in value["source_ids"]],
        "sequence_ids": [str(item) for item in value["sequence_ids"]],
        "source_manifest_records": torch.tensor(
            value["source_manifest_records"], dtype=torch.long
        ),
        "acoustic_rows": acoustic_rows,
        "teacher_bindings": [dict(binding) for binding in raw_bindings],
        "used_tokens": int(value["used_tokens"]),
        "supervised_tokens": int(value["supervised_tokens"]),
    }


class E2EPackedFamilyDataset(Dataset[dict[str, object]]):
    """Fork-safe random access to one immutable homogeneous task family."""

    def __init__(
        self,
        path: str | Path,
        *,
        family: str,
        seq_length: int = 18_000,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
        verify_sha256: bool = False,
        load_audio: bool = True,
        audio_loader: AudioLoader | None = None,
        teacher_readers: Mapping[str, TopKTeacherCacheReader] | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.family = str(family)
        self.seq_length = int(seq_length)
        self.load_audio = bool(load_audio)
        self.audio_loader = audio_loader or _default_audio_loader
        self.teacher_readers = dict(teacher_readers or {})
        if self.family not in TASK_FAMILIES:
            raise ValueError("unknown E2E task family")
        if self.seq_length != 18_000:
            raise ValueError("formal E2E packed datasets require seq-length 18000")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if expected_bytes is not None and self.path.stat().st_size != int(
            expected_bytes
        ):
            raise ValueError("E2E packed dataset byte count changed")
        if verify_sha256:
            if not expected_sha256 or _sha256(self.path) != expected_sha256:
                raise ValueError("E2E packed dataset SHA256 changed")
        self.offsets: Sequence[int] | None = load_index(self.path)
        if self.offsets is None or not self.offsets:
            raise ValueError("E2E packed dataset is missing its offset index")

    @classmethod
    def from_build_report(
        cls,
        report_path: str | Path,
        family: str,
        *,
        verify_sha256: bool = False,
        load_audio: bool = True,
        audio_loader: AudioLoader | None = None,
        teacher_readers: Mapping[str, TopKTeacherCacheReader] | None = None,
    ) -> "E2EPackedFamilyDataset":
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != BUILD_SCHEMA
            or report.get("status") != "passed"
            or int(report.get("seq_length", -1)) != 18_000
        ):
            raise ValueError("E2E task-pool build report is not a passed 18k build")
        families = report.get("families")
        if not isinstance(families, dict) or family not in families:
            raise ValueError("E2E task-pool build report is missing the family")
        metadata = families[family]
        if (
            not isinstance(metadata, dict)
            or metadata.get("family") != family
            or metadata.get("schema_version") != PACKED_TASK_SCHEMA
        ):
            raise ValueError("E2E task-pool family metadata is malformed")
        return cls(
            metadata["path"],
            family=family,
            seq_length=18_000,
            expected_bytes=int(metadata["bytes"]),
            expected_sha256=str(metadata["sha256"]),
            verify_sha256=verify_sha256,
            load_audio=load_audio,
            audio_loader=audio_loader,
            teacher_readers=teacher_readers,
        )

    def __len__(self) -> int:
        assert self.offsets is not None
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        assert self.offsets is not None
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("family") != self.family:
            raise ValueError("E2E packed record escaped its family dataset")
        item = packed_task_to_runtime_item(
            value,
            seq_length=self.seq_length,
            load_audio=self.load_audio,
            audio_loader=self.audio_loader,
        )
        if item["teacher_bindings"] and self.teacher_readers:
            item["teacher_posteriors"] = resolve_teacher_bindings(
                item["teacher_bindings"],
                self.teacher_readers,
                packed_labels=item["labels"],
            )
        else:
            item["teacher_posteriors"] = []
        return item


def collate_e2e_family(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty E2E batch")
    families = {str(value.get("family")) for value in batch}
    if len(families) != 1 or next(iter(families)) not in TASK_FAMILIES:
        raise ValueError("one optimizer microbatch cannot mix E2E task families")
    fixed = (
        "tokens",
        "labels",
        "loss_kinds",
        "loss_mask",
        "position_ids",
        "cu_seqlens",
        "max_seqlen",
    )
    result: dict[str, object] = {
        "family": next(iter(families)),
        "batch_size": len(batch),
        "sample_boundaries": [value["sample_boundaries"] for value in batch],
        "source_ids": [value["source_ids"] for value in batch],
        "sequence_ids": [value["sequence_ids"] for value in batch],
        "source_manifest_records": [
            value["source_manifest_records"] for value in batch
        ],
        "used_tokens": torch.tensor(
            [int(value["used_tokens"]) for value in batch], dtype=torch.long
        ),
        "supervised_tokens": torch.tensor(
            [int(value["supervised_tokens"]) for value in batch], dtype=torch.long
        ),
    }
    for name in fixed:
        result[name] = torch.stack([value[name] for value in batch])
    acoustic_rows = []
    teacher_bindings = []
    teacher_posteriors = []
    for batch_index, value in enumerate(batch):
        for raw in value["acoustic_rows"]:
            row = dict(raw)
            row["batch_index"] = batch_index
            acoustic_rows.append(row)
        for raw in value["teacher_bindings"]:
            binding = dict(raw)
            binding["batch_index"] = batch_index
            teacher_bindings.append(binding)
        for raw in value.get("teacher_posteriors", []):
            posterior = dict(raw)
            posterior["batch_index"] = batch_index
            teacher_posteriors.append(posterior)
    result["acoustic_rows"] = acoustic_rows
    result["teacher_bindings"] = teacher_bindings
    result["teacher_posteriors"] = teacher_posteriors
    return result


__all__ = [
    "E2EPackedFamilyDataset",
    "collate_e2e_family",
    "packed_task_to_runtime_item",
]
