"""Validated random access to merged V1 and Phase3 top-k teacher caches."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.cache import (
    CACHE_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.merge_phase3_cache import (
    MERGE_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.merge_v1_cache import (
    V1_MERGE_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_cache import (
    V1_CACHE_SCHEMA,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index


COMMON_ARRAYS = (
    "request_id",
    "reference_label",
    "indices",
    "probabilities",
    "top1",
    "confidence",
)
CACHE_SPECS = {
    "phase3": (MERGE_SCHEMA, CACHE_SCHEMA),
    "v1_asr": (V1_MERGE_SCHEMA, V1_CACHE_SCHEMA),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherPosterior:
    cache_kind: str
    sample_id: str
    source_manifest_record: int
    request_id: int
    indices: torch.Tensor
    probabilities: torch.Tensor
    reference_labels: torch.Tensor
    top1: torch.Tensor
    confidence: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.reference_labels.numel())


class TopKTeacherCacheReader:
    """Read request-local posterior slices from a fully audited merged cache."""

    def __init__(
        self,
        audit_path: str | Path,
        *,
        cache_kind: str,
        verify_manifest_sha256: bool = False,
        verify_bundle_sha256: bool = False,
        row_cache_size: int = 64,
    ) -> None:
        if cache_kind not in CACHE_SPECS:
            raise ValueError("unknown E2E teacher cache kind")
        if row_cache_size <= 0:
            raise ValueError("teacher row cache size must be positive")
        self.cache_kind = cache_kind
        self.audit_path = Path(audit_path).resolve()
        audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
        merge_schema, cache_schema = CACHE_SPECS[cache_kind]
        if (
            audit.get("schema_version") != merge_schema
            or audit.get("cache_schema") != cache_schema
            or audit.get("status") != "passed"
        ):
            raise ValueError("teacher cache audit is not a passed matching cache")
        self.cache_schema = cache_schema
        self.selection_start = int(audit["selection_start"])
        self.selection_stop = int(audit["selection_stop"])
        if not 0 <= self.selection_start < self.selection_stop:
            raise ValueError("teacher cache audit selection is invalid")
        self.path = Path(str(audit["output"])).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size != int(audit["output_bytes"]):
            raise ValueError("teacher cache manifest byte count changed")
        if verify_manifest_sha256 and _sha256(self.path) != audit["output_sha256"]:
            raise ValueError("teacher cache manifest SHA256 changed")
        self.offsets: Sequence[int] | None = load_index(self.path)
        if self.offsets is None or len(self.offsets) != (
            self.selection_stop - self.selection_start
        ):
            raise ValueError("teacher cache manifest offset coverage differs")
        self.verify_bundle_sha256 = bool(verify_bundle_sha256)
        self.row_cache_size = int(row_cache_size)
        self._rows: OrderedDict[
            tuple[Path, int], dict[str, np.ndarray]
        ] = OrderedDict()
        self._verified_bundles: set[Path] = set()

    def __len__(self) -> int:
        assert self.offsets is not None
        return len(self.offsets)

    def _manifest_row(
        self, source_manifest_record: int, sample_id: str
    ) -> dict[str, object]:
        ordinal = int(source_manifest_record) - self.selection_start
        if not 0 <= ordinal < len(self):
            raise IndexError("teacher cache does not cover source manifest record")
        assert self.offsets is not None
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[ordinal]))
            row = json.loads(handle.readline())
        if (
            row.get("schema_version") != self.cache_schema
            or int(row.get("source_manifest_record", -1))
            != int(source_manifest_record)
            or str(row.get("sample_id")) != sample_id
        ):
            raise ValueError("teacher cache manifest sample identity differs")
        return row

    def _arrays(self, row: Mapping[str, object]) -> dict[str, np.ndarray]:
        path = Path(str(row["bundle_path"])).resolve()
        bundle_row = int(row["bundle_row"])
        key = (path, bundle_row)
        cached = self._rows.get(key)
        if cached is not None:
            self._rows.move_to_end(key)
            return cached
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.verify_bundle_sha256 and path not in self._verified_bundles:
            if _sha256(path) != row.get("bundle_sha256"):
                raise ValueError("teacher cache bundle SHA256 changed")
            self._verified_bundles.add(path)
        prefix = f"row_{bundle_row}_"
        with np.load(path, allow_pickle=False) as bundle:
            if str(bundle["bundle_schema"][0]) != self.cache_schema:
                raise ValueError("teacher cache bundle schema differs")
            arrays = {
                name: np.asarray(bundle[f"{prefix}{name}"]).copy()
                for name in COMMON_ARRAYS
            }
        positions = len(arrays["reference_label"])
        if positions != int(row["teacher_positions"]) or any(
            len(value) != positions for value in arrays.values()
        ):
            raise ValueError("teacher cache row array lengths differ")
        if (
            arrays["indices"].ndim != 2
            or arrays["probabilities"].shape != arrays["indices"].shape
            or arrays["top1"].shape != (positions,)
            or arrays["confidence"].shape != (positions,)
        ):
            raise ValueError("teacher cache row top-k geometry differs")
        if not np.isfinite(arrays["probabilities"]).all() or not np.isfinite(
            arrays["confidence"]
        ).all():
            raise ValueError("teacher cache row contains NaN/Inf")
        if any(
            not np.issubdtype(arrays[name].dtype, np.integer)
            for name in ("request_id", "reference_label", "indices", "top1")
        ):
            raise ValueError("teacher cache row token metadata is not integral")
        probabilities = arrays["probabilities"].astype(np.float32)
        if np.any(probabilities < 0) or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=2e-3
        ):
            raise ValueError("teacher cache row probabilities are invalid")
        if np.any(arrays["confidence"] < 0) or np.any(
            arrays["confidence"] > 1
        ):
            raise ValueError("teacher cache row confidence is outside [0, 1]")
        for name in ("reference_label", "indices", "top1"):
            if np.any(arrays[name] < 0) or np.any(arrays[name] >= c.VOCAB_SIZE):
                raise ValueError("teacher cache row token escaped the vocabulary")
        self._rows[key] = arrays
        self._rows.move_to_end(key)
        while len(self._rows) > self.row_cache_size:
            self._rows.popitem(last=False)
        return arrays

    def read_binding(self, binding: Mapping[str, object]) -> TeacherPosterior:
        if binding.get("cache_kind") != self.cache_kind:
            raise ValueError("teacher binding was routed to the wrong cache reader")
        sample_id = str(binding["sample_id"])
        source_manifest_record = int(binding["source_manifest_record"])
        request_id = int(binding["request_id"])
        cache_start = int(binding["cache_position_start"])
        cache_stop = int(binding["cache_position_stop"])
        if not 0 <= cache_start < cache_stop:
            raise ValueError("teacher binding cache range is invalid")
        row = self._manifest_row(source_manifest_record, sample_id)
        descriptors = row.get("requests")
        if not isinstance(descriptors, list) or not 0 <= request_id < len(
            descriptors
        ):
            raise ValueError("teacher binding request ID is outside the cache row")
        descriptor = descriptors[request_id]
        if (
            not isinstance(descriptor, dict)
            or int(descriptor.get("request_id", -1)) != request_id
        ):
            raise ValueError("teacher cache request descriptor differs")
        request_start = int(descriptor["position_start"])
        request_stop = int(descriptor["position_stop"])
        if cache_stop > request_stop - request_start:
            raise ValueError("teacher binding exceeds its cache request")
        start = request_start + cache_start
        stop = request_start + cache_stop
        arrays = self._arrays(row)
        if not np.all(arrays["request_id"][start:stop] == request_id):
            raise ValueError("teacher binding slice crossed a request boundary")
        return TeacherPosterior(
            cache_kind=self.cache_kind,
            sample_id=sample_id,
            source_manifest_record=source_manifest_record,
            request_id=request_id,
            indices=torch.tensor(arrays["indices"][start:stop], dtype=torch.long),
            probabilities=torch.tensor(
                arrays["probabilities"][start:stop], dtype=torch.float32
            ),
            reference_labels=torch.tensor(
                arrays["reference_label"][start:stop], dtype=torch.long
            ),
            top1=torch.tensor(arrays["top1"][start:stop], dtype=torch.long),
            confidence=torch.tensor(
                arrays["confidence"][start:stop], dtype=torch.float32
            ),
        )


def resolve_teacher_bindings(
    bindings: Sequence[Mapping[str, object]],
    readers: Mapping[str, TopKTeacherCacheReader],
    *,
    packed_labels: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for raw in bindings:
        kind = str(raw.get("cache_kind"))
        reader = readers.get(kind)
        if reader is None:
            raise ValueError(f"missing teacher cache reader for {kind}")
        posterior = reader.read_binding(raw)
        packed_start = int(raw["packed_start"])
        packed_stop = int(raw["packed_stop"])
        if packed_stop - packed_start != posterior.positions:
            raise ValueError("teacher binding packed/cache position counts differ")
        if packed_labels is not None:
            if packed_labels.ndim == 1:
                labels = packed_labels[packed_start:packed_stop]
            elif packed_labels.ndim == 2 and "batch_index" in raw:
                labels = packed_labels[
                    int(raw["batch_index"]), packed_start:packed_stop
                ]
            else:
                raise ValueError("packed labels and teacher binding batch geometry differ")
            if not torch.equal(labels.cpu().long(), posterior.reference_labels):
                raise ValueError("teacher posterior labels differ from packed labels")
        value = dict(raw)
        value["posterior"] = posterior
        resolved.append(value)
    return resolved


__all__ = [
    "TeacherPosterior",
    "TopKTeacherCacheReader",
    "resolve_teacher_bindings",
]
