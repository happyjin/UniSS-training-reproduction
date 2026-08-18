#!/usr/bin/env python3
"""Stream-merge and fully audit contiguous Phase3 teacher cache parts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    atomic_json,
    file_sha256,
    selected_total,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.build_phase3_cache import (
    PART_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.cache import (
    CACHE_SCHEMA,
    FAMILY_IDS,
    HISTORY_IDS,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import write_index


MERGE_SCHEMA = "uniss_phase3_v4_e2e_phase3_teacher_merge_v1"
REQUIRED_ARRAYS = (
    "request_id",
    "event_index",
    "family_id",
    "history_id",
    "target_index",
    "reference_label",
    "indices",
    "probabilities",
    "top1",
    "confidence",
)


def _require_integer_array(value: np.ndarray, label: str) -> None:
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Phase3 teacher {label} is not an integer array")


def _descriptor_counts(
    descriptors: object,
    arrays: Mapping[str, np.ndarray],
    positions: int,
) -> Counter[str]:
    if not isinstance(descriptors, list):
        raise ValueError("Phase3 teacher request descriptors are missing")
    counts: Counter[str] = Counter()
    descriptor_cursor = 0
    for request_id, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ValueError("Phase3 teacher request descriptor is malformed")
        if int(descriptor["request_id"]) != request_id:
            raise ValueError("Phase3 teacher request IDs are not contiguous")
        start = int(descriptor["position_start"])
        stop = int(descriptor["position_stop"])
        if start != descriptor_cursor or not start < stop <= positions:
            raise ValueError("Phase3 teacher request positions contain a gap")
        if int(descriptor["positions"]) != stop - start:
            raise ValueError("Phase3 teacher request descriptor length differs")
        family = str(descriptor["family"])
        history = str(descriptor["history_kind"])
        if family not in FAMILY_IDS or history not in HISTORY_IDS:
            raise ValueError("Phase3 teacher request type is unknown")
        event_index = int(descriptor["event_index"])
        selection = slice(start, stop)
        expected = (
            ("request_id", request_id),
            ("event_index", event_index),
            ("family_id", FAMILY_IDS[family]),
            ("history_id", HISTORY_IDS[history]),
        )
        for name, value in expected:
            if not np.all(arrays[name][selection] == value):
                raise ValueError(
                    f"Phase3 teacher {name} does not match request descriptor"
                )
        target_indices = arrays["target_index"][selection]
        if np.any(target_indices < 0) or np.any(np.diff(target_indices) <= 0):
            raise ValueError("Phase3 teacher target positions are not increasing")
        top1_correct = int(
            np.count_nonzero(
                arrays["top1"][selection] == arrays["reference_label"][selection]
            )
        )
        reference_in_topk = int(
            np.count_nonzero(
                (
                    arrays["indices"][selection]
                    == arrays["reference_label"][selection, None]
                ).any(axis=1)
            )
        )
        if top1_correct != int(descriptor["teacher_top1_correct"]):
            raise ValueError("Phase3 teacher descriptor top-1 count differs")
        if reference_in_topk != int(descriptor["reference_in_topk"]):
            raise ValueError("Phase3 teacher descriptor top-k count differs")
        candidate_tokens = int(descriptor["content_candidate_tokens"])
        selected_tokens = int(descriptor["content_selected_tokens"])
        if not 0 <= selected_tokens <= candidate_tokens:
            raise ValueError("Phase3 teacher content mapping counts are invalid")
        counts[f"family:{family}"] += 1
        counts[f"history:{history}"] += 1
        counts["content_candidate_tokens"] += candidate_tokens
        counts["content_selected_tokens"] += selected_tokens
        descriptor_cursor = stop
    if not descriptors or descriptor_cursor != positions:
        raise ValueError("Phase3 teacher request positions do not cover the row")
    return counts


def _validate_bundle_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    if not rows:
        raise ValueError("Phase3 teacher bundle has no manifest rows")
    counts: Counter[str] = Counter()
    expected_sha = str(rows[0]["bundle_sha256"])
    if any(str(row["bundle_sha256"]) != expected_sha for row in rows):
        raise ValueError("Phase3 teacher bundle digest differs within its manifest rows")
    if file_sha256(path) != expected_sha:
        raise ValueError(f"Phase3 teacher bundle digest changed: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        if str(bundle["bundle_schema"][0]) != CACHE_SCHEMA:
            raise ValueError("unexpected Phase3 teacher bundle schema")
        for expected_bundle_row, row in enumerate(rows):
            if int(row["bundle_row"]) != expected_bundle_row:
                raise ValueError("Phase3 teacher bundle rows are not contiguous")
            prefix = f"row_{int(row['bundle_row'])}_"
            arrays = {name: bundle[f"{prefix}{name}"] for name in REQUIRED_ARRAYS}
            positions = len(arrays["reference_label"])
            if positions <= 0 or any(len(value) != positions for value in arrays.values()):
                raise ValueError("Phase3 teacher bundle arrays differ in length")
            if any(
                arrays[name].ndim != 1
                for name in REQUIRED_ARRAYS
                if name not in {"indices", "probabilities"}
            ):
                raise ValueError("Phase3 teacher metadata arrays are not one-dimensional")
            if (
                arrays["indices"].ndim != 2
                or arrays["indices"].shape[1] <= 0
                or arrays["probabilities"].shape != arrays["indices"].shape
            ):
                raise ValueError("Phase3 teacher top-k arrays differ")
            for name in (
                "request_id",
                "event_index",
                "family_id",
                "history_id",
                "target_index",
                "reference_label",
                "indices",
                "top1",
            ):
                _require_integer_array(arrays[name], name)
            if not np.isfinite(arrays["probabilities"]).all() or not np.isfinite(
                arrays["confidence"]
            ).all():
                raise ValueError("Phase3 teacher bundle contains NaN/Inf")
            if np.any(arrays["probabilities"] < 0) or np.any(
                (arrays["confidence"] < 0) | (arrays["confidence"] > 1)
            ):
                raise ValueError("Phase3 teacher probabilities are outside [0, 1]")
            if not np.allclose(
                arrays["probabilities"].astype(np.float32).sum(axis=1),
                1.0,
                atol=2e-3,
            ):
                raise ValueError("Phase3 teacher top-k probabilities do not sum to one")
            if positions != int(row["teacher_positions"]):
                raise ValueError("Phase3 teacher manifest position count differs")
            if np.any(arrays["indices"] < 0) or np.any(
                arrays["indices"] >= c.VOCAB_SIZE
            ):
                raise ValueError("Phase3 teacher top-k token is outside vocabulary")
            if np.any(arrays["reference_label"] < 0) or np.any(
                arrays["reference_label"] >= c.VOCAB_SIZE
            ):
                raise ValueError("Phase3 teacher reference is outside vocabulary")
            if np.any(arrays["top1"] < 0) or np.any(
                arrays["top1"] >= c.VOCAB_SIZE
            ):
                raise ValueError("Phase3 teacher top-1 token is outside vocabulary")
            descriptors = row.get("requests")
            if not isinstance(descriptors, list) or len(descriptors) != int(
                row["request_count"]
            ):
                raise ValueError("Phase3 teacher request descriptors differ")
            counts.update(_descriptor_counts(descriptors, arrays, positions))
            top1_correct = int(
                np.count_nonzero(arrays["top1"] == arrays["reference_label"])
            )
            reference_in_topk = int(
                np.count_nonzero(
                    (
                        arrays["indices"] == arrays["reference_label"][:, None]
                    ).any(axis=1)
                )
            )
            if top1_correct != int(row["teacher_top1_correct"]):
                raise ValueError("Phase3 teacher manifest top-1 count differs")
            if reference_in_topk != int(row["reference_in_topk"]):
                raise ValueError("Phase3 teacher manifest top-k count differs")
            counts["records"] += 1
            counts["requests"] += len(descriptors)
            counts["teacher_positions"] += positions
            counts["teacher_top1_correct"] += top1_correct
            counts["reference_in_topk"] += reference_in_topk
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def merge_phase3_cache(
    *,
    gold_path: Path,
    parts_root: Path,
    world_size: int,
    output_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    gold_path = gold_path.resolve()
    parts_root = parts_root.resolve()
    output_path = output_path.resolve()
    audit_path = audit_path.resolve()
    if world_size <= 0:
        raise ValueError("Phase3 teacher world size must be positive")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite merged Phase3 teacher cache")
    gold_offsets, gold_total = selected_total(gold_path, None)
    markers: list[dict[str, object]] = []
    invariant_keys = (
        "cache_schema",
        "world_size",
        "selection_start",
        "selection_stop",
        "gold",
        "rollouts",
        "model",
        "phase3_hf_sha256",
        "runtime_sha256",
        "topk",
        "temperature",
        "semantic_stride",
        "max_padded_tokens",
        "max_batch_size",
        "max_selected_positions",
    )
    for rank in range(world_size):
        expected_part_root = (parts_root / f"part_{rank:03d}").resolve()
        path = expected_part_root / "PART_COMPLETE.json"
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != PART_SCHEMA or marker.get("status") != "complete":
            raise ValueError(f"Phase3 teacher part is incomplete: {rank}")
        if int(marker["rank"]) != rank or int(marker["world_size"]) != world_size:
            raise ValueError("Phase3 teacher part rank geometry differs")
        if markers:
            for key in invariant_keys:
                if marker.get(key) != markers[0].get(key):
                    raise ValueError(f"Phase3 teacher part invariant differs: {key}")
        manifest = Path(str(marker["manifest"])).resolve()
        if manifest.parent != expected_part_root:
            raise ValueError("Phase3 teacher part references a foreign manifest")
        if manifest.stat().st_size != int(marker["manifest_bytes"]):
            raise ValueError("Phase3 teacher part manifest byte count changed")
        if file_sha256(manifest) != marker["manifest_sha256"]:
            raise ValueError("Phase3 teacher part manifest digest changed")
        markers.append(marker)
    if Path(str(markers[0]["gold"])).resolve() != gold_path:
        raise ValueError("Phase3 teacher parts were built from a different gold file")
    cursor = int(markers[0]["selection_start"])
    if cursor < 0 or int(markers[0]["selection_stop"]) <= cursor:
        raise ValueError("Phase3 teacher selection geometry is invalid")
    for marker in markers:
        if int(marker["assigned_start"]) != cursor:
            raise ValueError("Phase3 teacher parts contain a range gap or overlap")
        assigned_stop = int(marker["assigned_stop"])
        if assigned_stop <= cursor:
            raise ValueError("Phase3 teacher part range is empty")
        marker_counts = marker.get("counts")
        if not isinstance(marker_counts, dict) or int(
            marker_counts.get("records", -1)
        ) != assigned_stop - cursor:
            raise ValueError("Phase3 teacher part record count differs from its range")
        cursor = assigned_stop
    if cursor != int(markers[0]["selection_stop"]):
        raise ValueError("Phase3 teacher parts do not cover the registered selection")
    if cursor > gold_total:
        raise ValueError("Phase3 teacher selection exceeds gold data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = array("Q")
    byte_offset = 0
    expected_record = int(markers[0]["selection_start"])
    counts: Counter[str] = Counter()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with gold_path.open("rb") as gold_handle, os.fdopen(
            descriptor, "wb"
        ) as destination:
            for marker in markers:
                manifest = Path(str(marker["manifest"])).resolve()
                part_root = manifest.parent
                bundle_rows: list[dict[str, object]] = []
                current_bundle: Path | None = None
                seen_bundles: set[Path] = set()

                def flush_bundle() -> None:
                    nonlocal bundle_rows, current_bundle, counts
                    if current_bundle is not None:
                        counts.update(
                            _validate_bundle_rows(current_bundle, bundle_rows)
                        )
                        seen_bundles.add(current_bundle)
                    bundle_rows = []
                    current_bundle = None

                with manifest.open("rb") as source:
                    for line in source:
                        row = json.loads(line)
                        if row.get("schema_version") != CACHE_SCHEMA:
                            raise ValueError(
                                "Phase3 teacher manifest row schema differs"
                            )
                        record_index = int(row["source_manifest_record"])
                        if record_index != expected_record:
                            raise ValueError(
                                "Phase3 teacher records are not contiguous"
                            )
                        gold_handle.seek(int(gold_offsets[record_index]))
                        gold = E2ETrajectory.from_mapping(
                            json.loads(gold_handle.readline())
                        )
                        if (
                            row["sample_id"] != gold.sample_id
                            or row["split"] != gold.split
                        ):
                            raise ValueError(
                                "Phase3 teacher row differs from gold sample identity"
                            )
                        bundle = Path(str(row["bundle_path"])).resolve()
                        if part_root not in bundle.parents or not bundle.is_file():
                            raise ValueError(
                                "Phase3 teacher row references a foreign/missing bundle"
                            )
                        if current_bundle is not None and bundle != current_bundle:
                            flush_bundle()
                        if bundle in seen_bundles:
                            raise ValueError(
                                "Phase3 teacher bundle rows are not contiguous in manifest"
                            )
                        current_bundle = bundle
                        bundle_rows.append(row)
                        encoded = line if line.endswith(b"\n") else line + b"\n"
                        offsets.append(byte_offset)
                        destination.write(encoded)
                        byte_offset += len(encoded)
                        expected_record += 1
                flush_bundle()
            destination.flush()
            os.fsync(destination.fileno())
        if expected_record != int(markers[0]["selection_stop"]):
            raise ValueError(
                "Phase3 teacher merged record count differs from selection"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output_path, offsets)
    positions = counts["teacher_positions"]
    if counts["records"] != len(offsets) or positions <= 0:
        raise ValueError("Phase3 teacher merged denominator is zero or coverage differs")
    registered_counts: Counter[str] = Counter()
    for marker in markers:
        registered_counts.update(
            {
                str(key): int(value)
                for key, value in marker["counts"].items()
            }
        )
    if counts != registered_counts:
        raise ValueError("Phase3 teacher merged counts differ from part markers")
    audit = {
        "schema_version": MERGE_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "status": "passed",
        "gold": str(gold_path),
        "parts_root": str(parts_root),
        "world_size": world_size,
        "selection_start": markers[0]["selection_start"],
        "selection_stop": markers[0]["selection_stop"],
        "model": markers[0]["model"],
        "phase3_hf_sha256": markers[0]["phase3_hf_sha256"],
        "runtime_sha256": markers[0]["runtime_sha256"],
        "topk": markers[0]["topk"],
        "temperature": markers[0]["temperature"],
        "semantic_stride": markers[0]["semantic_stride"],
        "counts": dict(sorted(counts.items())),
        "teacher_top1_accuracy": counts["teacher_top1_correct"] / positions,
        "reference_in_topk_rate": counts["reference_in_topk"] / positions,
        "content_mapping_rate": counts["content_selected_tokens"]
        / max(1, counts["content_candidate_tokens"]),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": file_sha256(output_path),
        "index": index,
    }
    atomic_json(audit_path, audit)
    return audit


def main() -> None:
    args = parse_args()
    audit = merge_phase3_cache(
        gold_path=args.gold,
        parts_root=args.parts_root,
        world_size=args.world_size,
        output_path=args.output,
        audit_path=args.audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
