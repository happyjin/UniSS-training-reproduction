#!/usr/bin/env python3
"""Stream-merge and fully audit contiguous V1 ASR teacher cache parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from array import array
from collections import Counter, defaultdict
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
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.build_v1_cache import (
    V1_PART_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_cache import (
    V1_CACHE_SCHEMA,
    V1_HISTORY_IDS,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import write_index


V1_MERGE_SCHEMA = "uniss_phase3_v4_e2e_v1_asr_teacher_merge_v1"
REQUIRED_ARRAYS = (
    "request_id",
    "event_index",
    "history_id",
    "visible_glm_tokens",
    "target_index",
    "reference_label",
    "indices",
    "probabilities",
    "top1",
    "confidence",
)


def _hash_tokens(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _validate_descriptors(
    descriptors: object,
    arrays: Mapping[str, np.ndarray],
    positions: int,
) -> Counter[str]:
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("V1 teacher request descriptors are missing")
    counts: Counter[str] = Counter()
    cursor = 0
    previous_event: dict[str, int] = defaultdict(lambda: -1)
    previous_visible_glm: dict[str, int] = defaultdict(lambda: -1)
    final_by_history: Counter[str] = Counter()
    for request_id, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ValueError("V1 teacher request descriptor is malformed")
        if int(descriptor["request_id"]) != request_id:
            raise ValueError("V1 teacher request IDs are not contiguous")
        start = int(descriptor["position_start"])
        stop = int(descriptor["position_stop"])
        if start != cursor or not start < stop <= positions:
            raise ValueError("V1 teacher request positions contain a gap")
        if int(descriptor["positions"]) != stop - start:
            raise ValueError("V1 teacher request descriptor length differs")
        history = str(descriptor["history_kind"])
        if history not in V1_HISTORY_IDS:
            raise ValueError("V1 teacher history kind is unknown")
        if final_by_history[history]:
            raise ValueError("V1 teacher request appears after a final request")
        event_index = int(descriptor["event_index"])
        if event_index <= previous_event[history]:
            raise ValueError("V1 teacher event indices are not increasing")
        previous_event[history] = event_index
        visible_glm = int(descriptor["visible_glm_tokens"])
        if visible_glm < 0 or visible_glm < previous_visible_glm[history]:
            raise ValueError("V1 teacher visible GLM boundary rolled back")
        previous_visible_glm[history] = visible_glm
        selection = slice(start, stop)
        expected = (
            ("request_id", request_id),
            ("event_index", event_index),
            ("history_id", V1_HISTORY_IDS[history]),
            ("visible_glm_tokens", visible_glm),
        )
        for name, value in expected:
            if not np.all(arrays[name][selection] == value):
                raise ValueError(f"V1 teacher {name} differs from descriptor")
        target_indices = arrays["target_index"][selection]
        if not np.array_equal(
            target_indices,
            np.arange(stop - start, dtype=target_indices.dtype),
        ):
            raise ValueError("V1 teacher target positions are not contiguous")
        labels = arrays["reference_label"][selection]
        if str(descriptor["target_sha256"]) != _hash_tokens(labels.tolist()):
            raise ValueError("V1 teacher target digest differs")
        for key in ("prefix_sha256", "target_sha256"):
            value = str(descriptor[key])
            if len(value) != 64:
                raise ValueError("V1 teacher request digest is malformed")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError("V1 teacher request digest is malformed") from exc
        top1_correct = int(
            np.count_nonzero(arrays["top1"][selection] == labels)
        )
        reference_in_topk = int(
            np.count_nonzero(
                (arrays["indices"][selection] == labels[:, None]).any(axis=1)
            )
        )
        if top1_correct != int(descriptor["teacher_top1_correct"]):
            raise ValueError("V1 teacher descriptor top-1 count differs")
        if reference_in_topk != int(descriptor["reference_in_topk"]):
            raise ValueError("V1 teacher descriptor top-k count differs")
        final = bool(descriptor["final"])
        final_by_history[history] += int(final)
        counts[f"history:{history}"] += 1
        counts["final_requests"] += int(final)
        cursor = stop
    if cursor != positions:
        raise ValueError("V1 teacher request positions do not cover the row")
    if set(previous_event) != set(V1_HISTORY_IDS) or any(
        final_by_history[history] != 1 for history in V1_HISTORY_IDS
    ):
        raise ValueError("V1 teacher row does not cover both histories and final requests")
    return counts


def _validate_bundle_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> Counter[str]:
    if not rows:
        raise ValueError("V1 teacher bundle has no manifest rows")
    expected_sha = str(rows[0]["bundle_sha256"])
    if any(str(row["bundle_sha256"]) != expected_sha for row in rows):
        raise ValueError("V1 teacher bundle digest differs within manifest rows")
    if file_sha256(path) != expected_sha:
        raise ValueError(f"V1 teacher bundle digest changed: {path}")
    counts: Counter[str] = Counter()
    with np.load(path, allow_pickle=False) as bundle:
        if str(bundle["bundle_schema"][0]) != V1_CACHE_SCHEMA:
            raise ValueError("unexpected V1 teacher bundle schema")
        for expected_bundle_row, row in enumerate(rows):
            if int(row["bundle_row"]) != expected_bundle_row:
                raise ValueError("V1 teacher bundle rows are not contiguous")
            prefix = f"row_{expected_bundle_row}_"
            arrays = {name: bundle[f"{prefix}{name}"] for name in REQUIRED_ARRAYS}
            positions = len(arrays["reference_label"])
            if positions <= 0 or any(len(value) != positions for value in arrays.values()):
                raise ValueError("V1 teacher bundle arrays differ in length")
            if any(
                arrays[name].ndim != 1
                for name in REQUIRED_ARRAYS
                if name not in {"indices", "probabilities"}
            ):
                raise ValueError("V1 teacher metadata arrays are not one-dimensional")
            if (
                arrays["indices"].ndim != 2
                or arrays["indices"].shape[1] <= 0
                or arrays["probabilities"].shape != arrays["indices"].shape
            ):
                raise ValueError("V1 teacher top-k arrays differ")
            for name in (
                "request_id",
                "event_index",
                "history_id",
                "visible_glm_tokens",
                "target_index",
                "reference_label",
                "indices",
                "top1",
            ):
                if not np.issubdtype(arrays[name].dtype, np.integer):
                    raise ValueError(f"V1 teacher {name} is not an integer array")
            if not np.isfinite(arrays["probabilities"]).all() or not np.isfinite(
                arrays["confidence"]
            ).all():
                raise ValueError("V1 teacher bundle contains NaN/Inf")
            if np.any(arrays["probabilities"] < 0) or np.any(
                (arrays["confidence"] < 0) | (arrays["confidence"] > 1)
            ):
                raise ValueError("V1 teacher probabilities are outside [0, 1]")
            if not np.allclose(
                arrays["probabilities"].astype(np.float32).sum(axis=1),
                1.0,
                atol=2e-3,
            ):
                raise ValueError("V1 teacher top-k probabilities do not sum to one")
            for name in ("reference_label", "indices", "top1"):
                if np.any(arrays[name] < 0) or np.any(
                    arrays[name] >= c.VOCAB_SIZE
                ):
                    raise ValueError(f"V1 teacher {name} is outside vocabulary")
            if positions != int(row["teacher_positions"]):
                raise ValueError("V1 teacher manifest position count differs")
            descriptors = row.get("requests")
            if not isinstance(descriptors, list) or len(descriptors) != int(
                row["request_count"]
            ):
                raise ValueError("V1 teacher request descriptors differ")
            counts.update(_validate_descriptors(descriptors, arrays, positions))
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
                raise ValueError("V1 teacher manifest top-1 count differs")
            if reference_in_topk != int(row["reference_in_topk"]):
                raise ValueError("V1 teacher manifest top-k count differs")
            counts["records"] += 1
            counts["requests"] += len(descriptors)
            counts["teacher_positions"] += positions
            counts["teacher_top1_correct"] += top1_correct
            counts["reference_in_topk"] += reference_in_topk
    return counts


def merge_v1_cache(
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
        raise ValueError("V1 teacher world size must be positive")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite merged V1 teacher cache")
    gold_offsets, gold_total = selected_total(gold_path, None)
    invariant_keys = (
        "cache_schema",
        "world_size",
        "selection_start",
        "selection_stop",
        "gold",
        "rollouts",
        "checkpoint",
        "hf_model",
        "whispervq_model",
        "v1_hf_sha256",
        "runtime_sha256",
        "topk",
        "temperature",
    )
    markers: list[dict[str, object]] = []
    for rank in range(world_size):
        part_root = (parts_root / f"part_{rank:03d}").resolve()
        marker_path = part_root / "PART_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("schema_version") != V1_PART_SCHEMA
            or marker.get("status") != "complete"
        ):
            raise ValueError(f"V1 teacher part is incomplete: {rank}")
        if int(marker["rank"]) != rank or int(marker["world_size"]) != world_size:
            raise ValueError("V1 teacher part rank geometry differs")
        if markers:
            for key in invariant_keys:
                if marker.get(key) != markers[0].get(key):
                    raise ValueError(f"V1 teacher part invariant differs: {key}")
        manifest = Path(str(marker["manifest"])).resolve()
        if manifest.parent != part_root:
            raise ValueError("V1 teacher part references a foreign manifest")
        if manifest.stat().st_size != int(marker["manifest_bytes"]):
            raise ValueError("V1 teacher part manifest byte count changed")
        if file_sha256(manifest) != marker["manifest_sha256"]:
            raise ValueError("V1 teacher part manifest digest changed")
        markers.append(marker)
    if Path(str(markers[0]["gold"])).resolve() != gold_path:
        raise ValueError("V1 teacher parts were built from a different gold file")
    cursor = int(markers[0]["selection_start"])
    if cursor < 0 or int(markers[0]["selection_stop"]) <= cursor:
        raise ValueError("V1 teacher selection geometry is invalid")
    for marker in markers:
        if int(marker["assigned_start"]) != cursor:
            raise ValueError("V1 teacher parts contain a range gap or overlap")
        assigned_stop = int(marker["assigned_stop"])
        if assigned_stop <= cursor:
            raise ValueError("V1 teacher part range is empty")
        marker_counts = marker.get("counts")
        if not isinstance(marker_counts, dict) or int(
            marker_counts.get("records", -1)
        ) != assigned_stop - cursor:
            raise ValueError("V1 teacher part record count differs from its range")
        cursor = assigned_stop
    if cursor != int(markers[0]["selection_stop"]) or cursor > gold_total:
        raise ValueError("V1 teacher parts do not cover their registered selection")

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
                        if row.get("schema_version") != V1_CACHE_SCHEMA:
                            raise ValueError("V1 teacher manifest row schema differs")
                        record_index = int(row["source_manifest_record"])
                        if record_index != expected_record:
                            raise ValueError("V1 teacher records are not contiguous")
                        gold_handle.seek(int(gold_offsets[record_index]))
                        gold = E2ETrajectory.from_mapping(
                            json.loads(gold_handle.readline())
                        )
                        if (
                            row["sample_id"] != gold.sample_id
                            or row["split"] != gold.split
                        ):
                            raise ValueError(
                                "V1 teacher row differs from gold sample identity"
                            )
                        for request in row["requests"]:
                            if int(request["visible_glm_tokens"]) > gold.source_glm_length:
                                raise ValueError(
                                    "V1 teacher request exceeds visible source GLM"
                                )
                            if bool(request["final"]) and int(
                                request["visible_glm_tokens"]
                            ) != gold.source_glm_length:
                                raise ValueError(
                                    "V1 teacher final request lacks full source coverage"
                                )
                        bundle = Path(str(row["bundle_path"])).resolve()
                        if part_root not in bundle.parents or not bundle.is_file():
                            raise ValueError(
                                "V1 teacher row references a foreign/missing bundle"
                            )
                        if current_bundle is not None and bundle != current_bundle:
                            flush_bundle()
                        if bundle in seen_bundles:
                            raise ValueError(
                                "V1 teacher bundle rows are not contiguous in manifest"
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
            raise ValueError("V1 teacher merged record count differs from selection")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output_path, offsets)
    positions = counts["teacher_positions"]
    if counts["records"] != len(offsets) or positions <= 0:
        raise ValueError("V1 teacher merged denominator is zero or coverage differs")
    registered_counts: Counter[str] = Counter()
    for marker in markers:
        registered_counts.update(
            {str(key): int(value) for key, value in marker["counts"].items()}
        )
    if counts != registered_counts:
        raise ValueError("V1 teacher merged counts differ from part markers")
    audit = {
        "schema_version": V1_MERGE_SCHEMA,
        "cache_schema": V1_CACHE_SCHEMA,
        "status": "passed",
        "gold": str(gold_path),
        "parts_root": str(parts_root),
        "world_size": world_size,
        "selection_start": markers[0]["selection_start"],
        "selection_stop": markers[0]["selection_stop"],
        "checkpoint": markers[0]["checkpoint"],
        "hf_model": markers[0]["hf_model"],
        "v1_hf_sha256": markers[0]["v1_hf_sha256"],
        "runtime_sha256": markers[0]["runtime_sha256"],
        "topk": markers[0]["topk"],
        "temperature": markers[0]["temperature"],
        "counts": dict(sorted(counts.items())),
        "teacher_top1_accuracy": counts["teacher_top1_correct"] / positions,
        "reference_in_topk_rate": counts["reference_in_topk"] / positions,
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": file_sha256(output_path),
        "index": index,
    }
    atomic_json(audit_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = merge_v1_cache(
        gold_path=args.gold,
        parts_root=args.parts_root,
        world_size=args.world_size,
        output_path=args.output,
        audit_path=args.audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
