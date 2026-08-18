"""Compact immutable NPZ bundles for Phase3 MT/semantic teacher posteriors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.requests import (
    Phase3TeacherRequest,
)


CACHE_SCHEMA = "uniss_phase3_v4_e2e_phase3_teacher_cache_v1"
FAMILY_IDS = {"phase3_mt": 0, "phase3_semantic": 1}
HISTORY_IDS = {"gold_source": 0, "v1_source": 1, "gold_target": 2}


def hash_tokens(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def combine_sample(
    requests: Sequence[Phase3TeacherRequest],
    summaries: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    if len(requests) != len(summaries) or not requests:
        raise ValueError("Phase3 teacher request/result count differs")
    arrays: dict[str, list[np.ndarray]] = {
        "request_id": [],
        "event_index": [],
        "family_id": [],
        "history_id": [],
        "target_index": [],
        "reference_label": [],
        "indices": [],
        "probabilities": [],
        "top1": [],
        "confidence": [],
    }
    descriptors: list[dict[str, object]] = []
    cursor = 0
    topk_width: int | None = None
    for request_id, (request, summary) in enumerate(zip(requests, summaries)):
        count = len(request.selected_target_indices)
        indices = np.asarray(summary["indices"], dtype=np.int32)
        probabilities = np.asarray(summary["probabilities"], dtype=np.float16)
        top1 = np.asarray(summary["top1"], dtype=np.int32)
        confidence = np.asarray(summary["confidence"], dtype=np.float16)
        if indices.ndim != 2 or probabilities.shape != indices.shape:
            raise ValueError("Phase3 teacher top-k geometry differs")
        if len(indices) != count or top1.shape != (count,) or confidence.shape != (count,):
            raise ValueError("Phase3 teacher selected result count differs")
        if topk_width is None:
            topk_width = indices.shape[1]
        elif topk_width != indices.shape[1]:
            raise ValueError("Phase3 teacher top-k width changed within one sample")
        labels = np.asarray(request.reference_labels, dtype=np.int32)
        arrays["request_id"].append(np.full(count, request_id, dtype=np.int32))
        arrays["event_index"].append(np.full(count, request.event_index, dtype=np.int32))
        arrays["family_id"].append(np.full(count, FAMILY_IDS[request.family], dtype=np.int8))
        arrays["history_id"].append(
            np.full(count, HISTORY_IDS[request.history_kind], dtype=np.int8)
        )
        arrays["target_index"].append(
            np.asarray(request.selected_target_indices, dtype=np.int32)
        )
        arrays["reference_label"].append(labels)
        arrays["indices"].append(indices)
        arrays["probabilities"].append(probabilities)
        arrays["top1"].append(top1)
        arrays["confidence"].append(confidence)
        descriptors.append(
            {
                "request_id": request_id,
                "event_index": request.event_index,
                "family": request.family,
                "history_kind": request.history_kind,
                "position_start": cursor,
                "position_stop": cursor + count,
                "positions": count,
                "content_candidate_tokens": request.content_candidate_tokens,
                "content_selected_tokens": request.content_selected_tokens,
                "prompt_sha256": hash_tokens(request.prompt_ids),
                "target_sha256": hash_tokens(request.target_ids),
                "source_prefix_sha256": hashlib.sha256(
                    request.visible_source_prefix.encode("utf-8")
                ).hexdigest(),
                "target_prefix_sha256": hashlib.sha256(
                    request.visible_target_prefix.encode("utf-8")
                ).hexdigest(),
                "visible_semantic_tokens": request.visible_semantic_tokens,
                "reference_in_topk": int(
                    np.count_nonzero((indices == labels[:, None]).any(axis=1))
                ),
                "teacher_top1_correct": int(np.count_nonzero(top1 == labels)),
            }
        )
        cursor += count
    combined = {
        name: np.concatenate(values, axis=0)
        for name, values in arrays.items()
    }
    if len(combined["indices"]) != cursor:
        raise AssertionError("Phase3 teacher bundle position cursor did not close")
    return combined, descriptors


def save_bundle(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Phase3 teacher bundle: {path}")
    values: dict[str, np.ndarray] = {"bundle_schema": np.asarray([CACHE_SCHEMA])}
    manifest: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        arrays = row["arrays"]
        if not isinstance(arrays, dict):
            raise TypeError("Phase3 teacher bundle row arrays are missing")
        prefix = f"row_{row_index}"
        for name, value in arrays.items():
            values[f"{prefix}_{name}"] = np.asarray(value)
        descriptors = row["requests"]
        if not isinstance(descriptors, list):
            raise TypeError("Phase3 teacher request descriptors are missing")
        manifest.append(
            {
                "schema_version": CACHE_SCHEMA,
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "source_manifest_record": int(row["source_manifest_record"]),
                "bundle_path": str(path.resolve()),
                "bundle_row": row_index,
                "requests": descriptors,
                "request_count": len(descriptors),
                "teacher_positions": int(len(arrays["reference_label"])),
                "teacher_top1_correct": int(
                    np.count_nonzero(arrays["top1"] == arrays["reference_label"])
                ),
                "reference_in_topk": int(
                    np.count_nonzero(
                        (
                            arrays["indices"]
                            == arrays["reference_label"][:, None]
                        ).any(axis=1)
                    )
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)
    return manifest


def validate_bundle_row(row: Mapping[str, object]) -> dict[str, int]:
    path = Path(str(row["bundle_path"]))
    bundle_row = int(row["bundle_row"])
    with np.load(path, allow_pickle=False) as bundle:
        if str(bundle["bundle_schema"][0]) != CACHE_SCHEMA:
            raise ValueError("unexpected Phase3 teacher bundle schema")
        prefix = f"row_{bundle_row}_"
        required = (
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
        arrays = {name: bundle[f"{prefix}{name}"] for name in required}
    positions = len(arrays["reference_label"])
    if any(len(value) != positions for value in arrays.values()):
        raise ValueError("Phase3 teacher bundle arrays differ in length")
    if arrays["indices"].ndim != 2 or arrays["probabilities"].shape != arrays["indices"].shape:
        raise ValueError("Phase3 teacher bundle top-k geometry differs")
    if not np.isfinite(arrays["probabilities"]).all() or not np.isfinite(
        arrays["confidence"]
    ).all():
        raise ValueError("Phase3 teacher bundle contains NaN/Inf")
    if np.any(arrays["probabilities"] < 0):
        raise ValueError("Phase3 teacher probabilities are negative")
    if not np.allclose(arrays["probabilities"].astype(np.float32).sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("Phase3 teacher top-k probabilities do not sum to one")
    if positions != int(row["teacher_positions"]):
        raise ValueError("Phase3 teacher manifest position count differs")
    return {
        "positions": positions,
        "top1_correct": int(
            np.count_nonzero(arrays["top1"] == arrays["reference_label"])
        ),
        "reference_in_topk": int(
            np.count_nonzero(
                (arrays["indices"] == arrays["reference_label"][:, None]).any(axis=1)
            )
        ),
    }


__all__ = [
    "CACHE_SCHEMA",
    "FAMILY_IDS",
    "HISTORY_IDS",
    "combine_sample",
    "hash_tokens",
    "save_bundle",
    "validate_bundle_row",
]
