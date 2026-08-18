"""Immutable NPZ bundles for V1 same-prefix ASR top-k posteriors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_requests import (
    V1TeacherSequence,
)


V1_CACHE_SCHEMA = "uniss_phase3_v4_e2e_v1_asr_teacher_cache_v1"
V1_HISTORY_IDS = {"gold_asr": 0, "v1_asr": 1}


def combine_v1_sample(
    sequences: Sequence[V1TeacherSequence],
    summaries: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    if len(sequences) != len(summaries) or not sequences:
        raise ValueError("V1 teacher sequence/result count differs")
    arrays: dict[str, list[np.ndarray]] = {
        "request_id": [],
        "event_index": [],
        "history_id": [],
        "visible_glm_tokens": [],
        "target_index": [],
        "reference_label": [],
        "indices": [],
        "probabilities": [],
        "top1": [],
        "confidence": [],
    }
    descriptors: list[dict[str, object]] = []
    request_id = 0
    cursor = 0
    topk_width: int | None = None
    for sequence, summary in zip(sequences, summaries):
        sequence_positions = len(sequence.selected_predictor_positions)
        indices = np.asarray(summary["indices"], dtype=np.int32)
        probabilities = np.asarray(summary["probabilities"], dtype=np.float16)
        top1 = np.asarray(summary["top1"], dtype=np.int32)
        confidence = np.asarray(summary["confidence"], dtype=np.float16)
        if indices.ndim != 2 or probabilities.shape != indices.shape:
            raise ValueError("V1 teacher top-k geometry differs")
        if (
            len(indices) != sequence_positions
            or top1.shape != (sequence_positions,)
            or confidence.shape != (sequence_positions,)
        ):
            raise ValueError("V1 teacher selected result count differs")
        if topk_width is None:
            topk_width = indices.shape[1]
        elif topk_width != indices.shape[1]:
            raise ValueError("V1 teacher top-k width changed within one sample")
        sequence_cursor = 0
        for request in sequence.requests:
            count = len(request.reference_labels)
            stop = sequence_cursor + count
            labels = np.asarray(request.reference_labels, dtype=np.int32)
            request_indices = indices[sequence_cursor:stop]
            request_top1 = top1[sequence_cursor:stop]
            arrays["request_id"].append(
                np.full(count, request_id, dtype=np.int32)
            )
            arrays["event_index"].append(
                np.full(count, request.event_index, dtype=np.int32)
            )
            arrays["history_id"].append(
                np.full(
                    count,
                    V1_HISTORY_IDS[request.history_kind],
                    dtype=np.int8,
                )
            )
            arrays["visible_glm_tokens"].append(
                np.full(count, request.visible_glm_tokens, dtype=np.int32)
            )
            arrays["target_index"].append(
                np.asarray(request.target_indices, dtype=np.int32)
            )
            arrays["reference_label"].append(labels)
            arrays["indices"].append(request_indices)
            arrays["probabilities"].append(probabilities[sequence_cursor:stop])
            arrays["top1"].append(request_top1)
            arrays["confidence"].append(confidence[sequence_cursor:stop])
            descriptors.append(
                {
                    "request_id": request_id,
                    "event_index": request.event_index,
                    "history_kind": request.history_kind,
                    "visible_glm_tokens": request.visible_glm_tokens,
                    "visible_source_prefix": request.visible_source_prefix,
                    "position_start": cursor,
                    "position_stop": cursor + count,
                    "positions": count,
                    "prefix_sha256": request.prefix_sha256,
                    "target_sha256": request.target_sha256,
                    "final": request.final,
                    "reference_in_topk": int(
                        np.count_nonzero(
                            (request_indices == labels[:, None]).any(axis=1)
                        )
                    ),
                    "teacher_top1_correct": int(
                        np.count_nonzero(request_top1 == labels)
                    ),
                }
            )
            cursor += count
            sequence_cursor = stop
            request_id += 1
        if sequence_cursor != sequence_positions:
            raise AssertionError("V1 teacher request cursor did not close")
    combined = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    if len(combined["reference_label"]) != cursor:
        raise AssertionError("V1 teacher sample cursor did not close")
    return combined, descriptors


def save_v1_bundle(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite V1 teacher bundle: {path}")
    values: dict[str, np.ndarray] = {
        "bundle_schema": np.asarray([V1_CACHE_SCHEMA])
    }
    manifest: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        arrays = row["arrays"]
        descriptors = row["requests"]
        if not isinstance(arrays, dict) or not isinstance(descriptors, list):
            raise TypeError("V1 teacher row is malformed")
        prefix = f"row_{row_index}"
        for name, value in arrays.items():
            values[f"{prefix}_{name}"] = np.asarray(value)
        labels = arrays["reference_label"]
        manifest.append(
            {
                "schema_version": V1_CACHE_SCHEMA,
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "source_manifest_record": int(row["source_manifest_record"]),
                "bundle_path": str(path.resolve()),
                "bundle_row": row_index,
                "requests": descriptors,
                "request_count": len(descriptors),
                "teacher_positions": int(len(labels)),
                "teacher_top1_correct": int(
                    np.count_nonzero(arrays["top1"] == labels)
                ),
                "reference_in_topk": int(
                    np.count_nonzero(
                        (arrays["indices"] == labels[:, None]).any(axis=1)
                    )
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)
    return manifest


__all__ = [
    "V1_CACHE_SCHEMA",
    "V1_HISTORY_IDS",
    "combine_v1_sample",
    "save_v1_bundle",
]
