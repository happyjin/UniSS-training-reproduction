#!/usr/bin/env python3
"""Validate every packed source-GLM span against its immutable causal cache."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    PACKED_TRAJECTORY_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.dataset import (
    _parse_cache_reference,
    _source_glm_positions,
)
from training import constants_uniss as c


AUDIT_SCHEMA = "uniss_true_subsecond_pilot15_packed_causal_audit_v1"


class CausalBundleLRU:
    """Load only the two causal arrays instead of every teacher top-k array."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self.values: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def row(self, path: Path, index: int) -> list[int]:
        path = path.resolve()
        value = self.values.pop(path, None)
        if value is None:
            with np.load(path, allow_pickle=False) as bundle:
                value = (
                    bundle["causal_tokens"].astype(np.int64, copy=True),
                    bundle["causal_token_offsets"].astype(np.int64, copy=True),
                )
        self.values[path] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        tokens, offsets = value
        if not 0 <= index < len(offsets) - 1:
            raise IndexError(f"causal row {index} is outside {path}")
        start, end = int(offsets[index]), int(offsets[index + 1])
        return tokens[start:end].tolist()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def audit_part(path: Path) -> dict[str, object]:
    bundles = CausalBundleLRU(8)
    packed_records = annotations = numeric_collisions = 0
    errors: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            packed_records += 1
            if value.get("schema_version") != PACKED_TRAJECTORY_SCHEMA:
                errors.append({"line": line_number, "error": "schema"})
                continue
            tokens = value["tokens"]
            rows = zip(
                value["source_ids"],
                value["sample_boundaries"],
                value["trajectory_sidecars"],
            )
            for source_id, boundary, sidecar in rows:
                annotations += 1
                start, end = (int(boundary[0]), int(boundary[1]))
                try:
                    positions = _source_glm_positions(tokens, start, end)
                    candidate_count = sum(
                        c.GLM_SEMANTIC_OFFSET
                        <= int(tokens[position])
                        < c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
                        for position in range(start, end)
                    )
                    numeric_collisions += max(0, candidate_count - len(positions))
                    packed_codes = [
                        int(tokens[position]) - c.GLM_SEMANTIC_OFFSET
                        for position in positions
                    ]
                    cache_path, cache_index = _parse_cache_reference(
                        str(sidecar["frontend_token_cache"]), "causal"
                    )
                    cached = bundles.row(cache_path, cache_index)
                    cached = cached[: len(packed_codes)]
                    if packed_codes != cached:
                        raise ValueError("packed/cache causal codes differ")
                except Exception as exc:
                    if len(errors) < 32:
                        errors.append(
                            {
                                "line": line_number,
                                "source_id": str(source_id),
                                "error": str(exc),
                            }
                        )
    return {
        "part": path.parent.name,
        "path": str(path.resolve()),
        "packed_records": packed_records,
        "annotations": annotations,
        "numeric_id_collisions_excluded": numeric_collisions,
        "errors": errors,
        "passed": not errors,
    }


def audit(root: Path, workers: int = 15) -> dict[str, object]:
    paths = sorted(root.glob("part-*/packed_trajectory.jsonl"))
    if len(paths) != 15:
        raise ValueError(f"expected 15 packed parts, found {len(paths)}")
    with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        parts = list(executor.map(audit_part, paths))
    result = {
        "schema_version": AUDIT_SCHEMA,
        "parts": parts,
        "packed_records": sum(int(value["packed_records"]) for value in parts),
        "annotations": sum(int(value["annotations"]) for value in parts),
        "numeric_id_collisions_excluded": sum(
            int(value["numeric_id_collisions_excluded"]) for value in parts
        ),
        "error_count": sum(len(value["errors"]) for value in parts),
        "passed": all(bool(value["passed"]) for value in parts),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    result = audit(args.root, args.workers)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
