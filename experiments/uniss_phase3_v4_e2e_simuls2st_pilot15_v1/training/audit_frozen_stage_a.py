"""Bitwise audit of frozen Stage-A tensors across E2E canary checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


SCHEMA_VERSION = "uniss_e2e_frozen_stage_a_bitwise_audit_v1"
PREFIX = "stage_a_objective."


def _write_new_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _stage_a_metadata(checkpoint: str | Path) -> dict[str, object]:
    root = Path(checkpoint).resolve()
    if not (root / ".metadata").is_file():
        raise FileNotFoundError(root / ".metadata")
    metadata = FileSystemReader(str(root)).read_metadata().state_dict_metadata
    selected = {key: value for key, value in metadata.items() if key.startswith(PREFIX)}
    if not selected:
        raise ValueError(f"checkpoint has no frozen Stage-A tensors: {root}")
    if any(not hasattr(value, "size") or not hasattr(value, "properties") for value in selected.values()):
        raise TypeError(f"frozen Stage-A metadata contains non-tensors: {root}")
    return selected


def tensor_hashes(checkpoint: str | Path) -> dict[str, object]:
    root = Path(checkpoint).resolve()
    metadata = _stage_a_metadata(root)
    state = {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype, device="cpu")
        for key, value in metadata.items()
    }
    dcp.load(state, checkpoint_id=str(root))
    per_tensor: dict[str, str] = {}
    tree = hashlib.sha256()
    total_bytes = 0
    for key in sorted(state):
        tensor = state[key].detach().contiguous().view(torch.uint8)
        raw = tensor.numpy().tobytes(order="C")
        digest = hashlib.sha256(raw).hexdigest()
        per_tensor[key] = digest
        encoded_key = key.encode("utf-8")
        tree.update(len(encoded_key).to_bytes(8, "little"))
        tree.update(encoded_key)
        tree.update(bytes.fromhex(digest))
        total_bytes += len(raw)
    return {
        "checkpoint": str(root),
        "tensors": len(per_tensor),
        "bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
        "tensor_sha256": per_tensor,
    }


def audit_frozen_stage_a(
    reference: str | Path,
    candidates: Sequence[tuple[str, str | Path]],
) -> dict[str, object]:
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("frozen Stage-A audit requires unique named candidates")
    baseline = tensor_hashes(reference)
    baseline_hashes = baseline["tensor_sha256"]
    if not isinstance(baseline_hashes, dict):
        raise TypeError("reference Stage-A tensor hash map is malformed")
    results = []
    passed = True
    for name, checkpoint in candidates:
        candidate = tensor_hashes(checkpoint)
        candidate_hashes = candidate["tensor_sha256"]
        if not isinstance(candidate_hashes, dict):
            raise TypeError("candidate Stage-A tensor hash map is malformed")
        missing = sorted(set(baseline_hashes) - set(candidate_hashes))
        unexpected = sorted(set(candidate_hashes) - set(baseline_hashes))
        changed = sorted(
            key
            for key in set(baseline_hashes) & set(candidate_hashes)
            if baseline_hashes[key] != candidate_hashes[key]
        )
        exact = not missing and not unexpected and not changed
        passed = passed and exact
        results.append(
            {
                "name": name,
                "checkpoint": candidate["checkpoint"],
                "tensors": candidate["tensors"],
                "bytes": candidate["bytes"],
                "tree_sha256": candidate["tree_sha256"],
                "exact_bitwise_match": exact,
                "missing": missing,
                "unexpected": unexpected,
                "changed": changed,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "reference": {
            key: value for key, value in baseline.items() if key != "tensor_sha256"
        },
        "candidates": results,
        "exact_bitwise_match": passed,
    }


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_frozen_stage_a(args.reference, args.candidate)
    _write_new_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
