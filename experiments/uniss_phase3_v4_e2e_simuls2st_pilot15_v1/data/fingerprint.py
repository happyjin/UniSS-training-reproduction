#!/usr/bin/env python3
"""Parallel content fingerprints for immutable distributed checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


FINGERPRINT_SCHEMA = "uniss_checkpoint_tree_fingerprint_v1"


def _sha256_file(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint contains no regular files: {root}")
    return files


def fingerprint_checkpoint(root: Path, *, workers: int) -> dict[str, object]:
    root = root.resolve()
    files = _checkpoint_files(root)
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(files)))) as pool:
        digests = list(pool.map(_sha256_file, files))
    entries = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(files, digests)
    ]
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "path": str(root),
        "files": len(entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "entries": entries,
    }


def _parse_checkpoints(values: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("--checkpoint values must use LABEL=/absolute/path")
        if label in labels:
            raise ValueError(f"duplicate checkpoint label: {label}")
        labels.add(label)
        parsed.append((label, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite fingerprint report: {args.output}")
    checkpoints = _parse_checkpoints(args.checkpoint)
    report = {
        "schema_version": FINGERPRINT_SCHEMA,
        "status": "complete",
        "workers_per_checkpoint": max(1, int(args.workers)),
        "checkpoints": {
            label: fingerprint_checkpoint(path, workers=args.workers)
            for label, path in checkpoints
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "output": str(args.output.resolve()),
        "checkpoints": {
            label: {
                "path": value["path"],
                "files": value["files"],
                "bytes": value["bytes"],
                "sha256": value["sha256"],
            }
            for label, value in report["checkpoints"].items()  # type: ignore[union-attr]
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
