#!/usr/bin/env python3
"""Write a tiny immutable fingerprint for the Phase3 HF/native handoff."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from safetensors import safe_open


SCHEMA = "uniss_phase3_embedding_fingerprint_v1"
DEFAULT_ROWS = (0, 1, 151_643, 180_406)
DEFAULT_COLUMNS = (0, 1, 127, 895)


def build(source: Path, output: Path) -> dict[str, object]:
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        embedding = handle.get_slice("model.embed_tokens.weight")
        shape = embedding.get_shape()
        if shape != [180_480, 896]:
            raise ValueError(f"unexpected Phase3 embedding shape: {shape}")
        values = embedding[list(DEFAULT_ROWS), :][:, list(DEFAULT_COLUMNS)].float().tolist()
    payload = {
        "schema_version": SCHEMA,
        "source": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "rows": list(DEFAULT_ROWS),
        "columns": list(DEFAULT_COLUMNS),
        "values": values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
