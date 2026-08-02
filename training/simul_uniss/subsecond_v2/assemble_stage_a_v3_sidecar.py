"""Assemble ordered Stage-A-v3 sidecar part manifests without copying shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from pathlib import Path

from training.simul_uniss.jsonl_index import write_index


SCHEMA = "simul_uniss_stage_a_v3_causal_sidecar_assembled_v1"


def assemble(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root).resolve()
    markers = []
    for rank in range(args.world_size):
        path = root / f"part-{rank:02d}" / "PART_COMPLETE.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            raise ValueError(f"incomplete sidecar part: {path}")
        markers.append(value)
    markers.sort(key=lambda value: int(value["assigned_start"]))
    output = root / "manifest.jsonl"
    temporary = root / f".manifest.jsonl.tmp.{os.getpid()}"
    offsets = array("Q")
    position = 0
    records = 0
    try:
        with temporary.open("wb") as target:
            for marker in markers:
                with Path(str(marker["manifest"])).open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        offsets.append(position)
                        target.write(line)
                        position += len(line)
                        records += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "mode": markers[0]["mode"] if markers else None,
        "world_size": args.world_size,
        "records": records,
        "target_tokens": sum(int(value["target_tokens"]) for value in markers),
        "manifest": str(output),
        "index": write_index(output, offsets),
        "parts": [str(root / f"part-{rank:02d}") for rank in range(args.world_size)],
    }
    descriptor, name = tempfile.mkstemp(prefix=".STAGE_A_V3_COMPLETE.", dir=root)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, root / "STAGE_A_V3_COMPLETE.json")
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    assemble(parse_args())
