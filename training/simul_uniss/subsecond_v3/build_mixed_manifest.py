"""Interleave exact-prefix-hidden and clone supervision without copying shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "simul_uniss_stage_b_v3_mixed_manifest_v1"


def _row(path: Path, offset: int) -> dict[str, object]:
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def build(args: argparse.Namespace) -> dict[str, object]:
    selection = Path(args.selection_manifest).resolve()
    prefix = Path(args.prefix_manifest).resolve()
    clone = Path(args.clone_manifest).resolve()
    selection_offsets = load_index(selection)
    prefix_offsets = load_index(prefix)
    clone_offsets = load_index(clone)
    if selection_offsets is None or prefix_offsets is None or clone_offsets is None:
        raise ValueError("all input manifests require binary indexes")
    if len(selection_offsets) != len(prefix_offsets):
        raise ValueError(
            f"selection/prefix size mismatch: {len(selection_offsets)} != {len(prefix_offsets)}"
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(name)
    positions = array("Q")
    position = records = 0
    directions: dict[str, int] = {}
    supervision: dict[str, int] = {}
    try:
        with os.fdopen(descriptor, "wb") as target:
            for index, (selection_offset, prefix_offset) in enumerate(
                zip(selection_offsets, prefix_offsets)
            ):
                selected = _row(selection, selection_offset)
                prefix_row = _row(prefix, prefix_offset)
                source_index = int(selected["source_manifest_index"])
                if int(prefix_row["source_manifest_index"]) != source_index:
                    raise ValueError(f"prefix selection mismatch at row {index}")
                if source_index >= len(clone_offsets):
                    raise IndexError(source_index)
                clone_row = _row(clone, clone_offsets[source_index])
                if int(clone_row["source_manifest_index"]) != source_index:
                    raise ValueError(f"clone source index mismatch at {source_index}")
                direction = str(selected["direction"])
                directions[direction] = directions.get(direction, 0) + 2
                rows = (
                    (
                        "exact_prefix80_hidden",
                        {**prefix_row, "schema_version": SCHEMA},
                    ),
                    (
                        "streaming_clone_hidden",
                        {**clone_row, "schema_version": SCHEMA},
                    ),
                )
                for mode, row in rows:
                    row.update(
                        {
                            "src_lang": selected.get("src_lang"),
                            "tgt_lang": selected.get("tgt_lang"),
                            "direction": direction,
                            "supervision_mode": mode,
                            "selection_row": index,
                        }
                    )
                    supervision[mode] = supervision.get(mode, 0) + 1
                    encoded = (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode()
                    positions.append(position)
                    target.write(encoded)
                    position += len(encoded)
                    records += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "selection_manifest": str(selection),
        "prefix_manifest": str(prefix),
        "clone_manifest": str(clone),
        "mixed_manifest": str(output),
        "records": records,
        "directions": directions,
        "supervision": supervision,
        "index": write_index(output, positions),
    }
    marker = output.with_suffix(output.suffix + ".complete.json")
    marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--clone-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
