"""Materialize a small direction-balanced Phase3 sensitivity manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "simul_uniss_stage_b_v3_phase3_eval_manifest_v1"
DIRECTIONS = ("eng->cmn", "cmn->eng")


def _read_at(path: Path, offset: int) -> dict[str, object]:
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def build(args: argparse.Namespace) -> dict[str, object]:
    selection = Path(args.selection_manifest).resolve()
    source = Path(args.source_manifest).resolve()
    selection_offsets = load_index(selection)
    source_offsets = load_index(source)
    if selection_offsets is None or source_offsets is None:
        raise ValueError("selection and source manifests require binary indexes")

    selected: dict[str, list[dict[str, object]]] = {name: [] for name in DIRECTIONS}
    for offset in selection_offsets:
        row = _read_at(selection, offset)
        direction = str(row.get("direction"))
        if direction not in selected or len(selected[direction]) >= args.per_direction:
            continue
        source_index = int(row["source_manifest_index"])
        if source_index >= len(source_offsets):
            raise IndexError(source_index)
        source_row = _read_at(source, source_offsets[source_index])
        source_row["stage_b_v3_eval_direction"] = direction
        source_row["stage_b_v3_source_manifest_index"] = source_index
        selected[direction].append(source_row)
        if all(len(rows) >= args.per_direction for rows in selected.values()):
            break
    if any(len(rows) != args.per_direction for rows in selected.values()):
        raise RuntimeError(
            "insufficient balanced validation rows: "
            + json.dumps({name: len(rows) for name, rows in selected.items()})
        )

    rows: list[dict[str, object]] = []
    for index in range(args.per_direction):
        for direction in DIRECTIONS:
            rows.append(selected[direction][index])

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(name)
    positions = array("Q")
    position = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                encoded = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                positions.append(position)
                handle.write(encoded)
                position += len(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "selection_manifest": str(selection),
        "source_manifest": str(source),
        "output": str(output),
        "records": len(rows),
        "directions": {name: len(values) for name, values in selected.items()},
        "index": write_index(output, positions),
    }
    output.with_suffix(output.suffix + ".complete.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-direction", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
