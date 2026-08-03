"""Build deterministic direction-balanced source selections for Stage-B-v3."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from array import array
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "simul_uniss_stage_b_v3_balanced_selection_v1"


def _direction(record: dict[str, object]) -> str:
    source = str(record.get("src_lang", "")).lower()
    target = str(record.get("tgt_lang", "")).lower()
    if source in {"eng", "en"} and target in {"cmn", "zh", "zho"}:
        return "eng->cmn"
    if source in {"cmn", "zh", "zho"} and target in {"eng", "en"}:
        return "cmn->eng"
    raise ValueError(f"unsupported direction: {source}->{target}")


def build(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source_manifest).resolve()
    offsets = load_index(source)
    if offsets is None:
        raise ValueError(f"missing source index for {source}")
    order = list(range(len(offsets)))
    random.Random(args.seed).shuffle(order)
    selected: dict[str, list[dict[str, object]]] = {"eng->cmn": [], "cmn->eng": []}
    with source.open("rb") as handle:
        for index in order:
            if not args.all_records and all(
                len(values) >= args.per_direction for values in selected.values()
            ):
                break
            handle.seek(offsets[index])
            record = json.loads(handle.readline())
            direction = _direction(record)
            if not args.all_records and len(selected[direction]) >= args.per_direction:
                continue
            selected[direction].append(
                {
                    "schema_version": SCHEMA,
                    "source_manifest_index": index,
                    "source_manifest_offset": offsets[index],
                    "id": record.get("id"),
                    "src_lang": record.get("src_lang"),
                    "tgt_lang": record.get("tgt_lang"),
                    "direction": direction,
                }
            )
    if not args.all_records and any(
        len(values) < args.per_direction for values in selected.values()
    ):
        raise RuntimeError(
            "insufficient direction records: "
            + json.dumps({key: len(value) for key, value in selected.items()})
        )
    rows: list[dict[str, object]] = []
    maximum = max(len(values) for values in selected.values())
    for index in range(maximum):
        for direction in ("eng->cmn", "cmn->eng"):
            if index < len(selected[direction]):
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
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
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
        "source_manifest": str(source),
        "selection_manifest": str(output),
        "records": len(rows),
        "directions": {key: len(value) for key, value in selected.items()},
        "seed": args.seed,
        "index": write_index(output, positions),
    }
    marker = output.with_suffix(output.suffix + ".complete.json")
    marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-direction", type=int, default=50_000)
    parser.add_argument("--all-records", action="store_true")
    parser.add_argument("--seed", type=int, default=20_260_803)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
