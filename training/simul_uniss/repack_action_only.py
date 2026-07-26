"""Resumable action-only repacking for isolated long-context experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterator

from training.simul_uniss.full_data_pipeline import (
    atomic_write_json,
    atomic_write_text,
    concatenate,
    file_metadata,
)
from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.mask_action_samples import mask_action_sample
from training.simul_uniss.pack_sequences import (
    iter_jsonl,
    make_shifted_sample,
    pack_samples,
)


PACK_MARKER_SCHEMA = "simul_uniss_action_repack_part_v1"
ASSEMBLY_MARKER_SCHEMA = "simul_uniss_action_repack_assembly_v1"
READY_MARKER_SCHEMA = "simul_uniss_action_repack_ready_v1"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _validate_source(metadata: object, source: Path) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("missing source metadata")
    stat = source.stat()
    if Path(str(metadata.get("path"))).resolve() != source.resolve():
        raise ValueError(f"source path changed: {source}")
    if int(metadata.get("size_bytes", -1)) != stat.st_size:
        raise ValueError(f"source size changed: {source}")
    if int(metadata.get("mtime_ns", -1)) != stat.st_mtime_ns:
        raise ValueError(f"source mtime changed: {source}")


def verify_part(marker_path: Path) -> dict[str, object]:
    marker = _load_object(marker_path)
    if marker.get("schema_version") != PACK_MARKER_SCHEMA:
        raise ValueError(f"unexpected marker schema in {marker_path}")
    source = Path(str(marker["source"]["path"]))  # type: ignore[index]
    output = Path(str(marker["output"]["path"]))  # type: ignore[index]
    _validate_source(marker.get("source"), source)
    output_metadata = marker.get("output")
    if not isinstance(output_metadata, dict) or not output.is_file():
        raise FileNotFoundError(output)
    if output.stat().st_size != int(output_metadata.get("size_bytes", -1)):
        raise ValueError(f"output size changed: {output}")
    if int(marker.get("packed_records", 0)) <= 0:
        raise ValueError(f"empty packed part: {marker_path}")
    return marker


def pack_action(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.input)
    output = Path(args.output)
    marker_path = Path(args.marker)
    if marker_path.is_file():
        marker = verify_part(marker_path)
        if int(marker.get("seq_length", -1)) != args.seq_length:
            raise ValueError(f"sequence length mismatch in {marker_path}")
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing unmarked output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    input_records = 0
    represented_samples = 0
    dropped_overlong = 0

    def shifted_samples() -> Iterator:
        nonlocal input_records, dropped_overlong
        for sample in iter_jsonl(source):
            input_records += 1
            shifted = make_shifted_sample(mask_action_sample(sample))
            if shifted.length > args.seq_length:
                dropped_overlong += 1
            yield shifted

    packed_records = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for item in pack_samples(
                shifted_samples(),
                args.seq_length,
                drop_overlong=True,
            ):
                represented_samples += len(item["source_ids"])
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                packed_records += 1
            handle.flush()
            os.fsync(handle.fileno())
        if packed_records <= 0 or represented_samples + dropped_overlong != input_records:
            raise ValueError(
                "action repack accounting mismatch: "
                f"input={input_records}, represented={represented_samples}, "
                f"dropped={dropped_overlong}, packed={packed_records}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    marker = {
        "schema_version": PACK_MARKER_SCHEMA,
        "seq_length": args.seq_length,
        "source": file_metadata(source),
        "output": file_metadata(output),
        "input_records": input_records,
        "represented_samples": represented_samples,
        "dropped_overlong": dropped_overlong,
        "packed_records": packed_records,
    }
    atomic_write_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def verify_assembly(marker_path: Path) -> dict[str, object]:
    marker = _load_object(marker_path)
    if marker.get("schema_version") != ASSEMBLY_MARKER_SCHEMA:
        raise ValueError(f"unexpected assembly schema in {marker_path}")
    output = Path(str(marker["output"]["path"]))  # type: ignore[index]
    metadata = marker.get("output")
    if not isinstance(metadata, dict) or not output.is_file():
        raise FileNotFoundError(output)
    if output.stat().st_size != int(metadata.get("size_bytes", -1)):
        raise ValueError(f"assembled output size changed: {output}")
    offsets = load_index(output)
    if offsets is None or len(offsets) != int(marker.get("packed_records", -1)):
        raise ValueError(f"missing or stale assembled sidecar index: {output}")
    return marker


def assemble_action(args: argparse.Namespace) -> dict[str, object]:
    marker_path = Path(args.marker)
    if marker_path.is_file():
        marker = verify_assembly(marker_path)
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker

    parts_root = Path(args.parts_root)
    part_markers: list[dict[str, object]] = []
    part_paths: list[Path] = []
    for index in range(args.shard_start, args.shard_start + args.shard_count):
        part_dir = parts_root / f"train-{index:05d}"
        part_marker = verify_part(part_dir / "PACK_ACTION_COMPLETE.json")
        if int(part_marker.get("seq_length", -1)) != args.seq_length:
            raise ValueError(f"sequence length mismatch in {part_dir}")
        part_markers.append(part_marker)
        part_paths.append(Path(str(part_marker["output"]["path"])))  # type: ignore[index]

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing unmarked assembled output: {output}")
    index_metadata = concatenate(part_paths, output)
    marker = {
        "schema_version": ASSEMBLY_MARKER_SCHEMA,
        "seq_length": args.seq_length,
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "input_records": sum(int(value["input_records"]) for value in part_markers),
        "represented_samples": sum(int(value["represented_samples"]) for value in part_markers),
        "dropped_overlong": sum(int(value["dropped_overlong"]) for value in part_markers),
        "packed_records": sum(int(value["packed_records"]) for value in part_markers),
        "output": file_metadata(output),
        "offset_index": index_metadata,
    }
    atomic_write_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def generate_schedule(args: argparse.Namespace) -> dict[str, int]:
    marker = verify_assembly(Path(args.assembly_marker))
    records = int(marker["packed_records"])
    train_iters = max(1, math.ceil(records * args.epochs / args.global_batch_size))
    warmup_iters = max(1, math.ceil(train_iters * args.warmup_fraction))
    lines = [
        "# Generated from isolated long-context action repacking; do not edit by hand.",
        f'ACTION_SEQ_LENGTH="{int(marker["seq_length"])}"',
        f'ACTION_SOURCE_RECORDS="{int(marker["input_records"])}"',
        f'ACTION_REPRESENTED_SAMPLES="{int(marker["represented_samples"])}"',
        f'ACTION_DROPPED_OVERLONG="{int(marker["dropped_overlong"])}"',
        f'ACTION_PACKED_RECORDS="{records}"',
        f'STAGE3_TRAIN_ITERS="${{STAGE3_TRAIN_ITERS:-{train_iters}}}"',
        f'STAGE3_QWEN_WARMUP_ITERS="${{STAGE3_QWEN_WARMUP_ITERS:-{warmup_iters}}}"',
    ]
    atomic_write_text(Path(args.output), "\n".join(lines) + "\n")
    result = {"packed_records": records, "train_iters": train_iters, "warmup_iters": warmup_iters}
    print(json.dumps(result, sort_keys=True))
    return result


def finalize(args: argparse.Namespace) -> dict[str, object]:
    assembly = verify_assembly(Path(args.assembly_marker))
    validation = verify_part(Path(args.validation_marker))
    schedule = Path(args.schedule)
    if not schedule.is_file():
        raise FileNotFoundError(schedule)
    marker = {
        "schema_version": READY_MARKER_SCHEMA,
        "seq_length": int(assembly["seq_length"]),
        "assembly_marker": file_metadata(Path(args.assembly_marker)),
        "validation_marker": file_metadata(Path(args.validation_marker)),
        "schedule": file_metadata(schedule),
    }
    atomic_write_json(Path(args.output), marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--input", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--marker", required=True)
    pack.add_argument("--seq-length", type=int, required=True)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--parts-root", required=True)
    assemble.add_argument("--shard-start", type=int, required=True)
    assemble.add_argument("--shard-count", type=int, required=True)
    assemble.add_argument("--seq-length", type=int, required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--marker", required=True)

    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--assembly-marker", required=True)
    schedule.add_argument("--output", required=True)
    schedule.add_argument("--global-batch-size", type=int, required=True)
    schedule.add_argument("--epochs", type=float, default=1.0)
    schedule.add_argument("--warmup-fraction", type=float, default=0.05)

    ready = subparsers.add_parser("finalize")
    ready.add_argument("--assembly-marker", required=True)
    ready.add_argument("--validation-marker", required=True)
    ready.add_argument("--schedule", required=True)
    ready.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pack":
        pack_action(args)
    elif args.command == "assemble":
        assemble_action(args)
    elif args.command == "schedule":
        generate_schedule(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
