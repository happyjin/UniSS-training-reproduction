"""Resumable assembly and schedule generation for full Simul-UniSS data."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from array import array
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from training.simul_uniss import PACKED_SCHEMA_VERSION
from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.schema import sha256_file

PREPARED_MARKER = "PREPARE_COMPLETE.json"
PACKED_MARKER = "PACK_COMPLETE.json"


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def first_jsonl(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected JSON object in {path}")
                return value
    raise ValueError(f"empty JSONL file: {path}")


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def relative_file_metadata(path: Path) -> dict[str, object]:
    return {"relative_path": path.name, "size_bytes": path.stat().st_size}


def mark_prepared_part(
    source: Path, part_dir: Path, shard_index: int, published_dir: Path | None = None
) -> dict[str, object]:
    schedules = part_dir / "schedules.jsonl"
    samples = part_dir / "samples.jsonl"
    manifest_path = part_dir / "manifest.json"
    stats_path = part_dir / "stats.json"
    for path in (source, schedules, samples, manifest_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = load_json(manifest_path)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(shards[0], dict):
        raise ValueError(f"{manifest_path} must describe exactly one shard")
    source_entry = shards[0]
    if Path(str(source_entry.get("path"))).resolve() != source.resolve():
        raise ValueError(f"source path mismatch in {manifest_path}")
    source_sha256 = sha256_file(source)
    if source_entry.get("sha256") != source_sha256:
        raise ValueError(f"source SHA256 mismatch in {manifest_path}")

    stats = load_json(stats_path)
    if published_dir is not None:
        stats["schedules"] = str((published_dir / "schedules.jsonl").resolve())
        stats["samples"] = str((published_dir / "samples.jsonl").resolve())
        atomic_write_json(stats_path, stats)
    records = int(stats.get("records", 0))
    schedule_records = count_jsonl(schedules)
    sample_records = count_jsonl(samples)
    if records <= 0 or schedule_records != records or sample_records != records:
        raise ValueError(
            f"record mismatch for shard {shard_index}: stats={records}, "
            f"schedules={schedule_records}, samples={sample_records}"
        )
    marker = {
        "schema_version": "simul_uniss_full_prepared_part_v1",
        "shard_index": shard_index,
        "source": {**file_metadata(source), "sha256": source_sha256},
        "records": records,
        "events": int(stats.get("events", 0)),
        "wait_events": int(stats.get("wait_events", 0)),
        "write_events": int(stats.get("write_events", 0)),
        "schedules": relative_file_metadata(schedules),
        "samples": relative_file_metadata(samples),
    }
    atomic_write_json(part_dir / PREPARED_MARKER, marker)
    return marker


def verify_marker_files(marker_path: Path, fields: Iterable[str]) -> dict[str, object]:
    marker = load_json(marker_path)
    for field in fields:
        metadata = marker.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(f"missing {field} metadata in {marker_path}")
        path = Path(str(metadata.get("path")))
        if not path.is_file() or path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise ValueError(f"stale or missing {field} file recorded by {marker_path}")
    return marker


def verify_prepared_part(source: Path, part_dir: Path, shard_index: int) -> dict[str, object]:
    marker_path = part_dir / PREPARED_MARKER
    marker = load_json(marker_path)
    if int(marker.get("shard_index", -1)) != shard_index:
        raise ValueError(f"shard index mismatch in {marker_path}")
    source_metadata = marker["source"]
    assert isinstance(source_metadata, dict)
    if Path(str(source_metadata.get("path"))).resolve() != source.resolve():
        raise ValueError(f"source path mismatch in {marker_path}")
    source_stat = source.stat()
    if source_stat.st_mtime_ns != int(source_metadata.get("mtime_ns", -1)):
        raise ValueError(f"source mtime changed after preparing {part_dir}")
    for field in ("schedules", "samples"):
        metadata = marker.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(f"missing {field} metadata in {marker_path}")
        path = part_dir / str(metadata.get("relative_path"))
        if not path.is_file() or path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise ValueError(f"stale or missing {field} file recorded by {marker_path}")
    return marker


def mark_packed_part(prepared_part: Path, packed_part: Path, shard_index: int) -> dict[str, object]:
    prepared_marker = verify_prepared_part(
        Path(str(load_json(prepared_part / PREPARED_MARKER)["source"]["path"])),
        prepared_part,
        shard_index,
    )
    interleaved = packed_part / "packed_interleaved.jsonl"
    action = packed_part / "packed_action.jsonl"
    for path in (interleaved, action):
        if not path.is_file():
            raise FileNotFoundError(path)
        if first_jsonl(path).get("schema_version") != PACKED_SCHEMA_VERSION:
            raise ValueError(f"unexpected packed schema in {path}")
    interleaved_count = count_jsonl(interleaved)
    action_count = count_jsonl(action)
    if interleaved_count <= 0 or action_count <= 0:
        raise ValueError(f"empty packed output for shard {shard_index}")
    marker = {
        "schema_version": "simul_uniss_full_packed_part_v1",
        "shard_index": shard_index,
        "prepared_marker_sha256": sha256_file(prepared_part / PREPARED_MARKER),
        "source_records": int(prepared_marker["records"]),
        "packed_interleaved_records": interleaved_count,
        "packed_action_records": action_count,
        "packed_interleaved": relative_file_metadata(interleaved),
        "packed_action": relative_file_metadata(action),
    }
    atomic_write_json(packed_part / PACKED_MARKER, marker)
    return marker


def verify_packed_part(prepared_part: Path, packed_part: Path, shard_index: int) -> dict[str, object]:
    prepared_marker = verify_prepared_part(
        Path(str(load_json(prepared_part / PREPARED_MARKER)["source"]["path"])),
        prepared_part,
        shard_index,
    )
    marker_path = packed_part / PACKED_MARKER
    marker = load_json(marker_path)
    if int(marker.get("shard_index", -1)) != shard_index:
        raise ValueError(f"shard index mismatch in {marker_path}")
    expected = sha256_file(prepared_part / PREPARED_MARKER)
    if marker.get("prepared_marker_sha256") != expected:
        raise ValueError(f"prepared part changed after packing shard {shard_index}")
    if int(marker.get("source_records", -1)) != int(prepared_marker["records"]):
        raise ValueError(f"source record mismatch in {marker_path}")
    for field in ("packed_interleaved", "packed_action"):
        metadata = marker.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(f"missing {field} metadata in {marker_path}")
        path = packed_part / str(metadata.get("relative_path"))
        if not path.is_file() or path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise ValueError(f"stale or missing {field} file recorded by {marker_path}")
    return marker


def concatenate(paths: Iterable[Path], destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        offsets = array("Q")
        offset = 0
        with temporary.open("wb") as output:
            for path in paths:
                with path.open("rb") as source:
                    for line in source:
                        if line in {b"\n", b"\r\n"}:
                            continue
                        offsets.append(offset)
                        output.write(line)
                        offset += len(line)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return write_index(destination, offsets)
    finally:
        temporary.unlink(missing_ok=True)


def assemble(args: argparse.Namespace) -> dict[str, object]:
    prepared_parts = Path(args.prepared_parts)
    packed_parts = Path(args.packed_parts)
    schedules_output = Path(args.schedules_output)
    interleaved_output = Path(args.interleaved_output)
    action_output = Path(args.action_output)
    manifest_output = Path(args.manifest_output)
    marker_output = Path(args.marker_output)

    if marker_output.is_file():
        marker = verify_marker_files(marker_output, ("schedules", "packed_interleaved", "packed_action"))
        for field in ("schedules", "packed_interleaved", "packed_action"):
            metadata = marker[field]
            assert isinstance(metadata, dict)
            offsets = load_index(Path(str(metadata["path"])))
            expected = int(marker[f"{field}_records"])
            if offsets is None or len(offsets) != expected:
                raise ValueError(f"missing or stale offset index for {field}")
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker

    prepared_markers: list[dict[str, object]] = []
    packed_markers: list[dict[str, object]] = []
    schedule_paths: list[Path] = []
    interleaved_paths: list[Path] = []
    action_paths: list[Path] = []
    for index in range(args.shard_start, args.shard_start + args.shard_count):
        name = f"train-{index:05d}"
        prepared_part = prepared_parts / name
        packed_part = packed_parts / name
        prepared_marker = load_json(prepared_part / PREPARED_MARKER)
        source = Path(str(prepared_marker["source"]["path"]))
        prepared_markers.append(verify_prepared_part(source, prepared_part, index))
        packed_markers.append(verify_packed_part(prepared_part, packed_part, index))
        schedule_paths.append(prepared_part / "schedules.jsonl")
        interleaved_paths.append(packed_part / "packed_interleaved.jsonl")
        action_paths.append(packed_part / "packed_action.jsonl")

    schedule_index = concatenate(schedule_paths, schedules_output)
    interleaved_index = concatenate(interleaved_paths, interleaved_output)
    action_index = concatenate(action_paths, action_output)

    manifest = {
        "schema_version": "simul_uniss_full_data_manifest_v1",
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "prepared_parts": prepared_markers,
        "packed_parts": packed_markers,
    }
    atomic_write_json(manifest_output, manifest)
    marker = {
        "schema_version": "simul_uniss_full_data_assembly_v1",
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "source_records": sum(int(value["records"]) for value in prepared_markers),
        "schedules_records": sum(int(value["records"]) for value in prepared_markers),
        "packed_interleaved_records": sum(
            int(value["packed_interleaved_records"]) for value in packed_markers
        ),
        "packed_action_records": sum(int(value["packed_action_records"]) for value in packed_markers),
        "schedules": file_metadata(schedules_output),
        "schedules_index": schedule_index,
        "packed_interleaved": file_metadata(interleaved_output),
        "packed_interleaved_index": interleaved_index,
        "packed_action": file_metadata(action_output),
        "packed_action_index": action_index,
        "manifest": {**file_metadata(manifest_output), "sha256": sha256_file(manifest_output)},
    }
    atomic_write_json(marker_output, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def epoch_iterations(records: int, global_batch_size: int, epochs: str) -> int:
    if records <= 0 or global_batch_size <= 0 or Decimal(epochs) <= 0:
        raise ValueError("records, global batch size, and epochs must be positive")
    value = Decimal(records) * Decimal(epochs) / Decimal(global_batch_size)
    return max(1, int(math.ceil(value)))


def generate_schedule(args: argparse.Namespace) -> dict[str, int]:
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup fraction must be in [0, 1)")
    assembly = verify_marker_files(
        Path(args.assembly_marker), ("schedules", "packed_interleaved", "packed_action")
    )
    interleaved = int(assembly["packed_interleaved_records"])
    action = int(assembly["packed_action_records"])
    stage3 = epoch_iterations(action, args.global_batch_size, args.stage3_epochs)
    stage4 = epoch_iterations(interleaved, args.global_batch_size, args.stage4_epochs)
    stage6 = epoch_iterations(interleaved, args.global_batch_size, args.stage6_epochs)
    values = {
        "STAGE3_TRAIN_ITERS": stage3,
        "STAGE4_TRAIN_ITERS": stage4,
        "STAGE6_TRAIN_ITERS": stage6,
        "STAGE3_QWEN_WARMUP_ITERS": max(1, math.ceil(stage3 * args.warmup_fraction)),
        "STAGE4_QWEN_WARMUP_ITERS": max(1, math.ceil(stage4 * args.warmup_fraction)),
        "STAGE6_QWEN_WARMUP_ITERS": max(1, math.ceil(stage6 * args.warmup_fraction)),
    }
    manifest_hash = str(assembly["manifest"]["sha256"])
    lines = [
        "# Generated from the completed full-data assembly; do not edit by hand.",
        f'FULL_DATA_MANIFEST_SHA256="{manifest_hash}"',
        f'FULL_SOURCE_RECORDS="{int(assembly["source_records"])}"',
        f'FULL_PACKED_INTERLEAVED_RECORDS="{interleaved}"',
        f'FULL_PACKED_ACTION_RECORDS="{action}"',
    ]
    for name, value in values.items():
        lines.append(f'{name}="${{{name}:-{value}}}"')
    atomic_write_text(Path(args.output), "\n".join(lines) + "\n")
    result = {"source_records": int(assembly["source_records"]), **values}
    print(json.dumps(result, sort_keys=True))
    return result


def finalize(args: argparse.Namespace) -> None:
    assembly_marker = Path(args.assembly_marker)
    schedule_file = Path(args.schedule_file)
    stage0_metrics = Path(args.stage0_metrics)
    for path in (assembly_marker, schedule_file, stage0_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    value = {
        "schema_version": "simul_uniss_full_data_ready_v1",
        "assembly_marker": {**file_metadata(assembly_marker), "sha256": sha256_file(assembly_marker)},
        "training_schedule": {**file_metadata(schedule_file), "sha256": sha256_file(schedule_file)},
        "stage0_metrics": {**file_metadata(stage0_metrics), "sha256": sha256_file(stage0_metrics)},
    }
    atomic_write_json(Path(args.output), value)
    print(json.dumps(value, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepared = subparsers.add_parser("mark-prepared")
    prepared.add_argument("--source", required=True)
    prepared.add_argument("--part-dir", required=True)
    prepared.add_argument("--published-dir", default=None)
    prepared.add_argument("--shard-index", type=int, required=True)

    verify_prepared = subparsers.add_parser("verify-prepared")
    verify_prepared.add_argument("--source", required=True)
    verify_prepared.add_argument("--part-dir", required=True)
    verify_prepared.add_argument("--shard-index", type=int, required=True)

    packed = subparsers.add_parser("mark-packed")
    packed.add_argument("--prepared-part", required=True)
    packed.add_argument("--packed-part", required=True)
    packed.add_argument("--shard-index", type=int, required=True)

    verify_packed = subparsers.add_parser("verify-packed")
    verify_packed.add_argument("--prepared-part", required=True)
    verify_packed.add_argument("--packed-part", required=True)
    verify_packed.add_argument("--shard-index", type=int, required=True)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--prepared-parts", required=True)
    assemble_parser.add_argument("--packed-parts", required=True)
    assemble_parser.add_argument("--shard-start", type=int, required=True)
    assemble_parser.add_argument("--shard-count", type=int, required=True)
    assemble_parser.add_argument("--schedules-output", required=True)
    assemble_parser.add_argument("--interleaved-output", required=True)
    assemble_parser.add_argument("--action-output", required=True)
    assemble_parser.add_argument("--manifest-output", required=True)
    assemble_parser.add_argument("--marker-output", required=True)

    schedule_parser = subparsers.add_parser("schedule")
    schedule_parser.add_argument("--assembly-marker", required=True)
    schedule_parser.add_argument("--output", required=True)
    schedule_parser.add_argument("--global-batch-size", type=int, required=True)
    schedule_parser.add_argument("--stage3-epochs", required=True)
    schedule_parser.add_argument("--stage4-epochs", required=True)
    schedule_parser.add_argument("--stage6-epochs", required=True)
    schedule_parser.add_argument("--warmup-fraction", type=float, default=0.05)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--assembly-marker", required=True)
    finalize_parser.add_argument("--schedule-file", required=True)
    finalize_parser.add_argument("--stage0-metrics", required=True)
    finalize_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "mark-prepared":
        print(
            json.dumps(
                mark_prepared_part(
                    Path(args.source),
                    Path(args.part_dir),
                    args.shard_index,
                    Path(args.published_dir) if args.published_dir else None,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "verify-prepared":
        print(
            json.dumps(
                verify_prepared_part(Path(args.source), Path(args.part_dir), args.shard_index),
                sort_keys=True,
            )
        )
    elif args.command == "mark-packed":
        print(
            json.dumps(
                mark_packed_part(Path(args.prepared_part), Path(args.packed_part), args.shard_index),
                sort_keys=True,
            )
        )
    elif args.command == "verify-packed":
        print(
            json.dumps(
                verify_packed_part(Path(args.prepared_part), Path(args.packed_part), args.shard_index),
                sort_keys=True,
            )
        )
    elif args.command == "assemble":
        assemble(args)
    elif args.command == "schedule":
        generate_schedule(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
