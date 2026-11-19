#!/usr/bin/env python3
"""Atomically move invalid UniST V2 manifest rows into reject manifests.

The production WDS tar files are intentionally left unchanged: rejected source
members become unreachable byte ranges, avoiding a 2.2 TiB tar rewrite.  The
JSONL manifest, reject JSONL, per-Parquet state, and global summary/commit marker
are updated in place.  Dry-run is the default; ``--apply`` enables mutation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow.parquet as pq


FILTER_VERSION = "unist_quality_v1"
REJECT_SCHEMA_VERSION = "unist-conversion-reject-v1"
MIN_DURATION_RATIO = 0.1
MAX_DURATION_RATIO = 10.0
GENERATION_CEILING_MS = 59_000
GENERATION_CEILING_QWEN_FRAMES = 750
MIN_QWEN_FRAMES = 2
COLLAPSE_DOMINANCE = 0.95
COLLAPSE_ADJACENT_REPEAT = 0.95
# dominance >= .95 mathematically implies a high adjacent-repeat lower bound;
# this prefilter avoids per-row value_counts over billions of ordinary tokens.
DOMINANCE_PREFILTER_ADJACENT = 0.88

FIELD_PATTERNS = {
    "codec_len": re.compile(br'"codec_len":([0-9]+)'),
    "source_duration_ms": re.compile(br'"source_duration_ms":([0-9]+)'),
    "target_duration_ms": re.compile(br'"target_duration_ms":([0-9]+)'),
    "source_row_index": re.compile(br'"source_row_index":([0-9]+)'),
    "source_audio_encoded_bytes": re.compile(br'"source_audio_encoded_bytes":([0-9]+)'),
}


def parse_args() -> argparse.Namespace:
    workspace = Path("/opt/dlami/nvme/zeyangsong")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=workspace / "data/unist/converted/unist_qwen3_s2s_v2")
    parser.add_argument(
        "--raw-root", type=Path, default=workspace / "data/unist/raw")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--reconcile-only", action="store_true",
        help="repair/finalize metadata after an interrupted or older apply")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".quality.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _required_int(line: bytes, field: str) -> int:
    match = FIELD_PATTERNS[field].search(line)
    if match is None:
        raise ValueError(f"manifest row has no {field}: {line[:200]!r}")
    return int(match.group(1))


def bicodec_collapse_metrics(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return collapse flags, adjacent ratios, and dominance ratios by raw row."""
    chunked = pq.read_table(raw_path, columns=["target_bicodec"])["target_bicodec"]
    array = chunked.combine_chunks()
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = array.values.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets)
    adjacent = np.zeros(len(lengths), dtype=np.float32)
    if len(values) > 1:
        equal = (values[:-1] == values[1:]).astype(np.int32, copy=False)
        prefix = np.empty(len(equal) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(equal, dtype=np.int64, out=prefix[1:])
        valid = lengths > 1
        repeat_counts = np.zeros(len(lengths), dtype=np.int64)
        repeat_counts[valid] = (
            prefix[offsets[1:][valid] - 1] - prefix[offsets[:-1][valid]])
        adjacent[valid] = repeat_counts[valid] / (lengths[valid] - 1)

    dominance = np.zeros(len(lengths), dtype=np.float32)
    candidates = np.flatnonzero(adjacent >= DOMINANCE_PREFILTER_ADJACENT)
    for index in candidates:
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end <= start:
            continue
        _, counts = np.unique(values[start:end], return_counts=True)
        dominance[index] = float(counts.max() / (end - start))
    collapse = (
        (adjacent >= COLLAPSE_ADJACENT_REPEAT)
        | (dominance >= COLLAPSE_DOMINANCE))
    return collapse, adjacent, dominance


def quality_flags(
    codec_len: int, source_duration_ms: int, target_duration_ms: int,
    collapsed: bool,
) -> list[str]:
    ratio = target_duration_ms / source_duration_ms
    flags = []
    if codec_len < MIN_QWEN_FRAMES:
        flags.append("qwen_codec_too_short")
    if (target_duration_ms >= GENERATION_CEILING_MS
            or codec_len == GENERATION_CEILING_QWEN_FRAMES):
        flags.append("target_generation_ceiling")
    if ratio < MIN_DURATION_RATIO or ratio > MAX_DURATION_RATIO:
        flags.append("duration_ratio_out_of_bounds")
    if collapsed:
        flags.append("target_bicodec_collapse")
    return flags


def reject_from_manifest(
    row: dict[str, Any], raw_path: Path, flags: list[str],
    adjacent_repeat: float, dominance: float,
) -> dict[str, Any]:
    provenance = row["provenance"]
    ratio = row["target_duration_ms"] / row["source_duration_ms"]
    priority = (
        "qwen_codec_too_short", "target_generation_ceiling",
        "duration_ratio_out_of_bounds", "target_bicodec_collapse")
    reason = next(item for item in priority if item in flags)
    return {
        "schema_version": REJECT_SCHEMA_VERSION,
        "reason": reason,
        "quality_flags": flags,
        "quality_filter_version": FILTER_VERSION,
        "id": row.get("id"),
        "record_id": row.get("record_id"),
        "dataset_name": row.get("dataset_name"),
        "split": row.get("split"),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
        "source_text": str(row.get("source_transcript") or "").strip(),
        "target_text": str(row.get("target_text") or "").strip(),
        "source_parquet": raw_path.name,
        "source_parquet_path": str(raw_path.resolve()),
        "source_row_index": int(provenance["source_row_index"]),
        "source_bicodec_len": int(provenance["source_bicodec_len"]),
        "target_bicodec_len": int(provenance["target_bicodec_len"]),
        "bicodec_global_len": int(provenance["bicodec_global_len"]),
        "source_duration_ms": int(row["source_duration_ms"]),
        "target_duration_ms": int(row["target_duration_ms"]),
        "target_source_duration_ratio": ratio,
        "target_qwen_codec_len": int(row["codec_len"]),
        "source_audio_encoded_bytes": int(row["source_audio_encoded_bytes"]),
        "target_bicodec_adjacent_repeat_ratio": adjacent_repeat,
        "target_bicodec_dominant_ratio": dominance,
    }


def paths_for_manifest(dataset_root: Path, manifest: Path) -> tuple[Path, Path, Path]:
    key = manifest.stem.removeprefix("manifest-")
    return (
        dataset_root / "rejects" / f"reject-{key}.jsonl",
        dataset_root / "state" / f"{key}.json",
        dataset_root / "manifests" / f"manifest-{key}.jsonl",
    )


def process_manifest(
    dataset_root_s: str, raw_root_s: str, manifest_s: str, apply: bool,
) -> dict[str, Any]:
    dataset_root, raw_root, manifest = map(Path, (
        dataset_root_s, raw_root_s, manifest_s))
    reject_path, state_path, expected_manifest = paths_for_manifest(dataset_root, manifest)
    if manifest != expected_manifest:
        raise AssertionError((manifest, expected_manifest))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    raw_path = raw_root / Path(state["input_file"]).name
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    collapse, adjacent, dominance = bicodec_collapse_metrics(raw_path)
    if len(collapse) != int(state["input_rows"]):
        raise AssertionError(
            f"raw/state row mismatch: {raw_path} {len(collapse)} != {state['input_rows']}")

    manifest_partial = manifest.with_suffix(manifest.suffix + ".quality.partial")
    reject_partial = reject_path.with_suffix(reject_path.suffix + ".quality.partial")
    for partial in (manifest_partial, reject_partial):
        partial.unlink(missing_ok=True)

    counts: Counter[str] = Counter()
    kept_rows = removed_rows = removed_frames = orphaned_bytes = 0
    manifest_output = manifest_partial.open("wb", buffering=16 * 1024 * 1024) if apply else None
    reject_output = reject_partial.open("wb", buffering=4 * 1024 * 1024) if apply else None
    try:
        if apply and reject_path.is_file():
            with reject_path.open("rb", buffering=4 * 1024 * 1024) as existing:
                shutil.copyfileobj(existing, reject_output, length=4 * 1024 * 1024)
        with manifest.open("rb", buffering=16 * 1024 * 1024) as source:
            for line in source:
                codec_len = _required_int(line, "codec_len")
                source_ms = _required_int(line, "source_duration_ms")
                target_ms = _required_int(line, "target_duration_ms")
                source_index = _required_int(line, "source_row_index")
                if not 0 <= source_index < len(collapse):
                    raise IndexError((manifest, source_index, len(collapse)))
                flags = quality_flags(
                    codec_len, source_ms, target_ms, bool(collapse[source_index]))
                if not flags:
                    kept_rows += 1
                    if manifest_output is not None:
                        manifest_output.write(line)
                    continue
                removed_rows += 1
                removed_frames += codec_len
                orphaned_bytes += _required_int(line, "source_audio_encoded_bytes")
                for flag in flags:
                    counts[f"flag:{flag}"] += 1
                row = orjson.loads(line)
                reject = reject_from_manifest(
                    row, raw_path, flags,
                    float(adjacent[source_index]), float(dominance[source_index]))
                counts[f"reason:{reject['reason']}"] += 1
                if reject_output is not None:
                    reject_output.write(orjson.dumps(
                        reject, option=orjson.OPT_APPEND_NEWLINE))
    finally:
        if manifest_output is not None:
            manifest_output.close()
        if reject_output is not None:
            reject_output.close()

    if kept_rows + removed_rows != int(state["output_rows"]):
        raise AssertionError(
            f"manifest/state row mismatch: {manifest} "
            f"{kept_rows}+{removed_rows}!={state['output_rows']}")
    if apply and removed_rows:
        new_state = dict(state)
        new_state["output_rows"] = kept_rows
        # Early V2 shards with zero conversion rejects predate the explicit
        # rejected_rows field; their dataclass default is semantically zero.
        new_state["rejected_rows"] = int(state.get("rejected_rows", 0)) + removed_rows
        new_state["qwen_codec_frames"] = int(state["qwen_codec_frames"]) - removed_frames
        new_state["source_audio_bytes"] = int(state["source_audio_bytes"]) - orphaned_bytes
        new_state["quality_filter"] = {
            "version": FILTER_VERSION,
            "filtered_rows": removed_rows,
            "removed_qwen_frames": removed_frames,
            "orphaned_source_audio_bytes": orphaned_bytes,
        }
        new_state["reject_manifest"] = str(reject_path.resolve())
        if new_state["output_rows"] + new_state["rejected_rows"] != new_state["input_rows"]:
            raise AssertionError(f"state closure failed: {state_path}")
        state_partial = state_path.with_suffix(state_path.suffix + ".quality.partial")
        state_partial.write_text(
            json.dumps(new_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Reject first, manifest second, state last. Re-running is idempotent
        # once the manifest replacement succeeds.
        os.replace(reject_partial, reject_path)
        os.replace(manifest_partial, manifest)
        os.replace(state_partial, state_path)
    elif apply:
        manifest_partial.unlink(missing_ok=True)
        reject_partial.unlink(missing_ok=True)

    return {
        "manifest": str(manifest),
        "input_manifest_rows": kept_rows + removed_rows,
        "kept_rows": kept_rows,
        "removed_rows": removed_rows,
        "removed_qwen_frames": removed_frames,
        "orphaned_source_audio_bytes": orphaned_bytes,
        "counts": dict(counts),
    }


def reconcile_legacy_state_accounting(dataset_root: Path) -> None:
    """Complete accounting for states written by the first filter revision.

    That revision already removed rows/codec frames but left source_audio_bytes
    at its pre-filter value. Quality reject rows retain the exact encoded byte
    count, so the migration is deterministic and idempotent.
    """
    for state_path in sorted((dataset_root / "state").glob("*.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("quality_filter") or {}).get("version") == FILTER_VERSION:
            continue
        key = state_path.stem
        reject_path = dataset_root / "rejects" / f"reject-{key}.jsonl"
        filtered_rows = removed_frames = orphaned_bytes = 0
        if reject_path.is_file():
            with reject_path.open("rb") as handle:
                for line in handle:
                    if b'"quality_filter_version":"' + FILTER_VERSION.encode() + b'"' not in line:
                        continue
                    row = orjson.loads(line)
                    filtered_rows += 1
                    removed_frames += int(row["target_qwen_codec_len"])
                    orphaned_bytes += int(row["source_audio_encoded_bytes"])
        if not filtered_rows:
            continue
        # A quality reject can only be present after the manifest row and its
        # Qwen frames have already been removed by the earlier atomic rewrite.
        if int(state["output_rows"]) + int(state.get("rejected_rows", 0)) != int(state["input_rows"]):
            raise AssertionError(f"state closure failed before migration: {state_path}")
        state["source_audio_bytes"] = int(state["source_audio_bytes"]) - orphaned_bytes
        if state["source_audio_bytes"] < 0:
            raise AssertionError(f"negative source_audio_bytes: {state_path}")
        state["quality_filter"] = {
            "version": FILTER_VERSION,
            "filtered_rows": filtered_rows,
            "removed_qwen_frames": removed_frames,
            "orphaned_source_audio_bytes": orphaned_bytes,
        }
        atomic_json(state_path, state)


def quality_reject_summary(dataset_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = orphaned_bytes = 0
    for path in sorted((dataset_root / "rejects").glob("reject-*.jsonl")):
        with path.open("rb") as handle:
            for line in handle:
                if b'"quality_filter_version":"' + FILTER_VERSION.encode() + b'"' not in line:
                    continue
                row = orjson.loads(line)
                rows += 1
                orphaned_bytes += int(row.get("source_audio_encoded_bytes") or 0)
                counts[f"reason:{row['reason']}"] += 1
                for flag in row.get("quality_flags") or []:
                    counts[f"flag:{flag}"] += 1
    return {
        "version": FILTER_VERSION,
        "filtered_rows": rows,
        "orphaned_wds_members": rows,
        "orphaned_source_audio_bytes": orphaned_bytes,
        "reason_counts": {
            key.removeprefix("reason:"): value
            for key, value in sorted(counts.items()) if key.startswith("reason:")},
        "flag_counts": {
            key.removeprefix("flag:"): value
            for key, value in sorted(counts.items()) if key.startswith("flag:")},
        "thresholds": {
            "min_target_source_duration_ratio": MIN_DURATION_RATIO,
            "max_target_source_duration_ratio": MAX_DURATION_RATIO,
            "generation_ceiling_ms": GENERATION_CEILING_MS,
            "generation_ceiling_qwen_frames": GENERATION_CEILING_QWEN_FRAMES,
            "min_qwen_frames": MIN_QWEN_FRAMES,
            "target_bicodec_dominance": COLLAPSE_DOMINANCE,
            "target_bicodec_adjacent_repeat": COLLAPSE_ADJACENT_REPEAT,
        },
    }


def finalize(dataset_root: Path, raw_root: Path) -> dict[str, Any]:
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((dataset_root / "state").glob("*.json"))]
    input_files = sorted(raw_root.glob("*.parquet"))
    if len(states) != len(input_files):
        raise AssertionError(f"states/input files: {len(states)} != {len(input_files)}")
    for state in states:
        if state["output_rows"] + state.get("rejected_rows", 0) != state["input_rows"]:
            raise AssertionError(f"state closure failed: {state['input_file']}")
    previous = json.loads((dataset_root / "_COMMITTED.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": previous["schema_version"],
        "dataset_version": previous["dataset_version"],
        "process_version": previous["process_version"],
        "created_at_unix": previous["created_at_unix"],
        "quality_filtered_at_unix": time.time(),
        "input_dir": str(raw_root.resolve()),
        "input_files": len(input_files),
        "input_rows": sum(int(state["input_rows"]) for state in states),
        "rows": sum(int(state["output_rows"]) for state in states),
        "rejected_rows": sum(int(state.get("rejected_rows", 0)) for state in states),
        "manifest_shards": len(states),
        "reject_manifest_shards": sum(
            bool(state.get("reject_manifest")) for state in states),
        "tar_shards": sum(len(state["tar_shards"]) for state in states),
        "source_audio_bytes": sum(int(state["source_audio_bytes"]) for state in states),
        "qwen_codec_frames": sum(int(state["qwen_codec_frames"]) for state in states),
        "wds_audio_root": previous["wds_audio_root"],
        "manifest_glob": previous["manifest_glob"],
        "reject_manifest_glob": previous["reject_manifest_glob"],
        "qwen_tokenizer": previous["qwen_tokenizer"],
        "bicodec_model": previous["bicodec_model"],
        "license": previous["license"],
        "quality_filter": quality_reject_summary(dataset_root),
    }
    if summary["rows"] + summary["rejected_rows"] != summary["input_rows"]:
        raise AssertionError("global row closure failed")
    atomic_json(dataset_root / "_SUMMARY.json", summary)
    atomic_json(dataset_root / "_COMMITTED.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.reconcile_only:
        if not args.apply:
            raise ValueError("--reconcile-only requires --apply")
        reconcile_legacy_state_accounting(args.dataset_root)
        print(json.dumps({
            "mode": "reconcile-only",
            "committed": finalize(args.dataset_root, args.raw_root),
        }, ensure_ascii=False, indent=2), flush=True)
        return
    manifests = sorted((args.dataset_root / "manifests").glob("manifest-*.jsonl"))
    if args.max_files:
        manifests = manifests[:args.max_files]
    if not manifests:
        raise FileNotFoundError(args.dataset_root / "manifests")
    started = time.time()
    totals: Counter[str] = Counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(manifests))) as executor:
        futures = {
            executor.submit(
                process_manifest, str(args.dataset_root), str(args.raw_root),
                str(manifest), args.apply): manifest
            for manifest in manifests}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            totals["input_manifest_rows"] += result["input_manifest_rows"]
            totals["kept_rows"] += result["kept_rows"]
            totals["removed_rows"] += result["removed_rows"]
            totals["removed_qwen_frames"] += result["removed_qwen_frames"]
            totals["orphaned_source_audio_bytes"] += result["orphaned_source_audio_bytes"]
            totals.update(result["counts"])
            if completed % 8 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] removed={totals['removed_rows']} "
                    f"elapsed={time.time()-started:.1f}s", flush=True)
    report = {
        "filter_version": FILTER_VERSION,
        "mode": "apply" if args.apply else "dry-run",
        "files": len(manifests),
        "elapsed_seconds": time.time() - started,
        "totals": dict(totals),
    }
    if args.apply and not args.max_files:
        reconcile_legacy_state_accounting(args.dataset_root)
        report["committed"] = finalize(args.dataset_root, args.raw_root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
