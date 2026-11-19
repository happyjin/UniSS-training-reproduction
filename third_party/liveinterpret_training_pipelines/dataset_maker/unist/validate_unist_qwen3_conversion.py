#!/usr/bin/env python3
"""Validate a committed UniST BiCodec -> Qwen3 indexed-WDS conversion.

The validator is intentionally read-only with respect to the converted dataset.
It validates every manifest/reject row in parallel, checks per-input coverage and
WDS byte ranges, detects cross-shard record-id duplicates, and emits a small
stratified candidate set for GPU codec round-trip validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import orjson


EXPECTED_INPUT_ROWS = 19_826_439
EXPECTED_OUTPUT_ROWS = 19_227_252
EXPECTED_REJECT_ROWS = 599_187
EXPECTED_REJECT_REASONS = {
    "empty_target_bicodec": 499_920,
    "single_frame_target_bicodec": 895,
    "duration_ratio_out_of_bounds": 5_762,
    "qwen_codec_too_short": 390,
    "target_bicodec_collapse": 810,
    "target_generation_ceiling": 91_410,
}
SCHEMA_VERSION = "ms-swift-speech-simul-v1"
REJECT_SCHEMA_VERSION = "unist-conversion-reject-v1"
PROCESS_VERSION = "unist_bicodec_qwen3_v2"
TASK = "speech_to_speech_translation"
CODEBOOKS = 16
CODEBOOK_SIZE = 2048
WDS_URI_RE = re.compile(
    r"^wds://unist/audio/processed/shards/dataset=unist/"
    r"process_version=([^/]+)/(rank\d{2})/([^?]+\.tar)"
    r"\?offset=(\d+)&length=(\d+)&format=([A-Za-z0-9._-]+)$"
)


def parse_args() -> argparse.Namespace:
    workspace = Path("/opt/dlami/nvme/zeyangsong")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=workspace / "data/unist/converted/unist_qwen3_s2s_v2")
    parser.add_argument("--raw-root", type=Path, default=workspace / "data/unist/raw")
    parser.add_argument(
        "--report-dir", type=Path,
        default=workspace / "data/unist/validation/unist_qwen3_s2s_v2_20260730")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--roundtrip-samples", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_component(value: Any, max_length: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return (value[:max_length] or "unknown")


def task_prompt(target_lang: str) -> str:
    language = {"en": "English", "zh": "Chinese"}[target_lang]
    return (
        "Task: Listen to the input speech, translate it, and synthesize "
        f"the translation as {language} speech."
    )


def length_bin(duration_ms: int) -> str:
    if duration_ms < 1_000:
        return "lt1s"
    if duration_ms < 3_000:
        return "1to3s"
    if duration_ms < 8_000:
        return "3to8s"
    return "ge8s"


def add_error(result: dict[str, Any], code: str, detail: str) -> None:
    result["error_counts"][code] += 1
    if len(result["error_examples"]) < 50:
        result["error_examples"].append({"code": code, "detail": detail})


def id_hash(record_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest(), "little")


def validate_shard(job: dict[str, str]) -> dict[str, Any]:
    state_path = Path(job["state_path"])
    output_root = Path(job["output_root"])
    shard_report_dir = Path(job["shard_report_dir"])
    hash_dir = Path(job["hash_dir"])
    candidate_dir = Path(job["candidate_dir"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stem = state_path.stem
    manifest = Path(state["manifest"])
    reject_manifest = Path(state.get("reject_manifest") or output_root / "rejects" / f"reject-{stem}.jsonl")
    expected_tar_paths = {str(Path(path).resolve()) for path in state["tar_shards"]}
    input_rows = int(state["input_rows"])
    coverage = bytearray(input_rows)
    hashes = array("Q")
    tar_sizes: dict[str, int] = {}
    tar_last_end: dict[str, int] = {}
    referenced_tars: set[str] = set()
    candidate_best: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    started = time.time()
    result: dict[str, Any] = {
        "stem": stem,
        "state_path": str(state_path),
        "input_file": state["input_file"],
        "manifest": str(manifest),
        "reject_manifest": str(reject_manifest),
        "input_rows": input_rows,
        "manifest_rows": 0,
        "reject_rows": 0,
        "codec_frames": 0,
        "source_audio_bytes": 0,
        "direction_counts": Counter(),
        "dataset_counts": Counter(),
        "split_counts": Counter(),
        "reject_reason_counts": Counter(),
        "codec_length_bins": Counter(),
        "full_codec_rows_checked": 0,
        "sampled_flac_magic_rows": 0,
        "qwen_duration_delta_ms_max": 0,
        "error_counts": Counter(),
        "error_examples": [],
    }

    if not manifest.is_file():
        add_error(result, "missing_manifest", str(manifest))
    else:
        with manifest.open("rb", buffering=16 * 1024 * 1024) as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = orjson.loads(line)
                except Exception as exc:
                    add_error(result, "manifest_json", f"{manifest}:{line_number}: {exc}")
                    continue
                result["manifest_rows"] += 1
                label = f"{manifest.name}:{line_number}:{row.get('id')}"

                if row.get("schema_version") != SCHEMA_VERSION:
                    add_error(result, "schema_version", label)
                if row.get("task") != TASK or row.get("task_prompt") != "":
                    add_error(result, "task_contract", label)
                src_lang, tgt_lang = row.get("source_lang"), row.get("target_lang")
                if (src_lang, tgt_lang) not in {("en", "zh"), ("zh", "en")}:
                    add_error(result, "language_direction", label)
                direction = f"{src_lang}-{tgt_lang}"
                result["direction_counts"][direction] += 1
                result["dataset_counts"][str(row.get("dataset_name"))] += 1
                result["split_counts"][str(row.get("split"))] += 1

                messages = row.get("messages")
                expected_messages = [
                    {"role": "system", "content": task_prompt(tgt_lang)}
                    if tgt_lang in {"en", "zh"} else None,
                    {"role": "user", "content": "<audio>"},
                    {"role": "assistant", "content": row.get("target_text")},
                ]
                if messages != expected_messages:
                    add_error(result, "prompt_or_message_contract", label)
                if not str(row.get("source_transcript") or "").strip():
                    add_error(result, "empty_source_text", label)
                if not str(row.get("target_text") or "").strip():
                    add_error(result, "empty_target_text", label)

                provenance = row.get("provenance") or {}
                source_index = provenance.get("source_row_index")
                if not isinstance(source_index, int) or not 0 <= source_index < input_rows:
                    add_error(result, "source_row_index", f"{label}:{source_index}")
                elif coverage[source_index]:
                    add_error(result, "duplicate_source_row", f"{label}:{source_index}")
                else:
                    coverage[source_index] = 1
                if provenance.get("source_parquet") != Path(state["input_file"]).name:
                    add_error(result, "source_parquet", label)
                if provenance.get("bicodec_global_len") != 32:
                    add_error(result, "global_codec_len", label)
                source_bicodec_len = provenance.get("source_bicodec_len")
                target_bicodec_len = provenance.get("target_bicodec_len")
                if not isinstance(source_bicodec_len, int) or source_bicodec_len <= 0:
                    add_error(result, "source_bicodec_len", label)
                if not isinstance(target_bicodec_len, int) or target_bicodec_len <= 1:
                    add_error(result, "target_bicodec_len", label)
                if isinstance(source_bicodec_len, int) and row.get("source_duration_ms") != source_bicodec_len * 20:
                    add_error(result, "source_duration", label)
                if isinstance(target_bicodec_len, int) and row.get("target_duration_ms") != target_bicodec_len * 20:
                    add_error(result, "target_duration", label)

                codes = row.get("target_codec_codes")
                codec_len = row.get("codec_len")
                if not isinstance(codes, list) or not isinstance(codec_len, int) or codec_len <= 0:
                    add_error(result, "codec_container", label)
                    codes = []
                elif len(codes) != codec_len:
                    add_error(result, "codec_len", f"{label}:{codec_len}!={len(codes)}")
                result["codec_frames"] += len(codes)
                result["codec_length_bins"][length_bin(int(row.get("target_duration_ms") or 0))] += 1
                if codes:
                    positions = {0, len(codes) // 2, len(codes) - 1}
                    # Every row gets representative frame checks. Roughly one in
                    # 1024 rows receives a complete frame/token scan.
                    record_hash = id_hash(str(row.get("record_id")))
                    full_check = (record_hash & 1023) == 0
                    frames = codes if full_check else (codes[position] for position in positions)
                    if full_check:
                        result["full_codec_rows_checked"] += 1
                    for frame in frames:
                        if not isinstance(frame, list) or len(frame) != CODEBOOKS:
                            add_error(result, "codec_frame_shape", label)
                            break
                        if min(frame) < 0 or max(frame) >= CODEBOOK_SIZE:
                            add_error(result, "codec_token_range", label)
                            break
                    delta = abs(len(codes) * 80 - int(row.get("target_duration_ms") or 0))
                    result["qwen_duration_delta_ms_max"] = max(
                        result["qwen_duration_delta_ms_max"], delta)
                    if delta > 160:
                        add_error(result, "qwen_duration_delta", f"{label}:{delta}ms")
                else:
                    record_hash = id_hash(str(row.get("record_id")))

                record_id = str(row.get("record_id") or "")
                expected_record_id = ":".join((
                    "unist", safe_component(row.get("dataset_name")),
                    safe_component(row.get("id")), f"{src_lang}_{tgt_lang}",
                    safe_component(row.get("split"))))
                if record_id != expected_record_id:
                    add_error(result, "record_id", label)
                hashes.append(record_hash)

                audios = row.get("audios")
                if not isinstance(audios, list) or len(audios) != 1:
                    add_error(result, "audio_list", label)
                else:
                    match = WDS_URI_RE.fullmatch(str(audios[0]))
                    if not match:
                        add_error(result, "wds_uri", f"{label}:{audios[0]}")
                    else:
                        process_version, rank_dir, tar_name, offset_raw, length_raw, audio_format = match.groups()
                        offset, length = int(offset_raw), int(length_raw)
                        tar_path = (output_root / "speech" / process_version / rank_dir / tar_name).resolve()
                        tar_key = str(tar_path)
                        referenced_tars.add(tar_key)
                        if process_version != state["process_version"] or audio_format != state["audio_format"]:
                            add_error(result, "wds_setting", label)
                        if rank_dir != f"rank{int(state['rank']):02d}":
                            add_error(result, "wds_rank", label)
                        if tar_key not in expected_tar_paths:
                            add_error(result, "wds_tar_not_in_state", f"{label}:{tar_path}")
                        if tar_key not in tar_sizes:
                            try:
                                tar_sizes[tar_key] = tar_path.stat().st_size
                            except FileNotFoundError:
                                tar_sizes[tar_key] = -1
                                add_error(result, "missing_tar", str(tar_path))
                        if offset % 512 != 0 or offset < 512 or length <= 0:
                            add_error(result, "wds_range_alignment", f"{label}:{offset}:{length}")
                        if tar_sizes[tar_key] >= 0 and offset + length > tar_sizes[tar_key]:
                            add_error(result, "wds_range_bounds", f"{label}:{offset}:{length}")
                        previous_end = tar_last_end.get(tar_key)
                        if previous_end is not None and offset < previous_end:
                            add_error(result, "wds_range_overlap", f"{label}:{offset}<{previous_end}")
                        tar_last_end[tar_key] = offset + length
                        if row.get("source_audio_encoded_bytes") != length:
                            add_error(result, "source_audio_bytes", label)
                        result["source_audio_bytes"] += length
                        if (record_hash & 8191) == 0 and tar_sizes[tar_key] >= 0:
                            fd = os.open(tar_path, os.O_RDONLY)
                            try:
                                magic = os.pread(fd, 4, offset)
                            finally:
                                os.close(fd)
                            result["sampled_flac_magic_rows"] += 1
                            if magic != b"fLaC":
                                add_error(result, "flac_magic", f"{label}:{magic!r}")

                candidate_key = (direction, length_bin(int(row.get("target_duration_ms") or 0)))
                current = candidate_best.get(candidate_key)
                if current is None or record_hash < current[0]:
                    candidate_best[candidate_key] = (record_hash, {
                        "selection_hash": record_hash,
                        "record_id": record_id,
                        "id": row.get("id"),
                        "dataset_name": row.get("dataset_name"),
                        "split": row.get("split"),
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "source_text": row.get("source_transcript"),
                        "target_text": row.get("target_text"),
                        "source_duration_ms": row.get("source_duration_ms"),
                        "target_duration_ms": row.get("target_duration_ms"),
                        "codec_len": codec_len,
                        "target_codec_codes": codes,
                        "audios": audios,
                        "provenance": provenance,
                        "source_parquet_path": state["input_file"],
                    })

    if reject_manifest.is_file():
        with reject_manifest.open("rb", buffering=4 * 1024 * 1024) as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = orjson.loads(line)
                except Exception as exc:
                    add_error(result, "reject_json", f"{reject_manifest}:{line_number}: {exc}")
                    continue
                result["reject_rows"] += 1
                label = f"{reject_manifest.name}:{line_number}:{row.get('id')}"
                if row.get("schema_version") != REJECT_SCHEMA_VERSION:
                    add_error(result, "reject_schema_version", label)
                reason = row.get("reason")
                result["reject_reason_counts"][str(reason)] += 1
                if reason == "empty_target_bicodec" and row.get("target_bicodec_len") != 0:
                    add_error(result, "reject_reason_mismatch", label)
                elif reason == "single_frame_target_bicodec" and row.get("target_bicodec_len") != 1:
                    add_error(result, "reject_reason_mismatch", label)
                elif reason not in EXPECTED_REJECT_REASONS:
                    add_error(result, "reject_reason_unknown", label)
                elif reason not in {"empty_target_bicodec", "single_frame_target_bicodec"}:
                    if row.get("quality_filter_version") != "unist_quality_v1":
                        add_error(result, "reject_quality_filter_version", label)
                    if not row.get("quality_flags"):
                        add_error(result, "reject_quality_flags", label)
                source_index = row.get("source_row_index")
                if not isinstance(source_index, int) or not 0 <= source_index < input_rows:
                    add_error(result, "reject_source_row_index", f"{label}:{source_index}")
                elif coverage[source_index]:
                    add_error(result, "duplicate_source_row", f"{label}:{source_index}")
                else:
                    coverage[source_index] = 2
                if row.get("source_parquet") != Path(state["input_file"]).name:
                    add_error(result, "reject_source_parquet", label)
    elif int(state.get("rejected_rows", 0)):
        add_error(result, "missing_reject_manifest", str(reject_manifest))

    missing_coverage = coverage.count(0)
    result["missing_source_rows"] = missing_coverage
    if missing_coverage:
        add_error(result, "missing_source_rows", f"{stem}:{missing_coverage}")
    if result["manifest_rows"] != int(state["output_rows"]):
        add_error(result, "manifest_state_count", f"{stem}:{result['manifest_rows']}!={state['output_rows']}")
    if result["reject_rows"] != int(state.get("rejected_rows", 0)):
        add_error(result, "reject_state_count", f"{stem}:{result['reject_rows']}!={state.get('rejected_rows', 0)}")
    if result["manifest_rows"] + result["reject_rows"] != input_rows:
        add_error(result, "input_closure", stem)
    if result["codec_frames"] != int(state["qwen_codec_frames"]):
        add_error(result, "codec_frame_state_count", stem)
    if result["source_audio_bytes"] != int(state["source_audio_bytes"]):
        add_error(result, "audio_byte_state_count", stem)
    if not referenced_tars.issubset(expected_tar_paths):
        missing = sorted(expected_tar_paths - referenced_tars)
        extra = sorted(referenced_tars - expected_tar_paths)
        add_error(result, "tar_reference_set", f"{stem}:missing={missing[:3]} extra={extra[:3]}")
    elif referenced_tars != expected_tar_paths and not state.get("quality_filter"):
        missing = sorted(expected_tar_paths - referenced_tars)
        add_error(result, "tar_reference_set", f"{stem}:missing={missing[:3]}")

    hash_dir.mkdir(parents=True, exist_ok=True)
    hash_path = hash_dir / f"{stem}.u64"
    with hash_path.open("wb") as handle:
        hashes.tofile(handle)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{stem}.jsonl"
    with candidate_path.open("wb") as handle:
        for _, candidate in sorted(candidate_best.values(), key=lambda item: item[0]):
            handle.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))

    result["referenced_tars"] = len(referenced_tars)
    result["elapsed_seconds"] = time.time() - started
    result["error_counts"] = dict(result["error_counts"])
    for key in (
        "direction_counts", "dataset_counts", "split_counts",
        "reject_reason_counts", "codec_length_bins",
    ):
        result[key] = dict(result[key])
    shard_report_dir.mkdir(parents=True, exist_ok=True)
    (shard_report_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def merge_counter(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result[key])
    return dict(sorted(counter.items()))


def detect_hash_duplicates(hash_dir: Path) -> dict[str, int]:
    paths = sorted(hash_dir.glob("*.u64"))
    total = sum(path.stat().st_size // 8 for path in paths)
    values = np.empty(total, dtype=np.uint64)
    cursor = 0
    for path in paths:
        part = np.fromfile(path, dtype=np.uint64)
        values[cursor:cursor + len(part)] = part
        cursor += len(part)
    values.sort()
    duplicate_adjacent = int(np.count_nonzero(values[1:] == values[:-1])) if total else 0
    duplicate_groups = 0
    if duplicate_adjacent:
        equal = values[1:] == values[:-1]
        duplicate_groups = int(np.count_nonzero(equal & np.concatenate(([True], ~equal[:-1]))))
    return {
        "record_hashes": total,
        "duplicate_adjacent": duplicate_adjacent,
        "duplicate_groups": duplicate_groups,
        "hash_bits": 64,
    }


def select_roundtrip_candidates(candidate_dir: Path, count: int) -> list[dict[str, Any]]:
    candidates = []
    for path in sorted(candidate_dir.glob("*.jsonl")):
        with path.open("rb") as handle:
            candidates.extend(orjson.loads(line) for line in handle)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            f"{row['source_lang']}-{row['target_lang']}",
            str(row["dataset_name"]),
            length_bin(int(row["target_duration_ms"])),
        )
        groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["selection_hash"])
    selected = []
    ordered_keys = sorted(groups)
    depth = 0
    while len(selected) < count:
        added = False
        for key in ordered_keys:
            rows = groups[key]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def main() -> None:
    args = parse_args()
    committed_path = args.output_root / "_COMMITTED.json"
    if not committed_path.is_file():
        raise FileNotFoundError(f"conversion is not committed: {committed_path}")
    if args.report_dir.exists() and any(args.report_dir.iterdir()) and not args.force:
        raise FileExistsError(f"report directory is not empty: {args.report_dir}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    shard_report_dir = args.report_dir / "static_shards"
    hash_dir = args.report_dir / "record_hashes"
    candidate_dir = args.report_dir / "candidate_shards"

    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    state_paths = sorted((args.output_root / "state").glob("*.json"))
    raw_files = sorted(args.raw_root.glob("*.parquet"))
    preflight_errors = []
    if len(state_paths) != committed["input_files"] or len(raw_files) != committed["input_files"]:
        preflight_errors.append(
            f"files: states={len(state_paths)} raw={len(raw_files)} expected={committed['input_files']}")
    partials = list(args.output_root.rglob("*.partial"))
    if partials:
        preflight_errors.append(f"partial files remain: {len(partials)}")

    jobs = [{
        "state_path": str(path),
        "output_root": str(args.output_root),
        "shard_report_dir": str(shard_report_dir),
        "hash_dir": str(hash_dir),
        "candidate_dir": str(candidate_dir),
    } for path in state_paths]
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_shard, job): Path(job["state_path"]).stem for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:03d}/{len(futures)}] {result['stem']} "
                f"manifest={result['manifest_rows']} reject={result['reject_rows']} "
                f"errors={sum(result['error_counts'].values())} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results.sort(key=lambda result: result["stem"])
    error_counts = merge_counter(results, "error_counts")
    error_examples = [
        example for result in results for example in result["error_examples"]
    ][:200]
    duplicate_report = detect_hash_duplicates(hash_dir)
    all_state_tars = {
        str(Path(path).resolve())
        for state_path in state_paths
        for path in json.loads(state_path.read_text(encoding="utf-8"))["tar_shards"]
    }
    actual_tars = {
        str(path.resolve()) for path in (args.output_root / "speech").rglob("*.tar")
    }
    orphan_tars = sorted(actual_tars - all_state_tars)
    missing_tars = sorted(all_state_tars - actual_tars)
    if orphan_tars:
        error_counts["orphan_tar"] = len(orphan_tars)
    if missing_tars:
        error_counts["missing_tar_global"] = len(missing_tars)

    summary = {
        "schema_version": "unist-qwen3-validation-v1",
        "created_at_unix": time.time(),
        "output_root": str(args.output_root.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "workers": args.workers,
        "elapsed_seconds": time.time() - started,
        "preflight_errors": preflight_errors,
        "states": len(results),
        "input_rows": sum(result["input_rows"] for result in results),
        "manifest_rows": sum(result["manifest_rows"] for result in results),
        "reject_rows": sum(result["reject_rows"] for result in results),
        "codec_frames": sum(result["codec_frames"] for result in results),
        "source_audio_bytes": sum(result["source_audio_bytes"] for result in results),
        "tar_shards": len(actual_tars),
        "direction_counts": merge_counter(results, "direction_counts"),
        "dataset_counts": merge_counter(results, "dataset_counts"),
        "split_counts": merge_counter(results, "split_counts"),
        "reject_reason_counts": merge_counter(results, "reject_reason_counts"),
        "codec_length_bins": merge_counter(results, "codec_length_bins"),
        "full_codec_rows_checked": sum(result["full_codec_rows_checked"] for result in results),
        "sampled_flac_magic_rows": sum(result["sampled_flac_magic_rows"] for result in results),
        "qwen_duration_delta_ms_max": max(result["qwen_duration_delta_ms_max"] for result in results),
        "record_id_duplicate_check": duplicate_report,
        "orphan_tars": orphan_tars[:100],
        "missing_tars": missing_tars[:100],
        "error_counts": error_counts,
        "error_examples": error_examples,
        "expected": {
            "input_rows": EXPECTED_INPUT_ROWS,
            "manifest_rows": EXPECTED_OUTPUT_ROWS,
            "reject_rows": EXPECTED_REJECT_ROWS,
            "reject_reason_counts": EXPECTED_REJECT_REASONS,
        },
    }
    expected_checks = {
        "input_rows": summary["input_rows"] == EXPECTED_INPUT_ROWS,
        "manifest_rows": summary["manifest_rows"] == EXPECTED_OUTPUT_ROWS,
        "reject_rows": summary["reject_rows"] == EXPECTED_REJECT_ROWS,
        "reject_reasons": summary["reject_reason_counts"] == EXPECTED_REJECT_REASONS,
        "committed_input_rows": committed["input_rows"] == summary["input_rows"],
        "committed_manifest_rows": committed["rows"] == summary["manifest_rows"],
        "committed_reject_rows": committed["rejected_rows"] == summary["reject_rows"],
        "committed_codec_frames": committed["qwen_codec_frames"] == summary["codec_frames"],
        "committed_audio_bytes": committed["source_audio_bytes"] == summary["source_audio_bytes"],
        "record_ids_unique": duplicate_report["duplicate_adjacent"] == 0,
    }
    summary["expected_checks"] = expected_checks
    summary["passed"] = (
        not preflight_errors and not error_counts and all(expected_checks.values()))

    selected = select_roundtrip_candidates(candidate_dir, args.roundtrip_samples)
    roundtrip_path = args.report_dir / "roundtrip_samples.jsonl"
    with roundtrip_path.open("wb") as handle:
        for index, row in enumerate(selected):
            row["sample_index"] = index
            row["listening_sample"] = index < 10
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
    summary["roundtrip_samples"] = len(selected)
    summary["roundtrip_sample_path"] = str(roundtrip_path.resolve())

    report_path = args.report_dir / "static_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": summary["passed"],
        "report": str(report_path),
        "elapsed_seconds": summary["elapsed_seconds"],
        "error_counts": error_counts,
        "expected_checks": expected_checks,
        "roundtrip_samples": len(selected),
    }, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
