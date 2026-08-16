#!/usr/bin/env python3
"""Parallel full-split audit for Stage A append-only ASR supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.events import (
    build_asr_event_session,
)
from training.phase3_whisper_streamspeech_joint.tokenizer_maps import CompactCTCMap
from training.simul_uniss.jsonl_index import load_index


SCHEMA = "uniss_quality_first_stage_a_data_audit_v1"
SHARD_PATTERN = re.compile(r"train-(\d{5})\.parquet$")
DURATION_BINS = ((0, 4000, "lt4s"), (4000, 8000, "4to8s"), (8000, 15000, "8to15s"))


def _duration_bin(value: int) -> str:
    for lower, upper, name in DURATION_BINS:
        if lower <= value < upper:
            return name
    return "ge15s"


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _id_digest(value: str) -> np.uint64:
    raw = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return np.uint64(int.from_bytes(raw, "little", signed=False))


def _worker(
    manifest: str,
    start: int,
    stop: int,
    part: int,
    output_dir: str,
    model: str,
    ctc_map_dir: str,
    check_audio: bool,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    path = Path(manifest)
    offsets = np.memmap(str(path) + ".offsets.bin", mode="r", dtype=np.uint64)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    maps = {
        language: CompactCTCMap.load(Path(ctc_map_dir) / f"ctc_qwen_{language}.json")
        for language in ("eng", "cmn")
    }
    map_sets = {language: set(value.compact_to_qwen) for language, value in maps.items()}
    counters: Counter[str] = Counter()
    rejection: Counter[str] = Counter()
    ids = np.empty(stop - start, dtype=np.uint64)
    kept = 0
    with path.open("rb") as handle:
        for local, index in enumerate(range(start, stop)):
            handle.seek(int(offsets[index]))
            try:
                record = json.loads(handle.readline())
                if not bool(record.get("formal_a68_pass")):
                    raise ValueError("formal_a68_pass_false")
                source = str(record.get("source_parquet") or "")
                match = SHARD_PATTERN.search(source)
                if match is None or not 0 <= int(match.group(1)) <= 14:
                    raise ValueError("outside_train_00000_00014")
                session = build_asr_event_session(record)
                if check_audio and not Path(str(record.get("source_audio") or "")).is_file():
                    raise ValueError("missing_source_audio")
                token_ids = tokenizer.encode(session.normalized_transcript, add_special_tokens=False)
                if not token_ids:
                    raise ValueError("empty_qwen_asr_target")
                oov = sum(int(token) not in map_sets[session.src_lang] for token in token_ids)
                counters["ctc_oov_tokens"] += oov
                counters["ctc_target_tokens"] += len(token_ids)
                counters["records"] += 1
                counters[f"direction:{session.src_lang}->{record.get('tgt_lang')}"] += 1
                counters[f"shard:{int(match.group(1)):05d}"] += 1
                counters[f"duration:{_duration_bin(session.source_duration_ms)}"] += 1
                counters["source_words"] += len(session.words)
                counters["source_glm_tokens"] += session.source_glm_tokens
                counters["events"] += len(session.events)
                counters["events_with_text"] += sum(event.has_text_delta for event in session.events)
                counters["empty_text_events"] += sum(not event.has_text_delta for event in session.events)
                counters["prefinal_text_commit"] += int(session.prefinal_text_commit)
                counters["final_only_text"] += int(not session.prefinal_text_commit)
                counters["duration_ms"] += session.source_duration_ms
                counters["max_events"] = max(counters["max_events"], len(session.events))
                counters["max_words"] = max(counters["max_words"], len(session.words))
                counters["max_ctc_tokens"] = max(counters["max_ctc_tokens"], len(token_ids))
                ids[kept] = _id_digest(session.sample_id)
                kept += 1
            except Exception as exc:
                reason = str(exc).split(":", 1)[0]
                rejection[f"{type(exc).__name__}:{reason}"] += 1
            counters["input_records"] += 1
    ids = ids[:kept]
    id_path = Path(output_dir) / f"ids.part{part:03d}.npy"
    np.save(id_path, ids, allow_pickle=False)
    result = {
        "part": part,
        "start": start,
        "stop": stop,
        "counters": dict(sorted(counters.items())),
        "rejections": dict(sorted(rejection.items())),
        "id_hashes": str(id_path.resolve()),
    }
    _atomic_json(Path(output_dir) / f"audit.part{part:03d}.json", result)
    return result


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, total))
    return [
        (total * part // workers, total * (part + 1) // workers)
        for part in range(workers)
    ]


def _merge(parts: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], np.ndarray]:
    counters: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    arrays = []
    max_keys = {"max_events", "max_words", "max_ctc_tokens"}
    for part in parts:
        for key, value in part["counters"].items():
            if key in max_keys:
                counters[key] = max(counters[key], int(value))
            else:
                counters[key] += int(value)
        rejections.update({str(key): int(value) for key, value in part["rejections"].items()})
        arrays.append(np.load(part["id_hashes"], mmap_mode="r", allow_pickle=False))
    ids = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.uint64)
    return counters, rejections, ids


def _audit_split(
    name: str,
    manifest: Path,
    workers: int,
    output_dir: Path,
    model: Path,
    ctc_map_dir: Path,
    check_audio: bool,
) -> dict[str, Any]:
    offsets = load_index(manifest)
    if offsets is None:
        raise ValueError(f"missing validated offset index: {manifest}")
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=False)
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part, (start, stop) in enumerate(_ranges(len(offsets), workers)):
            futures.append(
                pool.submit(
                    _worker,
                    str(manifest),
                    start,
                    stop,
                    part,
                    str(split_dir),
                    str(model),
                    str(ctc_map_dir),
                    check_audio,
                )
            )
        parts = [future.result() for future in as_completed(futures)]
    parts.sort(key=lambda value: int(value["part"]))
    counters, rejections, ids = _merge(parts)
    ordered = np.sort(ids)
    duplicate_hashes = int(np.count_nonzero(ordered[1:] == ordered[:-1])) if len(ordered) > 1 else 0
    merged_ids = output_dir / f"{name}_id_sha256_u64.npy"
    np.save(merged_ids, ordered, allow_pickle=False)
    return {
        "manifest": str(manifest.resolve()),
        "records_in_index": len(offsets),
        "workers": len(parts),
        "counters": dict(sorted(counters.items())),
        "rejections": dict(sorted(rejections.items())),
        "duplicate_id_hashes": duplicate_hashes,
        "sorted_id_hashes": str(merged_ids.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ctc-map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-workers", type=int, default=30)
    parser.add_argument("--valid-workers", type=int, default=8)
    parser.add_argument("--skip-audio-exists", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage A audit: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    train = _audit_split(
        "train",
        args.train_manifest,
        args.train_workers,
        args.output_dir,
        args.model,
        args.ctc_map_dir,
        not args.skip_audio_exists,
    )
    valid = _audit_split(
        "valid",
        args.valid_manifest,
        args.valid_workers,
        args.output_dir,
        args.model,
        args.ctc_map_dir,
        not args.skip_audio_exists,
    )
    train_ids = np.load(train["sorted_id_hashes"], mmap_mode="r", allow_pickle=False)
    valid_ids = np.load(valid["sorted_id_hashes"], mmap_mode="r", allow_pickle=False)
    overlap = int(len(np.intersect1d(train_ids, valid_ids, assume_unique=True)))
    checks = {
        "train_all_records_pass": train["counters"].get("records", 0) == train["records_in_index"],
        "valid_all_records_pass": valid["counters"].get("records", 0) == valid["records_in_index"],
        "train_id_unique": train["duplicate_id_hashes"] == 0,
        "valid_id_unique": valid["duplicate_id_hashes"] == 0,
        "train_valid_disjoint": overlap == 0,
        "train_ctc_oov_zero": train["counters"].get("ctc_oov_tokens", 0) == 0,
        "valid_ctc_oov_zero": valid["counters"].get("ctc_oov_tokens", 0) == 0,
        "train_prefinal_commit_nonzero": train["counters"].get("prefinal_text_commit", 0) > 0,
        "valid_prefinal_commit_nonzero": valid["counters"].get("prefinal_text_commit", 0) > 0,
    }
    report = {
        "schema_version": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "train_valid_id_overlap": overlap,
        "train": train,
        "valid": valid,
    }
    _atomic_json(args.output_dir / "stage_a_data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
