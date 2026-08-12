#!/usr/bin/env python3
"""Full provenance and continuity audit for the fixed UniST 15-shard run.

This audit intentionally distinguishes forced/oracle timing from observed
natural timing.  A passing result proves that the released fixed-15 data are
continuous, causal and internally complete; it does not rename the timing
labels as ground-truth human READ/WRITE decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow.parquet as pq

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseSession,
    SCHEMA_VERSION as DENSE_SCHEMA,
    TICK_MS,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    canonical_runtime_pack,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MultiFilePackIndex,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index


AUDIT_SCHEMA = "uniss_event_rollout_pilot15_data_audit_v1"
EXPECTED_SHARDS = tuple(range(15))
_SHARD_PATTERN = re.compile(r"train-(\d{5})")


def split_for_id(sample_id: str, validation_modulus: int) -> str:
    digest = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16)
    return "valid" if digest % validation_modulus == 0 else "train"


def classify_timing_provenance(value: Mapping[str, object]) -> str:
    safe_kinds = {
        str(event.get("safe_label_kind", ""))
        for event in value.get("micro_write_events", [])  # type: ignore[union-attr]
    }
    forced = "forced_aligner" in str(value.get("source_alignment_kind", "")) or (
        "forced_aligner" in str(value.get("target_alignment_kind", ""))
    )
    oracle = any("oracle_bilingual_support" in item for item in safe_kinds)
    if forced and oracle:
        return "forced_word_alignment_plus_oracle_bilingual_support_pseudo_timing"
    if forced:
        return "forced_word_alignment_pseudo_timing"
    if oracle:
        return "oracle_bilingual_support_pseudo_timing"
    return "unclassified_non_exact_timing"


def _atomic_json(path: Path, value: object) -> None:
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


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _shard_from_path(value: str | Path) -> int:
    match = _SHARD_PATTERN.search(str(value))
    if match is None:
        raise ValueError(f"path does not identify a UniST train shard: {value}")
    return int(match.group(1))


def audit_stage_marker(stage_marker: Path) -> dict[str, object]:
    value = json.loads(stage_marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError("formal Stage A marker is incomplete")
    if int(value.get("validation_modulus", -1)) != 100:
        raise ValueError("formal split must retain validation_modulus=100")
    for key in ("a45_part_markers", "a68_part_markers"):
        paths = [Path(item).resolve() for item in value.get(key, [])]
        if len(paths) != 30 or any(not path.is_file() for path in paths):
            raise ValueError(f"formal marker {key} does not expose 30 complete parts")
        shards = sorted({_shard_from_path(path) for path in paths})
        if tuple(shards) != EXPECTED_SHARDS:
            raise ValueError(f"formal marker {key} uses shards {shards}")
        chunks = Counter(path.parent.name for path in paths)
        if chunks != {"chunk-00": 15, "chunk-01": 15}:
            raise ValueError(f"formal marker {key} chunk geometry changed: {chunks}")
    return {
        "status": "pass",
        "schema_version": value.get("schema_version"),
        "scope": value.get("scope"),
        "validation_modulus": 100,
        "shards": list(EXPECTED_SHARDS),
        "a45_parts": 30,
        "a68_parts": 30,
        "warning": value.get("warning"),
    }


def audit_formal_split(
    path: Path,
    *,
    split: str,
    validation_modulus: int,
    limit: int | None,
) -> tuple[dict[str, object], set[str]]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"formal manifest has no valid index: {path}")
    ids: set[str] = set()
    directions: Counter[str] = Counter()
    timing: Counter[str] = Counter()
    audio_origins: Counter[str] = Counter()
    source_alignments: Counter[str] = Counter()
    target_alignments: Counter[str] = Counter()
    support_status: Counter[str] = Counter()
    shards: Counter[int] = Counter()
    digest = hashlib.sha256()
    records = 0
    with path.open("rb") as handle:
        for line in handle:
            if limit is not None and records >= limit:
                break
            value = json.loads(line)
            sample_id = str(value["id"])
            if sample_id in ids:
                raise ValueError(f"duplicate {split} formal ID: {sample_id}")
            if split_for_id(sample_id, validation_modulus) != split:
                raise ValueError(f"formal ID {sample_id} violates deterministic {split} split")
            shard = _shard_from_path(str(value["source_parquet"]))
            if shard not in EXPECTED_SHARDS:
                raise ValueError(f"formal ID {sample_id} escaped fixed shard range")
            if not bool(value.get("formal_a68_pass")):
                raise ValueError(f"formal split contains rejected record {sample_id}")
            if int(value.get("source_row_index", -1)) < 0:
                raise ValueError(f"formal ID {sample_id} has invalid source row")
            micro = value.get("micro_write_events")
            if not isinstance(micro, list) or not micro:
                raise ValueError(f"formal ID {sample_id} has no Micro-WRITE events")
            if int(micro[0]["semantic_start"]) != 0:
                raise ValueError(f"formal ID {sample_id} semantic coverage does not start at 0")
            for left, right in zip(micro, micro[1:]):
                if int(left["semantic_end"]) != int(right["semantic_start"]):
                    raise ValueError(f"formal ID {sample_id} has a semantic gap/overlap")
            if int(micro[-1]["semantic_end"]) != len(value["target_bicodec"]):
                raise ValueError(f"formal ID {sample_id} semantic coverage is incomplete")
            ids.add(sample_id)
            digest.update(sample_id.encode("utf-8") + b"\n")
            directions[f"{value['src_lang']}->{value['tgt_lang']}"] += 1
            timing[classify_timing_provenance(value)] += 1
            audio_origins[str(value.get("audio_origin"))] += 1
            source_alignments[str(value.get("source_alignment_kind"))] += 1
            target_alignments[str(value.get("target_alignment_kind"))] += 1
            support_status[str(value.get("support_alignment_status"))] += 1
            shards[shard] += 1
            records += 1
    complete = limit is None
    if complete and records != len(offsets):
        raise ValueError(f"formal {split} scan read {records}, index has {len(offsets)}")
    return (
        {
            "status": "pass" if complete else "sampled_pass",
            "path": str(path),
            "records": records,
            "indexed_records": len(offsets),
            "unique_ids": len(ids),
            "ordered_id_sha256": digest.hexdigest(),
            "directions": dict(directions),
            "source_shards": {str(key): count for key, count in sorted(shards.items())},
            "timing_provenance": dict(timing),
            "timing_is_natural_exact": False,
            "audio_origins": dict(audio_origins),
            "source_alignment_kinds": dict(source_alignments),
            "target_alignment_kinds": dict(target_alignments),
            "support_alignment_status": dict(support_status),
        },
        ids,
    )


def _audit_dense_part(arguments: tuple[str, str, int | None]) -> dict[str, object]:
    directory_value, split, limit = arguments
    directory = Path(directory_value)
    marker = json.loads((directory / "PART_COMPLETE.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"dense part is incomplete: {directory}")
    path = directory / "dense.jsonl"
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"dense part lacks index: {directory}")
    expected_start = int(marker["source_start"])
    expected_end = int(marker["source_end"])
    ids: list[str] = []
    event_count = 0
    write_count = 0
    read_count = 0
    with path.open("rb") as handle:
        for local_index, line in enumerate(handle):
            if limit is not None and local_index >= limit:
                break
            session = DenseSession.from_dict(json.loads(line))
            if session.schema_version != DENSE_SCHEMA or session.split != split:
                raise ValueError(f"dense session split/schema mismatch in {directory}")
            if session.source_index != expected_start + local_index:
                raise ValueError(f"dense source_index is not contiguous in {directory}")
            if session.source_manifest.endswith(f"formal_{split}_manifest.jsonl") is False:
                raise ValueError(f"dense session points to the wrong formal split in {directory}")
            ids.append(session.sample_id)
            event_count += len(session.events)
            write_count += sum(event.action == "WRITE" for event in session.events)
            read_count += sum(event.action == "READ" for event in session.events)
    complete = limit is None
    if complete:
        expected_records = expected_end - expected_start
        if len(ids) != expected_records or len(ids) != len(offsets):
            raise ValueError(f"dense record count mismatch in {directory}")
        if int(marker["counts"]["sessions"]) != len(ids):
            raise ValueError(f"dense marker session count mismatch in {directory}")
    return {
        "part_id": directory.name,
        "status": "pass" if complete else "sampled_pass",
        "records": len(ids),
        "indexed_records": len(offsets),
        "source_start": expected_start,
        "source_end": expected_end,
        "ids": ids,
        "ordered_id_sha256": _sha256_lines(ids),
        "events": event_count,
        "writes": write_count,
        "reads": read_count,
    }


def audit_dense_parts(
    root: Path,
    *,
    split: str,
    expected_parts: int,
    workers: int,
    limit_per_part: int | None,
) -> tuple[dict[str, object], set[str]]:
    directories = sorted(path for path in root.glob("part-*") if path.is_dir())
    if len(directories) != expected_parts:
        raise ValueError(f"dense {split} has {len(directories)} parts, expected {expected_parts}")
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = list(
            executor.map(
                _audit_dense_part,
                [(str(path), split, limit_per_part) for path in directories],
            )
        )
    ids: set[str] = set()
    cursor = 0
    for result in results:
        if int(result["source_start"]) != cursor:
            raise ValueError(f"dense {split} part source ranges are not contiguous")
        cursor = int(result["source_end"])
        part_ids = set(result.pop("ids"))
        if len(part_ids) != int(result["records"]):
            raise ValueError(f"dense {split} part contains duplicate IDs")
        if ids.intersection(part_ids):
            raise ValueError(f"dense {split} IDs repeat across parts")
        ids.update(part_ids)
    return (
        {
            "status": "pass" if limit_per_part is None else "sampled_pass",
            "root": str(root),
            "part_count": len(results),
            "records": sum(int(value["records"]) for value in results),
            "events": sum(int(value["events"]) for value in results),
            "writes": sum(int(value["writes"]) for value in results),
            "reads": sum(int(value["reads"]) for value in results),
            "unique_ids": len(ids),
            "tick_ms": TICK_MS,
            "session_schema": DENSE_SCHEMA,
            "semantic_coverage": "gap_free_complete",
            "text_coverage": "exact_reconstruction",
            "parts": results,
        },
        ids,
    )


def _audit_pack_part(arguments: tuple[str, int | None]) -> dict[str, object]:
    packed_value, limit = arguments
    packed = Path(packed_value)
    offsets = load_index(packed)
    if offsets is None:
        raise ValueError(f"packed part lacks index: {packed}")
    ids: list[str] = []
    sessions = 0
    events = 0
    writes = 0
    with packed.open("rb") as handle:
        for record_index, line in enumerate(handle):
            if limit is not None and record_index >= limit:
                break
            value = json.loads(line)
            if value.get("schema_version") != PACK_SCHEMA:
                raise ValueError(f"unexpected pack schema in {packed}")
            lengths = {
                len(value[key])
                for key in ("tokens", "labels", "loss_mask", "token_roles", "position_ids")
            }
            if lengths != {18_000}:
                raise ValueError(f"pack tensors are not fixed at 18000 in {packed}")
            raw_sessions = value.get("sessions")
            if not isinstance(raw_sessions, list) or not raw_sessions:
                raise ValueError(f"pack has no complete sessions in {packed}")
            previous = 0
            raw_ids: list[str] = []
            for raw in raw_sessions:
                start, end = (int(item) for item in raw["boundary"])
                if start != previous or not start < end <= 18_000:
                    raise ValueError(f"session boundary gap/overlap in {packed}")
                previous = end
                raw_ids.append(str(raw["sample_id"]))
            if raw_ids != [str(item) for item in value["source_ids"]]:
                raise ValueError(f"pack source_ids differ from session IDs in {packed}")
            parsed = oracle_sessions_from_pack(value)
            canonical = canonical_runtime_pack(value)
            if len(canonical["tokens"]) != 18_000 or len(parsed) != len(raw_sessions):
                raise ValueError(f"canonical runtime parity failed in {packed}")
            for session in parsed:
                if not session.events[-1].source_finished:
                    raise ValueError(f"final runtime event has unfinished source in {packed}")
                if session.events[-1].continuation_token != c.TOKEN_EOS:
                    raise ValueError(f"final runtime event does not supervise EOS in {packed}")
                for index, event in enumerate(session.events):
                    if event.event_index != index or event.chunk_end_ms != (index + 1) * TICK_MS:
                        raise ValueError(f"runtime event timeline is not continuous in {packed}")
                    if event.action == "WRITE":
                        parsed_write = parse_write_outcome(event.outcome_tokens)
                        if not parsed_write.semantic_codes:
                            raise ValueError(f"runtime WRITE lacks semantic content in {packed}")
                        writes += 1
            ids.extend(raw_ids)
            sessions += len(parsed)
            events += sum(len(session.events) for session in parsed)
    complete = limit is None
    if complete and record_index + 1 != len(offsets):
        raise ValueError(f"packed record count differs from index: {packed}")
    return {
        "part_id": packed.parent.name,
        "status": "pass" if complete else "sampled_pass",
        "records": len(offsets) if complete else min(len(offsets), limit or 0),
        "indexed_records": len(offsets),
        "sessions": sessions,
        "events": events,
        "writes": writes,
        "ids": ids,
        "ordered_id_sha256": _sha256_lines(ids),
    }


def audit_pack_manifest(
    manifest: Path,
    *,
    split: str,
    workers: int,
    limit_per_part: int | None,
) -> tuple[dict[str, object], set[str]]:
    index = MultiFilePackIndex(manifest, expected_split=split)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = list(
            executor.map(
                _audit_pack_part,
                [(str(part.packed), limit_per_part) for part in index.parts],
            )
        )
    ids: set[str] = set()
    for result in results:
        part_ids = set(result.pop("ids"))
        if len(part_ids) != int(result["sessions"]):
            raise ValueError(f"packed {split} part contains duplicate sessions")
        if ids.intersection(part_ids):
            raise ValueError(f"packed {split} session appears in more than one part")
        ids.update(part_ids)
    records = sum(int(value["records"]) for value in results)
    if limit_per_part is None and records != len(index):
        raise ValueError(f"packed {split} global namespace count mismatch")
    return (
        {
            "status": "pass" if limit_per_part is None else "sampled_pass",
            "manifest": str(manifest),
            "part_count": len(results),
            "records": records,
            "global_records": len(index),
            "sessions": sum(int(value["sessions"]) for value in results),
            "events": sum(int(value["events"]) for value in results),
            "writes": sum(int(value["writes"]) for value in results),
            "unique_ids": len(ids),
            "runtime_parser": "oracle_sessions_from_pack",
            "canonical_runtime_parser": "canonical_runtime_pack",
            "session_cross_pack_count": 0,
            "parts": results,
        },
        ids,
    )


def raw_shard_ids(raw_root: Path) -> tuple[set[str], dict[str, object]]:
    ids: set[str] = set()
    per_shard: dict[str, int] = {}
    for shard in EXPECTED_SHARDS:
        path = raw_root / f"train-{shard:05d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = [str(item) for item in pq.read_table(path, columns=["id"])["id"].to_pylist()]
        if len(values) != len(set(values)):
            raise ValueError(f"raw shard {shard} contains duplicate IDs")
        if ids.intersection(values):
            raise ValueError(f"raw shard {shard} repeats IDs from an earlier shard")
        ids.update(values)
        per_shard[f"{shard:05d}"] = len(values)
    return ids, {
        "status": "pass",
        "shards": list(EXPECTED_SHARDS),
        "per_shard_records": per_shard,
        "unique_ids": len(ids),
    }


def audit_replay(
    packed: Path,
    *,
    allowed_ids: set[str],
    limit: int | None,
) -> dict[str, object]:
    offsets = load_index(packed)
    if offsets is None:
        # Historical replay uses an explicit .u64 index rather than jsonl_index.
        metadata_path = packed.parent / "packed_replay.offsets.u64.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        indexed_records = int(metadata["records"])
    else:
        indexed_records = len(offsets)
    source_ids: set[str] = set()
    task_counts: Counter[str] = Counter()
    records = 0
    source_occurrences = 0
    with packed.open("rb") as handle:
        for line in handle:
            if limit is not None and records >= limit:
                break
            value = json.loads(line)
            ids = [str(item) for item in value.get("source_ids", [])]
            tasks = [str(item) for item in value.get("tasks", [])]
            if not ids or len(ids) != len(tasks):
                raise ValueError(f"replay pack {records} has inconsistent IDs/tasks")
            escaped = set(ids).difference(allowed_ids)
            if escaped:
                raise ValueError(f"replay pack {records} escaped fixed shards: {sorted(escaped)[:5]}")
            if set(tasks).difference({"quality", "performance"}):
                raise ValueError(f"replay pack {records} contains a non-Phase3 task")
            source_ids.update(ids)
            task_counts.update(tasks)
            source_occurrences += len(ids)
            records += 1
    complete = limit is None
    if complete and records != indexed_records:
        raise ValueError(f"replay scan read {records}, index has {indexed_records}")
    return {
        "status": "pass" if complete else "sampled_pass",
        "packed": str(packed),
        "records": records,
        "indexed_records": indexed_records,
        "source_occurrences": source_occurrences,
        "unique_source_ids": len(source_ids),
        "out_of_scope_source_ids": 0,
        "task_counts": dict(task_counts),
        "allowed_raw_shards": list(EXPECTED_SHARDS),
        "scope_proof": "every replay source_id is a member of raw UniST shards 00000-00014",
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    stage_root = repo / "data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal"
    dense_root = repo / "data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1"
    experiment_data = repo / "data/megatron/uniss_phase3_event_rollout_joint_pilot15_v1"
    stage = audit_stage_marker(stage_root / "STAGE_A_COMPLETE.json")
    validation_modulus = int(stage["validation_modulus"])
    formal_train, formal_train_ids = audit_formal_split(
        stage_root / "formal_train_manifest.jsonl",
        split="train",
        validation_modulus=validation_modulus,
        limit=args.limit,
    )
    formal_valid, formal_valid_ids = audit_formal_split(
        stage_root / "formal_valid_manifest.jsonl",
        split="valid",
        validation_modulus=validation_modulus,
        limit=args.limit,
    )
    if formal_train_ids.intersection(formal_valid_ids):
        raise ValueError("formal train/validation IDs intersect")

    dense_train, dense_train_ids = audit_dense_parts(
        dense_root / "train_parts",
        split="train",
        expected_parts=32,
        workers=args.workers,
        limit_per_part=args.limit_per_part,
    )
    dense_valid, dense_valid_ids = audit_dense_parts(
        dense_root / "valid_parts",
        split="valid",
        expected_parts=4,
        workers=min(args.workers, 4),
        limit_per_part=args.limit_per_part,
    )
    packed_train, packed_train_ids = audit_pack_manifest(
        experiment_data / "train_parts_manifest.json",
        split="train",
        workers=args.workers,
        limit_per_part=args.limit_per_part,
    )
    packed_valid, packed_valid_ids = audit_pack_manifest(
        experiment_data / "valid_parts_manifest.json",
        split="valid",
        workers=min(args.workers, 4),
        limit_per_part=args.limit_per_part,
    )

    complete = args.limit is None and args.limit_per_part is None
    if complete:
        for name, actual, expected in (
            ("dense train", dense_train_ids, formal_train_ids),
            ("packed train", packed_train_ids, formal_train_ids),
            ("dense valid", dense_valid_ids, formal_valid_ids),
            ("packed valid", packed_valid_ids, formal_valid_ids),
        ):
            if actual != expected:
                raise ValueError(
                    f"{name} IDs differ from formal split: "
                    f"missing={len(expected - actual)} extra={len(actual - expected)}"
                )
    raw_ids, raw = raw_shard_ids(repo / "data/raw/UniST")
    replay = audit_replay(
        repo / "data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.jsonl",
        allowed_ids=raw_ids,
        limit=args.limit,
    )
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "pass" if complete else "sampled_pass",
        "scope": "fixed UniST train shards 00000-00014 only",
        "full198_training_authorized": False,
        "stage_a": stage,
        "formal": {
            "train": formal_train,
            "valid": formal_valid,
            "train_valid_intersection": 0,
            "split_rule": "sha256(id)[:16] modulo 100; remainder 0 is validation",
        },
        "dense_sessions": {"train": dense_train, "valid": dense_valid},
        "packed_trajectories": {"train": packed_train, "valid": packed_valid},
        "raw_source_scope": raw,
        "phase3_replay": replay,
        "timing_provenance_truth": {
            "classification": "pseudo_oracle_alignment",
            "natural_exact_timing": False,
            "source_word_timing": "Qwen3 forced aligner",
            "target_word_timing": "Qwen3 forced aligner",
            "write_support": "oracle bilingual future-monotonic support",
            "claim": "usable training supervision, not observed human READ/WRITE timing",
        },
        "gates": {
            "fixed_shards_only": True,
            "deterministic_split": True,
            "train_valid_intersection_zero": True,
            "complete_160ms_sessions": complete,
            "gap_free_text_and_semantic_coverage": complete,
            "sessions_never_cross_packs": complete,
            "runtime_parsers_accept_all_packs": complete,
            "phase3_replay_fixed15_only": replay["out_of_scope_source_ids"] == 0,
        },
    }
    if args.output:
        _atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-part", type=int)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    audit(args)


if __name__ == "__main__":
    main()
