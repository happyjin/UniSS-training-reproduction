#!/usr/bin/env python3
"""Regenerate repaired dense trajectory caches for fixed UniST shards 0..14.

Each rank owns disjoint shards. BiCodec decoding, causal WhisperVQ encoding and
Phase3 teacher inference stay on that rank's GPU. Teacher requests are
deduplicated by prefix time per row and evaluated in bounded sub-batches so a
larger decode batch can use H200 memory without creating a teacher OOM spike.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache import (
    BatchedBiCodecDecoder,
    Phase3Teacher,
    _prefix,
    _save_bundle,
    stable_prefix_length,
    teacher_bundle_reference,
    causal_bundle_reference,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schedule import plans_for_row
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import (
    Action,
    SCHEMA_VERSION,
    TrajectoryRecord,
)


CACHE_PART_SCHEMA = "uniss_true_subsecond_pilot15_cache_part_v2"
REQUIRED_COLUMNS = (
    "id",
    "transcription",
    "translation",
    "source_glm",
    "source_bicodec",
    "target_bicodec",
    "bicodec_global",
    "src_lang",
    "tgt_lang",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def try_claim_shard(output_root: Path, shard: int) -> Path | None:
    """Return an owned atomic lock directory, or None if claimed/completed."""

    part = output_root / f"part-{shard:03d}"
    part.mkdir(parents=True, exist_ok=True)
    if (part / "PART_COMPLETE.json").is_file():
        return None
    lock = part / ".worker_lock"
    try:
        lock.mkdir()
    except FileExistsError:
        return None
    if (part / "PART_COMPLETE.json").is_file():
        lock.rmdir()
        return None
    return lock


def _block_size(sample_id: str, tick_ms: int) -> int:
    value = int.from_bytes(
        hashlib.blake2b(f"{sample_id}:{tick_ms}:pilot15_v2".encode(), digest_size=2).digest(),
        "big",
    )
    return (8, 12, 16)[value % 3]


def _summarize_bounded(
    teacher: Phase3Teacher,
    requests: Sequence[tuple[list[int], list[int]]],
    request_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    if request_batch_size <= 0:
        raise ValueError("teacher request batch size must be positive")
    result: list[dict[str, np.ndarray]] = []
    for start in range(0, len(requests), request_batch_size):
        result.extend(teacher.summarize(requests[start : start + request_batch_size]))
    if len(result) != len(requests):
        raise AssertionError("teacher request accounting mismatch")
    return result


def build_records_for_row(
    *,
    shard: int,
    row_index: int,
    row: dict[str, Any],
    causal_tokens: Sequence[int],
    translation_ids: Sequence[int],
    summaries: Sequence[dict[str, np.ndarray]],
    request_by_end_ms: dict[int, int],
    cache_file: Path,
    cache_row_index: int,
    confidence_threshold: float,
) -> tuple[TrajectoryRecord, ...]:
    duration_ms = len(row["source_bicodec"]) * 20
    plans = plans_for_row(str(row["id"]), duration_ms)
    has_exact_deadline = any(plan.chunk_end_ms == 800 for plan in plans)
    previous_text = 0
    semantic_cursor = 0
    target_semantic = [int(value) for value in row["target_bicodec"]]
    records: list[TrajectoryRecord] = []

    for plan in plans:
        request_indices = (
            request_by_end_ms[plan.chunk_end_ms],
            request_by_end_ms[plan.future_1_end_ms],
            request_by_end_ms[plan.future_2_end_ms],
            request_by_end_ms[duration_ms],
        )
        group = [summaries[index] for index in request_indices]
        stable, safe = stable_prefix_length(
            translation_ids,
            [value["top1"] for value in group],
            [value["confidence"] for value in group],
            threshold=confidence_threshold,
        )
        stable = max(previous_text, stable)
        supported = stable - previous_text
        quality_flags: list[str] = []
        if supported > 0 and semantic_cursor >= len(target_semantic):
            # Never create a text WRITE whose audible semantic delta is empty.
            stable = previous_text
            supported = 0
            quality_flags.append("semantic_exhausted_commit_deferred")
        natural = Action.WRITE if supported > 0 else Action.READ
        target_exhausted = previous_text >= len(translation_ids)
        forced = (
            plan.chunk_end_ms == 800
            and natural is Action.READ
            and not target_exhausted
        )
        deadline = Action.WRITE if forced else natural

        semantic_start = semantic_cursor
        semantic_end = semantic_cursor
        if natural is Action.WRITE:
            semantic_end = min(
                len(target_semantic), semantic_cursor + _block_size(str(row["id"]), plan.chunk_end_ms)
            )
            if semantic_end <= semantic_start:
                raise AssertionError("natural WRITE lost its semantic micro-block")
        history_start = max(0, semantic_cursor - 200)
        if forced:
            quality_flags.append("hard_scheduler_soft_teacher_only")

        record = TrajectoryRecord(
            sample_id=str(row["id"]),
            shard=shard,
            row_index=row_index,
            src_lang=str(row["src_lang"]),
            tgt_lang=str(row["tgt_lang"]),
            source_duration_ms=duration_ms,
            chunk_end_ms=plan.chunk_end_ms,
            future_1_end_ms=plan.future_1_end_ms,
            future_2_end_ms=plan.future_2_end_ms,
            trajectory_kind=plan.kind,
            causal_source_glm=tuple(_prefix(causal_tokens, plan.chunk_end_ms)),
            future_1_source_glm=tuple(_prefix(causal_tokens, plan.future_1_end_ms)),
            future_2_source_glm=tuple(_prefix(causal_tokens, plan.future_2_end_ms)),
            frontend_token_cache=causal_bundle_reference(cache_file, cache_row_index),
            translation_ids=tuple(int(value) for value in translation_ids),
            teacher_prefix_topk_path=teacher_bundle_reference(
                cache_file, request_indices[0]
            ),
            teacher_future_1_topk_path=teacher_bundle_reference(
                cache_file, request_indices[1]
            ),
            teacher_future_2_topk_path=teacher_bundle_reference(
                cache_file, request_indices[2]
            ),
            teacher_full_topk_path=teacher_bundle_reference(
                cache_file, request_indices[3]
            ),
            previous_committed_length=previous_text,
            stable_target_length=stable,
            new_supported_count=supported,
            support_bucket=min(supported, 4),
            safe_commit_mask=tuple(bool(value) for value in safe),
            natural_action_target=natural,
            deadline_action_target=deadline,
            deadline_forced_target=forced,
            deadline_loss_enabled=has_exact_deadline,
            target_text_delta_ids=tuple(
                int(value) for value in translation_ids[previous_text:stable]
            ),
            semantic_history_start=history_start,
            semantic_history_end=semantic_cursor,
            semantic_target_start=semantic_start,
            semantic_target_end=semantic_end,
            speaker_global=tuple(int(value) for value in row["bicodec_global"]),
            quality_flags=tuple(quality_flags),
        ).with_checksum()
        records.append(record)
        if natural is Action.WRITE:
            previous_text = stable
            semantic_cursor = semantic_end

    return tuple(records)


def process_shard(args: argparse.Namespace, shard: int, decoder, whisper, teacher) -> dict[str, Any]:
    output_dir = Path(args.output_root) / f"part-{shard:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "PART_COMPLETE.json"
    output = output_dir / "trajectory_cache.jsonl"
    if marker.is_file() and output.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("schema_version") == CACHE_PART_SCHEMA:
            return value

    source = Path(args.raw_unist_dir) / f"train-{shard:05d}.parquet"
    accepted = np.sort(
        np.concatenate(
            (
                np.load(Path(args.index_root) / f"train-{shard:05d}.eng.npy", mmap_mode="r"),
                np.load(Path(args.index_root) / f"train-{shard:05d}.cmn.npy", mmap_mode="r"),
            )
        )
    )
    if args.max_rows_per_shard is not None:
        accepted = accepted[: args.max_rows_per_shard]
    table = pq.read_table(source, columns=list(REQUIRED_COLUMNS))
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    started = time.time()
    try:
        with temporary.open("wb") as handle:
            for batch_number, start in enumerate(range(0, len(accepted), args.batch_size)):
                row_indices = [int(value) for value in accepted[start : start + args.batch_size]]
                rows = table.take(pa.array(row_indices, type=pa.int64())).to_pylist()
                waveforms = decoder.decode(
                    [[int(value) for value in row["bicodec_global"]] for row in rows],
                    [[int(value) for value in row["source_bicodec"]] for row in rows],
                )
                whisper_outputs = whisper.encode([(waveform, 16_000) for waveform in waveforms])
                translations = [teacher.encode_text(str(row["translation"])) for row in rows]

                requests: list[tuple[list[int], list[int]]] = []
                row_request_maps: list[dict[int, int]] = []
                for row, output_row, text_ids in zip(rows, whisper_outputs, translations):
                    causal = [int(value) for value in output_row.tokens]
                    duration_ms = len(row["source_bicodec"]) * 20
                    unique_times = {duration_ms}
                    for plan in plans_for_row(str(row["id"]), duration_ms):
                        unique_times.update(
                            (plan.chunk_end_ms, plan.future_1_end_ms, plan.future_2_end_ms)
                        )
                    request_map: dict[int, int] = {}
                    for end_ms in sorted(unique_times):
                        request_map[end_ms] = len(requests)
                        requests.append((teacher.prompt(row, _prefix(causal, end_ms)), text_ids))
                    row_request_maps.append(request_map)

                summaries = _summarize_bounded(
                    teacher, requests, args.teacher_request_batch_size
                )
                cache_file = output_dir / f"bundle-{batch_number:06d}.npz"
                _save_bundle(
                    cache_file,
                    summaries,
                    [[int(value) for value in output_row.tokens] for output_row in whisper_outputs],
                )
                for cache_row_index, (row_index, row, output_row, text_ids, request_map) in enumerate(
                    zip(row_indices, rows, whisper_outputs, translations, row_request_maps)
                ):
                    records = build_records_for_row(
                        shard=shard,
                        row_index=row_index,
                        row=row,
                        causal_tokens=[int(value) for value in output_row.tokens],
                        translation_ids=text_ids,
                        summaries=summaries,
                        request_by_end_ms=request_map,
                        cache_file=cache_file,
                        cache_row_index=cache_row_index,
                        confidence_threshold=args.confidence_threshold,
                    )
                    counts["sessions"] += 1
                    if any(record.chunk_end_ms == 800 for record in records):
                        counts["exact_800_sessions"] += 1
                    for record in records:
                        encoded = (
                            json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode("utf-8")
                        handle.write(encoded)
                        digest.update(encoded)
                        counts["trajectories"] += 1
                        counts[f"direction:{record.src_lang}->{record.tgt_lang}"] += 1
                        counts[f"action:{record.natural_action_target.value}"] += 1
                        counts["deadline_forced"] += int(record.deadline_forced_target)
                        counts["teacher_requests_unique"] += 0
                counts["teacher_requests_unique"] += len(requests)
                counts["teacher_requests_naive"] += sum(
                    4 * len(plans_for_row(str(row["id"]), len(row["source_bicodec"]) * 20))
                    for row in rows
                )
                completed = start + len(rows)
                if args.progress_interval and completed % args.progress_interval < len(rows):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "rank": args.rank,
                                "shard": shard,
                                "rows": completed,
                                "rows_per_second": completed / elapsed,
                                "batch_size": args.batch_size,
                                "teacher_request_batch_size": args.teacher_request_batch_size,
                                "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                                "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                            }
                        ),
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    value = {
        "schema_version": CACHE_PART_SCHEMA,
        "trajectory_schema": SCHEMA_VERSION,
        "rank": args.rank,
        "shard": shard,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "accepted_rows": len(accepted),
        "session_count": counts["sessions"],
        "trajectory_count": counts["trajectories"],
        "exact_800_sessions": counts["exact_800_sessions"],
        "natural_write": counts["action:WRITE"],
        "natural_read": counts["action:READ"],
        "deadline_forced": counts["deadline_forced"],
        "directions": {
            key.removeprefix("direction:"): count
            for key, count in counts.items()
            if key.startswith("direction:")
        },
        "bundle_count": len(list(output_dir.glob("bundle-*.npz"))),
        "batch_size": args.batch_size,
        "teacher_request_batch_size": args.teacher_request_batch_size,
        "teacher_requests_unique": counts["teacher_requests_unique"],
        "teacher_requests_naive": counts["teacher_requests_naive"],
        "teacher_request_dedup_fraction": 1.0
        - counts["teacher_requests_unique"] / max(1, counts["teacher_requests_naive"]),
        "confidence_threshold": args.confidence_threshold,
        "elapsed_seconds": time.time() - started,
    }
    if value["session_count"] != value["accepted_rows"]:
        raise AssertionError("cache session accounting mismatch")
    _atomic_json(marker, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-unist-dir", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase3-model", required=True, type=Path)
    parser.add_argument("--whispervq-model", required=True, type=Path)
    parser.add_argument("--bicodec-checkpoint", required=True, type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-request-batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--max-rows-per-shard", type=int)
    parser.add_argument(
        "--dynamic-shard-queue",
        action="store_true",
        help="Atomically claim unfinished shards so faster GPUs take more work.",
    )
    parser.add_argument("--progress-interval", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.shard_count != 15:
        raise ValueError("this isolated pilot is frozen to shards 0..14")
    if args.batch_size <= 0 or args.teacher_request_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    torch.cuda.set_device(args.rank)
    device = torch.device(f"cuda:{args.rank}")
    from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
        StreamingWhisperVQTeacher,
    )

    decoder = BatchedBiCodecDecoder(args.bicodec_checkpoint, device)
    whisper = StreamingWhisperVQTeacher(
        args.whispervq_model,
        device=str(device),
        chunk_ms=160,
        right_context_ms=80,
    )
    teacher = Phase3Teacher(
        args.phase3_model,
        device,
        topk=args.topk,
        temperature=args.temperature,
    )
    if args.dynamic_shard_queue:
        shards = range(args.shard_count)
    else:
        shards = range(args.rank, args.shard_count, args.world_size)
    results = []
    for shard in shards:
        lock: Path | None = None
        if args.dynamic_shard_queue:
            lock = try_claim_shard(Path(args.output_root), shard)
            if lock is None:
                continue
        try:
            results.append(process_shard(args, shard, decoder, whisper, teacher))
        finally:
            if lock is not None:
                lock.rmdir()
    print(json.dumps({"rank": args.rank, "parts": results}, sort_keys=True))


if __name__ == "__main__":
    main()
