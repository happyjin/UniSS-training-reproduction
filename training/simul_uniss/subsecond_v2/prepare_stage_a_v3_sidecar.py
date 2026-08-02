"""Build isolated causal-teacher sidecars for Stage-B-v2 training."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import json
import math
import os
import tempfile
import time
from array import array
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torchaudio

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
    StreamingTeacherOutput,
    StreamingWhisperVQTeacher,
)
from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer


SCHEMA = "simul_uniss_stage_a_v3_causal_sidecar_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def partition_range(
    total_records: int,
    *,
    start_index: int,
    limit_records: int | None,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    selected_end = total_records
    if limit_records is not None:
        selected_end = min(selected_end, start_index + max(0, limit_records))
    selected_start = min(max(0, start_index), selected_end)
    count = selected_end - selected_start
    per_rank = math.ceil(count / world_size) if count else 0
    left = min(selected_end, selected_start + rank * per_rank)
    right = min(selected_end, left + per_rank)
    return left, right


def _read_record(manifest: Path, offset: int, index: int, max_samples: int) -> dict[str, object]:
    with manifest.open("rb") as handle:
        handle.seek(offset)
        value = json.loads(handle.readline())
    waveform, sample_rate = torchaudio.load(str(value["source_audio"]))
    waveform = waveform[:1]
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    value["_waveform"] = waveform[..., :max_samples]
    value["_source_manifest_index"] = index
    value["_source_manifest_offset"] = offset
    return value


def _prefix_targets(
    teacher: Glm4Tokenizer,
    waveform: torch.Tensor,
    token_end_ms: Sequence[int],
    *,
    chunk_ms: int,
    lookahead_ms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    duration_ms = int(round(waveform.shape[-1] / 16))
    commit_ends = list(range(chunk_ms, duration_ms + chunk_ms, chunk_ms))
    commit_ends = list(dict.fromkeys(min(value, duration_ms) for value in commit_ends))
    audio: list[tuple[torch.Tensor, int]] = []
    for committed_ms in commit_ends:
        visible_ms = min(duration_ms, committed_ms + lookahead_ms)
        visible_samples = max(400, min(waveform.shape[-1], visible_ms * 16))
        audio.append((waveform[..., :visible_samples], 16_000))
    predictions = [
        [int(value) for value in tokens] for tokens in teacher.bacth_tokenize(audio)
    ]
    tokens: list[int] = []
    stability: list[int] = []
    for tick, committed_ms in enumerate(commit_ends):
        required = bisect.bisect_right(token_end_ms, committed_ms)
        while len(tokens) < required:
            index = len(tokens)
            source_tick = next(
                (
                    candidate
                    for candidate in range(tick, len(predictions))
                    if index < len(predictions[candidate])
                ),
                None,
            )
            if source_tick is None:
                break
            value = predictions[source_tick][index]
            later = [
                predictions[min(len(predictions) - 1, source_tick + delta)]
                for delta in (0, 1, 2)
            ]
            stable = all(index < len(current) and current[index] == value for current in later)
            tokens.append(value)
            stability.append(int(stable))
        if len(tokens) < required:
            break
    return torch.tensor(tokens, dtype=torch.int32), torch.tensor(stability, dtype=torch.uint8)


@torch.inference_mode()
def nearest_codebook_topk(
    hidden: torch.Tensor,
    codebook: torch.Tensor,
    *,
    topk: int,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= topk <= codebook.shape[0]:
        raise ValueError("invalid codebook top-k")
    codebook = codebook.float()
    codebook_norm = codebook.square().sum(dim=1).reshape(1, -1)
    ids: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, len(hidden), chunk_size):
        current = hidden[start : start + chunk_size].to(codebook.device).float()
        distance = (
            current.square().sum(dim=1, keepdim=True)
            + codebook_norm
            - 2.0 * current @ codebook.T
        ) / hidden.shape[-1]
        values, indices = torch.topk(distance, k=topk, dim=1, largest=False, sorted=True)
        ids.append(indices.to(torch.int32).cpu())
        distances.append(values.to(torch.float16).cpu())
    return torch.cat(ids), torch.cat(distances)


def _save_shard(
    output_dir: Path,
    shard_index: int,
    mode: str,
    rows: Sequence[Mapping[str, object]],
) -> tuple[Path, list[dict[str, object]]]:
    shard = output_dir / "shards" / f"shard-{shard_index:06d}.pt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    target_offsets = [0]
    reference_offsets = [0]
    targets: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    stability: list[torch.Tensor] = []
    hidden: list[torch.Tensor] = []
    topk_ids: list[torch.Tensor] = []
    topk_distances: list[torch.Tensor] = []
    manifest_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        target = row["target_tokens"]
        reference = row["full_reference_tokens"]
        stable = row["stability"]
        if not all(isinstance(value, torch.Tensor) for value in (target, reference, stable)):
            raise TypeError("sidecar row tensors are missing")
        targets.append(target)  # type: ignore[arg-type]
        references.append(reference)  # type: ignore[arg-type]
        stability.append(stable)  # type: ignore[arg-type]
        target_offsets.append(target_offsets[-1] + len(target))  # type: ignore[arg-type]
        reference_offsets.append(reference_offsets[-1] + len(reference))  # type: ignore[arg-type]
        if "pre_vq_hidden" in row:
            hidden.append(row["pre_vq_hidden"])  # type: ignore[arg-type]
            topk_ids.append(row["topk_ids"])  # type: ignore[arg-type]
            topk_distances.append(row["topk_distances"])  # type: ignore[arg-type]
        manifest_rows.append(
            {
                "schema_version": SCHEMA,
                "id": row["id"],
                "mode": mode,
                "source_manifest_index": row["source_manifest_index"],
                "source_manifest_offset": row["source_manifest_offset"],
                "shard_path": str(shard.resolve()),
                "shard_row": row_index,
                "target_start": target_offsets[-2],
                "target_end": target_offsets[-1],
                "reference_start": reference_offsets[-2],
                "reference_end": reference_offsets[-1],
            }
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "mode": mode,
        "record_ids": [str(row["id"]) for row in rows],
        "source_manifest_indices": torch.tensor(
            [int(row["source_manifest_index"]) for row in rows], dtype=torch.int64
        ),
        "target_offsets": torch.tensor(target_offsets, dtype=torch.int64),
        "reference_offsets": torch.tensor(reference_offsets, dtype=torch.int64),
        "target_tokens": torch.cat(targets).to(torch.int32),
        "full_reference_tokens": torch.cat(references).to(torch.int32),
        "stability": torch.cat(stability).to(torch.uint8),
    }
    if hidden:
        payload.update(
            {
                "pre_vq_hidden": torch.cat(hidden).to(torch.bfloat16),
                "topk_ids": torch.cat(topk_ids).to(torch.int32),
                "topk_distances": torch.cat(topk_distances).to(torch.float16),
            }
        )
    temporary = shard.with_name(f".{shard.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, shard)
    return shard, manifest_rows


def prepare(args: argparse.Namespace) -> dict[str, object]:
    manifest = Path(args.manifest).resolve()
    offsets = load_index(manifest)
    if offsets is None:
        raise ValueError(f"missing index for {manifest}")
    left, right = partition_range(
        len(offsets),
        start_index=args.start_index,
        limit_records=args.limit_records,
        rank=args.rank,
        world_size=args.world_size,
    )
    output_dir = Path(args.output_dir).resolve() / f"part-{args.rank:02d}"
    marker = output_dir / "PART_COMPLETE.json"
    if marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = args.max_audio_seconds * 16_000
    clone = None
    prefix_teacher = None
    codebook = None
    if args.mode == "clone":
        clone = StreamingWhisperVQTeacher(
            args.whispervq_model,
            device=args.device,
            chunk_ms=args.chunk_ms,
            right_context_ms=args.lookahead_ms,
        )
        codebook = clone.model.codebook.weight.detach()
    elif args.mode == "prefix80":
        prefix_teacher = Glm4Tokenizer(args.whispervq_model, device=args.device)
    else:
        raise ValueError(args.mode)

    part_manifest = output_dir / "manifest.jsonl"
    part_offsets = array("Q")
    byte_offset = 0
    processed = 0
    target_tokens = 0
    shard_index = 0
    pending_rows: list[dict[str, object]] = []
    started = time.time()
    temporary_manifest = output_dir / f".manifest.jsonl.tmp.{os.getpid()}"
    try:
        with temporary_manifest.open("wb") as manifest_handle:
            for batch_start in range(left, right, args.teacher_batch_size):
                indices = list(range(batch_start, min(right, batch_start + args.teacher_batch_size)))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.audio_workers
                ) as executor:
                    records = list(
                        executor.map(
                            lambda pair: _read_record(
                                manifest, offsets[pair], pair, max_samples
                            ),
                            indices,
                        )
                    )
                clone_outputs: list[StreamingTeacherOutput] | None = None
                if clone is not None:
                    clone_outputs = clone.encode(
                        [record["_waveform"] for record in records]  # type: ignore[list-item]
                    )
                sidecar_rows: list[dict[str, object]] = []
                for local_index, record in enumerate(records):
                    waveform = record.pop("_waveform")
                    if not isinstance(waveform, torch.Tensor):
                        raise TypeError("waveform is not a tensor")
                    duration_ms = int(round(waveform.shape[-1] / 16))
                    reference = [int(value) for value in record[args.reference_field]]  # type: ignore[index]
                    reference_ends = [int(value) for value in record[args.reference_end_field]]  # type: ignore[index]
                    reference_count = bisect.bisect_right(reference_ends, duration_ms)
                    reference_tensor = torch.tensor(
                        reference[:reference_count], dtype=torch.int32
                    )
                    if clone_outputs is not None:
                        output = clone_outputs[local_index]
                        count = min(len(output.tokens), len(output.pre_vq_hidden))
                        target = output.tokens[:count].to(torch.int32)
                        hidden = output.pre_vq_hidden[:count].to(torch.bfloat16)
                        assert codebook is not None
                        hard_ids, hard_distances = nearest_codebook_topk(
                            hidden,
                            codebook,
                            topk=args.codebook_topk,
                            chunk_size=args.quantize_chunk_size,
                        )
                        stable = torch.ones(count, dtype=torch.uint8)
                        row = {
                            "pre_vq_hidden": hidden,
                            "topk_ids": hard_ids,
                            "topk_distances": hard_distances,
                        }
                    else:
                        assert prefix_teacher is not None
                        target, stable = _prefix_targets(
                            prefix_teacher,
                            waveform,
                            reference_ends[:reference_count],
                            chunk_ms=args.chunk_ms,
                            lookahead_ms=args.lookahead_ms,
                        )
                        row = {}
                    count = len(target)
                    if not count:
                        continue
                    sidecar_rows.append(
                        {
                            **row,
                            "id": record.get("id"),
                            "source_manifest_index": record["_source_manifest_index"],
                            "source_manifest_offset": record["_source_manifest_offset"],
                            "target_tokens": target,
                            "full_reference_tokens": reference_tensor,
                            "stability": stable[:count],
                        }
                    )
                pending_rows.extend(sidecar_rows)
                while len(pending_rows) >= args.records_per_shard:
                    current = pending_rows[: args.records_per_shard]
                    del pending_rows[: args.records_per_shard]
                    _, rows = _save_shard(output_dir, shard_index, args.mode, current)
                    shard_index += 1
                    for row in rows:
                        encoded = (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        ).encode("utf-8")
                        part_offsets.append(byte_offset)
                        manifest_handle.write(encoded)
                        byte_offset += len(encoded)
                    processed += len(current)
                    target_tokens += sum(int(row["target_end"]) - int(row["target_start"]) for row in rows)
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "processed": processed,
                            "assigned": right - left,
                            "records_per_second": processed / max(1e-6, time.time() - started),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if pending_rows:
                _, rows = _save_shard(output_dir, shard_index, args.mode, pending_rows)
                shard_index += 1
                for row in rows:
                    encoded = (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    part_offsets.append(byte_offset)
                    manifest_handle.write(encoded)
                    byte_offset += len(encoded)
                processed += len(pending_rows)
                target_tokens += sum(
                    int(row["target_end"]) - int(row["target_start"]) for row in rows
                )
                pending_rows.clear()
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        os.replace(temporary_manifest, part_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    index = write_index(part_manifest, part_offsets)
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "mode": args.mode,
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_start": left,
        "assigned_end": right,
        "assigned_records": right - left,
        "processed_records": processed,
        "target_tokens": target_tokens,
        "manifest": str(part_manifest),
        "index": index,
        "shards": shard_index,
        "chunk_ms": args.chunk_ms,
        "lookahead_ms": args.lookahead_ms,
        "codebook_topk": args.codebook_topk if args.mode == "clone" else None,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["clone", "prefix80"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--audio-workers", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--records-per-shard", type=int, default=512)
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, default=80)
    parser.add_argument("--codebook-topk", type=int, default=32)
    parser.add_argument("--quantize-chunk-size", type=int, default=256)
    parser.add_argument("--reference-field", default="teacher_source_glm")
    parser.add_argument("--reference-end-field", default="teacher_source_glm_end_ms")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
