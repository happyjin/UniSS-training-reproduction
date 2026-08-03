"""Export exact-prefix token and pre-VQ hidden supervision for Stage-B-v3."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from array import array
from pathlib import Path

import torch

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v2.prepare_stage_a_v3_sidecar import (
    _read_record,
    _save_shard,
    nearest_codebook_topk,
    partition_range,
)
from training.simul_uniss.subsecond_v3.prefix_hidden_teacher import (
    ExactPrefixWhisperVQTeacher,
    build_exact_prefix_hidden_targets,
)


SCHEMA = "simul_uniss_stage_b_v3_prefix_hidden_sidecar_v1"


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


def _selection_row(path: Path, offset: int) -> dict[str, object]:
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def prepare(args: argparse.Namespace) -> dict[str, object]:
    source_manifest = Path(args.source_manifest).resolve()
    selection_manifest = Path(args.selection_manifest).resolve()
    selection_offsets = load_index(selection_manifest)
    if selection_offsets is None:
        raise ValueError(f"missing selection index for {selection_manifest}")
    left, right = partition_range(
        len(selection_offsets),
        start_index=0,
        limit_records=None,
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
    teacher = ExactPrefixWhisperVQTeacher(args.whispervq_model, device=args.device)
    codebook = teacher.model.codebook.weight.detach()
    max_samples = round(args.max_audio_seconds * 16_000)
    selected_rows = [
        _selection_row(selection_manifest, selection_offsets[index])
        for index in range(left, right)
    ]
    part_manifest = output_dir / "manifest.jsonl"
    part_positions = array("Q")
    byte_offset = processed = target_tokens = hidden_tokens = shard_index = 0
    directions: dict[str, int] = {}
    pending: list[dict[str, object]] = []
    started = time.time()
    temporary_manifest = output_dir / f".manifest.jsonl.tmp.{os.getpid()}"
    try:
        with temporary_manifest.open("wb") as manifest_handle:
            for batch_start in range(0, len(selected_rows), args.record_batch_size):
                selection_batch = selected_rows[
                    batch_start : batch_start + args.record_batch_size
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.audio_workers
                ) as executor:
                    records = list(
                        executor.map(
                            lambda row: _read_record(
                                source_manifest,
                                int(row["source_manifest_offset"]),
                                int(row["source_manifest_index"]),
                                max_samples,
                            ),
                            selection_batch,
                        )
                    )
                for selection, record in zip(selection_batch, records):
                    waveform = record.pop("_waveform")
                    if not isinstance(waveform, torch.Tensor):
                        raise TypeError("waveform is not a tensor")
                    duration_ms = int(round(waveform.shape[-1] / 16))
                    reference = [int(value) for value in record[args.reference_field]]
                    reference_ends = [
                        int(value) for value in record[args.reference_end_field]
                    ]
                    reference_count = sum(value <= duration_ms for value in reference_ends)
                    target, stability, hidden = build_exact_prefix_hidden_targets(
                        teacher,
                        waveform,
                        reference_ends[:reference_count],
                        chunk_ms=args.chunk_ms,
                        lookahead_ms=args.lookahead_ms,
                    )
                    count = min(len(target), len(hidden))
                    if not count:
                        continue
                    target = target[:count]
                    stability = stability[:count]
                    hidden = hidden[:count].to(torch.bfloat16)
                    topk_ids, topk_distances = nearest_codebook_topk(
                        hidden,
                        codebook,
                        topk=args.codebook_topk,
                        chunk_size=args.quantize_chunk_size,
                    )
                    direction = str(selection["direction"])
                    directions[direction] = directions.get(direction, 0) + 1
                    pending.append(
                        {
                            "id": record.get("id"),
                            "src_lang": selection.get("src_lang"),
                            "tgt_lang": selection.get("tgt_lang"),
                            "direction": direction,
                            "source_manifest_index": record["_source_manifest_index"],
                            "source_manifest_offset": record["_source_manifest_offset"],
                            "target_tokens": target,
                            "full_reference_tokens": torch.tensor(
                                reference[:reference_count], dtype=torch.int32
                            ),
                            "stability": stability,
                            "pre_vq_hidden": hidden,
                            "topk_ids": topk_ids,
                            "topk_distances": topk_distances,
                        }
                    )
                while len(pending) >= args.records_per_shard:
                    current = pending[: args.records_per_shard]
                    del pending[: args.records_per_shard]
                    _, manifest_rows = _save_shard(
                        output_dir, shard_index, "prefix80_hidden", current
                    )
                    shard_index += 1
                    for row, source in zip(manifest_rows, current):
                        row.update(
                            {
                                "schema_version": SCHEMA,
                                "src_lang": source["src_lang"],
                                "tgt_lang": source["tgt_lang"],
                                "direction": source["direction"],
                                "supervision_mode": "exact_prefix80_hidden",
                            }
                        )
                        encoded = (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode()
                        part_positions.append(byte_offset)
                        manifest_handle.write(encoded)
                        byte_offset += len(encoded)
                    processed += len(current)
                    token_count = sum(
                        int(row["target_end"]) - int(row["target_start"])
                        for row in manifest_rows
                    )
                    target_tokens += token_count
                    hidden_tokens += token_count
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "processed": processed,
                            "assigned": right - left,
                            "records_per_second": processed
                            / max(1e-6, time.time() - started),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if pending:
                _, manifest_rows = _save_shard(
                    output_dir, shard_index, "prefix80_hidden", pending
                )
                shard_index += 1
                for row, source in zip(manifest_rows, pending):
                    row.update(
                        {
                            "schema_version": SCHEMA,
                            "src_lang": source["src_lang"],
                            "tgt_lang": source["tgt_lang"],
                            "direction": source["direction"],
                            "supervision_mode": "exact_prefix80_hidden",
                        }
                    )
                    encoded = (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode()
                    part_positions.append(byte_offset)
                    manifest_handle.write(encoded)
                    byte_offset += len(encoded)
                processed += len(pending)
                token_count = sum(
                    int(row["target_end"]) - int(row["target_start"])
                    for row in manifest_rows
                )
                target_tokens += token_count
                hidden_tokens += token_count
                pending.clear()
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        os.replace(temporary_manifest, part_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "mode": "prefix80_hidden",
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_start": left,
        "assigned_end": right,
        "assigned_records": right - left,
        "processed_records": processed,
        "directions": directions,
        "target_tokens": target_tokens,
        "hidden_tokens": hidden_tokens,
        "hidden_coverage": hidden_tokens / max(1, target_tokens),
        "manifest": str(part_manifest),
        "index": write_index(part_manifest, part_positions),
        "shards": shard_index,
        "chunk_ms": args.chunk_ms,
        "lookahead_ms": args.lookahead_ms,
        "codebook_topk": args.codebook_topk,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--audio-workers", type=int, default=4)
    parser.add_argument("--record-batch-size", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=512)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, default=80)
    parser.add_argument("--codebook-topk", type=int, default=32)
    parser.add_argument("--quantize-chunk-size", type=int, default=256)
    parser.add_argument("--reference-field", default="teacher_source_glm")
    parser.add_argument("--reference-end-field", default="teacher_source_glm_end_ms")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
