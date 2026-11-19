#!/usr/bin/env python3
"""Convert MiniMind-o Mimi codes into audio and Qwen3-TTS codec supervision.

The MiniMind-o parquet format stores each assistant audio as a flat sequence
with frame-major Mimi codes::

    [frame0_cb0, ..., frame0_cb7, frame1_cb0, ..., frame1_cb7, ...]

This script decodes that sequence with MiniMind-o's Mimi checkpoint, writes a
lossless PCM16 FLAC, then feeds the *saved file* through the same Qwen3-TTS
tokenizer used by bridge-alignment training.  Feeding the saved file (rather
than the in-memory float waveform) makes the precomputed codes reproducible
from ``target_audio``.

Outputs are transactional per input row-group and can be resumed safely::

    OUTPUT_ROOT/
      raw_audio/
        audio/rg_00000/row_000000000000.flac
        manifests/part-rg00000.jsonl
        text_to_speech_translation.jsonl         # with --finalize
      qwen3_tts_codec/
        shards/part-rg00000.parquet
        text_to_speech_translation.parquet       # with --finalize
      state/part-rg00000.json
      conversion_metadata.json

The raw JSONL follows the existing bridge-alignment contract.  The Qwen3
parquet uses the same semantic fields but replaces ``target_audio`` with
``target_codec_codes: list<list<int16>>`` in [T, 16] layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import MimiModel


MIMI_CODEBOOKS = 8
QWEN3_CODEBOOKS = 16
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1_920
SCHEMA_VERSION = "ms-swift-speech-simul-v1"
TASK = "text_to_speech_translation"


QWEN_SCHEMA = pa.schema([
    pa.field("sample_id", pa.string()),
    pa.field("source_row", pa.int64()),
    pa.field("schema_version", pa.string()),
    pa.field("task", pa.string()),
    pa.field("task_prompt", pa.string()),
    pa.field(
        "messages",
        pa.list_(pa.struct([
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
        ])),
    ),
    pa.field("audios", pa.list_(pa.string())),
    pa.field("target_text", pa.string()),
    pa.field("target_codec_codes", pa.list_(pa.list_(pa.int16()))),
    pa.field("source_lang", pa.string()),
    pa.field("target_lang", pa.string()),
    pa.field("latency_budget_ms", pa.int32()),
])


@dataclass
class PartRecord:
    key: str
    row_group: int
    source_start: int
    source_end: int
    num_rows: int
    raw_manifest: str
    qwen3_codec_shard: str
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        default=str(repo_root / "data/sft_t2a_translate_merged.parquet"),
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "data/sft_t2a_translate_converted"),
    )
    parser.add_argument(
        "--mimi-model",
        default=str(repo_root / "projects/minimind-o/model/mimi"),
    )
    parser.add_argument(
        "--qwen3-tokenizer",
        default=str(
            repo_root
            / "projects/liveinterpret_training_pipelines/models/Qwen3-TTS-Tokenizer-12Hz"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-batch-frames",
        type=int,
        default=None,
        help=(
            "Maximum padded Mimi frames per GPU batch (batch_rows * max_T). "
            "Use this with a large --batch-size to keep variable-length audio near a fixed memory budget."
        ),
    )
    parser.add_argument("--start-row-group", type=int, default=0)
    parser.add_argument("--end-row-group", type=int, default=None, help="Exclusive")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--audio-files-per-dir",
        type=int,
        default=10_000,
        help="Additional fan-out inside each row-group audio directory.",
    )
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Combine completed manifest/parquet parts into one file per subfolder.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only combine existing completed parts; do not load codec models.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def infer_languages(user_text: str) -> Tuple[str, str]:
    if "翻译成英文" in user_text:
        return "zh", "en"
    lowered = user_text.lower()
    if lowered.startswith("translate ") and " into chinese" in lowered:
        return "en", "zh"
    return "", ""


def task_prompt(target_lang: str) -> str:
    language = {"en": "English", "zh": "Chinese"}.get(target_lang, target_lang)
    if language:
        return f"Task: Translate the input text and synthesize the translation as {language} speech."
    return "Task: Translate the input text and synthesize the translation as speech."


def parse_conversations(raw: str) -> List[Dict[str, str]]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("conversations must be a non-empty JSON list")
    messages: List[Dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("each conversation message must be an object")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported conversation role: {role!r}")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final conversation message must be assistant")
    return messages


def reshape_mimi(flat_codes: Sequence[int]) -> torch.LongTensor:
    values = torch.as_tensor(flat_codes, dtype=torch.long)
    if values.numel() == 0 or values.numel() % MIMI_CODEBOOKS:
        raise ValueError(
            f"Mimi token count must be positive and divisible by {MIMI_CODEBOOKS}; "
            f"got {values.numel()}"
        )
    values = values.reshape(-1, MIMI_CODEBOOKS).transpose(0, 1).contiguous()
    if int(values.min()) < 0 or int(values.max()) >= 2048:
        raise ValueError(
            f"Mimi codes must be in [0, 2047], got [{int(values.min())}, {int(values.max())}]"
        )
    return values


def decode_mimi_batch(
    model: MimiModel,
    codes: Sequence[torch.LongTensor],
    device: str,
) -> List[np.ndarray]:
    max_frames = max(int(x.shape[1]) for x in codes)
    padded = torch.zeros(
        (len(codes), MIMI_CODEBOOKS, max_frames),
        dtype=torch.long,
        device=device,
    )
    for i, item in enumerate(codes):
        padded[i, :, : item.shape[1]] = item.to(device)
    with torch.inference_mode():
        decoded = model.decode(padded).audio_values.detach().float().cpu()
    result = []
    for i, item in enumerate(codes):
        expected_samples = int(item.shape[1]) * SAMPLES_PER_FRAME
        waveform = decoded[i].reshape(-1)[:expected_samples].numpy()
        result.append(np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False))
    return result


def load_models(mimi_path: Path, qwen_path: Path, device: str):
    if not (mimi_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Mimi weights not found: {mimi_path / 'model.safetensors'}")
    if not (qwen_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Qwen3-TTS tokenizer weights not found: {qwen_path / 'model.safetensors'}")

    mimi = MimiModel.from_pretrained(str(mimi_path)).eval().to(device)
    if device.startswith("cuda"):
        mimi = mimi.half()
    else:
        mimi = mimi.float()

    from qwen_tts import Qwen3TTSTokenizer

    qwen = Qwen3TTSTokenizer.from_pretrained(str(qwen_path), device_map=device)
    return mimi, qwen


def audio_path_for(raw_audio_dir: Path, row_group: int, source_row: int, files_per_dir: int) -> Path:
    bucket = source_row // files_per_dir
    return raw_audio_dir / f"rg_{row_group:05d}" / f"bucket_{bucket:06d}" / f"row_{source_row:012d}.flac"


def write_flac(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.flac")
    sf.write(tmp, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)


def raw_manifest_row(
    source_row: int,
    messages: List[Dict[str, str]],
    target_audio: Path,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    return {
        "sample_id": f"minimind-t2a-{source_row:012d}",
        "source_row": source_row,
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "task_prompt": task_prompt(target_lang),
        "messages": messages,
        "audios": [],
        "target_text": messages[-1]["content"],
        "target_audio": str(target_audio.resolve()),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "latency_budget_ms": 0,
    }


def qwen_codec_row(raw_row: Dict[str, Any], codes: torch.Tensor) -> Dict[str, Any]:
    if codes.ndim != 2 or int(codes.shape[1]) != QWEN3_CODEBOOKS:
        raise ValueError(f"Qwen3 codec must have shape [T, {QWEN3_CODEBOOKS}], got {tuple(codes.shape)}")
    return {
        "sample_id": raw_row["sample_id"],
        "source_row": raw_row["source_row"],
        "schema_version": raw_row["schema_version"],
        "task": raw_row["task"],
        "task_prompt": raw_row["task_prompt"],
        "messages": raw_row["messages"],
        "audios": raw_row["audios"],
        "target_text": raw_row["target_text"],
        "target_codec_codes": codes.detach().cpu().to(torch.int16).tolist(),
        "source_lang": raw_row["source_lang"],
        "target_lang": raw_row["target_lang"],
        "latency_budget_ms": raw_row["latency_budget_ms"],
    }


def row_group_offsets(parquet: pq.ParquetFile) -> List[int]:
    offsets = [0]
    for index in range(parquet.num_row_groups):
        offsets.append(offsets[-1] + parquet.metadata.row_group(index).num_rows)
    return offsets


def part_key(row_group: int, start: int, end: int, full_start: int, full_end: int) -> str:
    if start == full_start and end == full_end:
        return f"rg{row_group:05d}"
    return f"rg{row_group:05d}-{start:012d}-{end:012d}"


def split_by_frame_budget(
    records: Sequence[Tuple[str, Sequence[Sequence[int]]]],
    max_rows: int,
    max_padded_frames: Optional[int],
) -> Iterable[Sequence[Tuple[str, Sequence[Sequence[int]]]]]:
    """Greedily batch variable-length Mimi rows under a padded-frame budget."""
    batch: List[Tuple[str, Sequence[Sequence[int]]]] = []
    batch_max_frames = 0
    for record in records:
        audio_turns = record[1]
        if not audio_turns:
            frames = 0
        else:
            token_count = len(audio_turns[-1])
            frames = token_count // MIMI_CODEBOOKS if token_count % MIMI_CODEBOOKS == 0 else token_count
        candidate_max = max(batch_max_frames, frames)
        over_rows = len(batch) >= max_rows
        over_frames = (
            max_padded_frames is not None
            and batch
            and (len(batch) + 1) * candidate_max > max_padded_frames
        )
        if over_rows or over_frames:
            yield batch
            batch = []
            batch_max_frames = 0
        batch.append(record)
        batch_max_frames = max(batch_max_frames, frames)
    if batch:
        yield batch


def process_part(
    *,
    parquet: pq.ParquetFile,
    row_group: int,
    local_start: int,
    local_end: int,
    global_start: int,
    key: str,
    output_root: Path,
    raw_audio_dir: Path,
    mimi: MimiModel,
    qwen: Any,
    device: str,
    batch_size: int,
    max_batch_frames: Optional[int],
    files_per_dir: int,
    compression: Optional[str],
    overwrite: bool,
) -> PartRecord:
    state_path = output_root / "state" / f"part-{key}.json"
    manifest_path = output_root / "raw_audio/manifests" / f"part-{key}.jsonl"
    codec_path = output_root / "qwen3_tts_codec/shards" / f"part-{key}.parquet"
    if state_path.is_file() and manifest_path.is_file() and codec_path.is_file() and not overwrite:
        with state_path.open(encoding="utf-8") as f:
            return PartRecord(**json.load(f))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    codec_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    codec_tmp = codec_path.with_name(f".{codec_path.name}.{os.getpid()}.tmp")
    for tmp in (manifest_tmp, codec_tmp):
        if tmp.exists():
            tmp.unlink()

    target_rows = local_end - local_start
    processed = 0
    group_cursor = 0
    started = time.monotonic()
    writer = pq.ParquetWriter(codec_tmp, QWEN_SCHEMA, compression=compression)
    try:
        with manifest_tmp.open("w", encoding="utf-8") as manifest:
            batches = parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=["conversations", "answer_audios"],
            )
            for record_batch in batches:
                batch_start = group_cursor
                batch_end = batch_start + record_batch.num_rows
                group_cursor = batch_end
                if batch_end <= local_start:
                    continue
                if batch_start >= local_end:
                    break
                slice_start = max(local_start - batch_start, 0)
                slice_end = min(local_end - batch_start, record_batch.num_rows)
                selected = record_batch.slice(slice_start, slice_end - slice_start)
                conversations = selected.column(0).to_pylist()
                answer_audios = selected.column(1).to_pylist()
                records = list(zip(conversations, answer_audios))
                for batch in split_by_frame_budget(records, batch_size, max_batch_frames):
                    source_rows = [global_start + processed + i for i in range(len(batch))]
                    mimi_codes: List[torch.LongTensor] = []
                    messages_list: List[List[Dict[str, str]]] = []
                    for (conversation_raw, audio_turns), source_row in zip(batch, source_rows):
                        messages = parse_conversations(conversation_raw)
                        if not audio_turns:
                            raise ValueError(f"source row {source_row} has no answer_audios")
                        mimi_codes.append(reshape_mimi(audio_turns[-1]))
                        messages_list.append(messages)

                    waveforms = decode_mimi_batch(mimi, mimi_codes, device)
                    audio_paths: List[Path] = []
                    raw_rows: List[Dict[str, Any]] = []
                    for source_row, messages, waveform in zip(source_rows, messages_list, waveforms):
                        path = audio_path_for(raw_audio_dir, row_group, source_row, files_per_dir)
                        write_flac(path, waveform)
                        source_lang, target_lang = infer_languages(
                            next((m["content"] for m in messages if m["role"] == "user"), "")
                        )
                        raw_row = raw_manifest_row(
                            source_row, messages, path, source_lang, target_lang
                        )
                        manifest.write(
                            json.dumps(raw_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        audio_paths.append(path)
                        raw_rows.append(raw_row)

                    # Encode the saved PCM16 FLACs so precomputed supervision is
                    # bit-for-bit reproducible from target_audio.
                    encoded = qwen.encode([str(path) for path in audio_paths])
                    qwen_rows = [
                        qwen_codec_row(raw_row, codes)
                        for raw_row, codes in zip(raw_rows, encoded.audio_codes)
                    ]
                    if len(qwen_rows) != len(raw_rows):
                        raise ValueError(
                            f"Qwen3 tokenizer returned {len(qwen_rows)} rows for {len(raw_rows)} inputs")
                    writer.write_table(pa.Table.from_pylist(qwen_rows, schema=QWEN_SCHEMA))
                    processed += len(batch)
            if processed != target_rows:
                raise ValueError(f"expected to process {target_rows} rows, processed {processed}")
    except Exception:
        writer.close()
        for tmp in (manifest_tmp, codec_tmp):
            if tmp.exists():
                tmp.unlink()
        raise
    else:
        writer.close()

    os.replace(manifest_tmp, manifest_path)
    os.replace(codec_tmp, codec_path)
    record = PartRecord(
        key=key,
        row_group=row_group,
        source_start=global_start,
        source_end=global_start + processed,
        num_rows=processed,
        raw_manifest=str(manifest_path.resolve()),
        qwen3_codec_shard=str(codec_path.resolve()),
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_json(state_path, asdict(record))
    return record


def load_completed_parts(state_dir: Path) -> List[PartRecord]:
    parts = []
    for path in sorted(state_dir.glob("part-*.json")):
        with path.open(encoding="utf-8") as f:
            parts.append(PartRecord(**json.load(f)))
    parts.sort(key=lambda x: (x.source_start, x.source_end))
    for previous, current in zip(parts, parts[1:]):
        if current.source_start < previous.source_end:
            raise ValueError(
                f"overlapping completed parts: {previous.key} and {current.key}; "
                "remove the stale state file before finalizing"
            )
    return parts


def finalize_outputs(output_root: Path) -> None:
    parts = load_completed_parts(output_root / "state")
    if not parts:
        raise ValueError("no completed parts to finalize")

    raw_final = output_root / f"raw_audio/{TASK}.jsonl"
    raw_tmp = raw_final.with_name(f".{raw_final.name}.{os.getpid()}.tmp")
    with raw_tmp.open("wb") as dst:
        for part in parts:
            with open(part.raw_manifest, "rb") as src:
                for line_number, raw_line in enumerate(src, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON in {part.raw_manifest}:{line_number}"
                        ) from exc
                    # Workers started before this task was split from the S2S
                    # pipeline wrote the legacy task name into their completed
                    # parts. Canonicalise at finalization so no audio/codec
                    # recomputation is required and mixed old/new parts remain
                    # safe to resume.
                    row["task"] = TASK
                    dst.write(
                        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        .encode("utf-8")
                    )
    os.replace(raw_tmp, raw_final)

    codec_final = output_root / f"qwen3_tts_codec/{TASK}.parquet"
    codec_tmp = codec_final.with_name(f".{codec_final.name}.{os.getpid()}.tmp")
    writer = pq.ParquetWriter(codec_tmp, QWEN_SCHEMA, compression="zstd")
    try:
        for part in parts:
            source = pq.ParquetFile(part.qwen3_codec_shard)
            for batch in source.iter_batches(batch_size=256):
                task_index = batch.schema.get_field_index("task")
                if task_index < 0:
                    raise ValueError(f"codec shard is missing task column: {part.qwen3_codec_shard}")
                batch = batch.set_column(
                    task_index,
                    QWEN_SCHEMA.field("task"),
                    pa.array([TASK] * batch.num_rows, type=pa.string()),
                )
                writer.write_batch(batch)
    finally:
        writer.close()
    os.replace(codec_tmp, codec_final)

    atomic_json(
        output_root / "conversion_index.json",
        {
            "task": TASK,
            "num_parts": len(parts),
            "num_rows": sum(part.num_rows for part in parts),
            "source_start": parts[0].source_start,
            "source_end": parts[-1].source_end,
            "raw_manifest": str(raw_final.resolve()),
            "qwen3_codec_parquet": str(codec_final.resolve()),
            "parts": [asdict(part) for part in parts],
        },
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_batch_frames is not None and args.max_batch_frames < 1:
        raise ValueError("--max-batch-frames must be positive")
    if args.audio_files_per_dir < 1:
        raise ValueError("--audio-files-per-dir must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    input_path = Path(args.input_parquet).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    mimi_path = Path(args.mimi_model).expanduser().resolve()
    qwen_path = Path(args.qwen3_tokenizer).expanduser().resolve()
    raw_audio_dir = output_root / "raw_audio/audio"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        print("Finalizing existing completed parts...", flush=True)
        finalize_outputs(output_root)
        print(f"Output root: {output_root}")
        return 0

    source = pq.ParquetFile(input_path)
    expected_columns = {"conversations", "answer_audios"}
    missing = expected_columns - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"input parquet is missing columns: {sorted(missing)}")
    end_row_group = source.num_row_groups if args.end_row_group is None else args.end_row_group
    if not (0 <= args.start_row_group < end_row_group <= source.num_row_groups):
        raise ValueError(
            f"row-group range must satisfy 0 <= start < end <= {source.num_row_groups}"
        )

    device = resolve_device(args.device)
    compression = None if args.compression == "none" else args.compression
    atomic_json(
        output_root / "conversion_metadata.json",
        {
            "task": TASK,
            "input_parquet": str(input_path),
            "input_rows": source.metadata.num_rows,
            "input_row_groups": source.num_row_groups,
            "mimi_model": str(mimi_path),
            "qwen3_tokenizer": str(qwen_path),
            "device": device,
            "mimi_layout": "T,8 (input flattened frame-major)",
            "qwen3_layout": "T,16",
            "sample_rate": SAMPLE_RATE,
            "args": vars(args),
        },
    )

    print(f"Loading Mimi and Qwen3-TTS tokenizer on {device}...", flush=True)
    mimi, qwen = load_models(mimi_path, qwen_path, device)
    offsets = row_group_offsets(source)
    remaining = args.max_samples
    completed: List[PartRecord] = []
    for row_group in range(args.start_row_group, end_row_group):
        group_rows = source.metadata.row_group(row_group).num_rows
        take = group_rows if remaining is None else min(group_rows, remaining)
        if take <= 0:
            break
        full_start, full_end = offsets[row_group], offsets[row_group + 1]
        start, end = full_start, full_start + take
        key = part_key(row_group, start, end, full_start, full_end)
        print(f"Converting {key}: source rows [{start}, {end})", flush=True)
        part = process_part(
            parquet=source,
            row_group=row_group,
            local_start=0,
            local_end=take,
            global_start=start,
            key=key,
            output_root=output_root,
            raw_audio_dir=raw_audio_dir,
            mimi=mimi,
            qwen=qwen,
            device=device,
            batch_size=args.batch_size,
            max_batch_frames=args.max_batch_frames,
            files_per_dir=args.audio_files_per_dir,
            compression=compression,
            overwrite=args.overwrite,
        )
        completed.append(part)
        rate = part.num_rows / max(part.elapsed_seconds, 1e-9)
        memory_suffix = ""
        if device.startswith("cuda"):
            peak_allocated = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            memory_suffix = (
                f", peak_cuda_allocated={peak_allocated:.2f}GiB"
                f", peak_cuda_reserved={peak_reserved:.2f}GiB"
            )
        print(
            f"Completed {part.key}: {part.num_rows} rows in "
            f"{part.elapsed_seconds:.1f}s ({rate:.3f} rows/s){memory_suffix}",
            flush=True,
        )
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break

    if args.finalize:
        print("Finalizing combined JSONL and parquet...", flush=True)
        finalize_outputs(output_root)
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed row-group parts remain resumable.", file=sys.stderr)
        raise SystemExit(130)
