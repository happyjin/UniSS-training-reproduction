"""Build resumable Stage-A source-audio supervision for the causal Stage-B frontend.

This is intentionally an isolated, Stage-B-ready subset of the larger Stage-A
research plan.  It reconstructs source audio, preserves the released UniST GLM
teacher tokens, and records their fixed-rate frame timing.  It does not claim
that bilingual target-support alignment is complete, so its completion marker
is named ``STAGE_A_SOURCE_COMPLETE.json`` rather than ``STAGE_A_COMPLETE.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import soundfile as sf
import torch

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.schema import coerce_token_list, sha256_file
from training.simul_uniss.subsecond_v1 import (
    STAGE_A_ASSEMBLY_SCHEMA,
    STAGE_A_PART_SCHEMA,
    STAGE_A_SOURCE_SCHEMA,
)


PART_MARKER = "STAGE_A_SOURCE_PART_COMPLETE.json"
ASSEMBLY_MARKER = "STAGE_A_SOURCE_COMPLETE.json"
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


def _atomic_write_json(path: Path, value: object) -> None:
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


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def _file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    # ``ParquetSchema.names`` exposes physical leaf names (for example
    # repeated list elements), while ``schema_arrow.names`` preserves the
    # logical top-level columns requested by ``iter_batches``.
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    for batch in parquet.iter_batches(columns=list(REQUIRED_COLUMNS), batch_size=batch_size):
        yield from batch.to_pylist()


class DecoderOnlyBiCodec:
    """Load only the BiCodec decoder instead of the unused Wav2Vec2 encoder."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        from uniss.speech_tokenizer.bicodec.models.bicodec import BiCodec

        self.device = device
        self.model = BiCodec.load_from_checkpoint(checkpoint).to(device).eval()

    @torch.inference_mode()
    def decode(self, global_tokens: list[int], semantic_tokens: list[int]) -> torch.Tensor:
        global_tensor = torch.tensor([global_tokens], dtype=torch.long, device=self.device).unsqueeze(1)
        semantic_tensor = torch.tensor([semantic_tokens], dtype=torch.long, device=self.device)
        waveform = self.model.detokenize(semantic_tensor, global_tensor)
        return waveform.detach().float().reshape(-1).cpu()


def _glm_end_times(duration_ms: int, token_count: int) -> list[int]:
    if token_count <= 0:
        return []
    return [min(duration_ms, round(duration_ms * (index + 1) / token_count)) for index in range(token_count)]


def _part_is_current(marker_path: Path, source: Path, limit_records: int | None, side: str) -> bool:
    if not marker_path.is_file():
        return False
    marker = _read_json(marker_path)
    if marker.get("schema_version") != STAGE_A_PART_SCHEMA:
        return False
    source_meta = marker.get("source")
    manifest_meta = marker.get("manifest")
    if not isinstance(source_meta, dict) or not isinstance(manifest_meta, dict):
        return False
    stat = source.stat()
    if (
        Path(str(source_meta.get("path"))).resolve() != source.resolve()
        or int(source_meta.get("size_bytes", -1)) != stat.st_size
        or int(source_meta.get("mtime_ns", -1)) != stat.st_mtime_ns
        or marker.get("limit_records") != limit_records
        or marker.get("side") != side
    ):
        return False
    manifest = Path(str(manifest_meta.get("path")))
    if not manifest.is_file() or manifest.stat().st_size != int(manifest_meta.get("size_bytes", -1)):
        return False
    offsets = load_index(manifest)
    return offsets is not None and len(offsets) == int(marker.get("records", -1))


def prepare_part(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / PART_MARKER
    if _part_is_current(marker_path, source, args.limit_records, args.side):
        marker = _read_json(marker_path)
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    decoder = DecoderOnlyBiCodec(Path(args.bicodec_checkpoint), device)
    source_audio_dir = output_dir / "source_audio"
    target_audio_dir = output_dir / "target_audio"
    source_audio_dir.mkdir(parents=True, exist_ok=True)
    if args.side == "both":
        target_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    temporary_manifest = output_dir / f".manifest.jsonl.tmp.{os.getpid()}"
    offsets = array("Q")
    offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with temporary_manifest.open("wb") as manifest:
            for row_index, row in enumerate(_iter_rows(source, args.batch_size)):
                if args.limit_records is not None and counts["records"] >= args.limit_records:
                    break
                identifier = str(row["id"])
                source_glm = coerce_token_list(row["source_glm"], "source_glm")
                source_bicodec = coerce_token_list(row["source_bicodec"], "source_bicodec")
                target_bicodec = coerce_token_list(row["target_bicodec"], "target_bicodec")
                global_tokens = coerce_token_list(row["bicodec_global"], "bicodec_global")
                if not source_glm or not source_bicodec or len(global_tokens) != 32:
                    counts["rejected"] += 1
                    continue

                name = f"{row_index:07d}_{_safe_name(identifier)}"
                source_audio = source_audio_dir / f"{name}.flac"
                if not source_audio.is_file():
                    waveform = decoder.decode(global_tokens, source_bicodec)
                    sf.write(source_audio, waveform.numpy(), args.sample_rate, format="FLAC")
                source_info = sf.info(source_audio)
                source_duration_ms = round(1000 * source_info.frames / source_info.samplerate)

                target_audio: Path | None = None
                target_duration_ms: int | None = None
                if args.side == "both":
                    target_audio = target_audio_dir / f"{name}.flac"
                    if not target_audio.is_file():
                        waveform = decoder.decode(global_tokens, target_bicodec)
                        sf.write(target_audio, waveform.numpy(), args.sample_rate, format="FLAC")
                    target_info = sf.info(target_audio)
                    target_duration_ms = round(1000 * target_info.frames / target_info.samplerate)

                item: dict[str, object] = {
                    "schema_version": STAGE_A_SOURCE_SCHEMA,
                    "stage_a_scope": "stage_b_source_frontend_v1",
                    "alignment_kind": "fixed_rate_glm_frame_alignment_v1",
                    "id": identifier,
                    "src_lang": str(row["src_lang"]),
                    "tgt_lang": str(row["tgt_lang"]),
                    "transcription": str(row["transcription"]),
                    "translation": str(row["translation"]),
                    "source_glm": source_glm,
                    "source_glm_end_ms": _glm_end_times(source_duration_ms, len(source_glm)),
                    "target_bicodec": target_bicodec,
                    "bicodec_global": global_tokens,
                    "source_audio": str(source_audio),
                    "source_duration_ms": source_duration_ms,
                    "target_audio": None if target_audio is None else str(target_audio),
                    "target_duration_ms": target_duration_ms,
                    "audio_origin": "bicodec_reconstructed",
                    "source_parquet": str(source),
                    "source_row_index": row_index,
                    "support_alignment_status": "pending_stage_c",
                    "quality_flags": [],
                }
                encoded = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                offsets.append(offset)
                manifest.write(encoded)
                offset += len(encoded)
                counts["records"] += 1
                counts[f"direction:{item['src_lang']}->{item['tgt_lang']}"] += 1
                if args.progress_interval and counts["records"] % args.progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "records": counts["records"],
                                "records_per_second": counts["records"] / elapsed,
                                "source": source.name,
                            }
                        ),
                        flush=True,
                    )
            manifest.flush()
            os.fsync(manifest.fileno())
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    index_metadata = write_index(manifest_path, offsets)
    source_metadata = _file_metadata(source)
    if not args.skip_sha256:
        source_metadata["sha256"] = sha256_file(source)
    marker = {
        "schema_version": STAGE_A_PART_SCHEMA,
        "stage_a_scope": "stage_b_source_frontend_v1",
        "source": source_metadata,
        "manifest": _file_metadata(manifest_path),
        "manifest_index": index_metadata,
        "records": counts["records"],
        "rejected": counts["rejected"],
        "directions": {
            key.removeprefix("direction:"): value
            for key, value in sorted(counts.items())
            if key.startswith("direction:")
        },
        "limit_records": args.limit_records,
        "side": args.side,
        "sample_rate": args.sample_rate,
        "elapsed_seconds": time.time() - started,
        "bicodec_checkpoint": str(Path(args.bicodec_checkpoint).resolve()),
    }
    _atomic_write_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def _concatenate(paths: Iterable[Path], destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    offsets = array("Q")
    offset = 0
    try:
        with temporary.open("wb") as output:
            for path in paths:
                with path.open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        offsets.append(offset)
                        output.write(line)
                        offset += len(line)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return write_index(destination, offsets)


def assemble(args: argparse.Namespace) -> dict[str, object]:
    parts_root = Path(args.parts_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / ASSEMBLY_MARKER
    manifest_path = output_dir / "stage_a_source_manifest.jsonl"
    if marker_path.is_file():
        marker = _read_json(marker_path)
        offsets = load_index(manifest_path) if manifest_path.is_file() else None
        if (
            marker.get("schema_version") == STAGE_A_ASSEMBLY_SCHEMA
            and offsets is not None
            and len(offsets) == int(marker.get("records", -1))
        ):
            print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
            return marker

    manifests: list[Path] = []
    markers: list[dict[str, object]] = []
    for shard_index in range(args.shard_start, args.shard_start + args.shard_count):
        part_dir = parts_root / f"train-{shard_index:05d}"
        part_marker_path = part_dir / PART_MARKER
        if not part_marker_path.is_file():
            raise FileNotFoundError(part_marker_path)
        part_marker = _read_json(part_marker_path)
        if part_marker.get("schema_version") != STAGE_A_PART_SCHEMA:
            raise ValueError(f"unexpected part schema in {part_marker_path}")
        manifest = part_dir / "manifest.jsonl"
        offsets = load_index(manifest)
        if offsets is None or len(offsets) != int(part_marker.get("records", -1)):
            raise ValueError(f"invalid manifest index for {manifest}")
        manifests.append(manifest)
        markers.append(part_marker)

    index_metadata = _concatenate(manifests, manifest_path)
    directions: Counter[str] = Counter()
    for marker in markers:
        for key, value in dict(marker.get("directions", {})).items():
            directions[str(key)] += int(value)
    marker = {
        "schema_version": STAGE_A_ASSEMBLY_SCHEMA,
        "stage_a_scope": "stage_b_source_frontend_v1",
        "warning": "Source/frontend supervision complete; bilingual support alignment remains pending for Stage C/D.",
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "records": sum(int(marker["records"]) for marker in markers),
        "rejected": sum(int(marker.get("rejected", 0)) for marker in markers),
        "directions": dict(sorted(directions.items())),
        "manifest": _file_metadata(manifest_path),
        "manifest_index": index_metadata,
        "parts": [str((parts_root / f"train-{index:05d}" / PART_MARKER).resolve()) for index in range(args.shard_start, args.shard_start + args.shard_count)],
    }
    _atomic_write_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def validate(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    marker = _read_json(output_dir / ASSEMBLY_MARKER)
    manifest = Path(str(dict(marker["manifest"])["path"]))
    offsets = load_index(manifest)
    if offsets is None or len(offsets) != int(marker.get("records", -1)):
        raise ValueError("assembled manifest index mismatch")
    if not offsets:
        raise ValueError("assembled manifest is empty")
    checked = 0
    with manifest.open("rb") as handle:
        step = max(1, len(offsets) // max(1, args.samples))
        for index in range(0, len(offsets), step):
            handle.seek(offsets[index])
            item = json.loads(handle.readline())
            audio = Path(item["source_audio"])
            if not audio.is_file():
                raise FileNotFoundError(audio)
            info = sf.info(audio)
            if info.samplerate != 16000 or info.frames <= 0:
                raise ValueError(f"invalid audio file: {audio}")
            if len(item["source_glm"]) != len(item["source_glm_end_ms"]):
                raise ValueError(f"GLM timing mismatch for {item['id']}")
            checked += 1
            if checked >= args.samples:
                break
    result = {
        "status": "valid",
        "records": len(offsets),
        "sampled_records": checked,
        "manifest": str(manifest),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-part")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--bicodec-checkpoint", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--side", choices=("source", "both"), default="source")
    prepare.add_argument("--limit-records", type=int, default=None)
    prepare.add_argument("--batch-size", type=int, default=64)
    prepare.add_argument("--sample-rate", type=int, default=16000)
    prepare.add_argument("--progress-interval", type=int, default=100)
    prepare.add_argument("--skip-sha256", action="store_true")

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--parts-root", required=True)
    assemble_parser.add_argument("--output-dir", required=True)
    assemble_parser.add_argument("--shard-start", type=int, default=0)
    assemble_parser.add_argument("--shard-count", type=int, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", required=True)
    validate_parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-part":
        prepare_part(args)
    elif args.command == "assemble":
        assemble(args)
    elif args.command == "validate":
        validate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
