#!/usr/bin/env python3
"""Convert UniST Spark-TTS BiCodec rows to the project's Qwen3 S2S manifest.

The conversion is deliberately streaming and resumable at input-Parquet-file
granularity:

1. Decode source and target BiCodec semantic tokens with their shared global
   speaker tokens into 16 kHz waveforms.
2. Store source waveforms as indexed FLAC members in tar shards.  The manifest
   points at each member with the project's ``wds://...?offset=&length=`` URI.
3. Resample target waveforms in a GPU batch and encode them with the Qwen3-TTS
   12 Hz tokenizer into ``target_codec_codes`` with shape ``[T, 16]``.
4. Write canonical bidirectional S2S prompts and all text/provenance fields to
   one JSONL manifest shard per input Parquet file.

Rows with an empty or single-frame target BiCodec stream cannot provide usable
target-speech supervision and are written to transactional reject manifests
instead of being silently skipped. All other malformed rows fail fast. A state
file is committed only after the corresponding manifest, reject manifest, and
every tar shard have been closed and renamed. Re-running with the same
rank/world-size skips committed inputs.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import orjson
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF


SAMPLE_RATE = 16_000
QWEN_SAMPLE_RATE = 24_000
SAMPLES_PER_BICODEC_TOKEN = 320
QWEN_CODEBOOKS = 16
QWEN_CODEBOOK_SIZE = 2048
GLOBAL_TOKEN_COUNT = 32
GLOBAL_CODEBOOK_SIZE = 4096
TASK = "speech_to_speech_translation"
SCHEMA_VERSION = "ms-swift-speech-simul-v1"
REJECT_SCHEMA_VERSION = "unist-conversion-reject-v1"
DEFAULT_PROCESS_VERSION = "unist_bicodec_qwen3_v2"
REQUIRED_COLUMNS = (
    "id",
    "transcription",
    "translation",
    "source_bicodec",
    "target_bicodec",
    "bicodec_global",
    "dataset_name",
    "src_lang",
    "tgt_lang",
    "split",
)


@dataclass
class PartState:
    input_file: str
    input_size: int
    input_mtime_ns: int
    input_rows: int
    output_rows: int
    manifest: str
    tar_shards: list[str]
    source_audio_bytes: int
    qwen_codec_frames: int
    rank: int
    world_size: int
    process_version: str
    audio_format: str
    max_bicodec_padding_tokens: int
    qwen_tokenizer: str
    bicodec_model: str
    elapsed_seconds: float
    rows_per_second: float
    complete_input: bool
    bicodec_group_scope: str = "batch"
    bicodec_group_rows: int = 64
    qwen_encode_path: str = "wrapper"
    rejected_rows: int = 0
    reject_manifest: str = ""


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    workspace = repo.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "data/unist/raw")
    parser.add_argument(
        "--output-root", type=Path,
        default=workspace / "data/unist/converted/unist_qwen3_s2s_v2")
    parser.add_argument(
        "--spark-repo", type=Path, default=workspace / "projects/Spark-TTS")
    parser.add_argument(
        "--bicodec-model", type=Path,
        default=workspace / "projects/Spark-TTS-0.5B/BiCodec")
    parser.add_argument(
        "--qwen-tokenizer", type=Path,
        default=repo / "models/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--process-version", default=DEFAULT_PROCESS_VERSION)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batch-rows", type=int, default=64)
    parser.add_argument(
        "--max-bicodec-padding-tokens", type=int, default=0,
        help="Split each decode batch so semantic lengths differ by at most this many tokens.")
    parser.add_argument(
        "--bicodec-group-scope", choices=("batch", "block"), default="block",
        help="Group exact-length BiCodec inputs inside each Qwen batch or across a read block.")
    parser.add_argument(
        "--bicodec-group-rows", type=int, default=64,
        help="Maximum same-length source+target sequences per BiCodec forward in block mode.")
    parser.add_argument(
        "--qwen-encode-path", choices=("wrapper", "direct"), default="direct",
        help="Use public tokenizer wrapper or its bit-equivalent GPU-direct model path.")
    parser.add_argument(
        "--max-padded-seconds", type=float, default=512.0,
        help="Maximum batch_size * max(source/target duration) for BiCodec/Qwen3.")
    parser.add_argument("--read-batch-rows", type=int, default=8192)
    parser.add_argument("--audio-format", choices=("flac", "wav"), default="flac")
    parser.add_argument("--max-tar-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: rows per rank.")
    parser.add_argument("--max-files", type=int, default=0, help="Smoke test: files per rank.")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def safe_component(value: str, limit: int = 100) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return (normalized[:limit] or "sample").strip(".") or "sample"


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")
    if args.max_batch_rows <= 0 or args.read_batch_rows <= 0 or args.bicodec_group_rows <= 0:
        raise ValueError("batch row limits must be positive")
    if args.max_bicodec_padding_tokens < 0:
        raise ValueError("max BiCodec padding tokens must be >= 0")
    if args.max_padded_seconds <= 0 or args.max_tar_bytes <= 1024:
        raise ValueError("padded-seconds and tar size limits must be positive")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.process_version):
        raise ValueError(f"unsafe process version: {args.process_version!r}")


def input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found in {input_dir}")
    return files


def assigned_files(args: argparse.Namespace) -> list[Path]:
    files = [p for index, p in enumerate(input_files(args.input_dir))
             if index % args.world_size == args.rank]
    return files[:args.max_files] if args.max_files else files


def output_paths(args: argparse.Namespace, source: Path) -> tuple[Path, Path, Path, Path]:
    key = safe_component(source.stem)
    manifest = args.output_root / "manifests" / f"manifest-{key}.jsonl"
    reject_manifest = args.output_root / "rejects" / f"reject-{key}.jsonl"
    state = args.output_root / "state" / f"{key}.json"
    audio_dir = args.output_root / "speech" / args.process_version / f"rank{args.rank:02d}"
    return manifest, reject_manifest, state, audio_dir


def prepare_output(args: argparse.Namespace) -> None:
    for subdir in ("manifests", "rejects", "state", "speech", "logs"):
        (args.output_root / subdir).mkdir(parents=True, exist_ok=True)


def clean_uncommitted_part(
    manifest: Path, reject_manifest: Path, state: Path, audio_dir: Path, stem: str
) -> None:
    """Remove only converter-owned artifacts that have no committed state."""
    if state.exists():
        return
    candidates = [
        manifest,
        manifest.with_suffix(manifest.suffix + ".partial"),
        reject_manifest,
        reject_manifest.with_suffix(reject_manifest.suffix + ".partial"),
    ]
    candidates.extend(audio_dir.glob(f"{stem}-*.tar"))
    candidates.extend(audio_dir.glob(f"{stem}-*.tar.partial"))
    for path in candidates:
        if path.is_file():
            path.unlink()


def validate_tokens(row: dict[str, Any], source: Path, row_index: int) -> str | None:
    prefix = f"{source.name}:row={row_index}:id={row.get('id')}"
    for name in ("source_bicodec", "target_bicodec"):
        tokens = row.get(name)
        if not tokens:
            if name == "target_bicodec":
                return "empty_target_bicodec"
            raise ValueError(f"{prefix}: empty {name}")
        if name == "target_bicodec" and len(tokens) == 1:
            return "single_frame_target_bicodec"
        if min(tokens) < 0 or max(tokens) >= 8192:
            raise ValueError(f"{prefix}: {name} token outside [0, 8192)")
    global_tokens = row.get("bicodec_global")
    if not global_tokens or len(global_tokens) != GLOBAL_TOKEN_COUNT:
        raise ValueError(f"{prefix}: expected {GLOBAL_TOKEN_COUNT} global tokens")
    if min(global_tokens) < 0 or max(global_tokens) >= GLOBAL_CODEBOOK_SIZE:
        raise ValueError(f"{prefix}: global token outside [0, {GLOBAL_CODEBOOK_SIZE})")
    if not row.get("transcription") or not row.get("translation"):
        raise ValueError(f"{prefix}: empty source/target text")
    return None


def iter_blocks(source: Path, read_batch_rows: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    parquet = pq.ParquetFile(source)
    missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    offset = 0
    for record_batch in parquet.iter_batches(
        batch_size=read_batch_rows, columns=list(REQUIRED_COLUMNS), use_threads=True):
        rows = record_batch.to_pylist()
        for local_index, row in enumerate(rows):
            row["_source_row_index"] = offset + local_index
            row["_reject_reason"] = validate_tokens(row, source, offset + local_index)
        offset += len(rows)
        # Sorting only inside the streaming block bounds memory while reducing
        # padding substantially compared with source order.
        yield offset - len(rows), rows


def split_batches(
    rows: Sequence[dict[str, Any]], max_rows: int, max_padded_seconds: float
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    max_tokens = 0
    for row in rows:
        row_tokens = max(len(row["source_bicodec"]), len(row["target_bicodec"]))
        candidate_max = max(max_tokens, row_tokens)
        padded_seconds = (len(batch) + 1) * candidate_max * (
            SAMPLES_PER_BICODEC_TOKEN / SAMPLE_RATE)
        if batch and (len(batch) >= max_rows or padded_seconds > max_padded_seconds):
            yield batch
            batch = []
            max_tokens = 0
        batch.append(row)
        max_tokens = max(max_tokens, row_tokens)
    if batch:
        yield batch


def load_models(args: argparse.Namespace):
    sys.path.insert(0, str(args.spark_repo.resolve()))
    from sparktts.models.bicodec import BiCodec
    from qwen_tts import Qwen3TTSTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[rank{args.rank}] loading Spark-TTS BiCodec", flush=True)
    bicodec = BiCodec.load_from_checkpoint(args.bicodec_model.resolve()).to(device).eval()
    print(f"[rank{args.rank}] loading Qwen3 tokenizer", flush=True)
    qwen = Qwen3TTSTokenizer.from_pretrained(
        str(args.qwen_tokenizer.resolve()), device_map=str(device))
    actual_qwen_sr = int(qwen.get_input_sample_rate())
    if actual_qwen_sr != QWEN_SAMPLE_RATE:
        raise ValueError(
            f"expected Qwen3 input sample rate {QWEN_SAMPLE_RATE}, got {actual_qwen_sr}")
    return bicodec, qwen, device


@torch.inference_mode()
def decode_bicodec_batch(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    field: str,
    device: torch.device,
    max_padding_tokens: int,
) -> tuple[torch.Tensor, list[int]]:
    lengths = [len(row[field]) for row in rows]
    sample_lengths = [length * SAMPLES_PER_BICODEC_TOKEN for length in lengths]
    result = torch.zeros(
        (len(rows), max(sample_lengths)), dtype=torch.float32, device=device)

    # BiCodec has no padding mask. Very dissimilar semantic lengths in one
    # forward can therefore perturb a shorter waveform near its right boundary.
    # Regroup by the length of the field being decoded, while the downstream
    # Qwen3 tokenizer still receives the original large batch.
    order = sorted(range(len(rows)), key=lambda index: lengths[index])
    groups: list[list[int]] = []
    group: list[int] = []
    group_min = 0
    for index in order:
        if group and lengths[index] - group_min > max_padding_tokens:
            groups.append(group)
            group = []
        if not group:
            group_min = lengths[index]
        group.append(index)
    if group:
        groups.append(group)

    for indices in groups:
        max_length = max(lengths[index] for index in indices)
        semantic = torch.zeros((len(indices), max_length), dtype=torch.long, device=device)
        for local_index, row_index in enumerate(indices):
            semantic[local_index, :lengths[row_index]] = torch.as_tensor(
                rows[row_index][field], dtype=torch.long, device=device)
        global_tokens = torch.as_tensor(
            [rows[index]["bicodec_global"] for index in indices],
            dtype=torch.long, device=device).unsqueeze(1)
        decoded = model.detokenize(semantic, global_tokens).float()
        if decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        if decoded.ndim != 2:
            raise ValueError(f"unexpected BiCodec waveform shape: {tuple(decoded.shape)}")
        for local_index, row_index in enumerate(indices):
            sample_length = sample_lengths[row_index]
            if decoded.shape[1] < sample_length:
                raise ValueError(
                    f"BiCodec returned {decoded.shape[1]} samples, need {sample_length}")
            result[row_index, :sample_length] = decoded[local_index, :sample_length]
    return result, sample_lengths


@torch.inference_mode()
def qwen_encode_batch(qwen: Any, target_wavs_16k: torch.Tensor, lengths_16k: list[int]):
    # Batch resampling stays on GPU.  Passing 24 kHz numpy arrays to Qwen avoids
    # millions of per-sample librosa resampling calls during the full build.
    target_wavs_24k = AF.resample(
        target_wavs_16k, orig_freq=SAMPLE_RATE, new_freq=QWEN_SAMPLE_RATE)
    lengths_24k = [
        int(round(length * QWEN_SAMPLE_RATE / SAMPLE_RATE)) for length in lengths_16k]
    arrays = []
    for index, target_length in enumerate(lengths_24k):
        wav = target_wavs_24k[index, :target_length].clamp(-1, 1)
        arrays.append(wav.detach().cpu().numpy().astype(np.float32, copy=False))
    encoded = qwen.encode(arrays, sr=QWEN_SAMPLE_RATE)
    return normalize_qwen_outputs(encoded.audio_codes, len(arrays))


@torch.inference_mode()
def qwen_encode_batch_direct(
    qwen: Any, target_wavs_16k: torch.Tensor, lengths_16k: list[int]
):
    """Bit-equivalent Qwen3 encode without a GPU→NumPy→CPU-pad→GPU round trip."""
    target_wavs_24k = AF.resample(
        target_wavs_16k, orig_freq=SAMPLE_RATE, new_freq=QWEN_SAMPLE_RATE).clamp(-1, 1)
    lengths_24k = torch.as_tensor(
        [int(round(length * QWEN_SAMPLE_RATE / SAMPLE_RATE)) for length in lengths_16k],
        dtype=torch.long, device=target_wavs_24k.device)
    max_length = int(lengths_24k.max().item())
    values = target_wavs_24k[:, :max_length]
    padding_mask = torch.arange(max_length, device=values.device)[None, :] < lengths_24k[:, None]
    # Resampling a zero-padded batch leaves a short filter tail beyond each
    # waveform's logical end.  The feature extractor used by qwen.encode()
    # slices first and then pads with exact zeros, so clear that tail here as
    # well to keep the direct path token-equivalent to the wrapper path.
    values = values.masked_fill(~padding_mask, 0)
    encoded = qwen.model.encode(
        values.to(dtype=qwen.model.dtype), padding_mask.long(), return_dict=True)
    return normalize_qwen_outputs(encoded.audio_codes, len(lengths_16k))


def normalize_qwen_outputs(audio_codes: Sequence[Any], expected_rows: int):
    outputs = []
    for codes in audio_codes:
        tensor = torch.as_tensor(codes).detach().cpu().long()
        # Qwen commonly returns each sample as [1, 16, T].  Remove only leading
        # singleton wrapper dimensions: an unrestricted squeeze() also removes
        # T when a valid very-short sample has exactly one codec frame.
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1 and tensor.numel() == QWEN_CODEBOOKS:
            tensor = tensor.reshape(QWEN_CODEBOOKS, 1)
        if tensor.ndim != 2:
            raise ValueError(f"Qwen3 codec must be 2-D, got {tuple(tensor.shape)}")
        if tensor.shape[1] == QWEN_CODEBOOKS:
            pass
        elif tensor.shape[0] == QWEN_CODEBOOKS:
            tensor = tensor.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"Qwen3 codec must have {QWEN_CODEBOOKS} groups, got {tuple(tensor.shape)}")
        if tensor.numel() == 0 or tensor.min() < 0 or tensor.max() >= QWEN_CODEBOOK_SIZE:
            raise ValueError("Qwen3 codec is empty or contains an out-of-range token")
        outputs.append(tensor.to(torch.int16).tolist())
    if len(outputs) != expected_rows:
        raise ValueError(f"Qwen3 returned {len(outputs)} rows for batch size {expected_rows}")
    return outputs


@torch.inference_mode()
def decode_bicodec_block_exact(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    max_group_rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Decode source+target after pooling exact lengths across a read block."""
    groups: dict[int, list[tuple[str, int]]] = {}
    for row_index, row in enumerate(rows):
        for field in ("source_bicodec", "target_bicodec"):
            groups.setdefault(len(row[field]), []).append((field, row_index))

    source_outputs: list[np.ndarray | None] = [None] * len(rows)
    target_outputs: list[np.ndarray | None] = [None] * len(rows)
    for token_length in sorted(groups):
        items = groups[token_length]
        for start in range(0, len(items), max_group_rows):
            chunk = items[start:start + max_group_rows]
            semantic = torch.as_tensor(
                [rows[index][field] for field, index in chunk],
                dtype=torch.long, device=device)
            global_tokens = torch.as_tensor(
                [rows[index]["bicodec_global"] for _, index in chunk],
                dtype=torch.long, device=device).unsqueeze(1)
            decoded = model.detokenize(semantic, global_tokens).float()
            if decoded.ndim == 3 and decoded.shape[1] == 1:
                decoded = decoded[:, 0]
            expected_samples = token_length * SAMPLES_PER_BICODEC_TOKEN
            if decoded.ndim != 2 or decoded.shape[1] < expected_samples:
                raise ValueError(
                    f"unexpected block BiCodec shape {tuple(decoded.shape)} for {token_length} tokens")
            decoded_np = decoded[:, :expected_samples].cpu().numpy()
            for local_index, (field, row_index) in enumerate(chunk):
                waveform = decoded_np[local_index]
                if field == "source_bicodec":
                    source_outputs[row_index] = waveform
                else:
                    target_outputs[row_index] = waveform
    if any(waveform is None for waveform in source_outputs + target_outputs):
        raise RuntimeError("block BiCodec decode left an empty waveform slot")
    return source_outputs, target_outputs  # type: ignore[return-value]


def pad_numpy_waveforms(waveforms: Sequence[np.ndarray], device: torch.device) -> tuple[torch.Tensor, list[int]]:
    lengths = [int(waveform.shape[0]) for waveform in waveforms]
    padded = np.zeros((len(waveforms), max(lengths)), dtype=np.float32)
    for index, waveform in enumerate(waveforms):
        padded[index, :lengths[index]] = waveform
    return torch.from_numpy(padded).to(device=device), lengths


def encode_audio_bytes(waveform: np.ndarray, audio_format: str) -> bytes:
    buffer = io.BytesIO()
    waveform = np.clip(waveform, -1.0, 1.0)
    if audio_format == "flac":
        sf.write(buffer, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    else:
        sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class IndexedTarWriter:
    """Write complete audio payloads into transactional tar shards."""

    def __init__(
        self, audio_dir: Path, stem: str, process_version: str, rank: int,
        max_tar_bytes: int, audio_format: str
    ) -> None:
        self.audio_dir = audio_dir
        self.stem = stem
        self.process_version = process_version
        self.rank = rank
        self.max_tar_bytes = max_tar_bytes
        self.audio_format = audio_format
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.index = -1
        self.tar: tarfile.TarFile | None = None
        self.partial_paths: list[Path] = []
        self.final_paths: list[Path] = []
        self.members = 0
        self.payload_bytes = 0

    def _open(self) -> None:
        self.index += 1
        final = self.audio_dir / f"{self.stem}-{self.index:05d}.tar"
        partial = final.with_suffix(final.suffix + ".partial")
        if final.exists() or partial.exists():
            raise FileExistsError(f"uncommitted tar path already exists: {final}")
        self.tar = tarfile.open(partial, mode="w")
        self.partial_paths.append(partial)
        self.final_paths.append(final)

    def add(self, member_name: str, payload: bytes) -> str:
        estimated = 512 + math.ceil(len(payload) / 512) * 512
        if self.tar is None:
            self._open()
        assert self.tar is not None
        if self.tar.offset > 0 and self.tar.offset + estimated > self.max_tar_bytes:
            self.tar.close()
            self.tar = None
            self._open()
        assert self.tar is not None
        data_offset = self.tar.offset + 512
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        self.tar.addfile(info, io.BytesIO(payload))
        self.members += 1
        self.payload_bytes += len(payload)
        shard_name = self.final_paths[-1].name
        return (
            "wds://unist/audio/processed/shards/"
            f"dataset=unist/process_version={self.process_version}/rank{self.rank:02d}/"
            f"{shard_name}?offset={data_offset}&length={len(payload)}&format={self.audio_format}"
        )

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def commit(self) -> None:
        self.close()
        for partial, final in zip(self.partial_paths, self.final_paths):
            os.replace(partial, final)


def normalize_lang(value: str) -> str:
    value = (value or "").strip().lower().replace("_", "-")
    if value in {"eng", "english"} or value.startswith("en-"):
        return "en"
    if value in {"cmn", "zho", "chinese"} or value.startswith("zh-"):
        return "zh"
    if value in {"en", "zh"}:
        return value
    raise ValueError(f"unsupported UniST language: {value!r}")


def task_prompt(target_lang: str) -> str:
    language = {"en": "English", "zh": "Chinese"}[target_lang]
    return (
        "Task: Listen to the input speech, translate it, and synthesize "
        f"the translation as {language} speech.")


def manifest_row(
    row: dict[str, Any], source_file: Path, source_uri: str,
    source_audio_bytes: int, target_codes: list[list[int]],
) -> dict[str, Any]:
    src_lang = normalize_lang(row["src_lang"])
    tgt_lang = normalize_lang(row["tgt_lang"])
    source_text = str(row["transcription"]).strip()
    target_text = str(row["translation"]).strip()
    record_id = ":".join((
        "unist", safe_component(row["dataset_name"]), safe_component(row["id"]),
        f"{src_lang}_{tgt_lang}", safe_component(row["split"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "id": row["id"],
        "task": TASK,
        "task_prompt": "",
        "messages": [
            {"role": "system", "content": task_prompt(tgt_lang)},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": target_text},
        ],
        "audios": [source_uri],
        "source_duration_ms": int(round(
            len(row["source_bicodec"]) * SAMPLES_PER_BICODEC_TOKEN * 1000 / SAMPLE_RATE)),
        "target_text": target_text,
        "target_duration_ms": int(round(
            len(row["target_bicodec"]) * SAMPLES_PER_BICODEC_TOKEN * 1000 / SAMPLE_RATE)),
        "target_codec_codes": target_codes,
        "codec_len": len(target_codes),
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_transcript": source_text,
        "dataset_name": row["dataset_name"],
        "split": row["split"],
        "source_audio_encoded_bytes": source_audio_bytes,
        "provenance": {
            "source_parquet": source_file.name,
            "source_row_index": row["_source_row_index"],
            "source_bicodec_len": len(row["source_bicodec"]),
            "target_bicodec_len": len(row["target_bicodec"]),
            "bicodec_global_len": len(row["bicodec_global"]),
        },
    }


def reject_row(row: dict[str, Any], source_file: Path, reason: str) -> dict[str, Any]:
    """Return an auditable row-level rejection without copying large token lists."""
    return {
        "schema_version": REJECT_SCHEMA_VERSION,
        "reason": reason,
        "id": row.get("id"),
        "dataset_name": row.get("dataset_name"),
        "split": row.get("split"),
        "source_lang": normalize_lang(row.get("src_lang")),
        "target_lang": normalize_lang(row.get("tgt_lang")),
        "source_text": str(row.get("transcription") or "").strip(),
        "target_text": str(row.get("translation") or "").strip(),
        "source_parquet": source_file.name,
        "source_parquet_path": str(source_file.resolve()),
        "source_row_index": int(row["_source_row_index"]),
        "source_bicodec_len": len(row.get("source_bicodec") or []),
        "target_bicodec_len": len(row.get("target_bicodec") or []),
        "bicodec_global_len": len(row.get("bicodec_global") or []),
    }


def load_committed_state(
    state_path: Path, source: Path, args: argparse.Namespace | None = None
) -> PartState | None:
    if not state_path.is_file():
        return None
    state = PartState(**json.loads(state_path.read_text(encoding="utf-8")))
    stat = source.stat()
    if state.input_size != stat.st_size or state.input_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError(f"input changed after conversion: {source}")
    if args is not None:
        expected = {
            "process_version": args.process_version,
            "audio_format": args.audio_format,
            "max_bicodec_padding_tokens": args.max_bicodec_padding_tokens,
            "bicodec_group_scope": args.bicodec_group_scope,
            "bicodec_group_rows": args.bicodec_group_rows,
            "qwen_encode_path": args.qwen_encode_path,
            "qwen_tokenizer": str(args.qwen_tokenizer.resolve()),
            "bicodec_model": str(args.bicodec_model.resolve()),
        }
        for field, value in expected.items():
            if getattr(state, field) != value:
                raise RuntimeError(
                    f"conversion setting changed for {source}: "
                    f"{field}={getattr(state, field)!r}, expected {value!r}")
    manifest = Path(state.manifest)
    tar_paths = [Path(path) for path in state.tar_shards]
    if not manifest.is_file() or not all(path.is_file() for path in tar_paths):
        raise RuntimeError(f"committed state has missing outputs: {state_path}")
    if state.reject_manifest and not Path(state.reject_manifest).is_file():
        raise RuntimeError(f"committed state has missing reject manifest: {state_path}")
    return state


def process_file(args: argparse.Namespace, source: Path, models) -> PartState:
    bicodec, qwen, device = models
    manifest, reject_manifest, state_path, audio_dir = output_paths(args, source)
    committed = load_committed_state(state_path, source, args)
    if committed is not None:
        print(f"[rank{args.rank}] SKIP {source.name}: {committed.output_rows} rows", flush=True)
        return committed

    stem = safe_component(source.stem)
    clean_uncommitted_part(manifest, reject_manifest, state_path, audio_dir, stem)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    reject_manifest.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_partial = manifest.with_suffix(manifest.suffix + ".partial")
    reject_partial = reject_manifest.with_suffix(reject_manifest.suffix + ".partial")
    tar_writer = IndexedTarWriter(
        audio_dir, stem, args.process_version, args.rank,
        args.max_tar_bytes, args.audio_format)
    input_rows = pq.ParquetFile(source).metadata.num_rows
    target_rows = min(input_rows, args.limit) if args.limit else input_rows
    started = time.time()
    processed_rows = 0
    output_rows = 0
    rejected_rows = 0
    qwen_frames = 0

    print(f"[rank{args.rank}] START {source.name}: rows={target_rows}", flush=True)
    try:
        with (
            manifest_partial.open("wb", buffering=16 * 1024 * 1024) as manifest_fh,
            reject_partial.open("wb", buffering=4 * 1024 * 1024) as reject_fh,
        ):
            for _, block in iter_blocks(source, args.read_batch_rows):
                if args.limit:
                    remaining = target_rows - processed_rows
                    if remaining <= 0:
                        break
                    block = block[:remaining]
                processed_rows += len(block)

                valid_block = []
                for row in block:
                    reason = row.get("_reject_reason")
                    if reason:
                        reject_fh.write(orjson.dumps(
                            reject_row(row, source, str(reason)),
                            option=orjson.OPT_APPEND_NEWLINE))
                        rejected_rows += 1
                    else:
                        valid_block.append(row)
                block = valid_block
                if not block:
                    continue

                # Target length is the primary key because Qwen3 is the expensive
                # batched stage. Source decode is regrouped by its own length.
                block.sort(key=lambda row: (
                    len(row["target_bicodec"]), len(row["source_bicodec"])))
                for block_index, row in enumerate(block):
                    row["_block_index"] = block_index
                block_source_wavs = block_target_wavs = None
                if args.bicodec_group_scope == "block":
                    block_source_wavs, block_target_wavs = decode_bicodec_block_exact(
                        bicodec, block, device, args.bicodec_group_rows)
                for batch in split_batches(
                    block, args.max_batch_rows, args.max_padded_seconds):
                    if args.bicodec_group_scope == "block":
                        assert block_source_wavs is not None and block_target_wavs is not None
                        indices = [int(row["_block_index"]) for row in batch]
                        source_arrays = [block_source_wavs[index] for index in indices]
                        target_arrays = [block_target_wavs[index] for index in indices]
                        target_wavs, target_lengths = pad_numpy_waveforms(target_arrays, device)
                    else:
                        source_wavs, source_lengths = decode_bicodec_batch(
                            bicodec, batch, "source_bicodec", device,
                            args.max_bicodec_padding_tokens)
                        target_wavs, target_lengths = decode_bicodec_batch(
                            bicodec, batch, "target_bicodec", device,
                            args.max_bicodec_padding_tokens)
                        source_arrays = None
                    if args.qwen_encode_path == "direct":
                        target_codes = qwen_encode_batch_direct(qwen, target_wavs, target_lengths)
                    else:
                        target_codes = qwen_encode_batch(qwen, target_wavs, target_lengths)

                    for index, row in enumerate(batch):
                        if source_arrays is not None:
                            source_np = source_arrays[index]
                        else:
                            source_wav = source_wavs[index, :source_lengths[index]].clamp(-1, 1)
                            source_np = source_wav.detach().cpu().numpy().astype(np.float32, copy=False)
                        payload = encode_audio_bytes(source_np, args.audio_format)
                        member_name = (
                            f"{int(row['_source_row_index']):08d}-"
                            f"{safe_component(row['id'], 70)}.{args.audio_format}")
                        source_uri = tar_writer.add(member_name, payload)
                        record = manifest_row(
                            row, source, source_uri, len(payload), target_codes[index])
                        manifest_fh.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                        output_rows += 1
                        qwen_frames += len(target_codes[index])

                    if output_rows % args.progress_every < len(batch):
                        elapsed = time.time() - started
                        print(
                            f"[rank{args.rank}] {source.name} input={processed_rows}/{target_rows} "
                            f"output={output_rows} rejected={rejected_rows} "
                            f"rate={output_rows/max(elapsed, 1):.2f}/s "
                            f"tar_payload={tar_writer.payload_bytes/2**30:.2f}GiB",
                            flush=True)
                if processed_rows >= target_rows:
                    break
        if (processed_rows != target_rows or output_rows + rejected_rows != processed_rows
                or tar_writer.members != output_rows):
            raise RuntimeError(
                f"row invariant failed: expected={target_rows} processed={processed_rows} "
                f"output={output_rows} rejected={rejected_rows} "
                f"audio_members={tar_writer.members}")
        tar_writer.close()
        tar_writer.commit()
        os.replace(manifest_partial, manifest)
        os.replace(reject_partial, reject_manifest)

        elapsed = time.time() - started
        source_stat = source.stat()
        state = PartState(
            input_file=str(source.resolve()),
            input_size=source_stat.st_size,
            input_mtime_ns=source_stat.st_mtime_ns,
            input_rows=input_rows,
            output_rows=output_rows,
            manifest=str(manifest.resolve()),
            tar_shards=[str(path.resolve()) for path in tar_writer.final_paths],
            source_audio_bytes=tar_writer.payload_bytes,
            qwen_codec_frames=qwen_frames,
            rank=args.rank,
            world_size=args.world_size,
            process_version=args.process_version,
            audio_format=args.audio_format,
            max_bicodec_padding_tokens=args.max_bicodec_padding_tokens,
            bicodec_group_scope=args.bicodec_group_scope,
            bicodec_group_rows=args.bicodec_group_rows,
            qwen_encode_path=args.qwen_encode_path,
            qwen_tokenizer=str(args.qwen_tokenizer.resolve()),
            bicodec_model=str(args.bicodec_model.resolve()),
            elapsed_seconds=elapsed,
            rows_per_second=output_rows / max(elapsed, 1e-6),
            complete_input=(processed_rows == input_rows),
            rejected_rows=rejected_rows,
            reject_manifest=str(reject_manifest.resolve()),
        )
        state_partial = state_path.with_suffix(".json.partial")
        state_partial.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(state_partial, state_path)
        print(
            f"[rank{args.rank}] DONE {source.name}: input={processed_rows} "
            f"output={output_rows} rejected={rejected_rows} "
            f"elapsed={elapsed:.1f}s rate={state.rows_per_second:.2f}/s",
            flush=True)
        return state
    except BaseException:
        tar_writer.close()
        raise


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    files = input_files(args.input_dir)
    states: list[PartState] = []
    missing = []
    for source in files:
        # State names do not depend on rank.
        state_path = args.output_root / "state" / f"{safe_component(source.stem)}.json"
        state = load_committed_state(state_path, source, args)
        if state is None or not state.complete_input:
            missing.append(source.name)
        else:
            states.append(state)
    if missing:
        raise RuntimeError(
            f"cannot finalize: {len(missing)} input files incomplete; first={missing[:10]}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": args.output_root.name,
        "process_version": args.process_version,
        "created_at_unix": time.time(),
        "input_dir": str(args.input_dir.resolve()),
        "input_files": len(files),
        "input_rows": sum(state.input_rows for state in states),
        "rows": sum(state.output_rows for state in states),
        "rejected_rows": sum(state.rejected_rows for state in states),
        "manifest_shards": len(states),
        "reject_manifest_shards": sum(bool(state.reject_manifest) for state in states),
        "tar_shards": sum(len(state.tar_shards) for state in states),
        "source_audio_bytes": sum(state.source_audio_bytes for state in states),
        "qwen_codec_frames": sum(state.qwen_codec_frames for state in states),
        "wds_audio_root": str((args.output_root / "speech").resolve()),
        "manifest_glob": str((args.output_root / "manifests/manifest-*.jsonl").resolve()),
        "reject_manifest_glob": str(
            (args.output_root / "rejects/reject-*.jsonl").resolve()),
        "qwen_tokenizer": str(args.qwen_tokenizer.resolve()),
        "bicodec_model": str(args.bicodec_model.resolve()),
        "license": "CC-BY-NC-4.0",
    }
    summary_path = args.output_root / "_SUMMARY.json"
    committed_path = args.output_root / "_COMMITTED.json"
    summary_partial = summary_path.with_suffix(".json.partial")
    summary_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(summary_partial, summary_path)
    committed_partial = committed_path.with_suffix(".json.partial")
    committed_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(committed_partial, committed_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output(args)
    if args.finalize_only:
        finalize(args)
        return
    files = assigned_files(args)
    if not files:
        print(f"[rank{args.rank}] no assigned files", flush=True)
        return
    models = load_models(args)
    total = 0
    for source in files:
        state = process_file(args, source, models)
        total += state.output_rows
        if args.limit:
            break
    print(f"[rank{args.rank}] COMPLETE assigned_rows={total}", flush=True)


if __name__ == "__main__":
    main()
