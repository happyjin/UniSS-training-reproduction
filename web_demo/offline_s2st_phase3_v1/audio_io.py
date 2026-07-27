"""Audio validation, normalization, silence-based chunking, and stitching."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm"}


class AudioValidationError(ValueError):
    """Raised when a browser upload is not a supported bounded audio file."""


def _ffmpeg_executable() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _decode_with_ffmpeg(source: Path, destination: Path) -> None:
    executable = _ffmpeg_executable()
    if executable is None:
        raise AudioValidationError(
            "This audio format needs ffmpeg; install imageio-ffmpeg in the isolated demo environment"
        )
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AudioValidationError(
            f"ffmpeg could not decode the uploaded audio: {exc}"
        ) from exc


def decode_audio(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        return np.asarray(mono, dtype=np.float32), int(sample_rate)
    except (RuntimeError, sf.LibsndfileError):
        temporary = source.with_name(f".{source.stem}.{uuid.uuid4().hex}.decoded.wav")
        try:
            _decode_with_ffmpeg(source, temporary)
            audio, sample_rate = sf.read(temporary, dtype="float32", always_2d=True)
            return np.asarray(audio.mean(axis=1), dtype=np.float32), int(sample_rate)
        finally:
            temporary.unlink(missing_ok=True)


def normalize_uploaded_audio(
    source: str | Path,
    destination: str | Path,
    *,
    max_upload_bytes: int,
    min_audio_seconds: float,
    max_audio_seconds: float,
) -> dict[str, object]:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise AudioValidationError("Audio upload does not exist")
    suffix = source_path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise AudioValidationError(f"Unsupported audio extension: {suffix or '<none>'}")
    size = source_path.stat().st_size
    if size <= 0 or size > max_upload_bytes:
        raise AudioValidationError(
            f"Audio file size {size} bytes is outside (0, {max_upload_bytes}]"
        )
    waveform, sample_rate = decode_audio(source_path)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise AudioValidationError(
            "Decoded audio is empty or contains non-finite samples"
        )
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    duration = waveform.size / SAMPLE_RATE
    if duration < min_audio_seconds or duration > max_audio_seconds:
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s is outside [{min_audio_seconds}, {max_audio_seconds}]s"
        )
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / peak
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination_path, waveform, SAMPLE_RATE, subtype="PCM_16")
    return {
        "source_name": source_path.name,
        "source_suffix": suffix,
        "source_bytes": size,
        "sample_rate": SAMPLE_RATE,
        "samples": int(waveform.size),
        "duration_seconds": float(duration),
        "peak": peak,
    }


def _fixed_windows(start: int, end: int, maximum: int) -> list[tuple[int, int]]:
    windows = []
    cursor = start
    while cursor < end:
        next_end = min(end, cursor + maximum)
        windows.append((cursor, next_end))
        cursor = next_end
    return windows


def split_on_silence(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_chunk_seconds: float = 30.0,
    top_db: float = 35.0,
) -> list[np.ndarray]:
    """Create ordered offline chunks without introducing an external ASR/VAD model."""

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []
    maximum = max(1, round(max_chunk_seconds * sample_rate))
    if values.size <= maximum:
        return [values]
    intervals = librosa.effects.split(
        values, top_db=top_db, frame_length=1024, hop_length=256
    )
    if len(intervals) == 0:
        return [
            values[start:end] for start, end in _fixed_windows(0, values.size, maximum)
        ]
    padding = round(0.08 * sample_rate)
    merged: list[tuple[int, int]] = []
    for raw_start, raw_end in intervals:
        start = max(0, int(raw_start) - padding)
        end = min(values.size, int(raw_end) + padding)
        if merged and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.extend(_fixed_windows(start, end, maximum))
    chunks = [values[start:end] for start, end in merged if end > start]
    return chunks or [
        values[start:end] for start, end in _fixed_windows(0, values.size, maximum)
    ]


def stitch_audio(
    chunks: Sequence[np.ndarray],
    *,
    sample_rate: int = SAMPLE_RATE,
    silence_seconds: float = 0.12,
) -> np.ndarray:
    valid = [
        np.asarray(chunk, dtype=np.float32).reshape(-1)
        for chunk in chunks
        if len(chunk)
    ]
    if not valid:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(max(0, round(silence_seconds * sample_rate)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, chunk in enumerate(valid):
        if index and silence.size:
            pieces.append(silence)
        pieces.append(chunk)
    return np.concatenate(pieces)


def create_request_directory(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    request = root / time.strftime("%Y%m%d") / uuid.uuid4().hex
    request.mkdir(parents=True, exist_ok=False)
    return request


def cleanup_expired(output_root: str | Path, ttl_hours: float) -> int:
    root = Path(output_root)
    if not root.is_dir() or ttl_hours <= 0:
        return 0
    cutoff = time.time() - ttl_hours * 3600
    removed = 0
    for day in root.iterdir():
        if not day.is_dir():
            continue
        for request in day.iterdir():
            if request.is_dir() and request.stat().st_mtime < cutoff:
                shutil.rmtree(request)
                removed += 1
        if day.is_dir() and not any(day.iterdir()):
            day.rmdir()
    return removed


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, destination)
