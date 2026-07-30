"""Bounded audio I/O and session artifact helpers for the streaming demo."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm"}


class AudioValidationError(ValueError):
    """Raised when an input is not a supported, finite and bounded audio stream."""


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
        raise AudioValidationError("This audio format needs ffmpeg or imageio-ffmpeg")
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
        subprocess.run(command, check=True, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AudioValidationError(f"ffmpeg could not decode the audio: {exc}") from exc


def decode_audio(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        return np.asarray(audio.mean(axis=1), dtype=np.float32), int(sample_rate)
    except (RuntimeError, sf.LibsndfileError):
        temporary = source.with_name(f".{source.stem}.{uuid.uuid4().hex}.decoded.wav")
        try:
            _decode_with_ffmpeg(source, temporary)
            audio, sample_rate = sf.read(temporary, dtype="float32", always_2d=True)
            return np.asarray(audio.mean(axis=1), dtype=np.float32), int(sample_rate)
        finally:
            temporary.unlink(missing_ok=True)


def resample_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1 if values.shape[1] <= 8 else 0)
    values = values.reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise AudioValidationError("Audio is empty or contains non-finite samples")
    if sample_rate <= 0:
        raise AudioValidationError("Audio sample rate must be positive")
    if sample_rate != SAMPLE_RATE:
        values = librosa.resample(values, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(values)))
    if peak > 1.0:
        values = values / peak
    return values


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
    raw, sample_rate = decode_audio(source_path)
    waveform = resample_mono(raw, sample_rate)
    duration = waveform.size / SAMPLE_RATE
    if duration < min_audio_seconds or duration > max_audio_seconds:
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s is outside "
            f"[{min_audio_seconds}, {max_audio_seconds}]s"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination_path, waveform, SAMPLE_RATE, subtype="PCM_16")
    return {
        "source_name": source_path.name,
        "source_suffix": suffix,
        "source_bytes": size,
        "sample_rate": SAMPLE_RATE,
        "samples": int(waveform.size),
        "duration_seconds": float(duration),
        "peak": float(np.max(np.abs(waveform))),
    }


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def write_aligned_stereo(
    source: np.ndarray,
    translation: np.ndarray,
    destination: str | Path,
    *,
    translation_offset_ms: float,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    if translation_offset_ms < 0:
        raise ValueError("translation_offset_ms cannot be negative")
    left = np.asarray(source, dtype=np.float32).reshape(-1)
    right_audio = np.asarray(translation, dtype=np.float32).reshape(-1)
    offset = int(round(translation_offset_ms * sample_rate / 1000.0))
    total = max(len(left), offset + len(right_audio))
    stereo = np.zeros((total, 2), dtype=np.float32)
    stereo[: len(left), 0] = left
    stereo[offset : offset + len(right_audio), 1] = right_audio
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, sample_rate, subtype="PCM_16")
    return path


def concatenate_audio(chunks: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks if len(chunk)]
    return np.concatenate(valid) if valid else np.zeros(0, dtype=np.float32)
