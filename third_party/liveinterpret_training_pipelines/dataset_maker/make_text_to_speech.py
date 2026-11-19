#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert local TTS parquet shards into ms-swift TTS format.

Input  : /path/to/project/dataset/tts/dataset=*/examples.parquet
Output : text_to_speech_v2.jsonl

This v2 converter consumes all ``examples.parquet`` shards under each
``dataset=<name>/`` subdirectory.  A row is written when its text and audio
are available.  The TTS sample uses ``text`` (or ``transcription``) as input
text and ``audio_uri`` as output speech.

Output schema is identical to ``make_text_to_speech.py``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from collections import Counter

import shardio
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

# ISO 639-1/639-3 / common BCP-47 short codes -> English language name
# Covers all languages present in the dataset/tts curated datasets.

_AUDIO_RATE_CACHE: dict[str, Optional[int]] = {}

# Audio duration cache (seconds); populated lazily by audio_duration_sec().
_AUDIO_DURATION_CACHE: dict[str, Optional[float]] = {}

# FSx existence-check cache: avoid repeated os.path.isfile() round-trips to the
# network filesystem for paths we have already stat'd.
_FSX_EXISTS_CACHE: dict[str, bool] = {}

# Runtime verification: log the first few s3:// -> /mnt/fsx conversions so the
# S3 path format / mapping assumption can be eyeballed in the worker logs.
_FSX_SAMPLE_LIMIT = 5
_FSX_SAMPLE_LOGGED = 0

# Output directory for the generated jsonl manifests. Mirrors the worker's
# ${BASE_PATH}/datasets/jsonl/ so that the argparse default never NameErrors
# (the worker still passes --output explicitly).
DEFAULT_BASE_PATH = os.environ.get(
    "BASE_PATH",
    "/local/speech-language-model/liveinterpret_training_pipelines",
)
DEFAULT_JSONL_DIR = os.environ.get(
    "DEFAULT_JSONL_DIR",
    os.path.join(DEFAULT_BASE_PATH, "datasets", "jsonl"),
)

# Mount point where the S3 bucket is mirrored on each worker (HyperPod FSx).
DEFAULT_FSX_MOUNT = "/mnt/fsx"


def _detect_cpu_count() -> int:
    """Return usable CPU count, respecting cgroup limits (container-aware).

    Tries cgroup v2 first (``cpu.max``), then cgroup v1
    (``cpu.cfs_quota_us/cpu.cfs_period_us``), falls back to
    ``os.cpu_count()``.
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            parts = fh.read().strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, (quota + period - 1) // period)
    except (OSError, ValueError, IndexError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh_q:
            quota = int(fh_q.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh_p:
            period = int(fh_p.read().strip())
        if quota > 0 and period > 0:
            return max(1, (quota + period - 1) // period)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _log(msg: str) -> None:
    """Timestamped log to stderr (flushed immediately for real-time visibility)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _hms(seconds: float) -> str:
    """Format seconds as HhMmSs or MmSs."""
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


LANG_CODE_TO_NAME = {
    # East Asian
    "zh": "Chinese",
    "yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    # South Asian
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "si": "Sinhala",
    "ks": "Kashmiri",
    # Southeast Asian
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "jv": "Javanese",
    "su": "Sundanese",
    "km": "Khmer",
    "my": "Burmese",
    "iba": "Iban",
    # European
    "en": "English",
    "en-gb": "British English",
    "en-us": "American English",
    "en-ng": "Nigerian English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "es-ar": "Argentinian Spanish",
    "es-cl": "Chilean Spanish",
    "es-co": "Colombian Spanish",
    "es-pe": "Peruvian Spanish",
    "es-pr": "Puerto Rican Spanish",
    "es-ve": "Venezuelan Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "tr": "Turkish",
    "is": "Icelandic",
    "fo": "Faroese",
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    "hy": "Armenian",
    "kk": "Kazakh",
    # Middle East / North Africa
    "ar": "Arabic",
    "ar-tn": "Tunisian Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    # Sub-Saharan Africa
    "af": "Afrikaans",
    "am": "Amharic",
    "ee": "Ewe",
    "ha": "Hausa",
    "ln": "Lingala",
    "om": "Oromo",
    "sw": "Swahili",
    "tw": "Twi",
    "wo": "Wolof",
    "yo": "Yoruba",
    "pcm": "Nigerian Pidgin",
    # Central Asian / Sino-Tibetan
    "bo": "Tibetan",
    "ug": "Uyghur",
    # Indigenous American
    "azz": "Highland Puebla Nahuatl",
    "xty": "Yoloxochitl Mixtec",
    # Other
    "mul": "Multilingual",
}

def lang_code_to_name(code: str) -> str:
    """Map a short language code to its English name. Falls back to capitalized code."""
    if not code:
        return "Unknown"
    code_norm = code.strip().lower()
    if code_norm in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[code_norm]
    primary = code_norm.split("-")[0]
    if primary in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[primary]
    return code.capitalize()


def clean_str(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def natural_key(path: str) -> list[object]:
    """Sort key based on parent directory name (the ``dataset=...`` folder)."""
    parent = os.path.basename(os.path.dirname(path))
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", parent)]


def find_input_parquets(input_dir: str) -> list[str]:
    """Find all ``examples.parquet`` files under ``dataset=*/`` subdirectories."""
    files = []
    pattern = os.path.join(input_dir, "dataset=*", "examples.parquet")
    for path in glob.glob(pattern):
        if os.path.getsize(path) > 0:
            files.append(path)
    return sorted(files, key=natural_key)


def basename_from_uri(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    path = path.split("#", 1)[0].split("?", 1)[0]
    return os.path.basename(path)


def resolve_local_audio(audio_uri: str, audio_cache_dir: str) -> Optional[str]:
    """Try to map an S3 / remote audio URI to a local file on disk.

    Resolution order:
    1. If ``audio_uri`` is already a local file path, return it directly.
    2. Look for the basename in ``audio_cache_dir``.
    3. Return None (use the URI as-is downstream).
    """
    uri = clean_str(audio_uri)
    if not uri:
        return None
    if os.path.isfile(uri):
        return os.path.abspath(uri)
    basename = basename_from_uri(uri)
    if not basename:
        return None
    local_path = os.path.join(audio_cache_dir, basename)
    if os.path.isfile(local_path):
        return os.path.abspath(local_path)
    return None


def parse_s3_uri(uri: str) -> Optional[tuple[str, str]]:
    """Return ``(bucket, key)`` for a valid ``s3://bucket/key`` URI, else None."""
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path:
        return None
    return parsed.netloc, parsed.path.lstrip("/")


def s3_to_fsx(uri: str, fsx_mount: str) -> Optional[str]:
    """Map ``s3://<bucket>/<key>`` -> ``<fsx_mount>/<key>`` (bucket-insensitive).

    The S3 bucket is mirror-mounted at ``fsx_mount`` on each worker, so the
    object key (e.g. ``data/curated/...``) lives directly under it. Returns
    None when ``uri`` is not a recognizable ``s3://`` URI.
    """
    parsed = parse_s3_uri(uri)
    if parsed is None:
        return None
    _bucket, key = parsed
    if not key:
        return None
    return os.path.join(fsx_mount, key)


def fsx_file_exists(path: str) -> bool:
    """``os.path.isfile`` with a dedup cache to limit FSx network round-trips."""
    cached = _FSX_EXISTS_CACHE.get(path)
    if cached is None:
        cached = os.path.isfile(path)
        _FSX_EXISTS_CACHE[path] = cached
    return cached


def _maybe_log_fsx_sample(s3_uri: str, fsx_path: str) -> None:
    """Log the first few s3:// -> FSx conversions for runtime verification."""
    global _FSX_SAMPLE_LOGGED
    if _FSX_SAMPLE_LOGGED < _FSX_SAMPLE_LIMIT:
        _FSX_SAMPLE_LOGGED += 1
        _log(
            f"  [s3->fsx sample {_FSX_SAMPLE_LOGGED}/{_FSX_SAMPLE_LIMIT}] "
            f"{s3_uri} -> {fsx_path} (exists={fsx_file_exists(fsx_path)})"
        )


def audio_sample_rate(path: str) -> Optional[int]:
    """Read the sample rate from a local audio file (wave/flac)."""
    if path in _AUDIO_RATE_CACHE:
        return _AUDIO_RATE_CACHE[path]
    sample_rate = None
    try:
        with wave.open(path, "rb") as wav:
            sample_rate = int(wav.getframerate())
    except Exception:
        try:
            import soundfile as sf  # type: ignore
            sample_rate = int(sf.info(path).samplerate)
        except Exception:
            sample_rate = None
    _AUDIO_RATE_CACHE[path] = sample_rate
    return sample_rate


def has_sample_rate(path: str, expected_rate: int) -> bool:
    return audio_sample_rate(path) == expected_rate


def _ffmpeg_exe() -> Optional[str]:
    """Locate an ffmpeg binary: system PATH first, then imageio-ffmpeg.

    The slim worker image's libsndfile has no FLAC/mp3/ogg support, so we use
    ffmpeg (confirmed available in the runtime) as a format-agnostic fallback
    for reading audio duration.
    """
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def audio_duration_sec_ffmpeg(path: str) -> Optional[float]:
    """Read duration via ffmpeg; supports FLAC/mp3/ogg/opus etc.

    Parses the ``Duration: HH:MM:SS`` line from ffmpeg's stderr.  Returns
    ``None`` when ffmpeg is unavailable or the duration cannot be parsed.
    """
    exe = _ffmpeg_exe()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
        if not m:
            return None
        h, mi, s = m.groups()
        return float(h) * 3600 + float(mi) * 60 + float(s)
    except Exception:
        return None


def audio_duration_sec(path: str) -> Optional[float]:
    """Return audio duration in seconds from the file header (wave/soundfile).

    Only the file header is read (no sample data), so this is cheap even over a
    network filesystem.  Returns ``None`` when the duration cannot be determined.
    Results are cached per path.
    """
    if path in _AUDIO_DURATION_CACHE:
        return _AUDIO_DURATION_CACHE[path]
    duration: Optional[float] = None
    try:
        with wave.open(path, "rb") as wav:
            nframes = wav.getnframes()
            rate = wav.getframerate()
            duration = float(nframes / rate) if rate and rate > 0 else 0.0
    except Exception:
        try:
            import soundfile as sf  # type: ignore

            info = sf.info(path)
            duration = float(info.frames / info.samplerate) if info.samplerate > 0 else 0.0
        except Exception:
            try:
                duration = audio_duration_sec_ffmpeg(path)
            except Exception:
                duration = None
    _AUDIO_DURATION_CACHE[path] = duration
    return duration


def exceeds_max_duration(audio_paths, max_audio_duration_sec: float) -> bool:
    """True if any of ``audio_paths`` is longer than ``max_audio_duration_sec``.

    Deprecated in favour of :func:`duration_filter_reason` which also enforces
    the hard "known duration / parseable audio" rule.  Kept for call-site
    compatibility.
    """
    if not max_audio_duration_sec or max_audio_duration_sec <= 0:
        return False
    for p in audio_paths:
        if not p:
            continue
        dur = audio_duration_sec(p)
        if dur is not None and dur > max_audio_duration_sec:
            return True
    return False


def duration_filter_reason(
    audio_paths: list,
    record: dict,
    max_audio_duration_sec: float,
    drop_unknown_duration: bool,
) -> Optional[str]:
    """Hard duration/parseability gate.

    Returns ``None`` if the sample passes, otherwise a skip reason:

    * ``"audio_too_long"``          — duration (known) exceeds the limit.
    * ``"audio_duration_unknown"``  — duration could not be determined at all
      (audio is unparseable / corrupt / missing header).  Returned only when
      ``drop_unknown_duration`` is set, which is the default — i.e. unknown
      duration and unparseable audio are *hard-dropped* and never enter the
      training set.

    The precomputed ``duration_ms`` column from the curated parquet is trusted
    when present (no FSx audio open).  Only when it is absent do we read the
    audio header (wave -> soundfile -> ffmpeg) to learn the duration.
    """
    if not max_audio_duration_sec or max_audio_duration_sec <= 0:
        return None
    precomputed_ms = record.get("duration_ms")
    if precomputed_ms is not None:
        try:
            if (float(precomputed_ms) / 1000.0) > max_audio_duration_sec:
                return "audio_too_long"
            return None
        except (TypeError, ValueError):
            if drop_unknown_duration:
                return "audio_duration_unknown"
            return None
    # Fall back to reading the audio header for every referenced path.
    for p in audio_paths:
        if not p:
            continue
        dur = audio_duration_sec(p)
        if dur is None:
            if drop_unknown_duration:
                return "audio_duration_unknown"
            # legacy behaviour: keep undeterminable durations
        elif dur > max_audio_duration_sec:
            return "audio_too_long"
    return None


def convert_record(
    record: dict,
    audio_cache_dir: str,
    sampling_rate: int,
    require_local_audio: bool,
    audio_rate_filter: Optional[int],
    fsx_mount: str,
    max_audio_duration_sec: float = 0.0,
    drop_unknown_duration: bool = True,
) -> tuple[Optional[dict], str]:
    """Convert a single parquet row to the ms-swift TTS schema.

    Parameters
    ----------
    record : dict
        A row from ``examples.parquet``.
    audio_cache_dir : str
        Directory for local audio file lookup (fallback).
    sampling_rate : int
        Value written into ``chat_template_kwargs.sampling_rate``.
    require_local_audio : bool
        If True, skip rows whose audio cannot be resolved to a local file.
    audio_rate_filter : int or None
        If set, skip rows whose local audio sample rate does not match.
    fsx_mount : str
        Mount point of the mirrored S3 bucket; the raw ``audio_uri`` S3 URI is
        rewritten to ``<fsx_mount>/<key>`` so the manifest records a readable
        local path.
    """
    # -- text ----------------------------------------------------------------
    # Prefer ``text``; fall back to ``transcription``.
    target_text = clean_str(record.get("text") or record.get("transcription"))
    if not target_text:
        return None, "missing_text"

    # -- language ------------------------------------------------------------
    src_lang = clean_str(record.get("src_lang")).lower()
    if not src_lang:
        return None, "missing_src_lang"

    # -- audio ---------------------------------------------------------------
    audio_uri = clean_str(record.get("audio_uri"))
    if not audio_uri:
        return None, "missing_audio_uri"

    # 1) Resolve to an existing local file (direct path or --audio-cache-dir).
    local_audio = resolve_local_audio(audio_uri, audio_cache_dir)

    # 2) Otherwise rewrite the raw S3 URI to its mirrored FSx path and verify
    #    the file actually exists; drop the sample when it does not.
    if local_audio is None:
        fsx_audio = s3_to_fsx(audio_uri, fsx_mount)
        if fsx_audio is None:
            return None, "s3_uri_format_unexpected"
        _maybe_log_fsx_sample(audio_uri, fsx_audio)
        if not fsx_file_exists(fsx_audio):
            return None, "audio_not_on_fsx"
        local_audio = fsx_audio

    # Duration filter: prefer the precomputed ``duration_ms`` column from the
    # curated parquet (no FSx audio open needed at all).  Only fall back to
    # reading the audio header (wave -> soundfile -> ffmpeg) when that column
    # is absent.  This is what makes TTS processing fast on curated FLAC data.
    # The filter is HARD: unknown-duration / unparseable audio is dropped.
    if max_audio_duration_sec and max_audio_duration_sec > 0:
        reason = duration_filter_reason(
            [local_audio], record, max_audio_duration_sec, drop_unknown_duration
        )
        if reason is not None:
            return None, reason

    if require_local_audio and not local_audio:
        return None, "audio_not_local"

    target_audio_path = local_audio

    # if audio_rate_filter is not None:
    #     if not has_sample_rate(local_audio, audio_rate_filter):
    #         return None, f"audio_not_{audio_rate_filter}hz"

    # -- speaker profile (not available in this dataset) ----------------------
    speaker_id = clean_str(record.get("example_id"))
    gender = clean_str(record.get("gender"))

    tgt_lang_name = lang_code_to_name(src_lang)
    return {
        "schema_version": "ms-swift-speech-simul-v1",
        "task": "tts",
        "task_prompt": (
            f"Task: Speech Synthesis in {tgt_lang_name}; "
            f"the speaking speed is moderate; sentences end naturally."
        ),
        "messages": [{"role": "user", "content": target_text}],
        "target_text": target_text,
        "target_audio": target_audio_path,
        "source_lang": src_lang,
        "target_lang": src_lang,
        "speaker_profile": {
            "speaker_id": speaker_id if speaker_id else None,
            "gender": gender if gender else None,
            "style": None,
            "pitch": None,
        },
        "reference_audios": [],
        "chat_template_kwargs": {"sampling_rate": sampling_rate},
    }, "ok"


def _process_tts_file_subset(
    units: list[tuple[str, Optional[list[int]]]],
    output_path: str,
    audio_cache_dir: str,
    sampling_rate: int,
    require_local_audio: bool,
    audio_rate_filter: Optional[int],
    fsx_mount: str,
    max_audio_duration_sec: float = 0.0,
    drop_unknown_duration: bool = True,
    progress_interval: int = 5000,
    worker_id: int = 0,
) -> tuple[int, int, dict]:
    """Process a list of parquet work units, write JSONL to ``output_path``.

    Each unit is ``(file_path, rg_indices_or_None)``:
      - ``rg_indices_or_None is None`` -> read the whole parquet file.
      - otherwise                       -> read only the listed row groups
                                           (fine-grained, anti-straggler).

    Designed to run inside a ``ProcessPoolExecutor`` subprocess.  Each worker
    gets its own unit subset and writes to a private temp file; the main
    process concatenates all temp files afterwards.

    Returns ``(n_written, n_total, skip_reasons_dict)``.
    """
    import pyarrow.parquet as pq  # local import; only needed for row-group reads

    n_total = 0
    n_written = 0
    skip_reasons: Counter[str] = Counter()
    t0_worker = time.time()

    def _read_unit(parquet_path: str, rg_indices: Optional[list[int]]):
        if not rg_indices:
            return pd.read_parquet(parquet_path)
        pf = pq.ParquetFile(parquet_path)
        tables = [pf.read_row_group(i) for i in rg_indices]
        return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for unit_idx, (parquet_path, rg_indices) in enumerate(units, start=1):
            fname = os.path.basename(os.path.dirname(parquet_path))
            n_before = n_written
            scope = f"rg={rg_indices}" if rg_indices else "whole-file"
            _log(
                f"  [W{worker_id}][{unit_idx}/{len(units)}] {fname} "
                f"({scope}) starting (written={n_written:,})"
            )

            df = _read_unit(parquet_path, rg_indices)
            for record in df.to_dict(orient="records"):
                n_total += 1
                converted, reason = convert_record(
                    record,
                    audio_cache_dir=audio_cache_dir,
                    sampling_rate=sampling_rate,
                    require_local_audio=require_local_audio,
                    audio_rate_filter=audio_rate_filter,
                    fsx_mount=fsx_mount,
                    max_audio_duration_sec=max_audio_duration_sec,
                    drop_unknown_duration=drop_unknown_duration,
                )
                if converted is None:
                    skip_reasons[reason] += 1
                    continue
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % progress_interval == 0:
                    elapsed = time.time() - t0_worker
                    rate = n_written / elapsed if elapsed > 0 else 0
                    _log(
                        f"  [W{worker_id}] {n_written:,} written "
                        f"(total_rows={n_total:,}, rate={rate:.0f} rows/s)"
                    )

            file_rows = n_written - n_before
            _log(
                f"  [W{worker_id}][{unit_idx}/{len(units)}] {fname} done "
                f"({file_rows} written, total={n_written:,})"
            )

    elapsed = time.time() - t0_worker
    _log(
        f"  [W{worker_id}] ALL DONE: {n_written:,} written, "
        f"{n_total:,} total, {_hms(elapsed)}, "
        f"skip_reasons={dict(skip_reasons)}"
    )
    return n_written, n_total, dict(skip_reasons)


def parse_args() -> argparse.Namespace:
    default_input_dir = "../dataset/tts"
    default_output = os.path.abspath(
        os.path.join(DEFAULT_JSONL_DIR, "text_to_speech_v2.jsonl")
    )

    parser = argparse.ArgumentParser(
        description="Convert local TTS parquet shards to ms-swift TTS jsonl.",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default=default_input_dir,
        help=(
            f"Directory containing dataset=*/examples.parquet shards. "
            f"Default: {default_input_dir}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"Path to the output jsonl. Default: {default_output}",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help=(
            "Sampling rate written into chat_template_kwargs. "
            "Default: 16000"
        ),
    )
    parser.add_argument(
        "--audio-cache-dir",
        default="",
        help=(
            "Directory containing local audio files (for S3 URI fallback). "
            "By default no local lookup is attempted."
        ),
    )
    parser.add_argument(
        "--fsx-mount",
        default=DEFAULT_FSX_MOUNT,
        help=(
            "Mount point where the S3 bucket is mirrored. The raw s3://<bucket>/<key> "
            "audio_uri is rewritten to <fsx-mount>/<key> and verified to exist. "
            f"Default: {DEFAULT_FSX_MOUNT}"
        ),
    )
    parser.add_argument(
        "--require-local-audio",
        action="store_true",
        help="Skip rows whose audio URI cannot be resolved to a local file.",
    )
    parser.add_argument(
        "--audio-rate-filter",
        type=int,
        default=None,
        help=(
            "If set, only keep rows whose *local* audio sample rate matches. "
            "Ignored when audio is not resolved locally."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N parquet rows (useful for debugging).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5000,
        help="Log progress every N written samples. Default: 5000",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of parallel workers for processing parquet files. "
            "0 = auto-detect (os.cpu_count()). 1 = single-process. "
            "Ignored when --limit > 0 (single-process is always used with limit)."
        ),
    )
    parser.add_argument(
        "--max-audio-duration-sec",
        type=float,
        default=0.0,
        help=(
            "If > 0, drop samples whose (local) audio duration exceeds this many "
            "seconds. 0 = no duration filtering. Default: 0.0. The filter is HARD: "
            "unknown-duration / unparseable audio is dropped too (see "
            "--no-drop-unknown-duration)."
        ),
    )
    parser.add_argument(
        "--no-drop-unknown-duration",
        action="store_true",
        help=(
            "Disable the hard drop of unknown-duration / unparseable audio. "
            "By default such samples are excluded from the training set."
        ),
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "Explicit list of parquet files to process (overrides --input-dir "
            "discovery). Used by the rolling-prewarm orchestrator to process one "
            "batch at a time."
        ),
    )
    parser.add_argument(
        "--shard-max-lines",
        type=int,
        default=0,
        help=(
            "If > 0, split the output into multiple shards of at most this many "
            "lines each (written to <output_dir>/<name>/<name>_NNNNN.jsonl) so the "
            "training preprocess can read them in parallel. 0 = single file "
            "(backward compatible). Default: 0"
        ),
    )
    return parser.parse_args()


def main() -> int:
    t0_total = time.time()
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    audio_cache_dir = (
        os.path.abspath(args.audio_cache_dir) if args.audio_cache_dir else ""
    )

    _log(f"TTS maker starting, input={input_dir}")
    _log(f"  fsx_mount={args.fsx_mount} (s3://<bucket>/<key> -> {args.fsx_mount}/<key>)")

    # ---- Pre-flight ----
    if not os.path.isdir(input_dir):
        _log(f"[ERROR] input directory not found: {input_dir}")
        return 1

    # ---- Discover parquet files ----
    if args.input_files:
        parquet_files = [os.path.abspath(p) for p in args.input_files if os.path.isfile(p)]
        if not parquet_files:
            _log(f"[ERROR] none of --input-files exist: {args.input_files}")
            return 1
        _log(f"Using {len(parquet_files)} explicit --input-files (overrides discovery)")
    else:
        parquet_files = find_input_parquets(input_dir)
        if not parquet_files:
            _log(f"[ERROR] no non-empty examples.parquet found under: {input_dir}")
            return 1

    drop_unknown_duration = not args.no_drop_unknown_duration
    _log(f"  drop_unknown_duration={drop_unknown_duration} "
         f"(unknown/unparseable audio excluded)")

    # Quick row-count + row-group inventory by reading parquet metadata
    # (fast, doesn't load data). rg_items enumerates every row group so the
    # hybrid scheduler (large-file split, small/medium whole-file) can be built.
    _log(f"Scanning {len(parquet_files)} dataset=*/examples.parquet files (metadata only) ...")
    t_scan = time.time()
    import pyarrow.parquet as pq
    total_rows_est = 0
    rg_items: list[tuple[str, int, int]] = []
    for pf in parquet_files:
        try:
            meta = pq.read_metadata(pf)
        except Exception:
            continue
        for rg_idx in range(meta.num_row_groups):
            rg_rows = meta.row_group(rg_idx).num_rows
            rg_items.append((pf, rg_idx, rg_rows))
            total_rows_est += rg_rows
    _log(
        f"Scan done in {_hms(time.time()-t_scan)}: "
        f"{len(parquet_files)} files, {len(rg_items)} row groups, "
        f"~{total_rows_est:,} rows estimated"
    )
    if args.limit:
        _log(f"  (--limit={args.limit}, will stop early)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ---- Process ----
    n_total = 0
    n_written = 0
    skip_reasons: Counter[str] = Counter()
    t0_phase = time.time()

    # Determine effective number of workers.
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, _detect_cpu_count())
    _log(f"  num_workers={num_workers} (arg={args.num_workers})")
    # Never multiprocess when --limit is active — would distort the row budget.
    use_multiprocess = num_workers > 1 and len(parquet_files) > 1 and not args.limit

    if use_multiprocess:
        _log(
            f"Multiprocess mode: {num_workers} workers, "
            f"{len(parquet_files)} parquet files"
        )

        # ---- Hybrid scheduling: file-level for small/medium files,
        #     row-group-level for large files (>100k rows). ----
        #
        # Pure row-group scheduling (old) causes 64 workers to simultaneously
        # open different small parquet files after the large ones finish,
        # hammering FSx with metadata ops and dropping aggregate throughput
        # from ~7k to ~30 rows/s per worker.
        #
        # This hybrid approach keeps row-group parallelism for large files
        # (anti-straggler) while assigning whole small/medium files to
        # individual workers (anti-overhead).  Small files (<10k rows) are
        # batched together to further reduce concurrent FSx file opens.
        LARGE_THRESHOLD = 100_000  # rows: split file into per-row-group units
        MEDIUM_THRESHOLD = 10_000  # rows: whole file as one unit
        BATCH_ROWS_TARGET = 50_000  # rows: target size for small-file batches

        # Build file-level aggregates from the scan.
        file_total_rows: dict[str, int] = {}
        file_rg_list: dict[str, list[tuple[int, int]]] = {}
        for pf, rg_idx, rg_rows in rg_items:
            file_total_rows[pf] = file_total_rows.get(pf, 0) + rg_rows
            file_rg_list.setdefault(pf, []).append((rg_idx, rg_rows))

        # Uniform work units: (file_path, rg_indices_or_None, n_rows)
        #   - Large files  -> one unit per row group (fine-grained parallelism,
        #                     anti-straggler: the tail is spread over all CPUs).
        #   - Medium files -> one whole-file unit.
        #   - Small files  -> batched by row count; each file is its own
        #                     whole-file unit (keeps row-group accounting simple).
        units: list[tuple[str, Optional[list[int]], int]] = []
        small_pool: list[tuple[str, int]] = []
        for pf, total in file_total_rows.items():
            if total >= LARGE_THRESHOLD:
                for rg_idx, rg_rows in file_rg_list[pf]:
                    units.append((pf, [rg_idx], rg_rows))
            elif total >= MEDIUM_THRESHOLD:
                units.append((pf, None, total))
            else:
                small_pool.append((pf, total))

        if small_pool:
            small_pool.sort(key=lambda x: x[0])  # deterministic order
            batch: list[str] = []
            batch_rows = 0
            for pf, total in small_pool:
                batch.append(pf)
                batch_rows += total
                if batch_rows >= BATCH_ROWS_TARGET:
                    for bf in batch:
                        units.append((bf, None, file_total_rows[bf]))
                    batch = []
                    batch_rows = 0
            for bf in batch:
                units.append((bf, None, file_total_rows[bf]))

        # Greedy bin-packing of units by row count across all workers.  Because
        # large files are already split into per-row-group units, the long tail
        # at the end is distributed over every CPU instead of stranding a few
        # big files on a handful of processes.
        worker_loads = [0] * num_workers
        worker_assignments: list[list[tuple[str, Optional[list[int]]]]] = [
            [] for _ in range(num_workers)
        ]

        for pf, rg_indices, n_rows in sorted(units, key=lambda x: -x[2]):
            w = min(range(num_workers), key=lambda i: worker_loads[i])
            worker_assignments[w].append((pf, rg_indices))
            worker_loads[w] += n_rows

        # Log distribution skew.
        max_load = max(worker_loads) if worker_loads else 0
        min_load = min(worker_loads) if worker_loads else 0
        _log(
            f"  load: {len(units)} units, max={max_load:,} rows, "
            f"min={min_load:,} rows, "
            f"imbalance={max_load/max(1, min_load):.1f}x"
        )

        temp_outputs: list[str] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(worker_assignments)
        ) as executor:
            futures: dict[concurrent.futures.Future, int] = {}
            for worker_id, assign in enumerate(worker_assignments):
                temp_out = f"{args.output}.part_{worker_id}"
                temp_outputs.append(temp_out)
                fut = executor.submit(
                    _process_tts_file_subset,
                    assign,
                    temp_out,
                    audio_cache_dir,
                    args.sampling_rate,
                    args.require_local_audio,
                    args.audio_rate_filter,
                    args.fsx_mount,
                    args.max_audio_duration_sec,
                    drop_unknown_duration,
                    args.progress_interval,
                    worker_id,
                )
                futures[fut] = worker_id

            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                try:
                    n_w, n_t, skips = future.result()
                    n_written += n_w
                    n_total += n_t
                    for reason, count in skips.items():
                        skip_reasons[reason] += count
                    _log(f"  [W{worker_id}] result collected: {n_w:,} written")
                except Exception as exc:
                    _log(f"[ERROR] Worker {worker_id} failed: {exc}")
                    raise

        # Concatenate temp files in worker order to keep output deterministic.
        # When sharding is enabled the concatenated stream is written out as
        # multiple medium-sized shards via ShardWriter.
        _log(f"Concatenating {len(temp_outputs)} temp files → {args.output} ...")
        writer = shardio.ShardWriter(args.output, args.shard_max_lines)
        try:
            for temp_out in temp_outputs:
                if os.path.isfile(temp_out) and os.path.getsize(temp_out) > 0:
                    with open(temp_out, "r", encoding="utf-8") as fin:
                        for line in fin:
                            writer.write(line.rstrip("\n"))
                    os.remove(temp_out)
                else:
                    # The worker produced nothing (all rows skipped).
                    if os.path.isfile(temp_out):
                        os.remove(temp_out)
        finally:
            writer.close()
    else:
        # ---- Single-process path (original behaviour) ----
        writer = shardio.ShardWriter(args.output, args.shard_max_lines)
        try:
            for file_idx, parquet_path in enumerate(parquet_files, start=1):
                fname = os.path.basename(os.path.dirname(parquet_path))  # dataset=... dir
                t0_file = time.time()
                n_before = n_written

                _log(
                    f"  [{file_idx}/{len(parquet_files)}] {fname} starting "
                    f"(written={n_written:,}, rate_cache={len(_AUDIO_RATE_CACHE):,})"
                )

                df = pd.read_parquet(parquet_path)
                for record in df.to_dict(orient="records"):
                    if args.limit and n_total >= args.limit:
                        break
                    n_total += 1
                    converted, reason = convert_record(
                        record,
                        audio_cache_dir=audio_cache_dir,
                        sampling_rate=args.sampling_rate,
                        require_local_audio=args.require_local_audio,
                        audio_rate_filter=args.audio_rate_filter,
                        fsx_mount=args.fsx_mount,
                        max_audio_duration_sec=args.max_audio_duration_sec,
                        drop_unknown_duration=drop_unknown_duration,
                    )
                    if converted is None:
                        skip_reasons[reason] += 1
                        continue
                    writer.write(json.dumps(converted, ensure_ascii=False))
                    n_written += 1
                    if n_written % args.progress_interval == 0:
                        elapsed = time.time() - t0_phase
                        rate = n_written / elapsed if elapsed > 0 else 0
                        _log(
                            f"    {n_written:,} samples written "
                            f"(total_rows={n_total:,}, skip_rate={sum(skip_reasons.values())/max(1,n_total)*100:.1f}%, "
                            f"rate={rate:.0f} rows/s, elapsed={_hms(elapsed)})"
                        )

                # Per-file footer
                file_rows = n_written - n_before
                file_elapsed = time.time() - t0_file
                file_rate = file_rows / file_elapsed if file_elapsed > 0 else 0
                _log(
                    f"  [{file_idx}/{len(parquet_files)}] {fname} done "
                    f"({file_rows} written in {_hms(file_elapsed)}, "
                    f"{file_rate:.0f} rows/s, total_written={n_written:,})"
                )

                if args.limit and n_total >= args.limit:
                    _log(f"  --limit={args.limit} reached, stopping.")
                    break

        finally:
            writer.close()

    # ---- Summary ----
    total_elapsed = time.time() - t0_total
    n_skipped = sum(skip_reasons.values())

    _log("=" * 60)
    _log(f"DONE in {_hms(total_elapsed)}")
    _log(
        f"  files={len(parquet_files)}, total_rows={n_total:,}, "
        f"written={n_written:,}, skipped={n_skipped:,}"
    )
    if n_written > 0:
        _log(f"  overall_rate={n_written/total_elapsed:.0f} rows/s")
    _log(f"  skip_reasons={dict(skip_reasons)}")
    if args.shard_max_lines and args.shard_max_lines > 0:
        out_desc = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            os.path.basename(args.output).replace(".jsonl", ""),
        ) + "/ (shards)"
    else:
        out_desc = args.output
    _log(f"  output -> {out_desc}")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
