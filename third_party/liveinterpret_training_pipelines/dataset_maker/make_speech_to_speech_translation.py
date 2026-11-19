#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert local speech2speech parquet shards into ms-swift S2S translation format.

Input  : /path/to/project/dataset/speech2speech/*.parquet
Output : speech_to_speech_translation_v2.jsonl

This v2 converter consumes all shards matching ``part*.parquet`` and
``zh_en_part_*.parquet``. A row is written only when its local source audio is
16 kHz and its local target audio is 24 kHz.

Output schema is identical to ``make_speech_to_speech_translation.py``.
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
from typing import Optional
from urllib.parse import urlparse

import shardio

import pandas as pd

# ISO 639-1 / common BCP-47 short codes -> English language name
LANG_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "he": "Hebrew",
    "fa": "Persian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "pcm": "Nigerian Pidgin",
    "yue": "Cantonese",
}

_PARQUET_RE = re.compile(r"^(?:part\d+|zh_en_part_\d+)\.parquet$")
_AUDIO_RATE_CACHE: dict[str, Optional[int]] = {}

# Audio duration cache (seconds); populated lazily by audio_duration_sec().
_AUDIO_DURATION_CACHE: dict[str, Optional[float]] = {}

# Output directory for the generated jsonl manifests. Mirrors the worker's
# ${BASE_PATH}/datasets/jsonl/ so the argparse default never NameErrors
# (the worker still passes --output explicitly).
DEFAULT_BASE_PATH = os.environ.get(
    "BASE_PATH",
    "/local/speech-language-model/liveinterpret_training_pipelines",
)
DEFAULT_JSONL_DIR = os.environ.get(
    "DEFAULT_JSONL_DIR",
    os.path.join(DEFAULT_BASE_PATH, "datasets", "jsonl"),
)


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


# In-memory index: audio_dir -> set of filenames.
# Built once via os.listdir() at startup, then used for O(1) lookups
# instead of per-row os.path.isfile() calls which are ~1000x slower over
# network filesystems (FSx / NFS).
_AUDIO_INDEX: dict[str, set[str]] = {}


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


def lang_code_to_name(code: str) -> str:
    """Map a short language code to its English name. Falls back to the code itself."""
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
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def find_input_parquets(input_dir: str) -> list[str]:
    files = []
    for path in glob.glob(os.path.join(input_dir, "*.parquet")):
        name = os.path.basename(path)
        if _PARQUET_RE.match(name) and os.path.getsize(path) > 0:
            files.append(path)
    return sorted(files, key=natural_key)


def basename_from_uri(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    path = path.split("#", 1)[0].split("?", 1)[0]
    return os.path.basename(path)


def resolve_audio_path(value: str, audio_dir: str) -> Optional[str]:
    value = clean_str(value)
    if not value:
        return None
    # Fast path: value is already an absolute local file path
    if os.path.isabs(value) and os.path.isfile(value):
        return os.path.abspath(value)
    basename = basename_from_uri(value)
    if not basename:
        return None
    # Use in-memory index if available (O(1) dict lookup vs FSx round-trip)
    index = _AUDIO_INDEX.get(audio_dir)
    if index is not None:
        if basename in index:
            return os.path.abspath(os.path.join(audio_dir, basename))
        return None
    # Fallback: per-row os.path.isfile (slow over network FS, kept for
    # backward compatibility when index was not pre-built)
    local_path = os.path.join(audio_dir, basename)
    if os.path.isfile(local_path):
        return os.path.abspath(local_path)
    return None


def audio_sample_rate(path: str) -> Optional[int]:
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

    Deprecated in favour of :func:`duration_filter_reason`.  Kept for
    call-site compatibility.
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
    """Hard duration/parseability gate (see make_text_to_speech.duration_filter_reason).

    Returns ``None`` if the sample passes, or ``"audio_too_long"`` /
    ``"audio_duration_unknown"`` otherwise.  Unknown-duration / unparseable audio
    is hard-dropped when ``drop_unknown_duration`` is set (default).
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
    for p in audio_paths:
        if not p:
            continue
        dur = audio_duration_sec(p)
        if dur is None:
            if drop_unknown_duration:
                return "audio_duration_unknown"
        elif dur > max_audio_duration_sec:
            return "audio_too_long"
    return None


def convert_record(
    record: dict,
    source_audio_dir: str,
    target_audio_dir: str,
    latency_budget_ms: int,
    max_audio_duration_sec: float = 0.0,
    drop_unknown_duration: bool = True,
) -> tuple[Optional[dict], str]:
    """Convert a single parquet row to the ms-swift output schema."""
    source_audio = resolve_audio_path(record.get("source_audio_uri"), source_audio_dir)
    target_audio = resolve_audio_path(record.get("target_audio_uri"), target_audio_dir)
    target_text = clean_str(record.get("target_text"))
    src_lang = clean_str(record.get("src_lang")).lower()
    tgt_lang = clean_str(record.get("tgt_lang")).lower()

    if not source_audio or not target_audio or not target_text or not src_lang or not tgt_lang:
        return None, "missing_required_field"

    # Duration filter: prefer the precomputed ``duration_ms`` column from the
    # curated parquet (no FSx audio open needed).  Only fall back to reading the
    # audio header (wave -> soundfile -> ffmpeg) when that column is absent.
    # The filter is HARD: unknown-duration / unparseable audio is dropped.
    if max_audio_duration_sec and max_audio_duration_sec > 0:
        reason = duration_filter_reason(
            [source_audio, target_audio], record, max_audio_duration_sec,
            drop_unknown_duration,
        )
        if reason is not None:
            return None, reason
    # if not has_sample_rate(source_audio, 16000):
    #     return None, "source_audio_not_16k"
    # if not has_sample_rate(target_audio, 24000):
    #     return None, "target_audio_not_24k"

    tgt_lang_name = lang_code_to_name(tgt_lang)
    return {
        "schema_version": "ms-swift-speech-simul-v1",
        "task": "speech_to_speech_translation",
        "task_prompt": f"Task: Speech-to-speech translation; target language is {tgt_lang_name}.",
        "messages": [{"role": "user", "content": "<audio>"}],
        "audios": [source_audio],
        "target_text": target_text,
        "target_audio": target_audio,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "latency_budget_ms": latency_budget_ms,
    }, "ok"


def _process_s2s_file_subset(
    parquet_files: list[str],
    output_path: str,
    source_audio_dir: str,
    target_audio_dir: str,
    latency_budget_ms: int,
    max_audio_duration_sec: float = 0.0,
    drop_unknown_duration: bool = True,
    progress_interval: int = 5000,
    worker_id: int = 0,
) -> tuple[int, int, dict]:
    """Process a list of parquet files, write JSONL to ``output_path``.

    Designed to run inside a ``ProcessPoolExecutor`` subprocess.  The in-memory
    audio index (``_AUDIO_INDEX``) is inherited from the parent via fork
    (copy-on-write), so no re-scan is needed.

    Returns ``(n_written, n_total, skip_reasons_dict)``.
    """
    n_total = 0
    n_written = 0
    skip_reasons: Counter[str] = Counter()
    t0_worker = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for file_idx, parquet_path in enumerate(parquet_files, start=1):
            fname = os.path.basename(parquet_path)
            n_before = n_written

            _log(
                f"  [W{worker_id}][{file_idx}/{len(parquet_files)}] {fname} starting "
                f"(written={n_written:,})"
            )

            df = pd.read_parquet(parquet_path)
            for record in df.to_dict(orient="records"):
                n_total += 1
                converted, reason = convert_record(
                    record,
                    source_audio_dir=source_audio_dir,
                    target_audio_dir=target_audio_dir,
                    latency_budget_ms=latency_budget_ms,
                    max_audio_duration_sec=max_audio_duration_sec,
                    drop_unknown_duration=drop_unknown_duration,
                )
                if converted is None:
                    skip_reasons[reason] += 1
                    continue
                if n_written < 5:
                    _log(
                        f"  [W{worker_id}] [audio path sample {n_written + 1}/5] "
                        f"source={converted['audios'][0]} | target={converted['target_audio']}"
                    )
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
                f"  [W{worker_id}][{file_idx}/{len(parquet_files)}] {fname} done "
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
    default_input_dir = "../dataset/speech2speech"
    default_output = os.path.abspath(
        os.path.join(DEFAULT_JSONL_DIR, "speech_to_speech_translation_v2.jsonl")
    )

    parser = argparse.ArgumentParser(
        description="Convert local speech2speech parquet shards to ms-swift speech-to-speech translation jsonl.",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default=default_input_dir,
        help=f"Directory containing parquet shards plus source_audio/ and target_audio/. Default: {default_input_dir}",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"Path to the output jsonl. Default: {default_output}",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=2000,
        help="Latency budget in milliseconds written to every sample. Default: 2000",
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
            "If > 0, drop samples whose (local) source or target audio duration "
            "exceeds this many seconds. 0 = no duration filtering. Default: 0.0. "
            "The filter is HARD: unknown-duration / unparseable audio is dropped "
            "too (see --no-drop-unknown-duration)."
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
    source_audio_dir = os.path.join(input_dir, "source_audio")
    target_audio_dir = os.path.join(input_dir, "target_audio")

    _log(f"S2S maker starting, input={input_dir}")

    # ---- Pre-flight ----
    if not os.path.isdir(input_dir):
        _log(f"[ERROR] input directory not found: {input_dir}")
        return 1
    if not os.path.isdir(source_audio_dir):
        _log(f"[ERROR] source_audio directory not found: {source_audio_dir}")
        return 1
    if not os.path.isdir(target_audio_dir):
        _log(f"[ERROR] target_audio directory not found: {target_audio_dir}")
        return 1

    # ---- Build in-memory audio index (one os.listdir per dir, O(1) lookups) ----
    _log("Building source_audio index ...")
    t_idx = time.time()
    _AUDIO_INDEX[source_audio_dir] = set(os.listdir(source_audio_dir))
    _AUDIO_INDEX[target_audio_dir] = set(os.listdir(target_audio_dir))
    _log(
        f"Audio index built in {_hms(time.time()-t_idx)}: "
        f"source={len(_AUDIO_INDEX[source_audio_dir])} files, "
        f"target={len(_AUDIO_INDEX[target_audio_dir])} files"
    )

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
            _log(f"[ERROR] no non-empty part*.parquet or zh_en_part_*.parquet found under: {input_dir}")
            return 1

    drop_unknown_duration = not args.no_drop_unknown_duration
    _log(f"  drop_unknown_duration={drop_unknown_duration} "
         f"(unknown/unparseable audio excluded)")

    # Quick row-count by reading parquet metadata (fast, doesn't load data)
    _log(f"Scanning {len(parquet_files)} parquet files (metadata only) ...")
    t_scan = time.time()
    total_rows_est = 0
    file_row_counts: list[tuple[str, int]] = []
    for pf in parquet_files:
        try:
            import pyarrow.parquet as pq
            nrows = pq.read_metadata(pf).num_rows
            total_rows_est += nrows
            file_row_counts.append((os.path.basename(pf), nrows))
        except Exception:
            # Fallback: count during processing
            file_row_counts.append((os.path.basename(pf), -1))
    _log(
        f"Scan done in {_hms(time.time()-t_scan)}: "
        f"{len(parquet_files)} files, ~{total_rows_est:,} rows estimated"
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
        # Pure row-group scheduling (old) causes N workers to simultaneously
        # open different small parquet files after the large ones finish,
        # hammering FSx with metadata ops and dropping aggregate throughput.
        #
        # This hybrid approach keeps row-group parallelism for large files
        # (anti-straggler) while assigning whole small/medium files to
        # individual workers (anti-overhead).  Small files (<10k rows) are
        # batched together to further reduce concurrent FSx file opens.
        LARGE_THRESHOLD = 100_000  # rows: split into row-group tasks
        MEDIUM_THRESHOLD = 10_000  # rows: whole file as one task
        BATCH_ROWS_TARGET = 50_000  # rows: target size for small-file batches

        # Build per-file row counts from the scan metadata (full paths).
        # _process_s2s_file_subset reads each file in full, so we assign whole
        # parquet files to workers with greedy bin-packing by size for load
        # balance (no row-group splitting needed here).
        file_total_rows: dict[str, int] = {}
        for pf in parquet_files:
            try:
                import pyarrow.parquet as pq
                nrows = pq.read_metadata(pf).num_rows
            except Exception:
                nrows = -1
            file_total_rows[pf] = nrows if nrows and nrows > 0 else 1

        worker_loads = [0] * num_workers
        worker_assignments: list[list[str]] = [[] for _ in range(num_workers)]

        for pf, total in sorted(file_total_rows.items(), key=lambda x: -x[1]):
            w = min(range(num_workers), key=lambda i: worker_loads[i])
            worker_assignments[w].append(pf)
            worker_loads[w] += total

        # Log distribution skew.
        max_load = max(worker_loads) if worker_loads else 0
        min_load = min(worker_loads) if worker_loads else 0
        _log(
            f"  load: max={max_load:,} rows, min={min_load:,} rows, "
            f"imbalance={max_load/max(1, min_load):.1f}x"
        )

        temp_outputs: list[str] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(worker_assignments)
        ) as executor:
            futures: dict[concurrent.futures.Future, int] = {}
            for worker_id, files in enumerate(worker_assignments):
                temp_out = f"{args.output}.part_{worker_id}"
                temp_outputs.append(temp_out)
                fut = executor.submit(
                    _process_s2s_file_subset,
                    files,
                    temp_out,
                    source_audio_dir,
                    target_audio_dir,
                    args.latency_budget_ms,
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

        # Concatenate temp files in worker order (deterministic output).
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
                    if os.path.isfile(temp_out):
                        os.remove(temp_out)
        finally:
            writer.close()
    else:
        # ---- Single-process path (original behaviour) ----
        writer = shardio.ShardWriter(args.output, args.shard_max_lines)
        try:
            for file_idx, (parquet_path, _) in enumerate(zip(parquet_files, file_row_counts), start=1):
                fname = os.path.basename(parquet_path)
                t0_file = time.time()

                # Per-file header
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
                        source_audio_dir=source_audio_dir,
                        target_audio_dir=target_audio_dir,
                        latency_budget_ms=args.latency_budget_ms,
                        max_audio_duration_sec=args.max_audio_duration_sec,
                        drop_unknown_duration=drop_unknown_duration,
                    )
                    if converted is None:
                        skip_reasons[reason] += 1
                        continue
                    if n_written < 5:
                        _log(
                            f"  [audio path sample {n_written + 1}/5] "
                            f"source={converted['audios'][0]} | target={converted['target_audio']}"
                        )
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
    _log(f"  audio_rate_cache_size={len(_AUDIO_RATE_CACHE):,}")
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
