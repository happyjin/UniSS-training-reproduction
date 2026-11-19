#!/usr/bin/env python3.12
"""
Filter text_to_speech.jsonl by:
1. Audio duration <= 2 minutes (120 seconds)
2. Qwen3-TTS supported languages: en, zh, ja, ko, de, fr, es, ru, pt, it

Strategy (two-pass, memory-efficient):
  Pass 1: Scan full file, filter by language, collect unique audio paths.
  Pass 2: Check duration of each unique audio file (header only, parallel).
  Pass 3: Re-read file, write entries where lang supported AND audio <= 2min.
"""

import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import soundfile as sf

INPUT_FILE = "/opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/datasets/jsonl/text_to_speech.jsonl"
OUTPUT_FILE = "/opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/datasets/jsonl/text_to_speech_filtered.jsonl"

SUPPORTED_LANGS = {"en", "zh", "ja", "ko", "de", "fr", "es", "ru", "pt", "it"}
MAX_DURATION_SEC = 120.0  # 2 minutes


def get_audio_duration(audio_path):
    """Get audio duration in seconds by reading header only."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return -1.0


def check_duration_batch(paths):
    """Check duration for a batch of audio paths."""
    results = {}
    for p in paths:
        dur = get_audio_duration(p)
        results[p] = dur
    return results


def main():
    t0 = time.time()

    # === Pass 1: Filter by language, collect unique audio paths ===
    print("Pass 1: Scanning for supported languages and collecting unique audio paths...")
    lang_counts = defaultdict(int)
    unique_audio_paths = set()
    total_lines = 0

    with open(INPUT_FILE, "r") as f:
        for i, line in enumerate(f):
            total_lines = i + 1
            if i % 2_000_000 == 0 and i > 0:
                print(f"  Scanned {i:,} lines... ({time.time()-t0:.1f}s)")
            d = json.loads(line)
            lang = d.get("target_lang", "")
            if lang in SUPPORTED_LANGS:
                lang_counts[lang] += 1
                unique_audio_paths.add(d["target_audio"])

    supported_total = sum(lang_counts.values())
    print(f"\nPass 1 done in {time.time()-t0:.1f}s")
    print(f"  Total entries: {total_lines:,}")
    print(f"  Supported lang entries: {supported_total:,}")
    print(f"  Unique audio files to check: {len(unique_audio_paths):,}")
    print(f"  Per-language:")
    for lang in sorted(SUPPORTED_LANGS):
        print(f"    {lang}: {lang_counts[lang]:,}")

    # === Pass 2: Check audio durations in parallel ===
    print(f"\nPass 2: Checking {len(unique_audio_paths):,} audio file durations...")
    t1 = time.time()

    audio_paths_list = list(unique_audio_paths)
    del unique_audio_paths  # free memory

    num_workers = min(cpu_count(), 32)
    batch_size = 500
    batches = [audio_paths_list[i:i+batch_size] for i in range(0, len(audio_paths_list), batch_size)]

    print(f"  Using {num_workers} workers, {len(batches)} batches of {batch_size} files")

    duration_map = {}
    checked = 0
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(check_duration_batch, batches):
            duration_map.update(result)
            checked += len(result)
            if checked % 200_000 < batch_size:
                print(f"  Checked {checked:,}/{len(audio_paths_list):,} files... ({time.time()-t1:.1f}s)")

    print(f"  Done in {time.time()-t1:.1f}s")

    # Stats on durations
    valid_count = sum(1 for d in duration_map.values() if 0 < d <= MAX_DURATION_SEC)
    invalid_count = sum(1 for d in duration_map.values() if d > MAX_DURATION_SEC)
    error_count = sum(1 for d in duration_map.values() if d < 0)
    print(f"  Valid (<= 2min): {valid_count:,}")
    print(f"  Too long (> 2min): {invalid_count:,}")
    print(f"  Errors (missing/unreadable): {error_count:,}")

    # Build set of valid audio paths for fast lookup
    valid_audio = {p for p, d in duration_map.items() if 0 < d <= MAX_DURATION_SEC}
    del duration_map  # free memory

    # === Pass 3: Re-read file, write filtered entries ===
    print(f"\nPass 3: Writing filtered entries to {OUTPUT_FILE}...")
    t2 = time.time()

    written = 0
    lang_written = defaultdict(int)
    skipped_duration = 0
    skipped_lang = 0

    with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "w") as f_out:
        for i, line in enumerate(f_in):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  Re-scanned {i:,} lines... written {written:,} ({time.time()-t2:.1f}s)")
            d = json.loads(line)
            lang = d.get("target_lang", "")
            if lang not in SUPPORTED_LANGS:
                skipped_lang += 1
                continue
            audio = d["target_audio"]
            if audio in valid_audio:
                f_out.write(line)
                written += 1
                lang_written[lang] += 1
            else:
                skipped_duration += 1

    print(f"\nPass 3 done in {time.time()-t2:.1f}s")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"Input entries: {total_lines:,}")
    print(f"Skipped (unsupported lang): {skipped_lang:,}")
    print(f"Skipped (duration > 2min or error): {skipped_duration:,}")
    print(f"Written: {written:,}")
    print(f"\nPer-language written:")
    for lang in sorted(SUPPORTED_LANGS):
        total = lang_counts[lang]
        kept = lang_written.get(lang, 0)
        pct = (kept / total * 100) if total > 0 else 0
        print(f"  {lang}: {kept:>10,} / {total:>10,} ({pct:.1f}%)")

    out_size = os.path.getsize(OUTPUT_FILE)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Output size: {out_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
