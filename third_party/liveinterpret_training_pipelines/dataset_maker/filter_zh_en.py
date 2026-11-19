#!/usr/bin/env python3.12
"""
Filter text_to_speech.jsonl → jsonl_zeyang/text_to_speech_zh_en.jsonl

Filters:
  1. Only keep target_lang == "en" or "zh" (exact match, no regional suffixes)
  2. Problem 1: Remove multi-segment shared audio (same audio, different texts) → remove all
  3. Problem 2: Deduplicate by (speaker_id, target_text), keep first occurrence (no path verification)
  4. Log duplicate source datasets for summary
"""

import json
import os
import re
import time
from collections import defaultdict

INPUT_FILE = "/opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/datasets/jsonl/text_to_speech.jsonl"
OUTPUT_DIR = "/opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/datasets/jsonl_zeyang"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "text_to_speech_zh_en.jsonl")

SUPPORTED_LANGS = {"en", "zh"}


def extract_dataset_name(audio_path):
    """Extract dataset=... from audio path."""
    m = re.search(r"dataset=([^/]+)", audio_path)
    return m.group(1) if m else "unknown"


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Pass 1: Build metadata ===
    print("Pass 1: Scanning for en/zh entries and building metadata...")

    audio_text_hashes = defaultdict(set)
    content_first_path = {}       # key -> first audio path seen
    content_first_dataset = {}    # key -> dataset name of first path

    total_lines = 0
    en_zh_count = 0
    lang_counts = defaultdict(int)

    with open(INPUT_FILE, "r") as f:
        for i, line in enumerate(f):
            total_lines = i + 1
            if i % 2_000_000 == 0 and i > 0:
                print(f"  Scanned {i:,} lines... ({time.time()-t0:.1f}s)")

            d = json.loads(line)
            lang = d.get("target_lang", "")
            if lang not in SUPPORTED_LANGS:
                continue

            en_zh_count += 1
            lang_counts[lang] += 1

            audio = d["target_audio"]
            text = d["target_text"]
            speaker_id = d["speaker_profile"]["speaker_id"] or ""
            text_h = hash(text)
            key = (speaker_id, text_h)

            audio_text_hashes[audio].add(text_h)

            if key not in content_first_path:
                content_first_path[key] = audio
                content_first_dataset[key] = extract_dataset_name(audio)

    print(f"\nPass 1 done in {time.time()-t0:.1f}s")
    print(f"  Total lines: {total_lines:,}")
    print(f"  en+zh entries: {en_zh_count:,}")
    for lang in sorted(SUPPORTED_LANGS):
        print(f"    {lang}: {lang_counts[lang]:,}")
    print(f"  Unique audio paths: {len(audio_text_hashes):,}")
    print(f"  Unique content keys: {len(content_first_path):,}")

    # === Identify Problem 1 ===
    problem1_audios = set()
    for audio, text_hashes in audio_text_hashes.items():
        if len(text_hashes) > 1:
            problem1_audios.add(audio)

    print(f"\nProblem 1 (multi-segment shared audio): {len(problem1_audios):,} audio files")
    del audio_text_hashes

    # === Determine keep set ===
    keep_path = {}
    removed_p1 = 0
    for key, path in content_first_path.items():
        if path in problem1_audios:
            removed_p1 += 1
            continue
        keep_path[key] = path

    print(f"  Removed (Problem 1): {removed_p1:,}")
    print(f"  To keep: {len(keep_path):,}")

    # === Pass 2: Write filtered output ===
    print(f"\nPass 2: Writing filtered entries...")
    t2 = time.time()

    written = 0
    seen_keys = set()
    skipped_p1 = 0
    skipped_dup = 0
    written_lang = defaultdict(int)

    # Track which datasets had duplicates removed
    dup_removed_datasets = defaultdict(int)
    kept_datasets = defaultdict(int)

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for i, line in enumerate(fin):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  Re-scanned {i:,} lines... written={written:,} ({time.time()-t2:.1f}s)")

            d = json.loads(line)
            lang = d.get("target_lang", "")
            if lang not in SUPPORTED_LANGS:
                continue

            audio = d["target_audio"]
            if audio in problem1_audios:
                skipped_p1 += 1
                continue

            text = d["target_text"]
            speaker_id = d["speaker_profile"]["speaker_id"] or ""
            text_h = hash(text)
            key = (speaker_id, text_h)

            if key not in keep_path:
                continue

            if audio != keep_path[key]:
                skipped_dup += 1
                dup_removed_datasets[extract_dataset_name(audio)] += 1
                continue

            if key in seen_keys:
                skipped_dup += 1
                dup_removed_datasets[extract_dataset_name(audio)] += 1
                continue

            fout.write(line)
            seen_keys.add(key)
            written += 1
            written_lang[lang] += 1
            kept_datasets[extract_dataset_name(audio)] += 1

    print(f"\nPass 2 done in {time.time()-t2:.1f}s")

    # === Summary ===
    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Input: {total_lines:,} entries")
    print(f"  en: {lang_counts['en']:,}")
    print(f"  zh: {lang_counts['zh']:,}")
    print(f"  en+zh total: {en_zh_count:,}")
    print(f"")
    print(f"Filtered out:")
    print(f"  Problem 1 (multi-segment): {skipped_p1:,}")
    print(f"  Problem 2 (duplicates): {skipped_dup:,}")
    print(f"")
    print(f"Written: {written:,}")
    for lang in sorted(SUPPORTED_LANGS):
        print(f"  {lang}: {written_lang[lang]:,}")

    print(f"\n{'='*60}")
    print(f"DUPLICATE REMOVAL BY DATASET")
    print(f"{'='*60}")
    print(f"{'Dataset':<55} {'Removed':>10} {'Kept':>10}")
    print(f"{'-'*55} {'-'*10} {'-'*10}")
    all_datasets = set(dup_removed_datasets.keys()) | set(kept_datasets.keys())
    for ds in sorted(all_datasets):
        removed = dup_removed_datasets.get(ds, 0)
        kept = kept_datasets.get(ds, 0)
        if removed > 0:
            print(f"{ds:<55} {removed:>10,} {kept:>10,}")

    print(f"\nDatasets with NO duplicates (all kept):")
    for ds in sorted(all_datasets):
        removed = dup_removed_datasets.get(ds, 0)
        kept = kept_datasets.get(ds, 0)
        if removed == 0 and kept > 0:
            print(f"  {ds}: {kept:,}")

    out_size = os.path.getsize(OUTPUT_FILE)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Output size: {out_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
