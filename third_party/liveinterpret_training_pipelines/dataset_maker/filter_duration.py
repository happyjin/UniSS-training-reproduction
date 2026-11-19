#!/usr/bin/env python3.12
"""
Filter text_to_speech_zh_en_clean.jsonl -> text_to_speech_zh_en_final.jsonl

Filters:
  1. Remove thchs_30 OCR-noise entries (dataset=thchs_30 with ASCII letters in
     target_text). Expected: 12 entries (Problem 10).
  2. Remove audio longer than MAX_DURATION_SEC (default 120s = 2 min).

Audio duration is read from the file header via soundfile.info (no decoding),
parallelized across processes for speed.
"""

import json
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool

import soundfile as sf

BASE = "/opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines"
INPUT_FILE = os.path.join(BASE, "datasets/jsonl_zeyang/text_to_speech_zh_en_clean.jsonl")
OUTPUT_FILE = os.path.join(BASE, "datasets/jsonl_zeyang/text_to_speech_zh_en_final.jsonl")

MAX_DURATION_SEC = 120.0
NUM_WORKERS = 96
ASCII_RE = re.compile(r"[a-zA-Z]")


def extract_dataset_name(audio_path):
    m = re.search(r"dataset=([^/]+)", audio_path)
    return m.group(1) if m else "unknown"


def is_thchs_ocr(d):
    audio = d["target_audio"]
    return "dataset=thchs_30" in audio and bool(ASCII_RE.search(d["target_text"]))


def get_duration(path):
    try:
        info = sf.info(path)
        if info.samplerate <= 0:
            return path, -1.0
        return path, info.frames / float(info.samplerate)
    except Exception:
        return path, -1.0


def main():
    t0 = time.time()

    print("Pass 1: scanning input & collecting unique audio paths...", flush=True)
    unique_paths = set()
    total = 0
    thchs_ocr = 0
    with open(INPUT_FILE) as f:
        for i, line in enumerate(f):
            total = i + 1
            d = json.loads(line)
            if is_thchs_ocr(d):
                thchs_ocr += 1
                continue
            unique_paths.add(d["target_audio"])
    print(f"  Total input entries: {total:,}", flush=True)
    print(f"  thchs_30 OCR entries removed: {thchs_ocr:,}", flush=True)
    print(f"  Unique audio paths to probe: {len(unique_paths):,}", flush=True)
    print(f"  Pass 1 done ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nProbing durations with {NUM_WORKERS} workers...", flush=True)
    tp = time.time()
    duration = {}
    paths = list(unique_paths)
    done = 0
    with Pool(NUM_WORKERS) as pool:
        for path, dur in pool.imap_unordered(get_duration, paths, chunksize=256):
            duration[path] = dur
            done += 1
            if done % 200_000 == 0:
                print(f"  probed {done:,}/{len(paths):,} ({time.time()-tp:.1f}s)", flush=True)
    print(f"  Duration probe done ({time.time()-tp:.1f}s)", flush=True)

    n_err = sum(1 for v in duration.values() if v < 0)
    n_long = sum(1 for v in duration.values() if v > MAX_DURATION_SEC)
    print(f"  Unreadable audio paths: {n_err:,}", flush=True)
    print(f"  Paths > {MAX_DURATION_SEC:.0f}s: {n_long:,}", flush=True)

    print("\nPass 2: writing filtered output...", flush=True)
    t2 = time.time()
    written = 0
    skipped_thchs = 0
    skipped_long = 0
    skipped_err = 0
    written_lang = defaultdict(int)
    long_by_dataset = defaultdict(int)
    dur_buckets = defaultdict(int)

    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            d = json.loads(line)
            if is_thchs_ocr(d):
                skipped_thchs += 1
                continue
            audio = d["target_audio"]
            dur = duration.get(audio, -1.0)
            if dur < 0:
                skipped_err += 1
                continue
            if dur > MAX_DURATION_SEC:
                skipped_long += 1
                long_by_dataset[extract_dataset_name(audio)] += 1
                continue
            fout.write(line)
            written += 1
            written_lang[d.get("target_lang", "")] += 1
            dur_buckets[int(dur // 10) * 10] += 1

    print(f"  Pass 2 done ({time.time()-t2:.1f}s)", flush=True)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    print(f"Input entries:            {total:,}", flush=True)
    print(f"Removed thchs_30 OCR:     {skipped_thchs:,}", flush=True)
    print(f"Removed >{MAX_DURATION_SEC:.0f}s audio:      {skipped_long:,}", flush=True)
    print(f"Removed unreadable audio: {skipped_err:,}", flush=True)
    print(f"Written:                  {written:,}", flush=True)
    for lang in sorted(written_lang):
        print(f"  {lang}: {written_lang[lang]:,}", flush=True)

    if long_by_dataset:
        print(f"\n>{MAX_DURATION_SEC:.0f}s removed by dataset:", flush=True)
        for ds, c in sorted(long_by_dataset.items(), key=lambda x: -x[1]):
            print(f"  {ds}: {c:,}", flush=True)

    print(f"\nKept duration histogram (10s buckets):", flush=True)
    for b in sorted(dur_buckets):
        print(f"  {b:>4d}-{b+10:<4d}s: {dur_buckets[b]:,}", flush=True)

    print(f"\nOutput: {OUTPUT_FILE}", flush=True)
    print(f"Output size: {os.path.getsize(OUTPUT_FILE)/(1024**3):.2f} GB", flush=True)


if __name__ == "__main__":
    main()
