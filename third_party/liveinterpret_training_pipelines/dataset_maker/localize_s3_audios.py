#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Localize an ms-swift speech jsonl whose audios point to s3://.

Pipeline:
  1. Read an input jsonl (one record per line). For each record:
     - rewrite ``audios[*]`` and ``target_audio`` from ``s3://`` to local
       paths under ``--audio-root`` (preserving the bucket/key directory
       structure).
     - fix the well-known Common-Voice path issue where ``lang=pcm`` in the
       jsonl does NOT match the real S3 layout (real layout uses
       ``lang=<src_lang>``). The src lang is extracted from the path
       segment ``cv-corpus-XX/<lang>/clips/...``.
  2. Optionally download every referenced S3 object to its local path
     (skipping files that already exist).
  3. Write the rewritten records to ``--output-jsonl``.

Designed to be safe for re-runs (download skips existing files), and to
support sub-sampling via ``--limit`` for smoke tests before going full
scale.

Examples
--------
Smoke (200 samples, download + rewrite):
  python3 localize_s3_audios.py \\
      --input  speech_to_speech_translation.jsonl \\
      --output-jsonl speech_to_speech_translation.local.smoke.jsonl \\
      --audio-root /path/to/project/datasets/speech2speech_v2/audio_local \\
      --limit 200 --download

Full mirror:
  python3 localize_s3_audios.py \\
      --input  speech_to_speech_translation.jsonl \\
      --output-jsonl speech_to_speech_translation.jsonl \\
      --audio-root /path/to/project/datasets/speech2speech_v2/audio_local \\
      --download --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from urllib.parse import urlparse


_CV_LANG_RE = re.compile(r"/cv-corpus-[^/]+/([A-Za-z][A-Za-z0-9-]{1,7})/clips/")


def parse_s3_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Return (bucket, key) for a valid s3://bucket/key URI, else None."""
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path:
        return None
    return parsed.netloc, parsed.path.lstrip("/")


def fix_lang_pcm(s3_uri: str) -> str:
    """Rewrite ``lang=pcm`` in a Common-Voice path to ``lang=<src_lang>``.

    The real S3 layout for the upstream Common-Voice mirror uses
    ``lang=<src_lang>`` (e.g. ``lang=en``), while the manifest writes a
    constant ``lang=pcm`` (referring to the audio sample format). We can
    recover the true ``src_lang`` from the deeper path segment
    ``cv-corpus-XX/<src_lang>/clips/<file>``.
    """
    if "lang=pcm" not in s3_uri:
        return s3_uri
    m = _CV_LANG_RE.search(s3_uri)
    if not m:
        return s3_uri
    src_lang = m.group(1)
    return s3_uri.replace("lang=pcm", f"lang={src_lang}", 1)


def s3_to_local(s3_uri: str, audio_root: str) -> str:
    """Map ``s3://bucket/key`` → ``<audio_root>/bucket/key``."""
    parsed = parse_s3_uri(s3_uri)
    if parsed is None:
        return s3_uri
    bucket, key = parsed
    return os.path.join(audio_root, bucket, key)


def aws_cp(s3_uri: str, local_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Download a single S3 object to ``local_path``.

    Returns (ok, message). Skips when the local file already exists with
    non-zero size.
    """
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return True, f"skip (exists): {local_path}"
    if dry_run:
        return True, f"dry-run: would download {s3_uri} -> {local_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = ["aws", "s3", "cp", s3_uri, local_path, "--only-show-errors"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, f"timeout: {s3_uri}"
    if proc.returncode != 0:
        # Cleanup partial file if any.
        if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
            try:
                os.remove(local_path)
            except OSError:
                pass
        return False, f"failed [{proc.returncode}]: {s3_uri} :: {proc.stderr.strip()[:200]}"
    return True, f"ok: {s3_uri} -> {local_path}"


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Rewrite s3:// audio paths to local paths and optionally download them."
    )
    parser.add_argument("--input", "-i", required=True, help="Input jsonl with s3:// audio paths.")
    parser.add_argument("--output-jsonl", "-o", required=True, help="Output jsonl with local paths.")
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Local root directory. Files are mirrored at <root>/<bucket>/<key>.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N records (0 = all).")
    parser.add_argument("--download", action="store_true",
                        help="If set, download referenced S3 objects in addition to rewriting paths.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --download, only print what would be downloaded.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent S3 downloads. Default: 8.")
    parser.add_argument("--skip-missing", action="store_true",
                        help="After download, drop records whose local audio still does not exist.")
    parser.add_argument("--fix-lang-pcm", dest="fix_lang_pcm", action="store_true", default=True,
                        help="Rewrite lang=pcm to the real Common-Voice src lang (default: on).")
    parser.add_argument("--no-fix-lang-pcm", dest="fix_lang_pcm", action="store_false",
                        help="Disable lang=pcm fixup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] input not found: {args.input}", file=sys.stderr)
        return 1
    if not shutil.which("aws") and args.download:
        print("[ERROR] aws CLI not found in PATH but --download is requested.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    os.makedirs(args.audio_root, exist_ok=True)

    # Pass 1: read, fix, build local-rewrite list and the unique download set.
    records = []
    download_pairs = {}  # s3_uri -> local_path (deduped)

    with open(args.input, "r", encoding="utf-8") as fin:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            if args.limit and len(records) >= args.limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {ln}: bad JSON: {e}", file=sys.stderr)
                continue

            # audios[]
            new_audios = []
            for a in row.get("audios") or []:
                if isinstance(a, str) and a.startswith("s3://"):
                    fixed = fix_lang_pcm(a) if args.fix_lang_pcm else a
                    local = s3_to_local(fixed, args.audio_root)
                    download_pairs[fixed] = local
                    new_audios.append(local)
                else:
                    new_audios.append(a)
            if "audios" in row:
                row["audios"] = new_audios

            # target_audio
            ta = row.get("target_audio")
            if isinstance(ta, str) and ta.startswith("s3://"):
                fixed = fix_lang_pcm(ta) if args.fix_lang_pcm else ta
                local = s3_to_local(fixed, args.audio_root)
                download_pairs[fixed] = local
                row["target_audio"] = local

            # reference_audios[] (TTS schema)
            ra = row.get("reference_audios")
            if isinstance(ra, list):
                new_ra = []
                for a in ra:
                    if isinstance(a, str) and a.startswith("s3://"):
                        fixed = fix_lang_pcm(a) if args.fix_lang_pcm else a
                        local = s3_to_local(fixed, args.audio_root)
                        download_pairs[fixed] = local
                        new_ra.append(local)
                    else:
                        new_ra.append(a)
                row["reference_audios"] = new_ra

            records.append(row)

    print(f"[INFO] records={len(records)}, unique S3 objects={len(download_pairs)}",
          file=sys.stderr)

    # Pass 2: download.
    failed = set()
    if args.download:
        print(f"[INFO] downloading with {args.workers} workers ...", file=sys.stderr)
        n_ok = 0
        n_fail = 0
        n_done = 0
        items = list(download_pairs.items())
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(aws_cp, s3, local, args.dry_run): (s3, local)
                    for s3, local in items}
            for fut in as_completed(futs):
                s3, local = futs[fut]
                ok, msg = fut.result()
                n_done += 1
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failed.add(local)
                    print(f"[WARN] {msg}", file=sys.stderr)
                if n_done % 200 == 0 or n_done == len(items):
                    print(f"[INFO] download progress {n_done}/{len(items)} "
                          f"ok={n_ok} fail={n_fail}", file=sys.stderr)

    # Pass 3: write output jsonl, optionally drop rows with missing audio.
    n_written = 0
    n_dropped = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for row in records:
            if args.skip_missing and args.download:
                paths = list(row.get("audios") or [])
                if isinstance(row.get("target_audio"), str):
                    paths.append(row["target_audio"])
                paths.extend(row.get("reference_audios") or [])
                missing = [p for p in paths
                           if isinstance(p, str)
                           and (p in failed or not os.path.isfile(p))]
                if missing:
                    n_dropped += 1
                    continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DONE] wrote {n_written} records, dropped {n_dropped} -> {args.output_jsonl}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
