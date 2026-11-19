#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prewarm every audio referenced by a configured task jsonl into the FSx cache.

Invoked (once, on a single worker node) after the main prepare_data pipeline
finishes. Reads the jsonl (single file or shard directory), extracts every
referenced audio path, and ``lfs hsm_restore``s them into the FSx SSD cache so
downstream training jobs hit the cache instead of cold HSM recalls.

Audio-path extraction reuses ``prewarm.extract_audio_paths_from_jsonl`` (the same
key logic as ``audit_audio_sizes``). When ``lfs`` is unavailable (container not
privileged / lustre-client missing) the warm degrades to a no-op with a warning.

Single-node coordination is the caller's responsibility (``worker.sh`` uses an
atomic marker on the shared ``FSX_MOUNT`` so only one node actually runs this).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# dataset_maker is on sys.path so shardio (bare import) resolves, mirroring
# run_prepare.py.
if os.path.join(REPO_ROOT, "dataset_maker") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "dataset_maker"))

from prewarm import (  # noqa: E402
    extract_audio_paths_from_jsonl,
    warm_paths,
)


def _resolve_one(path: str) -> Optional[str]:
    """Resolve a jsonl target to a real file or shard directory.

    - existing directory => shard directory (preferred);
    - existing file => used as-is (covers explicit ``*.jsonl`` paths);
    - otherwise a bare base name => ``<base>.jsonl`` if that exists, else None.
    Lets callers pass a logical base (e.g. ``OUTPUT_DIR/bridge_alignment_joint``)
    without knowing whether the pipeline emitted shards or a single file, so the
    caller never needs to branch on ``SHARD_MAX_LINES``.
    """
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return path
    alt = path + ".jsonl"
    if os.path.isfile(alt):
        return alt
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prewarm all audio referenced by a task jsonl into FSx."
    )
    ap.add_argument(
        "--jsonl", action="append", dest="jsonl", default=[],
        help="task jsonl to prewarm (single file or shard directory). May be "
             "given multiple times. A bare base name (no .jsonl, not a dir) is "
             "resolved as a shard directory if it exists, else '<base>.jsonl'.",
    )
    ap.add_argument("--parallel", type=int, default=16,
                    help="concurrent lfs hsm_restore calls.")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="seconds between lfs hsm_action polls.")
    ap.add_argument("--timeout", type=int, default=7200,
                    help="max seconds to wait for one warm batch.")
    ap.add_argument(
        "--chunk", type=int, default=0,
        help="warm in batches of N paths (0 = single pass over the full list). "
             "Batching keeps the poll set small for very large jsonls.",
    )
    args = ap.parse_args()

    if not args.jsonl:
        print("[prewarm_task_jsonl] no --jsonl given; nothing to do.", flush=True)
        return 0

    resolved = []
    for j in args.jsonl:
        r = _resolve_one(j)
        if r is None:
            print(f"[prewarm_task_jsonl] WARNING: not found (nor '<base>.jsonl'): {j}",
                  file=sys.stderr, flush=True)
            continue
        resolved.append(r)
    if not resolved:
        print("[prewarm_task_jsonl] no valid jsonl target found; nothing to do.",
              flush=True)
        return 0

    paths = extract_audio_paths_from_jsonl(resolved)
    if not paths:
        print(f"[prewarm_task_jsonl] no audio paths found in {resolved}; "
              f"nothing to do.", flush=True)
        return 0

    print(f"[prewarm_task_jsonl] {len(paths)} unique audio paths from {resolved}",
          flush=True)

    if args.chunk and args.chunk > 0:
        total = 0
        for i in range(0, len(paths), args.chunk):
            batch = paths[i:i + args.chunk]
            total += warm_paths(batch, args.parallel, args.poll_interval, args.timeout)
        print(f"[prewarm_task_jsonl] warmed {total}/{len(paths)} files", flush=True)
    else:
        n = warm_paths(paths, args.parallel, args.poll_interval, args.timeout)
        print(f"[prewarm_task_jsonl] warmed {n}/{len(paths)} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
