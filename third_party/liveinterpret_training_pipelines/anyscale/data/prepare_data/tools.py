#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_data plumbing CLI used by the shell orchestrator (worker.sh).

These are the operations that are awkward or unsafe to do in bash but contain no
FSx-specific logic (that lives in ``prewarm.py``):

  find-inputs     Enumerate the parquet inputs of a dataset by importing the
                  dataset_maker maker modules (same discovery the makers use).
  concat          Append per-batch part files into the final, shard-aware
                  output (single file "a"-append, or rolling new shard).
  manifest-diff   Print the inputs NOT yet recorded in the incremental manifest
                  (current inputs minus processed inputs).
  manifest-save   Atomically union the given inputs into the incremental
                  manifest (call AFTER a successful append).
  ready           Exit 0 iff the output path already has content (dir with a
                  non-empty jsonl, or a non-empty file) — used by bash `if`.
  count-lines     Print the number of jsonl lines in an output path.

All commands are stateless and read/write only their explicit arguments.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

# Make the repo (and its dataset_maker package) importable so we can import the
# maker modules' find_input_parquets and shardio (bare import inside makers).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_DATASET_MAKER = os.path.join(REPO_ROOT, "dataset_maker")
if _DATASET_MAKER not in sys.path:
    sys.path.insert(0, _DATASET_MAKER)

import shardio  # noqa: E402

from dataset_maker.make_text_to_speech import find_input_parquets as _find_tts
from dataset_maker.make_speech_to_speech_translation import (
    find_input_parquets as _find_s2s,
)


# ---------------------------------------------------------------------------
# output state helpers (mirror the previous run_prepare.py)
# ---------------------------------------------------------------------------
def output_ready(p: str) -> bool:
    if os.path.isdir(p):
        for f in glob.glob(os.path.join(p, "*.jsonl")):
            if os.path.isfile(f) and os.path.getsize(f) > 0:
                return True
        return False
    return os.path.isfile(p) and os.path.getsize(p) > 0


def total_lines(p: str) -> int:
    if os.path.isdir(p):
        n = 0
        for f in glob.glob(os.path.join(p, "*.jsonl")):
            if os.path.isfile(f):
                n += sum(1 for _ in open(f, "r", encoding="utf-8"))
        return n
    if os.path.isfile(p):
        return sum(1 for _ in open(p, "r", encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# append parts into the final output (shard-aware)
# ---------------------------------------------------------------------------
def append_parts(parts: list[str], out_base: str, shard_max_lines: int) -> int:
    if shard_max_lines and shard_max_lines > 0:
        return _append_parts_sharded(parts, out_base, shard_max_lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    n = 0
    with open(out_base, "a", encoding="utf-8") as fout:
        for p in parts:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                with open(p, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        n += 1
                os.remove(p)
            elif os.path.isfile(p):
                os.remove(p)
    return n


def _append_parts_sharded(parts: list[str], out_base: str, shard_max_lines: int) -> int:
    os.makedirs(out_base, exist_ok=True)
    stem = os.path.basename(out_base)
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    existing = sorted(glob.glob(os.path.join(out_base, f"{stem}_*.jsonl")))
    idx = 0
    cur_path = None
    cur_count = 0
    if existing:
        m = __import__("re").search(r"_(\d+)\.jsonl$", os.path.basename(existing[-1]))
        idx = int(m.group(1)) + 1 if m else len(existing)
        cur_path = existing[-1]
        cur_count = sum(1 for _ in open(cur_path, "r", encoding="utf-8"))
    n = 0
    fout = None
    try:
        for p in parts:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                with open(p, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if fout is None or cur_count >= shard_max_lines:
                            if fout is not None:
                                fout.close()
                            cur_path = os.path.join(out_base, f"{stem}_{idx:05d}.jsonl")
                            fout = open(cur_path, "w", encoding="utf-8")
                            cur_count = 0
                            idx += 1
                        fout.write(line)
                        cur_count += 1
                        n += 1
                os.remove(p)
            elif os.path.isfile(p):
                os.remove(p)
    finally:
        if fout is not None:
            fout.close()
    return n


# ---------------------------------------------------------------------------
# incremental manifest
# ---------------------------------------------------------------------------
def _load_manifest(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("inputs", []))
    except Exception:  # noqa: BLE001
        return set()


def _save_manifest(path: str, inputs_set: set) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"inputs": sorted(inputs_set)}, f)
    os.replace(tmp, path)


def _read_lines(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_find_inputs(args: argparse.Namespace) -> int:
    fn = _find_tts if args.kind == "tts" else _find_s2s
    for p in fn(args.input_dir):
        print(os.path.abspath(p))
    return 0


def _cmd_concat(args: argparse.Namespace) -> int:
    parts = [p for p in args.parts if p]
    if not parts:
        return 0
    append_parts(parts, args.out, args.shard_max_lines)
    return 0


def _cmd_manifest_diff(args: argparse.Namespace) -> int:
    processed = _load_manifest(args.manifest)
    current = set(_read_lines(args.inputs_file))
    for p in sorted(current - processed):
        print(p)
    return 0


def _cmd_manifest_save(args: argparse.Namespace) -> int:
    processed = _load_manifest(args.manifest)
    current = set(_read_lines(args.inputs_file))
    _save_manifest(args.manifest, processed | current)
    return 0


def _cmd_ready(args: argparse.Namespace) -> int:
    return 0 if output_ready(args.path) else 1


def _cmd_count_lines(args: argparse.Namespace) -> int:
    print(total_lines(args.path))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="prepare_data plumbing CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fi = sub.add_parser("find-inputs", help="print dataset parquet inputs (one/line)")
    fi.add_argument("--kind", required=True, choices=["tts", "s2s"])
    fi.add_argument("--input-dir", required=True)
    fi.set_defaults(func=_cmd_find_inputs)

    cc = sub.add_parser("concat", help="append part files into the final output")
    cc.add_argument("--out", required=True)
    cc.add_argument("--shard-max-lines", type=int, default=0)
    cc.add_argument("parts", nargs="+", help="part files to append")
    cc.set_defaults(func=_cmd_concat)

    md = sub.add_parser("manifest-diff", help="print inputs not yet in the manifest")
    md.add_argument("--manifest", required=True)
    md.add_argument("--inputs-file", required=True)
    md.set_defaults(func=_cmd_manifest_diff)

    ms = sub.add_parser("manifest-save", help="union inputs into the manifest (atomic)")
    ms.add_argument("--manifest", required=True)
    ms.add_argument("--inputs-file", required=True)
    ms.set_defaults(func=_cmd_manifest_save)

    rd = sub.add_parser("ready", help="exit 0 iff output path has content")
    rd.add_argument("--path", required=True)
    rd.set_defaults(func=_cmd_ready)

    cl = sub.add_parser("count-lines", help="print jsonl line count of an output path")
    cl.add_argument("--path", required=True)
    cl.set_defaults(func=_cmd_count_lines)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
