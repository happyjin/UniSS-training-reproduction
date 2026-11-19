#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for sharded JSONL output and directory/file input expansion.

Used by the dataset_maker scripts so that TTS / S2S / merge stages can all emit
(and later re-read) a dataset as a set of medium-sized ``*.jsonl`` shards rather
than one giant file.  Sharding lets the training preprocess read many files in
parallel, while keeping each shard large enough to stay efficient over the
network.

Contract
--------
* A ``--output /path/to/name.jsonl`` is interpreted as:
  - ``shard_max_lines == 0`` -> a single file ``/path/to/name.jsonl``
    (backward compatible with the old behaviour).
  - ``shard_max_lines  > 0`` -> a directory ``/path/to/name/`` containing
    ``name_00000.jsonl``, ``name_00001.jsonl``, ...
* An *input* to the merge step may be either a single ``.jsonl`` file or a
  directory of shards; ``resolve_input_paths`` expands both to a flat file list.
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Sequence


class ShardWriter:
    """Write JSONL lines into one file or a set of sharded files.

    Parameters
    ----------
    output_path : str
        Base path such as ``/dir/name.jsonl``.
    shard_max_lines : int
        Maximum number of lines per shard. ``<= 0`` disables sharding and writes
        a single file at ``output_path``.
    pad : int
        Zero-padding width for shard indices.
    """

    def __init__(self, output_path: str, shard_max_lines: int = 0, pad: int = 5):
        output_path = os.path.abspath(output_path)
        self.out_dir = os.path.dirname(output_path)
        stem = os.path.basename(output_path)
        if stem.endswith(".jsonl"):
            stem = stem[:-6]
        self.stem = stem
        self.shard_max_lines = shard_max_lines if shard_max_lines and shard_max_lines > 0 else 0
        self.pad = pad
        self._index = 0
        self._count = 0
        self._f = None
        os.makedirs(self.out_dir, exist_ok=True)

        if self.shard_max_lines == 0:
            # Single-file (legacy) mode.
            self._base_dir = None
            self._single = open(os.path.join(self.out_dir, self.stem + ".jsonl"), "w", encoding="utf-8")
        else:
            # Sharded mode: one sub-directory named after the dataset.
            self._base_dir = os.path.join(self.out_dir, self.stem)
            os.makedirs(self._base_dir, exist_ok=True)
            self._single = None

    def _open_next_shard(self) -> None:
        if self._f is not None:
            self._f.close()
        name = f"{self.stem}_{self._index:0{self.pad}d}.jsonl"
        self._f = open(os.path.join(self._base_dir, name), "w", encoding="utf-8")
        self._index += 1
        self._count = 0

    def write(self, line: str) -> None:
        """Write one JSONL line (without trailing newline)."""
        if self.shard_max_lines == 0:
            self._single.write(line + "\n")
            return
        if self._f is None or self._count >= self.shard_max_lines:
            self._open_next_shard()
        self._f.write(line + "\n")
        self._count += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        if self._single is not None:
            self._single.close()
            self._single = None

    def shard_paths(self) -> List[str]:
        """Return the list of produced shard files (single file in legacy mode)."""
        if self.shard_max_lines == 0:
            return [os.path.join(self.out_dir, self.stem + ".jsonl")]
        return sorted(glob.glob(os.path.join(self._base_dir, f"{self.stem}_*.jsonl")))

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resolve_input_paths(inputs: Sequence[str]) -> List[str]:
    """Expand each input (file or directory) into a sorted list of ``.jsonl`` paths.

    A directory input is expanded via ``*.jsonl`` (sorted) so that a dataset
    written as shards is read back in deterministic order.
    """
    files: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, "*.jsonl"))))
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            # Keep as-is so the caller can surface a clear "not found" error.
            files.append(inp)
    return files


def iter_valid_lines_multi(
    file_list: Sequence[str], validate_json: bool
) -> Iterable[str]:
    """Yield stripped non-empty lines from a list of ``.jsonl`` files."""
    import json
    import sys

    for path in file_list:
        with open(path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                if validate_json:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(
                            f"[WARN] Skipping malformed JSON in {path}:{line_no}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                yield line


def count_valid_lines_multi(file_list: Sequence[str], validate_json: bool) -> int:
    return sum(1 for _ in iter_valid_lines_multi(file_list, validate_json))
