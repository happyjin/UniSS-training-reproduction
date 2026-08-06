#!/usr/bin/env python3
"""Build joint manifests in parallel across immutable Stage-A shard files.

This module is deliberately separate from ``build_joint_manifests`` so the
existing smoke and reproduction paths keep their original single-process
behavior.  Each worker transforms one source shard into private temporary
parts.  The parent merges those parts in argument order, preserving the exact
record order and JSON encoding of the reference builder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase3_whisper_streamspeech_joint.build_joint_manifests import (
    _ManifestWriter,
    _atomic_json,
    _is_validation,
    _iter_jsonl,
    _record,
    _validate_source,
)
from training.phase3_whisper_streamspeech_joint.tokenizer_maps import build_compact_map


_WORKER_TOKENIZER: Any = None
_WORKER_CHECK_AUDIO = True


def _worker_init(phase3_model: str, check_audio: bool) -> None:
    global _WORKER_CHECK_AUDIO, _WORKER_TOKENIZER
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        phase3_model,
        local_files_only=True,
    )
    _WORKER_CHECK_AUDIO = check_audio


def _write_source_part(task: tuple[int, str, str | None, int, int | None, str]) -> dict[str, Any]:
    task_index, source_value, forced_split, validation_per_mille, limit, parts_root_value = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("parallel manifest worker tokenizer is not initialized")

    source = Path(source_value).resolve()
    parts_root = Path(parts_root_value)
    prefix = parts_root / f"part-{task_index:06d}"
    paths = {split: Path(f"{prefix}.{split}.jsonl") for split in ("train", "valid")}
    handles = {split: path.open("wb") for split, path in paths.items()}
    offsets = {split: array("Q") for split in ("train", "valid")}
    byte_offsets = {split: 0 for split in ("train", "valid")}
    vocab: dict[str, set[int]] = {"eng": set(), "cmn": set()}
    counts: Counter[str] = Counter()
    direction_indices = {
        (split, direction): array("Q")
        for split in ("train", "valid")
        for direction in ("eng->cmn", "cmn->eng")
    }

    try:
        source_records = 0
        for path, line_number, value in _iter_jsonl([source]):
            if limit is not None and source_records >= limit:
                break
            source_records += 1
            counts[f"input:{forced_split or 'hash'}"] += 1
            _validate_source(path, line_number, value, _WORKER_CHECK_AUDIO)
            split = forced_split
            if split is None:
                split = (
                    "valid"
                    if _is_validation(str(value["id"]), validation_per_mille)
                    else "train"
                )
            item = _record(value, _WORKER_TOKENIZER, split)
            encoded = (
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            record_index = counts[f"written:{split}"]
            offsets[split].append(byte_offsets[split])
            handles[split].write(encoded)
            byte_offsets[split] += len(encoded)
            vocab[str(item["src_lang"])].update(item["source_qwen_ids"])
            vocab[str(item["tgt_lang"])].update(item["target_qwen_ids"])
            counts[f"written:{split}"] += 1
            direction = f"{item['src_lang']}->{item['tgt_lang']}"
            counts[f"direction:{split}:{direction}"] += 1
            direction_indices[(split, direction)].append(record_index)

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise

    return {
        "task_index": task_index,
        "source": str(source),
        "paths": {split: str(path) for split, path in paths.items()},
        "offsets": offsets,
        "vocab": {language: sorted(ids) for language, ids in vocab.items()},
        "counts": dict(counts),
        "direction_indices": {
            f"{split}:{direction}": values
            for (split, direction), values in direction_indices.items()
        },
    }


def _append_part(writer: _ManifestWriter, path: Path, offsets: array) -> None:
    base = writer.offset
    writer.offsets.extend(base + int(offset) for offset in offsets)
    with path.open("rb") as source:
        shutil.copyfileobj(source, writer.handle, length=16 * 1024 * 1024)
    writer.offset += path.stat().st_size


def build_manifests_parallel(
    *,
    train_sources: list[str | Path],
    valid_sources: list[str | Path] | None,
    output_dir: str | Path,
    phase3_model: str | Path,
    workers: int,
    validation_per_mille: int = 10,
    limit: int | None = None,
    check_audio: bool = True,
) -> dict[str, object]:
    if not train_sources:
        raise ValueError("at least one train source is required")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 0 <= validation_per_mille < 1000:
        raise ValueError("validation_per_mille must be in [0,1000)")
    if valid_sources and validation_per_mille:
        validation_per_mille = 0

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parts_root = Path(tempfile.mkdtemp(prefix=".parallel_joint_parts.", dir=root))
    writers = {
        split: _ManifestWriter(root / f"joint_{split}.jsonl")
        for split in ("train", "valid")
    }
    vocab: dict[str, set[int]] = {"eng": set(), "cmn": set()}
    counts: Counter[str] = Counter()
    direction_indices = {
        (split, direction): array("Q")
        for split in ("train", "valid")
        for direction in ("eng->cmn", "cmn->eng")
    }

    tasks: list[tuple[int, str, str | None, int, int | None, str]] = []
    forced_train = None if not valid_sources else "train"
    for source in train_sources:
        tasks.append(
            (
                len(tasks),
                str(Path(source).resolve()),
                forced_train,
                validation_per_mille,
                limit,
                str(parts_root),
            )
        )
    if valid_sources:
        for source in valid_sources:
            tasks.append(
                (
                    len(tasks),
                    str(Path(source).resolve()),
                    "valid",
                    validation_per_mille,
                    limit,
                    str(parts_root),
                )
            )

    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_worker_init,
            initargs=(str(Path(phase3_model).resolve()), check_audio),
        ) as executor:
            for expected_index, result in enumerate(executor.map(_write_source_part, tasks)):
                if int(result["task_index"]) != expected_index:
                    raise RuntimeError("parallel manifest results arrived out of order")
                split_bases = {
                    split: counts[f"written:{split}"] for split in ("train", "valid")
                }
                for split in ("train", "valid"):
                    part_path = Path(result["paths"][split])
                    _append_part(writers[split], part_path, result["offsets"][split])
                    part_path.unlink()
                for language in ("eng", "cmn"):
                    vocab[language].update(result["vocab"][language])
                for key, value in result["counts"].items():
                    counts[key] += int(value)
                for split in ("train", "valid"):
                    for direction in ("eng->cmn", "cmn->eng"):
                        values = result["direction_indices"][f"{split}:{direction}"]
                        base = split_bases[split]
                        direction_indices[(split, direction)].extend(
                            base + int(index) for index in values
                        )

        if not counts["written:train"] or not counts["written:valid"]:
            raise ValueError(f"both train and valid must be non-empty: {dict(counts)}")
        indices = {split: writer.finish() for split, writer in writers.items()}
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    maps = {}
    maps_dir = root / "tokenizer_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for language, ids in vocab.items():
        mapping = build_compact_map(language, [ids])
        path = maps_dir / f"ctc_qwen_{language}.json"
        mapping.save(path)
        maps[language] = {
            "path": str(path),
            "vocabulary": len(mapping.compact_to_qwen),
            "blank_id": mapping.blank_id,
        }

    direction_files = {}
    indices_dir = root / "direction_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    for (split, direction), values in direction_indices.items():
        path = indices_dir / f"{split}_{direction.replace('->', '_to_')}.npy"
        np.save(path, np.asarray(values, dtype=np.uint64), allow_pickle=False)
        direction_files[f"{split}:{direction}"] = str(path)

    summary = {
        "schema_version": "uniss_phase3_whisper_streamspeech_joint_manifest_summary_v1",
        "status": "complete",
        "builder": "parallel_order_preserving_v1",
        "workers": min(workers, len(tasks)),
        "train_sources": [str(Path(path).resolve()) for path in train_sources],
        "valid_sources": []
        if not valid_sources
        else [str(Path(path).resolve()) for path in valid_sources],
        "validation_per_mille": validation_per_mille,
        "counts": dict(sorted(counts.items())),
        "indices": indices,
        "tokenizer_maps": maps,
        "direction_indices": direction_files,
    }
    _atomic_json(root / "manifest_summary.json", summary)
    parts_root.rmdir()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", action="append", required=True)
    parser.add_argument("--valid-source", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase3-model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--validation-per-mille", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-audio-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifests_parallel(
        train_sources=args.train_source,
        valid_sources=args.valid_source,
        output_dir=args.output_dir,
        phase3_model=args.phase3_model,
        workers=args.workers,
        validation_per_mille=args.validation_per_mille,
        limit=args.limit,
        check_audio=not args.skip_audio_check,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
