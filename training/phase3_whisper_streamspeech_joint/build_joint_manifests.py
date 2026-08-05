#!/usr/bin/env python3
"""Build immutable joint manifests with Phase3-Qwen CTC token positions.

The builder never modifies its source Stage-A manifests or audio.  A formal
run should pass full198 train Stage-A manifests via ``--train-source`` and the
official UniST dev Stage-A manifest via ``--valid-source``.  If no validation
source is supplied, a deterministic hash holdout is available for pilot runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase3_whisper_streamspeech_joint.tokenizer_maps import build_compact_map
from training.simul_uniss.jsonl_index import write_index


SCHEMA = "uniss_phase3_whisper_streamspeech_joint_record_v1"
REQUIRED = (
    "id",
    "src_lang",
    "tgt_lang",
    "transcription",
    "translation",
    "source_glm",
    "target_bicodec",
    "bicodec_global",
    "source_audio",
)


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_jsonl(paths: Iterable[str | Path]):
    for path_value in paths:
        path = Path(path_value).resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected object at {path}:{line_number}")
                yield path, line_number, value


def _is_validation(identifier: str, per_mille: int) -> bool:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 1000 < per_mille


def _validate_source(path: Path, line_number: int, value: dict[str, object], check_audio: bool) -> None:
    missing = [name for name in REQUIRED if name not in value]
    if missing:
        raise KeyError(f"{path}:{line_number} missing {missing}")
    if (str(value["src_lang"]), str(value["tgt_lang"])) not in {
        ("eng", "cmn"),
        ("cmn", "eng"),
    }:
        raise ValueError(f"unsupported direction at {path}:{line_number}")
    if not value["source_glm"] or not value["target_bicodec"]:
        raise ValueError(f"empty speech tokens at {path}:{line_number}")
    if len(value["bicodec_global"]) != 32:  # type: ignore[arg-type]
        raise ValueError(f"bicodec_global must contain 32 tokens at {path}:{line_number}")
    if check_audio and not Path(str(value["source_audio"])).is_file():
        raise FileNotFoundError(str(value["source_audio"]))


def _record(value: dict[str, object], tokenizer: Tokenizer, split: str) -> dict[str, object]:
    source_ids = [int(token) for token in tokenizer.encode(str(value["transcription"]), add_special_tokens=False)]
    target_ids = [int(token) for token in tokenizer.encode(str(value["translation"]), add_special_tokens=False)]
    if not source_ids or not target_ids:
        raise ValueError(f"empty Qwen text tokenization for {value['id']}")
    return {
        "schema_version": SCHEMA,
        "split": split,
        "id": str(value["id"]),
        "src_lang": str(value["src_lang"]),
        "tgt_lang": str(value["tgt_lang"]),
        "transcription": str(value["transcription"]),
        "translation": str(value["translation"]),
        "source_qwen_ids": source_ids,
        "target_qwen_ids": target_ids,
        "source_glm": [int(token) for token in value["source_glm"]],  # type: ignore[union-attr]
        "target_bicodec": [int(token) for token in value["target_bicodec"]],  # type: ignore[union-attr]
        "bicodec_global": [int(token) for token in value["bicodec_global"]],  # type: ignore[union-attr]
        "source_audio": str(Path(str(value["source_audio"])).resolve()),
        "source_duration_ms": int(value.get("source_duration_ms", 0)),
        "source_manifest": str(value.get("source_parquet", "")),
        "source_row_index": int(value.get("source_row_index", -1)),
    }


class _ManifestWriter:
    def __init__(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        self.temporary = Path(name)
        self.handle = os.fdopen(descriptor, "wb")
        self.offsets = array("Q")
        self.offset = 0

    def write(self, value: dict[str, object]) -> None:
        encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.offsets.append(self.offset)
        self.handle.write(encoded)
        self.offset += len(encoded)

    def finish(self) -> dict[str, object]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary, self.destination)
        return write_index(self.destination, self.offsets)

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.temporary.unlink(missing_ok=True)


def build_manifests(
    *,
    train_sources: list[str | Path],
    output_dir: str | Path,
    tokenizer: Tokenizer,
    valid_sources: list[str | Path] | None = None,
    validation_per_mille: int = 10,
    limit: int | None = None,
    check_audio: bool = True,
) -> dict[str, object]:
    if not train_sources:
        raise ValueError("at least one train source is required")
    if not 0 <= validation_per_mille < 1000:
        raise ValueError("validation_per_mille must be in [0,1000)")
    if valid_sources and validation_per_mille:
        # Explicit official validation always wins; avoid accidental train loss.
        validation_per_mille = 0
    root = Path(output_dir).resolve()
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

    def consume(paths: list[str | Path], forced_split: str | None) -> None:
        for source_path in paths:
            source_records = 0
            for path, line_number, value in _iter_jsonl([source_path]):
                if limit is not None and source_records >= limit:
                    break
                source_records += 1
                counts[f"input:{forced_split or 'hash'}"] += 1
                _validate_source(path, line_number, value, check_audio)
                split = forced_split
                if split is None:
                    split = "valid" if _is_validation(str(value["id"]), validation_per_mille) else "train"
                item = _record(value, tokenizer, split)
                record_index = counts[f"written:{split}"]
                writers[split].write(item)
                vocab[str(item["src_lang"])].update(item["source_qwen_ids"])  # type: ignore[arg-type]
                vocab[str(item["tgt_lang"])].update(item["target_qwen_ids"])  # type: ignore[arg-type]
                counts[f"written:{split}"] += 1
                direction = f"{item['src_lang']}->{item['tgt_lang']}"
                counts[f"direction:{split}:{direction}"] += 1
                direction_indices[(split, direction)].append(record_index)

    try:
        consume(train_sources, None if not valid_sources else "train")
        if valid_sources:
            consume(valid_sources, "valid")
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
        "train_sources": [str(Path(path).resolve()) for path in train_sources],
        "valid_sources": [] if not valid_sources else [str(Path(path).resolve()) for path in valid_sources],
        "validation_per_mille": validation_per_mille,
        "counts": dict(sorted(counts.items())),
        "indices": indices,
        "tokenizer_maps": maps,
        "direction_indices": direction_files,
    }
    _atomic_json(root / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", action="append", required=True)
    parser.add_argument("--valid-source", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase3-model", required=True)
    parser.add_argument("--validation-per-mille", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-audio-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    summary = build_manifests(
        train_sources=args.train_source,
        valid_sources=args.valid_source,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        validation_per_mille=args.validation_per_mille,
        limit=args.limit,
        check_audio=not args.skip_audio_check,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
