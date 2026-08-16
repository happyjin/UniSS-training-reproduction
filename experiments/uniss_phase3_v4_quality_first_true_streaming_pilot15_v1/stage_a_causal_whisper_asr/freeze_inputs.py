#!/usr/bin/env python3
"""Freeze immutable Stage A source/checkpoint/map provenance without copying 40 GB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.ctc_targets import (
    UTF8ByteCTCMap,
    load_ctc_map,
)


SCHEMA = "uniss_quality_first_stage_a_source_snapshot_v2"


def sha256(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
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


def manifest_info(path: Path) -> dict[str, object]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing validated uint64 index: {path}")
    return {
        "path": str(path.resolve()),
        "records": len(offsets),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "offset_index_sha256": sha256(Path(str(path) + ".offsets.bin")),
    }


def first_record(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--ctc-map-dir", type=Path, required=True)
    parser.add_argument("--stage00-gate", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.stage00_gate.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise ValueError("Stage 00 gate did not pass")
    speaker_record = first_record(args.valid_manifest)
    speaker = [int(value) for value in speaker_record.get("bicodec_global", [])]
    if len(speaker) != 32:
        raise ValueError("fixed system speaker source must contain 32 global tokens")
    maps = {}
    for language in ("eng", "cmn"):
        path = args.ctc_map_dir / f"ctc_qwen_{language}.json"
        mapping = load_ctc_map(path)
        maps[language] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "target_kind": "utf8_byte" if isinstance(mapping, UTF8ByteCTCMap) else "qwen_compact",
            "classes_without_blank": mapping.blank_id,
            "blank_id": mapping.blank_id,
        }
    native = args.native_checkpoint.resolve()
    if native.name != "iter_0009075" or not native.is_dir():
        raise ValueError("Stage A must initialize from native Phase3 iter_0009075")
    snapshot = {
        "schema_version": SCHEMA,
        "stage00_gate": str(args.stage00_gate.resolve()),
        "stage00_gate_sha256": sha256(args.stage00_gate),
        "native_checkpoint": str(native),
        "train": manifest_info(args.train_manifest),
        "valid": manifest_info(args.valid_manifest),
        "ctc_maps": maps,
        "fixed_system_speaker": {
            "source_sample_id": str(speaker_record.get("id")),
            "future_leakage": False,
            "global_tokens": speaker,
            "policy": "one immutable cross-session prompt condition; never derived from the current utterance",
        },
    }
    atomic_json(args.output, snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
