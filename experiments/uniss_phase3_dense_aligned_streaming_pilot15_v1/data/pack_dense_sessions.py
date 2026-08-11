#!/usr/bin/env python3
"""Tokenize and pack one dense-session manifest/part for Megatron training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
    build_session_token_sample,
    pack_session_samples,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseSession,
)
from training.simul_uniss.jsonl_index import load_index, write_index


PACK_PART_SCHEMA = "uniss_dense_aligned_streaming_pack_part_v3"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FormalReader:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.offsets = None
        self.handle = None

    def read(self, path_value: str, index: int) -> dict[str, object]:
        path = Path(path_value).resolve()
        if path != self.path:
            if self.handle is not None:
                self.handle.close()
            offsets = load_index(path)
            if offsets is None:
                raise ValueError(f"missing formal index for {path}")
            self.path = path
            self.offsets = offsets
            self.handle = path.open("rb")
        if self.offsets is None or self.handle is None:
            raise AssertionError("formal reader was not initialized")
        self.handle.seek(int(self.offsets[index]))
        return json.loads(self.handle.readline())

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def pack(args: argparse.Namespace) -> dict[str, object]:
    dense = Path(args.dense_manifest).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    dense_offsets = load_index(dense)
    if dense_offsets is None:
        raise ValueError(f"missing dense index for {dense}")
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError(f"unexpected pack marker: {marker_path}")
        if output.is_file():
            print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
            return marker
        raise FileNotFoundError(output)
    if output.exists():
        raise FileExistsError(f"refusing unmarked packed output: {output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )

    def encode(value: str) -> list[int]:
        if not value:
            return []
        return [
            int(token)
            for token in tokenizer.encode(value, add_special_tokens=False)
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    reader = FormalReader()
    started = time.time()
    try:
        def samples():
            with dense.open("rb") as handle:
                for record_index, offset in enumerate(dense_offsets):
                    handle.seek(int(offset))
                    session = DenseSession.from_dict(json.loads(handle.readline()))
                    formal = reader.read(session.source_manifest, session.source_index)
                    sample = build_session_token_sample(session, formal, encode)
                    counts["sessions"] += 1
                    counts["session_tokens"] += sample.length
                    counts["annotations"] += len(sample.annotations)
                    counts[f"direction:{session.src_lang}-{session.tgt_lang}"] += 1
                    if args.progress_interval and (record_index + 1) % args.progress_interval == 0:
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            json.dumps(
                                {
                                    "processed": record_index + 1,
                                    "sessions_per_second": (record_index + 1) / elapsed,
                                    "packed_records": counts["packed_records"],
                                }
                            ),
                            flush=True,
                        )
                    yield sample

        with temporary.open("wb") as target:
            for value in pack_session_samples(samples(), seq_length=args.seq_length):
                if value.get("schema_version") != PACK_SCHEMA:
                    raise AssertionError("dense pack schema changed")
                encoded = (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                output_offsets.append(byte_offset)
                target.write(encoded)
                digest.update(encoded)
                byte_offset += len(encoded)
                counts["packed_records"] += 1
                counts["padded_tokens"] += args.seq_length - sum(
                    int(end) - int(start) for start, end in value["sample_boundaries"]
                )
            target.flush()
            os.fsync(target.fileno())
        if counts["packed_records"] <= 0:
            raise ValueError("dense packing produced no records")
        os.replace(temporary, output)
    finally:
        reader.close()
        temporary.unlink(missing_ok=True)
    index = write_index(output, output_offsets)
    marker = {
        "schema_version": PACK_PART_SCHEMA,
        "pack_schema_version": PACK_SCHEMA,
        "status": "complete",
        "dense_manifest": str(dense),
        "dense_records": len(dense_offsets),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "seq_length": args.seq_length,
        "output": str(output),
        "output_sha256": digest.hexdigest(),
        "index": index,
        "counts": dict(counts),
        "packing_efficiency": counts["session_tokens"]
        / max(1, counts["packed_records"] * args.seq_length),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
