#!/usr/bin/env python3
"""Pack runtime-parity dense sessions using causal-clone source tokens."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from experiments.uniss_phase3_runtime_parity_streaming_v2.data.causal_sidecar import (
    CausalCloneSidecarReader,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.exact_trace import (
    ExactRuntimeTraceReader,
)
from training.simul_uniss.jsonl_index import load_index, write_index


PACK_PART_SCHEMA = "uniss_runtime_parity_dense_pack_part_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
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
            self.close()
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
        self.handle = None


def pack(args: argparse.Namespace) -> dict[str, object]:
    dense = Path(args.dense_manifest).resolve()
    sidecar_manifest = (
        Path(args.causal_sidecar_manifest).resolve()
        if args.causal_sidecar_manifest
        else None
    )
    trace_manifest = (
        Path(args.runtime_trace_manifest).resolve()
        if args.runtime_trace_manifest
        else None
    )
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    dense_offsets = load_index(dense)
    if dense_offsets is None:
        raise ValueError(f"missing dense index for {dense}")
    if marker_path.is_file() and output.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != PACK_PART_SCHEMA:
            raise ValueError(f"unexpected pack marker: {marker_path}")
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value
    if output.exists():
        raise FileExistsError(f"refusing unmarked packed output: {output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )

    def encode(value: str) -> list[int]:
        return (
            []
            if not value
            else [
                int(token)
                for token in tokenizer.encode(value, add_special_tokens=False)
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    formal_reader = FormalReader()
    started = time.time()
    try:
        with ExitStack() as stack:
            if trace_manifest is not None:
                trace_reader = stack.enter_context(
                    ExactRuntimeTraceReader(trace_manifest)
                )
                sidecar = None
                source_domain = "exact_deployment_pcm_trace"
            elif sidecar_manifest is not None:
                sidecar = stack.enter_context(
                    CausalCloneSidecarReader(sidecar_manifest)
                )
                trace_reader = None
                source_domain = "stage_a_v3_chunk_causal_clone"
            else:
                raise AssertionError("one runtime token source is required")

            def samples():
                with dense.open("rb") as handle:
                    for record_index, offset in enumerate(dense_offsets):
                        handle.seek(int(offset))
                        session = DenseSession.from_dict(json.loads(handle.readline()))
                        formal = formal_reader.read(
                            session.source_manifest, session.source_index
                        )
                        if trace_reader is not None:
                            causal, _ = trace_reader.tokens_and_times_for_source_index(
                                session.source_index, expected_id=session.sample_id
                            )
                        else:
                            assert sidecar is not None
                            causal = sidecar.tokens(
                                session.source_index, expected_id=session.sample_id
                            )
                        if len(causal) != session.source_glm_length:
                            raise ValueError(
                                f"dense/causal source lengths differ for {session.sample_id}"
                            )
                        runtime_formal = dict(formal)
                        runtime_formal["source_glm"] = causal
                        sample = build_session_token_sample(
                            session, runtime_formal, encode
                        )
                        counts["sessions"] += 1
                        counts["session_tokens"] += sample.length
                        counts["annotations"] += len(sample.annotations)
                        counts[f"direction:{session.src_lang}-{session.tgt_lang}"] += 1
                        if (
                            args.progress_interval
                            and (record_index + 1) % args.progress_interval == 0
                        ):
                            print(
                                json.dumps(
                                    {
                                        "processed": record_index + 1,
                                        "sessions_per_second": (record_index + 1)
                                        / max(time.time() - started, 1e-6),
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
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    output_offsets.append(byte_offset)
                    target.write(encoded)
                    digest.update(encoded)
                    byte_offset += len(encoded)
                    counts["packed_records"] += 1
                    counts["padded_tokens"] += args.seq_length - sum(
                        int(end) - int(start)
                        for start, end in value["sample_boundaries"]
                    )
                target.flush()
                os.fsync(target.fileno())
            if counts["packed_records"] <= 0:
                raise ValueError("runtime-parity packer produced no records")
            os.replace(temporary, output)
    finally:
        formal_reader.close()
        temporary.unlink(missing_ok=True)

    result = {
        "schema_version": PACK_PART_SCHEMA,
        "pack_schema_version": PACK_SCHEMA,
        "status": "complete",
        "dense_manifest": str(dense),
        "causal_sidecar_manifest": (
            str(sidecar_manifest) if sidecar_manifest is not None else None
        ),
        "runtime_trace_manifest": (
            str(trace_manifest) if trace_manifest is not None else None
        ),
        "source_token_domain": source_domain,
        "dense_records": len(dense_offsets),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "seq_length": args.seq_length,
        "output": str(output),
        "output_sha256": digest.hexdigest(),
        "index": write_index(output, output_offsets),
        "counts": dict(counts),
        "packing_efficiency": counts["session_tokens"]
        / max(1, counts["packed_records"] * args.seq_length),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-manifest", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--causal-sidecar-manifest")
    source.add_argument("--runtime-trace-manifest")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    pack(parse_args())
