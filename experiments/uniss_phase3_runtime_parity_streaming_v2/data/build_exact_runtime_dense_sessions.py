#!/usr/bin/env python3
"""Build dense sessions only from zero-revision deployment PCM traces."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions import (
    build_dense_session,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.exact_trace import (
    ExactRuntimeTraceReader,
)
from training.simul_uniss.jsonl_index import load_index, write_index


SCHEMA = "uniss_exact_runtime_dense_part_v1"


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


def _read(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def build(args: argparse.Namespace) -> dict[str, object]:
    formal_path = Path(args.formal_manifest).resolve()
    trace_path = Path(args.runtime_trace_manifest).resolve()
    speaker_formal_path = Path(
        args.speaker_formal_manifest or args.formal_manifest
    ).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    formal_offsets = load_index(formal_path)
    if formal_offsets is None:
        raise ValueError(f"missing formal index for {formal_path}")
    if marker_path.is_file() and output.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value
    if output.exists():
        raise FileExistsError(f"refusing unmarked dense output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    speaker_offsets = load_index(speaker_formal_path)
    if speaker_offsets is None:
        raise ValueError(
            f"missing fixed-speaker formal index for {speaker_formal_path}"
        )
    if not 0 <= args.speaker_source_index < len(speaker_offsets):
        raise ValueError(
            f"fixed-speaker source index is out of range: {args.speaker_source_index}"
        )
    with speaker_formal_path.open("rb") as speaker_handle:
        fixed = _read(
            speaker_handle, speaker_offsets[args.speaker_source_index]
        )
    fixed_speaker = [int(value) for value in fixed["bicodec_global"]]  # type: ignore[index]
    if len(fixed_speaker) != 32:
        raise ValueError("fixed low-latency speaker must contain 32 tokens")

    try:
        with ExactRuntimeTraceReader(trace_path) as traces, formal_path.open(
            "rb"
        ) as formal_handle, temporary.open("wb") as target:
            for trace_row in range(len(traces)):
                trace = traces.record(trace_row)
                source_index = int(trace["source_index"])
                if not 0 <= source_index < len(formal_offsets):
                    raise ValueError(f"trace source index is out of range: {source_index}")
                formal = _read(formal_handle, formal_offsets[source_index])
                tokens, times = traces.tokens_and_times(
                    trace_row, expected_id=str(formal["id"])
                )
                runtime_formal = dict(formal)
                runtime_formal["source_glm"] = tokens
                runtime_formal["source_glm_end_ms"] = times
                session = build_dense_session(
                    runtime_formal,
                    source_manifest=formal_path,
                    source_index=source_index,
                    split=args.split,
                    speaker_global=fixed_speaker,
                    low_watermark_ms=args.low_watermark_ms,
                    target_buffer_ms=args.target_buffer_ms,
                    semantic_history_tokens=args.semantic_history_tokens,
                )
                encoded = (
                    json.dumps(
                        session.to_dict(), ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode("utf-8")
                output_offsets.append(byte_offset)
                target.write(encoded)
                byte_offset += len(encoded)
                counts["sessions"] += 1
                counts["events"] += len(session.events)
                counts["writes"] += sum(
                    event.action == "WRITE" for event in session.events
                )
                counts["runtime_source_tokens"] += len(tokens)
                counts[f"direction:{session.src_lang}-{session.tgt_lang}"] += 1
            target.flush()
            os.fsync(target.fileno())
        if counts["sessions"] != len(output_offsets) or counts["sessions"] <= 0:
            raise ValueError("exact runtime dense build lost sessions")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "formal_manifest": str(formal_path),
        "runtime_trace_manifest": str(trace_path),
        "source_token_domain": "exact_deployment_pcm_trace",
        "fixed_low_latency_speaker_manifest": str(speaker_formal_path),
        "fixed_low_latency_speaker_source_index": args.speaker_source_index,
        "fixed_low_latency_speaker_sample_id": str(fixed["id"]),
        "fixed_low_latency_speaker_global": fixed_speaker,
        "output": str(output),
        "index": write_index(output, output_offsets),
        "counts": dict(counts),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--runtime-trace-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument(
        "--speaker-formal-manifest",
        help=(
            "Formal manifest that owns the one fixed low-latency speaker. "
            "Pass the train manifest for both train and validation builds."
        ),
    )
    parser.add_argument("--speaker-source-index", type=int, default=0)
    parser.add_argument("--low-watermark-ms", type=int, default=240)
    parser.add_argument("--target-buffer-ms", type=int, default=400)
    parser.add_argument("--semantic-history-tokens", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
