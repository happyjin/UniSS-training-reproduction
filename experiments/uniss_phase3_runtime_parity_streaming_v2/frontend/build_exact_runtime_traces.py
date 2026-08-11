#!/usr/bin/env python3
"""Replay raw PCM through the exact deployment frontend at fixed 160 ms ticks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
    StreamingWhisperVQTeacher,
)
from web_demo.true_subsecond_pilot15_streaming_v1.causal_frontend import (
    BoundedCausalWhisperVQFrontend,
    SAMPLE_RATE,
)

from .exact_trace import TRACE_SCHEMA


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


def _pcm(path: Path) -> tuple[np.ndarray, int, str]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(values, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.reshape(-1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"runtime trace requires 16kHz PCM, found {sample_rate}")
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("runtime trace received empty or non-finite PCM")
    digest = hashlib.sha256(waveform.astype("<f4", copy=False).tobytes()).hexdigest()
    return waveform, sample_rate, digest


def build(args: argparse.Namespace) -> dict[str, object]:
    formal = Path(args.formal_manifest).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    offsets = load_index(formal)
    if offsets is None:
        raise ValueError(f"missing formal index for {formal}")
    if marker_path.is_file() and output.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value
    if output.exists():
        raise FileExistsError(f"refusing unmarked runtime trace: {output}")

    teacher = StreamingWhisperVQTeacher(
        args.whispervq_model,
        device=args.device,
        chunk_ms=args.acoustic_chunk_ms,
        right_context_ms=args.right_context_ms,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    accepted = 0
    scanned = 0
    try:
        with formal.open("rb") as source, temporary.open("wb") as target:
            for source_index in range(args.start_index, len(offsets)):
                if args.limit_accepted is not None and accepted >= args.limit_accepted:
                    break
                if args.maximum_scanned is not None and scanned >= args.maximum_scanned:
                    break
                scanned += 1
                record = _read(source, offsets[source_index])
                duration_ms = int(record["source_duration_ms"])
                if duration_ms > args.maximum_audio_ms:
                    counts["skipped_too_long"] += 1
                    continue
                try:
                    audio_path = Path(str(record["source_audio"])).resolve()
                    waveform, sample_rate, pcm_sha256 = _pcm(audio_path)
                    actual_ms = int(round(len(waveform) * 1000 / sample_rate))
                    if abs(actual_ms - duration_ms) > args.duration_tolerance_ms:
                        raise ValueError(
                            f"manifest/PCM durations differ: {duration_ms} vs {actual_ms}"
                        )
                    frontend = BoundedCausalWhisperVQFrontend(
                        teacher,
                        chunk_ms=args.acoustic_chunk_ms,
                        right_context_ms=args.right_context_ms,
                        window_ms=args.frontend_window_ms,
                    )
                    step_samples = args.ingest_tick_ms * SAMPLE_RATE // 1000
                    token_values: list[int] = []
                    commit_times: list[int] = []
                    steps: list[dict[str, object]] = []
                    for start in range(0, len(waveform), step_samples):
                        end = min(len(waveform), start + step_samples)
                        result = frontend.push(
                            waveform[start:end], is_final=end == len(waveform)
                        )
                        new_tokens = [int(value) for value in result.new_tokens]
                        token_start = len(token_values)
                        token_values.extend(new_tokens)
                        commit_times.extend([result.source_end_ms] * len(new_tokens))
                        steps.append(
                            {
                                "source_start_sample": start,
                                "source_end_sample": end,
                                "source_end_ms": result.source_end_ms,
                                "is_final": end == len(waveform),
                                "window_start_ms": result.window_start_ms,
                                "stable_end_ms": result.stable_end_ms,
                                "token_start": token_start,
                                "token_end": len(token_values),
                                "new_tokens": new_tokens,
                            }
                        )
                    if token_values != list(frontend.committed):
                        raise AssertionError("runtime trace token concatenation changed")
                    if frontend.committed_revision_violations:
                        raise ValueError(
                            "deployment frontend revised committed tokens: "
                            f"{frontend.committed_revision_violations}"
                        )
                    expected = math.ceil(len(waveform) / (SAMPLE_RATE * 0.08))
                    if len(token_values) != expected:
                        raise ValueError(
                            f"runtime trace coverage differs: {len(token_values)} != {expected}"
                        )
                    value = {
                        "schema_version": TRACE_SCHEMA,
                        "id": record["id"],
                        "source_manifest": str(formal),
                        "source_index": source_index,
                        "source_audio": str(audio_path),
                        "source_duration_ms": duration_ms,
                        "sample_rate": sample_rate,
                        "pcm_f32le_sha256": pcm_sha256,
                        "runtime_source_glm": token_values,
                        "runtime_source_glm_commit_ms": commit_times,
                        "steps": steps,
                        "ingest_tick_ms": args.ingest_tick_ms,
                        "acoustic_chunk_ms": args.acoustic_chunk_ms,
                        "right_context_ms": args.right_context_ms,
                        "frontend_window_ms": args.frontend_window_ms,
                        "committed_revision_violations": 0,
                        "maximum_frontend_buffer_ms": frontend.maximum_buffer_samples
                        * 1000
                        / SAMPLE_RATE,
                    }
                    encoded = (
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    output_offsets.append(byte_offset)
                    target.write(encoded)
                    byte_offset += len(encoded)
                    accepted += 1
                    counts["accepted"] += 1
                    counts["tokens"] += len(token_values)
                    counts[f"direction:{record['src_lang']}-{record['tgt_lang']}"] += 1
                    print(
                        json.dumps(
                            {
                                "accepted": accepted,
                                "source_index": source_index,
                                "id": record["id"],
                                "duration_ms": duration_ms,
                                "tokens": len(token_values),
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    counts["rejected"] += 1
                    if args.fail_fast:
                        raise RuntimeError(
                            f"runtime trace failed at formal row {source_index}"
                        ) from error
                    print(
                        json.dumps(
                            {
                                "source_index": source_index,
                                "id": record.get("id"),
                                "error": f"{type(error).__name__}: {error}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            target.flush()
            os.fsync(target.fileno())
        if accepted <= 0:
            raise ValueError("runtime trace builder accepted no records")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    result = {
        "schema_version": "uniss_runtime_exact_frontend_trace_build_v1",
        "status": "complete",
        "formal_manifest": str(formal),
        "output": str(output),
        "index": write_index(output, output_offsets),
        "counts": dict(counts),
        "start_index": args.start_index,
        "scanned": scanned,
        "accepted": accepted,
        "ingest_tick_ms": args.ingest_tick_ms,
        "acoustic_chunk_ms": args.acoustic_chunk_ms,
        "right_context_ms": args.right_context_ms,
        "frontend_window_ms": args.frontend_window_ms,
        "maximum_audio_ms": args.maximum_audio_ms,
        "future_pcm_visible": False,
        "requires_zero_committed_revisions": True,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-accepted", type=int)
    parser.add_argument("--maximum-scanned", type=int)
    parser.add_argument("--maximum-audio-ms", type=int, default=4_600)
    parser.add_argument("--duration-tolerance-ms", type=int, default=40)
    parser.add_argument("--ingest-tick-ms", type=int, default=160)
    parser.add_argument("--acoustic-chunk-ms", type=int, default=160)
    parser.add_argument("--right-context-ms", type=int, default=80)
    parser.add_argument("--frontend-window-ms", type=int, default=4_800)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
