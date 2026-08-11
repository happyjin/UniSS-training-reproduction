#!/usr/bin/env python3
"""Build dense sessions from the existing causal-clone token domain.

This is intentionally separate from the frozen v1 builder.  The v1 builder
reads released ``source_glm`` values, while deployment observes causal
WhisperVQ clone tokens.  This builder joins the formal record with the already
completed Stage-A-v3 clone sidecar and derives deployment-clock commit times.
"""

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
from experiments.uniss_phase3_runtime_parity_streaming_v2.data.causal_sidecar import (
    CausalCloneSidecarReader,
    runtime_commit_end_times,
)
from training.simul_uniss.jsonl_index import load_index, write_index


PART_SCHEMA = "uniss_runtime_parity_dense_part_v1"


def _part_range(records: int, part_index: int, num_parts: int) -> tuple[int, int]:
    if not 0 <= part_index < num_parts:
        raise ValueError("part_index must be in [0,num_parts)")
    return records * part_index // num_parts, records * (part_index + 1) // num_parts


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


def _record(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def build_part(args: argparse.Namespace) -> dict[str, object]:
    formal_path = Path(args.formal_manifest).resolve()
    sidecar_path = Path(args.causal_sidecar_manifest).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    formal_offsets = load_index(formal_path)
    if formal_offsets is None:
        raise ValueError(f"missing formal index for {formal_path}")
    with CausalCloneSidecarReader(sidecar_path) as sidecar:
        if len(sidecar) != len(formal_offsets):
            raise ValueError(
                f"formal/causal record counts differ: {len(formal_offsets)} != {len(sidecar)}"
            )
        start, end = _part_range(len(formal_offsets), args.part_index, args.num_parts)
        if args.limit is not None:
            end = min(end, start + int(args.limit))
        if marker_path.is_file() and output.is_file():
            value = json.loads(marker_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != PART_SCHEMA:
                raise ValueError(f"unexpected existing marker: {marker_path}")
            print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
            return value
        if output.exists():
            raise FileExistsError(f"refusing unmarked output: {output}")

        with formal_path.open("rb") as handle:
            fixed = _record(handle, formal_offsets[args.speaker_source_index])
        fixed_speaker = [int(value) for value in fixed["bicodec_global"]]  # type: ignore[index]
        if len(fixed_speaker) != 32:
            raise ValueError("fixed low-latency speaker does not contain 32 tokens")

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        offsets = array("Q")
        byte_offset = 0
        counts: Counter[str] = Counter()
        started = time.time()
        try:
            with formal_path.open("rb") as source, temporary.open("wb") as target:
                for index in range(start, end):
                    try:
                        formal = _record(source, formal_offsets[index])
                        causal = sidecar.tokens(index, expected_id=str(formal["id"]))
                        duration_ms = int(formal["source_duration_ms"])
                        coverage = min(1.0, len(causal) * args.token_hop_ms / duration_ms)
                        if coverage < args.minimum_source_token_coverage:
                            raise ValueError(
                                "causal sidecar was truncated before source EOS: "
                                f"coverage={coverage:.4f}"
                            )
                        runtime_formal = dict(formal)
                        runtime_formal["source_glm"] = causal
                        runtime_formal["source_glm_end_ms"] = runtime_commit_end_times(
                            duration_ms,
                            len(causal),
                            chunk_ms=args.chunk_ms,
                            right_context_ms=args.right_context_ms,
                            token_hop_ms=args.token_hop_ms,
                        )
                        session = build_dense_session(
                            runtime_formal,
                            source_manifest=formal_path,
                            source_index=index,
                            split=args.split,
                            speaker_global=fixed_speaker,
                            low_watermark_ms=args.low_watermark_ms,
                            target_buffer_ms=args.target_buffer_ms,
                            semantic_history_tokens=args.semantic_history_tokens,
                        )
                        encoded = (
                            json.dumps(
                                session.to_dict(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        offsets.append(byte_offset)
                        target.write(encoded)
                        byte_offset += len(encoded)
                        counts["sessions"] += 1
                        counts["events"] += len(session.events)
                        counts["writes"] += sum(
                            event.action == "WRITE" for event in session.events
                        )
                        counts["causal_source_tokens"] += len(causal)
                        counts[f"direction:{session.src_lang}-{session.tgt_lang}"] += 1
                    except Exception as error:
                        counts["rejected"] += 1
                        if args.fail_fast:
                            raise RuntimeError(
                                f"runtime-parity build failed at row {index}"
                            ) from error
                        if counts["rejected"] <= args.maximum_logged_rejections:
                            print(
                                json.dumps(
                                    {
                                        "source_index": index,
                                        "error": f"{type(error).__name__}: {error}",
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                    processed = index - start + 1
                    if args.progress_interval and processed % args.progress_interval == 0:
                        print(
                            json.dumps(
                                {
                                    "part": args.part_index,
                                    "processed": processed,
                                    "accepted": counts["sessions"],
                                    "rejected": counts["rejected"],
                                    "records_per_second": processed
                                    / max(time.time() - started, 1e-6),
                                }
                            ),
                            flush=True,
                        )
                target.flush()
                os.fsync(target.fileno())
            if counts["sessions"] <= 0:
                raise ValueError("runtime-parity builder produced no sessions")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    result = {
        "schema_version": PART_SCHEMA,
        "status": "complete",
        "formal_manifest": str(formal_path),
        "causal_sidecar_manifest": str(sidecar_path),
        "source_token_domain": "stage_a_v3_chunk_causal_clone",
        "source_timing_domain": "deployment_complete_chunk_commit",
        "chunk_ms": args.chunk_ms,
        "right_context_ms": args.right_context_ms,
        "token_hop_ms": args.token_hop_ms,
        "fixed_low_latency_speaker_global": fixed_speaker,
        "minimum_source_token_coverage": args.minimum_source_token_coverage,
        "source_start": start,
        "source_end": end,
        "output": str(output),
        "index": write_index(output, offsets),
        "counts": dict(counts),
        "acceptance_rate": counts["sessions"] / max(1, end - start),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--causal-sidecar-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--speaker-source-index", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--right-context-ms", type=int, default=80)
    parser.add_argument("--token-hop-ms", type=int, default=80)
    parser.add_argument("--minimum-source-token-coverage", type=float, default=0.95)
    parser.add_argument("--low-watermark-ms", type=int, default=240)
    parser.add_argument("--target-buffer-ms", type=int, default=400)
    parser.add_argument("--semantic-history-tokens", type=int, default=200)
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument("--maximum-logged-rejections", type=int, default=20)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_part(parse_args())
