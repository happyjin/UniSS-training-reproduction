#!/usr/bin/env python3
"""Derive compact 160ms dense READ/WRITE sessions from formal A4--A8 records."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseEvent,
    DenseSession,
    SCHEMA_VERSION,
    TICK_MS,
    visible_prefix_length,
)
from training.simul_uniss.jsonl_index import load_index, write_index


PART_SCHEMA = "uniss_dense_aligned_streaming_part_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_at(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def _target_word_spans(
    target_text: str, target_words: Sequence[Mapping[str, object]]
) -> list[tuple[int, int]]:
    """Locate formal lexical words without normalizing the released translation."""

    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, word in enumerate(target_words):
        token = str(word.get("text", word.get("word", "")))
        if not token:
            raise ValueError(f"empty target word at index {index}")
        start = target_text.find(token, cursor)
        if start < 0:
            lowered_start = target_text.casefold().find(token.casefold(), cursor)
            if lowered_start >= 0:
                start = lowered_start
        if start < 0:
            raise ValueError(
                f"target word {token!r} at index {index} was not found after character {cursor}"
            )
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    if not spans:
        raise ValueError("target_words is empty")
    return spans


def _exact_text_deltas(
    target_text: str,
    target_words: Sequence[Mapping[str, object]],
    micro_events: Sequence[Mapping[str, object]],
) -> list[str]:
    """Assign every original character exactly once to a lexical Micro-WRITE."""

    spans = _target_word_spans(target_text, target_words)
    character_cursor = 0
    word_cursor = 0
    deltas: list[str] = []
    for index, event in enumerate(micro_events):
        word_start = int(event["target_word_start"])
        word_end = int(event["target_word_end"])
        semantic_continuation = bool(event.get("semantic_continuation", False))
        has_lexical_delta = bool(str(event.get("text", ""))) and not semantic_continuation
        if not has_lexical_delta:
            deltas.append("")
            continue
        if word_start != word_cursor:
            raise ValueError(
                f"Micro-WRITE {index} starts at word {word_start}, expected {word_cursor}"
            )
        if not word_start < word_end <= len(spans):
            raise ValueError(f"Micro-WRITE {index} has an invalid target word span")
        character_end = spans[word_end][0] if word_end < len(spans) else len(target_text)
        if character_end < character_cursor:
            raise ValueError("target character cursor moved backwards")
        deltas.append(target_text[character_cursor:character_end])
        character_cursor = character_end
        word_cursor = word_end
    if word_cursor != len(spans):
        raise ValueError(
            f"Micro-WRITEs cover {word_cursor}/{len(spans)} target lexical words"
        )
    if character_cursor != len(target_text):
        raise ValueError("Micro-WRITEs do not consume the complete target text")
    if "".join(deltas) != target_text:
        raise AssertionError("exact target text delta reconstruction failed")
    return deltas


def _audio_boundary(semantic: int, semantic_count: int, duration_ms: int) -> int:
    return min(
        duration_ms,
        max(0, int(round(int(semantic) * int(duration_ms) / int(semantic_count)))),
    )


def build_dense_session(
    record: Mapping[str, object],
    *,
    source_manifest: Path,
    source_index: int,
    split: str,
    speaker_global: Sequence[int],
    low_watermark_ms: int = 240,
    target_buffer_ms: int = 400,
    semantic_history_tokens: int = 200,
) -> DenseSession:
    """Build one ordered session; session events are never independently shuffled."""

    if not bool(record.get("formal_a68_pass")):
        raise ValueError("dense sessions require formal_a68_pass records")
    source_glm = [int(value) for value in record["source_glm"]]  # type: ignore[index]
    source_glm_end_ms = [int(value) for value in record["source_glm_end_ms"]]  # type: ignore[index]
    target_bicodec = [int(value) for value in record["target_bicodec"]]  # type: ignore[index]
    target_words = [dict(value) for value in record["target_words"]]  # type: ignore[index]
    micro = [dict(value) for value in record["micro_write_events"]]  # type: ignore[index]
    if len(source_glm) != len(source_glm_end_ms):
        raise ValueError("source_glm and source_glm_end_ms lengths differ")
    if not source_glm or not target_bicodec or not micro:
        raise ValueError("formal record has an empty source, target, or Micro-WRITE sequence")
    for left, right in zip(micro, micro[1:]):
        if int(left["semantic_end"]) != int(right["semantic_start"]):
            raise ValueError("formal Micro-WRITEs have a semantic gap or overlap")
    if int(micro[0]["semantic_start"]) != 0:
        raise ValueError("formal semantic sequence does not start at zero")
    if int(micro[-1]["semantic_end"]) != len(target_bicodec):
        raise ValueError("formal semantic sequence does not cover the full target")

    target_text = str(record["translation"])
    deltas = _exact_text_deltas(target_text, target_words, micro)
    source_duration_ms = int(record["source_duration_ms"])
    target_duration_ms = int(record["target_duration_ms"])
    playback_buffer_ms = 0
    micro_cursor = 0
    events: list[DenseEvent] = []
    maximum_ticks = (
        math.ceil(source_duration_ms / TICK_MS) + len(micro) * 4 + 32
    )
    for tick_index in range(maximum_ticks):
        wall_time_ms = (tick_index + 1) * TICK_MS
        source_end_ms = min(wall_time_ms, source_duration_ms)
        source_finished = wall_time_ms >= source_duration_ms
        visible = (
            len(source_glm)
            if source_finished
            else visible_prefix_length(source_glm_end_ms, source_end_ms)
        )
        buffer_before = max(0, playback_buffer_ms - TICK_MS)

        safe_pending = 0
        for candidate_index in range(micro_cursor, len(micro)):
            candidate = micro[candidate_index]
            safe = source_finished or int(candidate["earliest_safe_ms"]) <= source_end_ms
            if not safe:
                break
            safe_pending += 1

        # The final target fragment is delayed until source EOS.  This makes
        # the unique FINAL WRITE both causally complete and the last event.
        if micro_cursor == len(micro) - 1 and not source_finished:
            safe_pending = 0

        should_write = (
            micro_cursor < len(micro)
            and safe_pending > 0
            and (buffer_before < low_watermark_ms or source_finished)
        )
        if should_write:
            item = micro[micro_cursor]
            semantic_start = int(item["semantic_start"])
            semantic_end = int(item["semantic_end"])
            audio_start = _audio_boundary(
                semantic_start, len(target_bicodec), target_duration_ms
            )
            audio_end = _audio_boundary(
                semantic_end, len(target_bicodec), target_duration_ms
            )
            playback_after = buffer_before + max(1, audio_end - audio_start)
            final_write = micro_cursor == len(micro) - 1
            event = DenseEvent(
                event_index=tick_index,
                wall_time_ms=wall_time_ms,
                source_end_ms=source_end_ms,
                visible_source_token_end=visible,
                action="WRITE",
                playback_buffer_before_ms=buffer_before,
                playback_buffer_after_ms=playback_after,
                support_bucket=min(safe_pending, 4),
                safe_pending_count=safe_pending,
                text_delta=deltas[micro_cursor],
                target_word_start=int(item["target_word_start"]),
                target_word_end=int(item["target_word_end"]),
                semantic_start=semantic_start,
                semantic_end=semantic_end,
                target_audio_start_ms=audio_start,
                target_audio_end_ms=audio_end,
                earliest_safe_ms=int(item["earliest_safe_ms"]),
                final_write=final_write,
                source_finished=source_finished,
            )
            micro_cursor += 1
        else:
            playback_after = buffer_before
            event = DenseEvent(
                event_index=tick_index,
                wall_time_ms=wall_time_ms,
                source_end_ms=source_end_ms,
                visible_source_token_end=visible,
                action="READ",
                playback_buffer_before_ms=buffer_before,
                playback_buffer_after_ms=playback_after,
                support_bucket=min(safe_pending, 4),
                safe_pending_count=safe_pending,
                semantic_start=(
                    int(micro[micro_cursor]["semantic_start"])
                    if micro_cursor < len(micro)
                    else len(target_bicodec)
                ),
                semantic_end=(
                    int(micro[micro_cursor]["semantic_start"])
                    if micro_cursor < len(micro)
                    else len(target_bicodec)
                ),
                source_finished=source_finished,
            )
        events.append(event)
        playback_buffer_ms = playback_after
        if micro_cursor == len(micro):
            break
    else:
        raise RuntimeError("dense scheduler exceeded its bounded tick budget")

    return DenseSession(
        sample_id=str(record["id"]),
        source_manifest=str(source_manifest.resolve()),
        source_index=source_index,
        split=split,
        src_lang=str(record["src_lang"]),
        tgt_lang=str(record["tgt_lang"]),
        source_duration_ms=source_duration_ms,
        target_duration_ms=target_duration_ms,
        source_glm_length=len(source_glm),
        target_semantic_length=len(target_bicodec),
        target_word_count=len(target_words),
        target_text=target_text,
        speaker_global=tuple(int(value) for value in speaker_global),
        events=tuple(events),
        low_watermark_ms=low_watermark_ms,
        target_buffer_ms=target_buffer_ms,
        semantic_history_tokens=semantic_history_tokens,
    ).with_checksum()


def _part_range(records: int, part_index: int, num_parts: int) -> tuple[int, int]:
    if not 0 <= part_index < num_parts:
        raise ValueError("part_index must be in [0, num_parts)")
    return records * part_index // num_parts, records * (part_index + 1) // num_parts


def build_part(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.input_manifest).resolve()
    output = Path(args.output).resolve()
    marker_path = Path(args.marker).resolve()
    offsets = load_index(source)
    if offsets is None:
        raise ValueError(f"missing JSONL index for {source}")
    start, end = _part_range(len(offsets), args.part_index, args.num_parts)
    if args.limit is not None:
        end = min(end, start + int(args.limit))
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != PART_SCHEMA:
            raise ValueError(f"unexpected existing marker schema: {marker_path}")
        if output.is_file():
            print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
            return marker
        raise FileNotFoundError(output)
    if output.exists():
        raise FileExistsError(f"refusing unmarked output: {output}")

    with source.open("rb") as handle:
        fixed_speaker = [
            int(value) for value in _record_at(handle, offsets[args.speaker_source_index])["bicodec_global"]
        ]
    if len(fixed_speaker) != 32:
        raise ValueError("fixed speaker source does not contain 32 global tokens")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output_offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as target:
            for source_index in range(start, end):
                try:
                    record = _record_at(source_handle, offsets[source_index])
                    session = build_dense_session(
                        record,
                        source_manifest=source,
                        source_index=source_index,
                        split=args.split,
                        speaker_global=fixed_speaker,
                        low_watermark_ms=args.low_watermark_ms,
                        target_buffer_ms=args.target_buffer_ms,
                        semantic_history_tokens=args.semantic_history_tokens,
                    )
                    value = session.to_dict()
                    encoded = (
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    output_offsets.append(byte_offset)
                    target.write(encoded)
                    byte_offset += len(encoded)
                    counts["sessions"] += 1
                    counts["events"] += len(session.events)
                    writes = [event for event in session.events if event.action == "WRITE"]
                    counts["writes"] += len(writes)
                    counts["reads"] += len(session.events) - len(writes)
                    counts["semantic_tokens"] += session.target_semantic_length
                    counts[f"direction:{session.src_lang}-{session.tgt_lang}"] += 1
                except Exception as error:
                    counts["rejected"] += 1
                    if args.fail_fast:
                        raise RuntimeError(
                            f"dense build failed at source index {source_index}"
                        ) from error
                    if counts["rejected"] <= args.maximum_logged_rejections:
                        print(
                            json.dumps(
                                {
                                    "source_index": source_index,
                                    "error": f"{type(error).__name__}: {error}",
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                processed = source_index - start + 1
                if args.progress_interval and processed % args.progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "part": args.part_index,
                                "processed": processed,
                                "accepted": counts["sessions"],
                                "rejected": counts["rejected"],
                                "records_per_second": processed / elapsed,
                            }
                        ),
                        flush=True,
                    )
            target.flush()
            os.fsync(target.fileno())
        if counts["sessions"] <= 0:
            raise ValueError("dense builder produced no sessions")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output, output_offsets)
    marker = {
        "schema_version": PART_SCHEMA,
        "dense_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "input_manifest": str(source),
        "input_records": len(offsets),
        "part_index": args.part_index,
        "num_parts": args.num_parts,
        "source_start": start,
        "source_end": end,
        "output": str(output),
        "index": index,
        "fixed_speaker_source_index": args.speaker_source_index,
        "fixed_speaker_global": fixed_speaker,
        "counts": dict(counts),
        "acceptance_rate": counts["sessions"] / max(1, end - start),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--speaker-source-index", type=int, default=0)
    parser.add_argument("--low-watermark-ms", type=int, default=240)
    parser.add_argument("--target-buffer-ms", type=int, default=400)
    parser.add_argument("--semantic-history-tokens", type=int, default=200)
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument("--maximum-logged-rejections", type=int, default=20)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_part(parse_args())


if __name__ == "__main__":
    main()
