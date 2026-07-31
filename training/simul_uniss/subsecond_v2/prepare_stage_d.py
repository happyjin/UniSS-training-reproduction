"""Prepare formal Stage-D Micro-WRITE SFT data with Phase3 replay."""

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
from typing import Callable, Iterator, Mapping

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.pack_sequences import make_shifted_sample, pack_samples
from training.simul_uniss.sample_builders import WeightedSample, build_interleaved_sample


SCHEMA = "simul_uniss_subsecond_stage_d_formal_data_v2"


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


def formal_schedule(
    record: Mapping[str, object], text_encoder: Callable[[str], list[int]], *, tick_ms: int = 160
) -> dict[str, object]:
    if not bool(record.get("formal_a68_pass")):
        raise ValueError("Stage-D formal schedule requires an accepted A8 record")
    source = [int(value) for value in record["teacher_source_glm"]]  # type: ignore[index]
    source_ends = [int(value) for value in record["teacher_source_glm_end_ms"]]  # type: ignore[index]
    target = [int(value) for value in record["target_bicodec"]]  # type: ignore[index]
    micro = [dict(value) for value in record["micro_write_events"]]  # type: ignore[index]
    if len(source) != len(source_ends) or not source or not target or not micro:
        raise ValueError("formal Stage-D record has incomplete source/target supervision")
    source_duration_ms = int(record["source_duration_ms"])
    events: list[dict[str, object]] = []
    source_cursor = 0
    micro_cursor = 0
    tick_count = max(1, math.ceil(source_duration_ms / tick_ms))
    for tick_index in range(tick_count):
        source_end_ms = min(source_duration_ms, (tick_index + 1) * tick_ms)
        next_source = source_cursor
        while next_source < len(source_ends) and source_ends[next_source] <= source_end_ms:
            next_source += 1
        event: dict[str, object] = {
            "chunk_index": len(events),
            "source_start_ms": tick_index * tick_ms,
            "source_end_ms": source_end_ms,
            "source_glm_start": source_cursor,
            "source_glm_end": next_source,
            "source_glm": source[source_cursor:next_source],
            "source_is_final": False,
        }
        source_cursor = next_source
        if micro_cursor < len(micro) and int(micro[micro_cursor]["earliest_safe_ms"]) <= source_end_ms:
            value = micro[micro_cursor]
            event.update(
                {
                    "action": "write",
                    "target_text_ids": text_encoder(str(value["text"])),
                    "target_semantic_start": int(value["semantic_start"]),
                    "target_semantic_end": int(value["semantic_end"]),
                    "target_semantic": target[
                        int(value["semantic_start"]) : int(value["semantic_end"])
                    ],
                    "support_end_ms": int(value["support_end_ms"]),
                    "earliest_safe_ms": int(value["earliest_safe_ms"]),
                    "semantic_continuation": bool(value.get("semantic_continuation", False)),
                }
            )
            micro_cursor += 1
        else:
            event["action"] = "wait"
        events.append(event)

    # Flush target micro-transactions after source exhaustion.  Empty source
    # chunks are valid cached-Qwen transactions; only the final transaction is
    # marked source_is_final so the protocol never ends on WAIT.
    while micro_cursor < len(micro):
        value = micro[micro_cursor]
        events.append(
            {
                "chunk_index": len(events),
                "source_start_ms": source_duration_ms,
                "source_end_ms": source_duration_ms,
                "source_glm_start": source_cursor,
                "source_glm_end": source_cursor,
                "source_glm": [],
                "source_is_final": False,
                "action": "write",
                "target_text_ids": text_encoder(str(value["text"])),
                "target_semantic_start": int(value["semantic_start"]),
                "target_semantic_end": int(value["semantic_end"]),
                "target_semantic": target[int(value["semantic_start"]) : int(value["semantic_end"])],
                "support_end_ms": int(value["support_end_ms"]),
                "earliest_safe_ms": int(value["earliest_safe_ms"]),
                "semantic_continuation": bool(value.get("semantic_continuation", False)),
                "final_flush": True,
            }
        )
        micro_cursor += 1
    if source_cursor < len(source):
        events[-1]["source_glm"] = [*events[-1]["source_glm"], *source[source_cursor:]]  # type: ignore[index]
        events[-1]["source_glm_end"] = len(source)
    if events[-1]["action"] == "wait":
        # If every target transaction was emitted before source exhaustion,
        # move the last transaction onto the final source tick.  This preserves
        # exact target coverage without duplicating semantic tokens.
        write_index = max(index for index, value in enumerate(events[:-1]) if value["action"] == "write")
        previous = events[write_index]
        payload_keys = (
            "target_text_ids",
            "target_semantic_start",
            "target_semantic_end",
            "target_semantic",
            "support_end_ms",
            "earliest_safe_ms",
            "semantic_continuation",
            "final_flush",
        )
        payload = {key: previous.pop(key) for key in payload_keys if key in previous}
        previous["action"] = "wait"
        events[-1].update({"action": "write", **payload})
    events[-1]["source_is_final"] = True
    emitted_source = [token for event in events for token in event["source_glm"]]  # type: ignore[index]
    emitted_target = [
        token
        for event in events
        if event["action"] == "write"
        for token in event["target_semantic"]  # type: ignore[index]
    ]
    if emitted_source != source:
        raise AssertionError("formal schedule does not cover teacher source GLM exactly")
    if emitted_target != target:
        raise AssertionError("formal schedule does not cover target semantic exactly")
    return {
        "schema_version": "simul_uniss_formal_schedule_v2",
        "alignment_kind": str(record["alignment_kind"]),
        "id": str(record["id"]),
        "src_lang": str(record["src_lang"]),
        "tgt_lang": str(record["tgt_lang"]),
        "transcription": str(record["transcription"]),
        "translation": str(record["translation"]),
        "chunk_ms": tick_ms,
        "source_glm_length": len(source),
        "target_bicodec_length": len(target),
        "speaker_tokens": [int(value) for value in record["bicodec_global"]],  # type: ignore[index]
        "events": events,
    }


def _iter_indexed(path: Path) -> Iterator[dict[str, object]]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing JSONL offset index for {path}")
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            yield json.loads(handle.readline())


def _iter_jsonl(paths: list[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _phase3_replay(value: Mapping[str, object]) -> WeightedSample:
    prompt = [int(token) for token in value["prompt_ids"]]  # type: ignore[index]
    target = [int(token) for token in value["target_ids"]]  # type: ignore[index]
    return WeightedSample(
        sample_id=f"phase3_replay:{value.get('id', '')}",
        input_ids=[*prompt, *target],
        token_weights=[*([0.0] * len(prompt)), *([1.0] * len(target))],
        task="phase3_replay",
    )


def prepare(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.tick_ms <= 0:
        raise ValueError("tick_ms must be positive")
    if args.seq_length <= 1:
        raise ValueError("seq_length must be greater than one")
    if not 0.0 <= args.replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    input_manifest = Path(args.input_manifest).resolve()
    replay_paths = [Path(value).resolve() for value in args.phase3_replay]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "packed.jsonl"
    marker_path = output_dir / "STAGE_D_FORMAL_DATA_READY.json"
    if marker_path.is_file() and output.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False) if text else []
    replay = _iter_jsonl(replay_paths)
    replay_per_formal = args.replay_ratio / max(1e-9, 1.0 - args.replay_ratio)
    replay_credit = 0.0
    counts: Counter[str] = Counter()

    def samples() -> Iterator[dict[str, object]]:
        nonlocal replay_credit
        for index, record in enumerate(_iter_indexed(input_manifest)):
            if args.limit_records is not None and index >= args.limit_records:
                break
            schedule = formal_schedule(record, encode, tick_ms=args.tick_ms)
            yield build_interleaved_sample(schedule).to_json()
            counts["formal_samples"] += 1
            counts["write_events"] += sum(
                event["action"] == "write" for event in schedule["events"]  # type: ignore[index]
            )
            replay_credit += replay_per_formal
            while replay_credit >= 1.0:
                try:
                    value = next(replay)
                except StopIteration:
                    counts["replay_exhausted"] = 1
                    replay_credit = 0.0
                    break
                yield _phase3_replay(value).to_json()
                counts["replay_samples"] += 1
                replay_credit -= 1.0

    shifted = (make_shifted_sample(value) for value in samples())
    packed = pack_samples(shifted, args.seq_length, drop_overlong=True)
    temporary = output_dir / f".packed.jsonl.tmp.{os.getpid()}"
    offsets = array("Q")
    offset = 0
    started = time.time()
    try:
        with temporary.open("wb") as handle:
            for value in packed:
                encoded = (json.dumps(value, separators=(",", ":")) + "\n").encode()
                offsets.append(offset)
                handle.write(encoded)
                offset += len(encoded)
                counts["packed_sequences"] += 1
                if (
                    args.progress_interval
                    and counts["packed_sequences"] % args.progress_interval == 0
                ):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "formal_samples": counts["formal_samples"],
                                "replay_samples": counts["replay_samples"],
                                "packed_sequences": counts["packed_sequences"],
                                "records_per_second": (
                                    counts["formal_samples"] + counts["replay_samples"]
                                )
                                / elapsed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output, offsets)
    marker = {
        "schema_version": SCHEMA,
        "status": "ready",
        "scope": "formal_bilingual_micro_write_stage_d_v2",
        "input_manifest": str(input_manifest),
        "phase3_replay": [str(path) for path in replay_paths],
        "output": str(output),
        "index": index,
        "seq_length": args.seq_length,
        "replay_ratio": args.replay_ratio,
        "counts": dict(counts),
        "suggested_train_iters_gbs128": math.ceil(counts["packed_sequences"] / 128),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    (output_dir / "training_schedule.env").write_text(
        f'STAGE_D_TRAIN_ITERS="${{STAGE_D_TRAIN_ITERS:-{marker["suggested_train_iters_gbs128"]}}}"\n',
        encoding="utf-8",
    )
    print(json.dumps(marker, sort_keys=True))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--phase3-replay", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tick-ms", type=int, default=160)
    parser.add_argument("--seq-length", type=int, default=18000)
    parser.add_argument("--replay-ratio", type=float, default=0.30)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
