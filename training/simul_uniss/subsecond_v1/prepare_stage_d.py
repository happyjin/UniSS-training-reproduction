"""Prepare Stage-D proxy Micro-WRITE data with Phase3 replay.

The current 15-shard bootstrap schedules use proportional alignment.  This
module therefore creates an explicitly named proxy dataset: it validates the
Micro-WRITE training path but is not a replacement for formal bilingual
support and target-word timestamps.
"""

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
from typing import Iterator, Mapping

from training.simul_uniss.jsonl_index import write_index
from training.simul_uniss.pack_sequences import make_shifted_sample, pack_samples
from training.simul_uniss.sample_builders import WeightedSample, build_interleaved_sample


SCHEMA = "simul_uniss_subsecond_stage_d_proxy_data_v1"


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


def _iter_jsonl(paths: list[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected object at {path}:{line_number}")
                yield value


def _balanced_spans(length: int, maximum: int, minimum: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    chunks = max(1, math.ceil(length / maximum))
    while chunks > 1 and length // chunks < minimum:
        chunks -= 1
    return [
        (length * index // chunks, length * (index + 1) // chunks)
        for index in range(chunks)
    ]


def micro_split_schedule(
    schedule: Mapping[str, object], *, minimum_semantic: int = 8, maximum_semantic: int = 16
) -> dict[str, object]:
    events = schedule.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("schedule events must be a non-empty list")
    result_events: list[dict[str, object]] = []
    micro_writes = 0
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("schedule event must be an object")
        if event.get("action") != "write":
            result_events.append(dict(event))
            continue
        text = [int(value) for value in event.get("target_text_ids", [])]
        semantic = [int(value) for value in event.get("target_semantic", [])]
        spans = _balanced_spans(len(semantic), maximum_semantic, minimum_semantic)
        if not spans:
            raise ValueError(f"WRITE event has no semantic tokens: {schedule.get('id')}")
        for micro_index, (semantic_start, semantic_end) in enumerate(spans):
            text_start = len(text) * micro_index // len(spans)
            text_end = len(text) * (micro_index + 1) // len(spans)
            micro = dict(event)
            micro["micro_write_index"] = micro_index
            micro["micro_write_count"] = len(spans)
            micro["source_glm"] = list(event.get("source_glm", [])) if micro_index == 0 else []
            micro["target_text_ids"] = text[text_start:text_end]
            micro["target_semantic"] = semantic[semantic_start:semantic_end]
            micro["target_semantic_start"] = int(event.get("target_semantic_start", 0)) + semantic_start
            micro["target_semantic_end"] = int(event.get("target_semantic_start", 0)) + semantic_end
            micro["source_is_final"] = bool(event.get("source_is_final", False)) and micro_index == len(spans) - 1
            micro["alignment_status"] = "proxy_proportional_micro_split"
            result_events.append(micro)
            micro_writes += 1
    value = dict(schedule)
    value["schema_version"] = "simul_uniss_stage_d_proxy_schedule_v1"
    value["alignment_kind"] = "proxy_proportional_micro_write_not_formal_support_alignment"
    value["events"] = result_events
    value["micro_write_events"] = micro_writes
    return value


def _phase3_replay_sample(value: Mapping[str, object]) -> WeightedSample:
    prompt = [int(token) for token in value["prompt_ids"]]  # type: ignore[index]
    target = [int(token) for token in value["target_ids"]]  # type: ignore[index]
    if not prompt or not target:
        raise ValueError("Phase3 replay requires non-empty prompt and target")
    return WeightedSample(
        sample_id=f"phase3_replay:{value.get('id', '')}",
        input_ids=[*prompt, *target],
        token_weights=[*([0.0] * len(prompt)), *([1.0] * len(target))],
        task="phase3_replay",
    )


def mixed_samples(
    schedules: Iterator[dict[str, object]],
    replay: Iterator[dict[str, object]],
    *,
    replay_ratio: float,
    limit_schedules: int | None,
    counts: Counter[str],
) -> Iterator[dict[str, object]]:
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    replay_per_micro = replay_ratio / max(1e-9, 1.0 - replay_ratio)
    replay_credit = 0.0
    replay_exhausted = False
    for schedule_index, schedule in enumerate(schedules):
        if limit_schedules is not None and schedule_index >= limit_schedules:
            break
        proxy = micro_split_schedule(schedule)
        counts["schedules"] += 1
        counts["micro_write_events"] += int(proxy["micro_write_events"])
        counts[f"direction:{proxy['src_lang']}->{proxy['tgt_lang']}"] += 1
        yield build_interleaved_sample(proxy).to_json()
        counts["micro_samples"] += 1
        replay_credit += replay_per_micro
        while replay_credit >= 1.0 and not replay_exhausted:
            try:
                replay_value = next(replay)
            except StopIteration:
                replay_exhausted = True
                counts["replay_exhausted"] = 1
                break
            yield _phase3_replay_sample(replay_value).to_json()
            counts["replay_samples"] += 1
            replay_credit -= 1.0


def prepare(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / "STAGE_D_PROXY_DATA_READY.json"
    output = output_dir / "packed.jsonl"
    if marker_path.is_file() and output.is_file():
        value = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value

    schedule_paths = [Path(value).resolve() for value in args.schedules]
    replay_paths = [Path(value).resolve() for value in args.phase3_replay]
    for path in [*schedule_paths, *replay_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    temporary = output_dir / f".packed.jsonl.tmp.{os.getpid()}"
    counts: Counter[str] = Counter()
    started = time.time()
    raw = mixed_samples(
        _iter_jsonl(schedule_paths),
        _iter_jsonl(replay_paths),
        replay_ratio=args.replay_ratio,
        limit_schedules=args.limit_schedules,
        counts=counts,
    )
    shifted = (make_shifted_sample(value) for value in raw)
    packed = pack_samples(shifted, args.seq_length, drop_overlong=True)
    offsets = array("Q")
    offset = 0
    try:
        with temporary.open("wb") as handle:
            for item in packed:
                encoded = (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")
                offsets.append(offset)
                handle.write(encoded)
                offset += len(encoded)
                counts["packed_sequences"] += 1
                if args.progress_interval and counts["packed_sequences"] % args.progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "packed_sequences": counts["packed_sequences"],
                                "schedules": counts["schedules"],
                                "elapsed_seconds": elapsed,
                            }
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
        "scope": "stage_d_micro_write_proxy_not_formal_bilingual_alignment",
        "warning": "Uses proportional bootstrap alignment; formal Stage D requires Stage-A A7/A8 support and target timestamps.",
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "index": index,
        "seq_length": args.seq_length,
        "replay_ratio_requested": args.replay_ratio,
        "limit_schedules": args.limit_schedules,
        "counts": dict(counts),
        "suggested_train_iters_gbs128": math.ceil(counts["packed_sequences"] / 128),
        "elapsed_seconds": time.time() - started,
        "schedule_inputs": [str(path) for path in schedule_paths],
        "phase3_replay_inputs": [str(path) for path in replay_paths],
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
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--phase3-replay", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=18000)
    parser.add_argument("--replay-ratio", type=float, default=0.30)
    parser.add_argument("--limit-schedules", type=int)
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
