#!/usr/bin/env python3
"""Build deterministic 45--90 s same-direction episodes from 15-shard gold data.

The source JSONL is accessed through its immutable uint64 offset index; the
22 GB trajectory file is never copied or rewritten.  Each output episode only
stores component references and frozen teacher text, so construction is CPU
and metadata bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path
from typing import BinaryIO


def load_offsets(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 8:
        raise ValueError("offset index is not uint64 aligned")
    return list(struct.unpack(f"<{len(raw) // 8}Q", raw))


def read_record(handle: BinaryIO, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def normalized_join(values: list[str], language: str) -> str:
    if language == "cmn":
        return "。".join(value.strip().rstrip("。！？") for value in values if value.strip()) + "。"
    return ". ".join(value.strip().rstrip(".!?") for value in values if value.strip()) + "."


def build_episodes(
    source: Path,
    index: Path,
    *,
    count: int,
    minimum_ms: int,
    target_ms: int,
    maximum_ms: int,
    seed: int,
) -> list[dict[str, object]]:
    offsets = load_offsets(index)
    rng = random.Random(int(seed))
    order = list(range(len(offsets)))
    rng.shuffle(order)
    pools: dict[str, list[dict[str, object]]] = {"cmn->eng": [], "eng->cmn": []}
    with source.open("rb") as handle:
        for record_index in order:
            row = read_record(handle, offsets[record_index])
            direction = f"{row['src_lang']}->{row['tgt_lang']}"
            if direction not in pools:
                continue
            duration = int(row["source_duration_ms"])
            if not 2_000 <= duration <= 20_000:
                continue
            pools[direction].append(
                {
                    "record_index": record_index,
                    "sample_id": str(row["sample_id"]),
                    "source_audio": str(row["source_audio"]),
                    "source_audio_sha256": str(row["source_audio_sha256"]),
                    "duration_ms": duration,
                    "transcription": str(row["full_transcription"]),
                    "translation": str(row["full_translation"]),
                    "speaker_global": [int(value) for value in row["speaker_global"]],
                    "source_glm_length": int(row["source_glm_length"]),
                    "target_semantic_length": int(row["target_semantic_length"]),
                }
            )
            if all(len(values) >= count * 6 for values in pools.values()):
                break
    episodes: list[dict[str, object]] = []
    cursors = {key: 0 for key in pools}
    for episode_index in range(count):
        direction = "cmn->eng" if episode_index % 2 == 0 else "eng->cmn"
        components: list[dict[str, object]] = []
        duration = 0
        values = pools[direction]
        while duration < target_ms and cursors[direction] < len(values):
            candidate = values[cursors[direction]]
            cursors[direction] += 1
            proposed = duration + int(candidate["duration_ms"])
            if proposed > maximum_ms and duration >= minimum_ms:
                break
            components.append(candidate)
            duration = proposed
        if duration < minimum_ms or not components:
            raise RuntimeError(f"insufficient {direction} records for episode {episode_index}")
        src_lang, tgt_lang = direction.split("->")
        component_ids = [str(value["sample_id"]) for value in components]
        digest = hashlib.sha256("\n".join(component_ids).encode("utf-8")).hexdigest()
        episodes.append(
            {
                "episode_id": f"episode_{episode_index:06d}_{direction.replace('->', '_')}",
                "direction": direction,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "duration_ms": duration,
                "components": components,
                "component_count": len(components),
                "component_ids_sha256": digest,
                "teacher_transcription": normalized_join(
                    [str(value["transcription"]) for value in components], src_lang
                ),
                "teacher_translation": normalized_join(
                    [str(value["translation"]) for value in components], tgt_lang
                ),
            }
        )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--minimum-ms", type=int, default=45_000)
    parser.add_argument("--target-ms", type=int, default=60_000)
    parser.add_argument("--maximum-ms", type=int, default=90_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    episodes = build_episodes(
        args.source,
        args.index,
        count=args.count,
        minimum_ms=args.minimum_ms,
        target_ms=args.target_ms,
        maximum_ms=args.maximum_ms,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = {
        "schema_version": "uniss_stateful_longepisode_manifest_v1",
        "status": "passed",
        "source": str(args.source.resolve()),
        "source_index": str(args.index.resolve()),
        "output": str(args.output.resolve()),
        "episodes": len(episodes),
        "directions": {
            direction: sum(row["direction"] == direction for row in episodes)
            for direction in ("cmn->eng", "eng->cmn")
        },
        "duration_ms": {
            "minimum": min(int(row["duration_ms"]) for row in episodes),
            "maximum": max(int(row["duration_ms"]) for row in episodes),
            "mean": sum(int(row["duration_ms"]) for row in episodes) / len(episodes),
        },
        "unique_components": len(
            {str(value["sample_id"]) for row in episodes for value in row["components"]}
        ),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

