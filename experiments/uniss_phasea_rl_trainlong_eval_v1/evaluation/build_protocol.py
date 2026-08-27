#!/usr/bin/env python3
"""Select the longest formal-RL train episodes with auditable provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIRECTIONS = ("cmn->eng", "eng->cmn")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_records(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    rollout: dict[str, Any],
    *,
    per_direction: int,
) -> list[dict[str, Any]]:
    summaries = list(rollout.get("summaries", []))
    rollout_ids = {str(row["episode_id"]) for row in summaries}
    if int(rollout.get("episodes", len(summaries))) != len(rollout_ids):
        raise ValueError("formal rollout episode count/IDs are inconsistent")
    train_by_id = {str(row["episode_id"]): row for row in train_rows}
    missing = sorted(rollout_ids - set(train_by_id))
    if missing:
        raise ValueError(f"formal rollout IDs missing from train manifest: {missing[:3]}")

    valid_audio_hashes = {str(row["source_audio_sha256"]) for row in valid_rows}
    valid_component_ids = {
        str(component["sample_id"])
        for row in valid_rows
        for component in row.get("components", [])
    }
    selected: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        candidates = [
            train_by_id[episode_id]
            for episode_id in rollout_ids
            if str(train_by_id[episode_id]["direction"]) == direction
        ]
        candidates.sort(
            key=lambda row: (-int(row["source_duration_ms"]), str(row["episode_id"]))
        )
        if len(candidates) < per_direction:
            raise ValueError(f"not enough formal rollout episodes for {direction}")
        for row in candidates[:per_direction]:
            if str(row["source_audio_sha256"]) in valid_audio_hashes:
                raise ValueError(f"train/valid episode audio overlap: {row['episode_id']}")
            component_ids = {
                str(component["sample_id"]) for component in row.get("components", [])
            }
            overlap = sorted(component_ids & valid_component_ids)
            if overlap:
                raise ValueError(
                    f"train/valid component overlap for {row['episode_id']}: {overlap[:3]}"
                )
            audio = Path(str(row["source_audio"]))
            if not audio.is_file():
                raise FileNotFoundError(audio)
            selected.append(
                {
                    "sample_id": str(row["episode_id"]),
                    "episode_id": str(row["episode_id"]),
                    "src_lang": str(row["src_lang"]),
                    "tgt_lang": str(row["tgt_lang"]),
                    "direction": direction,
                    "source_audio": str(audio.resolve()),
                    "source_duration_ms": int(row["source_duration_ms"]),
                    "reference_transcription": str(row["teacher_transcription"]),
                    "reference_translation": str(row["teacher_translation"]),
                    "component_count": int(row["component_count"]),
                    "component_ids_sha256": str(row["component_ids_sha256"]),
                    "source_audio_sha256": str(row["source_audio_sha256"]),
                    "rl_train_seen": True,
                    "formal_rollout_seen": True,
                    "validation_overlap": False,
                }
            )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--valid-episodes", type=Path, required=True)
    parser.add_argument("--formal-rollout", type=Path, required=True)
    parser.add_argument("--per-direction", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.per_direction <= 0:
        raise ValueError("per-direction must be positive")
    rollout = json.loads(args.formal_rollout.read_text(encoding="utf-8"))
    records = select_records(
        read_jsonl(args.train_episodes),
        read_jsonl(args.valid_episodes),
        rollout,
        per_direction=args.per_direction,
    )
    payload = {
        "schema_version": "uniss_phasea_rl_train_seen_long_protocol_v1",
        "status": "complete",
        "selection": "longest formal-RL rollout episodes per direction",
        "claim_boundary": (
            "All records are train-seen/in-domain. This protocol tests whether RL learned "
            "its training objective and must not be used as a generalization claim."
        ),
        "per_direction": args.per_direction,
        "records": records,
        "sources": {
            "train_episodes": str(args.train_episodes.resolve()),
            "valid_episodes": str(args.valid_episodes.resolve()),
            "formal_rollout": str(args.formal_rollout.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "sample_id": row["sample_id"],
                    "direction": row["direction"],
                    "duration_seconds": row["source_duration_ms"] / 1000.0,
                    "components": row["component_count"],
                }
                for row in records
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
