#!/usr/bin/env python3
"""Pack one fresh rollout round exactly once with Phase3 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.pack_trajectories import (
    finalize_pack,
    offsets,
    read_indexed,
    replay_pack as historical_replay_pack,
    write_jsonl,
)


FAMILY_IDS = {"mt": 2, "tts": 3, "control": 4}


def replay_pack(row: dict[str, object], seq_length: int) -> dict[str, object]:
    value = historical_replay_pack(row, seq_length)
    value["family_ids"] = [
        5 if int(family) == 4 else int(family) for family in value["family_ids"]
    ]
    return value


def is_valid_episode(episode_id: str) -> bool:
    digest = hashlib.blake2b(episode_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 8 == 0


def trajectory_sample(row: dict[str, object]) -> dict[str, object] | None:
    family = str(row["family"])
    if family == "asr":
        return None
    if family not in FAMILY_IDS:
        raise ValueError(f"unknown trajectory family {family}")
    prompt = [int(value) for value in row["prompt_ids"]]
    generated = [int(value) for value in row["generated_ids"]]
    old = [float(value) for value in row["old_log_probs"]]
    if not generated or len(generated) != len(old):
        raise ValueError("trajectory generation/log-prob geometry differs")
    sequence = [*prompt, *generated]
    tokens, labels = sequence[:-1], sequence[1:]
    response_start = len(prompt) - 1
    response_mask = [0.0] * len(tokens)
    old_log_probs = [0.0] * len(tokens)
    advantages = [0.0] * len(tokens)
    advantage = float(row["advantage"])
    for offset, value in enumerate(old):
        position = response_start + offset
        response_mask[position] = 1.0
        old_log_probs[position] = value
        advantages[position] = advantage
    return {
        "tokens": tokens,
        "labels": labels,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "replay_mask": [0.0] * len(tokens),
        "family_ids": [FAMILY_IDS[family]] * len(tokens),
        "identity": (
            f"{row['episode_id']}:g{row['group_index']}:"
            f"e{row['event_index']}:t{row['trace_index']}:{family}"
        ),
    }


def pack_samples(samples, seq_length: int):
    current = None
    for sample in samples:
        if sample is None:
            continue
        length = len(sample["tokens"])
        if length > seq_length:
            raise ValueError(f"sample exceeds sequence length: {sample['identity']}")
        if current is None or len(current["tokens"]) + length > seq_length:
            if current is not None:
                yield finalize_pack(current, seq_length)
            current = {
                "tokens": [],
                "labels": [],
                "response_mask": [],
                "old_log_probs": [],
                "advantages": [],
                "replay_mask": [],
                "family_ids": [],
                "position_ids": [],
                "sample_boundaries": [],
                "identities": [],
            }
        start = len(current["tokens"])
        for field in (
            "tokens",
            "labels",
            "response_mask",
            "old_log_probs",
            "advantages",
            "replay_mask",
            "family_ids",
        ):
            current[field].extend(sample[field])
        current["position_ids"].extend(range(length))
        current["sample_boundaries"].append([start, start + length])
        current["identities"].append(sample["identity"])
    if current is not None:
        yield finalize_pack(current, seq_length)


def load(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def add_event_family_summary(
    summary: dict[str, object], rows: list[dict[str, object]]
) -> dict[str, object]:
    """Report event-family response tokens, including the new control family."""
    family_tokens = {
        name: sum(
            int(int(family) == family_id and float(mask) > 0)
            for row in rows
            for family, mask in zip(row["family_ids"], row["response_mask"])
        )
        for name, family_id in FAMILY_IDS.items()
    }
    summary["family_response_tokens"] = family_tokens
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--phase3-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = list(load(args.trajectory))
    train_rows = [row for row in rows if not is_valid_episode(str(row["episode_id"]))]
    valid_rows = [row for row in rows if is_valid_episode(str(row["episode_id"]))]
    train_rl = list(pack_samples((trajectory_sample(row) for row in train_rows), args.seq_length))
    valid_rl = list(pack_samples((trajectory_sample(row) for row in valid_rows), args.seq_length))
    replay_count = min((len(train_rl) + 1) // 2, len(offsets(args.phase3_replay)))
    replay = [
        replay_pack(row, args.seq_length)
        for row in read_indexed(args.phase3_replay, range(replay_count))
    ]
    train = []
    replay_index = 0
    for index, row in enumerate(train_rl):
        train.append(row)
        if (index + 1) % 2 == 0 and replay_index < len(replay):
            train.append(replay[replay_index])
            replay_index += 1
    train.extend(replay[replay_index:])
    args.output.mkdir(parents=True)
    train_summary = add_event_family_summary(
        write_jsonl(args.output / "train_packs.jsonl", train), train
    )
    valid_summary = add_event_family_summary(
        write_jsonl(args.output / "valid_packs.jsonl", valid_rl), valid_rl
    )
    report = {
        "schema_version": "uniss_event_constrained_dataset_v2",
        "status": "passed",
        "seq_length": args.seq_length,
        "trajectories": len(rows),
        "train_trajectories": len(train_rows),
        "valid_trajectories": len(valid_rows),
        "train_rl_packs": len(train_rl),
        "valid_rl_packs": len(valid_rl),
        "phase3_replay_packs": len(replay),
        "fresh_rollout_consumption_epochs": 1,
        "train": train_summary,
        "valid": valid_summary,
    }
    (args.output / "AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
