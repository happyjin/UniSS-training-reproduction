#!/usr/bin/env python3
"""Pack free-running policy traces and immutable Phase3 replay for Megatron."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_REPLAY,
)
from training import constants_uniss as c


def offsets(path: Path) -> list[int]:
    raw = path.with_suffix(path.suffix + ".offsets.bin").read_bytes()
    if len(raw) % 8:
        raise ValueError("offset index is malformed")
    return list(struct.unpack(f"<{len(raw)//8}Q", raw))


def read_indexed(path: Path, record_indexes: Iterable[int]):
    values = offsets(path)
    with path.open("rb") as handle:
        for index in record_indexes:
            handle.seek(values[int(index)])
            yield json.loads(handle.readline())


def trajectory_sample(row: dict[str, object]) -> dict[str, object]:
    prompt = [int(value) for value in row["prompt_ids"]]
    generated = [int(value) for value in row["generated_ids"]]
    old = [float(value) for value in row["old_log_probs"]]
    if not generated or len(old) != len(generated):
        raise ValueError("trajectory generation/log-prob geometry differs")
    sequence = [*prompt, *generated]
    tokens = sequence[:-1]
    labels = sequence[1:]
    response_start = len(prompt) - 1
    response_mask = [0.0] * len(tokens)
    old_log_probs = [0.0] * len(tokens)
    advantages = [0.0] * len(tokens)
    for offset, value in enumerate(old):
        position = response_start + offset
        response_mask[position] = 1.0
        old_log_probs[position] = value
        advantages[position] = float(row["advantage"])
    return {
        "tokens": tokens,
        "labels": labels,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "replay_mask": [0.0] * len(tokens),
        "family_ids": [1 if row["family"] == "mt" else 2] * len(tokens),
        "identity": f"{row['episode_id']}:g{row['group_index']}:t{row['trace_index']}",
    }


def replay_pack(row: dict[str, object], seq_length: int) -> dict[str, object]:
    used = int(row["used_tokens"])
    if len(row["tokens"]) != seq_length or len(row["labels"]) != seq_length:
        raise ValueError("Phase3 replay pack length differs from seq_length")
    tokens = [int(value) for value in row["tokens"]]
    labels = [int(value) for value in row["labels"]]
    kinds = [int(value) for value in row["loss_kinds"]]
    mask = [1.0 if value == LOSS_REPLAY else 0.0 for value in kinds]
    if not any(mask):
        raise ValueError("Phase3 replay record has no replay positions")
    return {
        "schema_version": "uniss_free_running_episode_grpo_pack_v1",
        "tokens": tokens,
        "labels": labels,
        "position_ids": [int(value) for value in row["position_ids"]],
        "response_mask": [0.0] * seq_length,
        "old_log_probs": [0.0] * seq_length,
        "advantages": [0.0] * seq_length,
        "replay_mask": mask,
        "family_ids": [3 if index < used else 0 for index in range(seq_length)],
        "loss_mask": mask,
        "sample_boundaries": row["sample_boundaries"],
        "identities": [f"phase3:{value}" for value in row["source_ids"]],
        "used_tokens": used,
        "rl_tokens": 0,
        "replay_tokens": int(sum(mask)),
    }


def pack_samples(samples: Iterable[dict[str, object]], seq_length: int):
    current = None
    for sample in samples:
        length = len(sample["tokens"])
        if length > seq_length:
            raise ValueError(f"sample exceeds seq length: {sample['identity']} {length}")
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


def finalize_pack(pack: dict[str, object], seq_length: int) -> dict[str, object]:
    used = len(pack["tokens"])
    padding = seq_length - used
    pack["tokens"].extend([c.TOKEN_EOS] * padding)
    pack["labels"].extend([c.TOKEN_EOS] * padding)
    for field in ("response_mask", "old_log_probs", "advantages", "replay_mask"):
        pack[field].extend([0.0] * padding)
    pack["family_ids"].extend([0] * padding)
    pack["position_ids"].extend(range(padding))
    pack.update(
        {
            "schema_version": "uniss_free_running_episode_grpo_pack_v1",
            "used_tokens": used,
            "rl_tokens": int(sum(pack["response_mask"])),
            "replay_tokens": int(sum(pack["replay_mask"])),
            "loss_mask": [
                float(bool(left) or bool(right))
                for left, right in zip(pack["response_mask"], pack["replay_mask"])
            ],
        }
    )
    return pack


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions: list[int] = []
    count = rl = replay = used = 0
    with path.open("wb") as handle:
        for row in rows:
            positions.append(handle.tell())
            handle.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            count += 1
            rl += int(row["rl_tokens"])
            replay += int(row["replay_tokens"])
            used += int(row["used_tokens"])
    index = path.with_suffix(path.suffix + ".offsets.bin")
    index.write_bytes(struct.pack(f"<{len(positions)}Q", *positions))
    return {
        "path": str(path.resolve()),
        "records": count,
        "used_tokens": used,
        "rl_tokens": rl,
        "replay_tokens": replay,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "offset_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--phase3-replay", type=Path, required=True)
    parser.add_argument("--phase3-replay-records", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    args = parser.parse_args()
    trajectories = []
    for path in args.trajectory:
        trajectories.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    rl_samples = [trajectory_sample(row) for row in trajectories]
    replay_rows = read_indexed(
        args.phase3_replay,
        range(min(args.phase3_replay_records, len(offsets(args.phase3_replay)))),
    )
    replay_packs = [replay_pack(row, args.seq_length) for row in replay_rows]
    rl_packs = list(pack_samples(rl_samples, args.seq_length))
    interleaved = []
    maximum = max(len(rl_packs), len(replay_packs))
    for index in range(maximum):
        if index < len(rl_packs):
            interleaved.append(rl_packs[index])
        if index < len(replay_packs):
            interleaved.append(replay_packs[index])
    summary = write_jsonl(args.output, interleaved)
    summary.update(
        {
            "schema_version": "uniss_free_running_episode_grpo_dataset_v1",
            "status": "passed",
            "seq_length": args.seq_length,
            "trajectories": len(trajectories),
            "rl_packs": len(rl_packs),
            "phase3_replay_packs": len(replay_packs),
            "source_trajectory_files": [str(path.resolve()) for path in args.trajectory],
            "phase3_replay_source": str(args.phase3_replay.resolve()),
        }
    )
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
