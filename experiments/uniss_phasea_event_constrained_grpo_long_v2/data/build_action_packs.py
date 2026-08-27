#!/usr/bin/env python3
"""Pack audited WAIT/WRITE events and immutable Phase3 replay for Megatron."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable, Iterator

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_REPLAY,
)
from training import constants_uniss as c


SCHEMA = "uniss_event_action_warmup_pack_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def indexed_offsets(path: Path) -> list[int]:
    raw = path.with_suffix(path.suffix + ".offsets.bin").read_bytes()
    if len(raw) % 8:
        raise ValueError("offset index is malformed")
    return list(struct.unpack(f"<{len(raw)//8}Q", raw))


def read_indexed(path: Path, indexes: Iterable[int]):
    offsets = indexed_offsets(path)
    with path.open("rb") as handle:
        for index in indexes:
            handle.seek(offsets[int(index)])
            yield json.loads(handle.readline())


def event_identity(row: dict[str, object]) -> str:
    return f"{row['sample_id']}:{row['chunk_end_ms']}"


def action_sample(row: dict[str, object]) -> dict[str, object]:
    action = str(row["natural_action_target"])
    if action not in {"READ", "WRITE"}:
        raise ValueError(f"invalid natural action {action}")
    header = [
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
        c.language_token_id(str(row["tgt_lang"])),
        c.speed_token_id(1.0),
        *c.wrap_global_tokens([int(value) for value in row["speaker_global"]]),
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic([int(value) for value in row["causal_source_glm"]]),
        c.TOKEN_END_GLM,
    ]
    action_token = c.TOKEN_WAIT_READ if action == "READ" else c.TOKEN_WRITE_GENERATE
    sequence = [*header, action_token]
    response_positions = []
    action_label_position = len(sequence) - 2
    if action == "WRITE":
        delta = [int(value) for value in row["target_text_delta_ids"]]
        if not delta:
            raise ValueError("natural WRITE has empty target delta")
        suffix = [
            c.language_token_id(str(row["tgt_lang"])),
            c.speed_token_id(1.0),
            c.TOKEN_START_CONTENT,
            *delta,
            c.TOKEN_END_CONTENT,
        ]
        start = len(sequence) - 1
        sequence.extend(suffix)
        response_positions.extend(range(start, len(sequence) - 1))
    sequence.append(c.TOKEN_EOS)
    tokens = sequence[:-1]
    labels = sequence[1:]
    response_mask = [0.0] * len(tokens)
    action_mask = [0.0] * len(tokens)
    action_mask[action_label_position] = 1.0
    for position in response_positions:
        if 0 <= position < len(response_mask) and position != action_label_position:
            response_mask[position] = 1.0
    return {
        "tokens": tokens,
        "labels": labels,
        "position_ids": list(range(len(tokens))),
        "response_mask": response_mask,
        "action_mask": action_mask,
        "replay_mask": [0.0] * len(tokens),
        "family_ids": [4] * len(tokens),
        "identity": event_identity(row),
        "action": action,
    }


def replay_pack(row: dict[str, object], seq_length: int) -> dict[str, object]:
    if len(row["tokens"]) != seq_length or len(row["labels"]) != seq_length:
        raise ValueError("Phase3 replay pack length differs from seq_length")
    kinds = [int(value) for value in row["loss_kinds"]]
    mask = [1.0 if value == LOSS_REPLAY else 0.0 for value in kinds]
    if not any(mask):
        raise ValueError("Phase3 replay record has no replay tokens")
    used = int(row["used_tokens"])
    return {
        "schema_version": SCHEMA,
        "tokens": [int(value) for value in row["tokens"]],
        "labels": [int(value) for value in row["labels"]],
        "position_ids": [int(value) for value in row["position_ids"]],
        "response_mask": [0.0] * seq_length,
        "action_mask": [0.0] * seq_length,
        "replay_mask": mask,
        "family_ids": [5 if index < used else 0 for index in range(seq_length)],
        "loss_mask": mask,
        "sample_boundaries": row["sample_boundaries"],
        "identities": [f"phase3:{value}" for value in row["source_ids"]],
        "used_tokens": used,
        "action_tokens": 0,
        "response_tokens": 0,
        "replay_tokens": int(sum(mask)),
    }


def finalize_pack(pack: dict[str, object], seq_length: int) -> dict[str, object]:
    used = len(pack["tokens"])
    padding = seq_length - used
    pack["tokens"].extend([c.TOKEN_PAD] * padding)
    pack["labels"].extend([c.TOKEN_PAD] * padding)
    pack["position_ids"].extend([0] * padding)
    for field in ("response_mask", "action_mask", "replay_mask"):
        pack[field].extend([0.0] * padding)
    pack["family_ids"].extend([0] * padding)
    pack.update(
        {
            "schema_version": SCHEMA,
            "used_tokens": used,
            "action_tokens": int(sum(pack["action_mask"])),
            "response_tokens": int(sum(pack["response_mask"])),
            "replay_tokens": int(sum(pack["replay_mask"])),
            "loss_mask": [
                float(bool(action) or bool(response) or bool(replay))
                for action, response, replay in zip(
                    pack["action_mask"], pack["response_mask"], pack["replay_mask"]
                )
            ],
        }
    )
    return pack


def pack_samples(samples: Iterable[dict[str, object]], seq_length: int):
    current = None
    for sample in samples:
        length = len(sample["tokens"])
        if length > seq_length:
            raise ValueError(f"action sample exceeds seq_length: {sample['identity']}")
        if current is None or len(current["tokens"]) + length > seq_length:
            if current is not None:
                yield finalize_pack(current, seq_length)
            current = {
                "tokens": [],
                "labels": [],
                "position_ids": [],
                "response_mask": [],
                "action_mask": [],
                "replay_mask": [],
                "family_ids": [],
                "sample_boundaries": [],
                "identities": [],
            }
        start = len(current["tokens"])
        for field in (
            "tokens",
            "labels",
            "position_ids",
            "response_mask",
            "action_mask",
            "replay_mask",
            "family_ids",
        ):
            current[field].extend(sample[field])
        current["sample_boundaries"].append([start, start + length])
        current["identities"].append(sample["identity"])
    if current is not None:
        yield finalize_pack(current, seq_length)


def split_events(path: Path, valid_modulus: int = 20):
    train: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    counts = {"train": {"READ": 0, "WRITE": 0}, "valid": {"READ": 0, "WRITE": 0}}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.blake2b(event_identity(row).encode(), digest_size=8).digest()
            split = "valid" if int.from_bytes(digest, "big") % valid_modulus == 0 else "train"
            (valid if split == "valid" else train).append(row)
            counts[split][str(row["natural_action_target"])] += 1
    if not train or not valid:
        raise ValueError("event split produced an empty partition")
    return train, valid, counts


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(path)
    positions: list[int] = []
    counts = {"records": 0, "used_tokens": 0, "action_tokens": 0, "response_tokens": 0, "replay_tokens": 0}
    with path.open("wb") as handle:
        for row in rows:
            positions.append(handle.tell())
            handle.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            counts["records"] += 1
            for name in ("used_tokens", "action_tokens", "response_tokens", "replay_tokens"):
                counts[name] += int(row[name])
    index = path.with_suffix(path.suffix + ".offsets.bin")
    index.write_bytes(struct.pack(f"<{len(positions)}Q", *positions))
    return {
        **counts,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "offset_sha256": file_sha256(index),
    }


def interleave(action_packs: list[dict[str, object]], replay_packs: list[dict[str, object]]):
    replay_index = 0
    for index, row in enumerate(action_packs):
        yield row
        if (index + 1) % 2 == 0 and replay_index < len(replay_packs):
            yield replay_packs[replay_index]
            replay_index += 1
    yield from replay_packs[replay_index:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--phase3-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--valid-modulus", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    train_events, valid_events, counts = split_events(args.events, args.valid_modulus)
    train_action = list(pack_samples((action_sample(row) for row in train_events), args.seq_length))
    valid_action = list(pack_samples((action_sample(row) for row in valid_events), args.seq_length))
    replay_available = indexed_offsets(args.phase3_replay)
    replay_count = min((len(train_action) + 1) // 2, len(replay_available))
    replay = [
        replay_pack(row, args.seq_length)
        for row in read_indexed(args.phase3_replay, range(replay_count))
    ]
    train_summary = write_jsonl(
        args.output / "train_packs.jsonl", interleave(train_action, replay)
    )
    valid_summary = write_jsonl(args.output / "valid_packs.jsonl", valid_action)
    report = {
        "schema_version": "uniss_event_action_warmup_dataset_v2",
        "status": "passed",
        "seq_length": args.seq_length,
        "source_events": str(args.events.resolve()),
        "event_counts": counts,
        "train_action_packs": len(train_action),
        "valid_action_packs": len(valid_action),
        "phase3_replay_packs": len(replay),
        "train": train_summary,
        "valid": valid_summary,
    }
    (args.output / "AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
