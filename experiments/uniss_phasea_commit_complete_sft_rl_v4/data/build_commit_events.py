#!/usr/bin/env python3
"""Derive auditable non-empty COMMIT supervision from immutable event cache.

The source rows are never modified.  A WRITE label is retained only when its
teacher-supported target delta is both safe and sufficiently long to be a
usable speech unit; every other event becomes WAIT.  This explicitly prevents
the empty-WRITE target that dominated the v3 free-running policy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def split_for(identity: str, valid_modulus: int) -> str:
    digest = hashlib.blake2b(identity.encode(), digest_size=8).digest()
    return "valid" if int.from_bytes(digest, "big") % valid_modulus == 0 else "train"


def relabel(row: dict[str, object], minimum_delta_tokens: int) -> dict[str, object]:
    value = dict(row)
    delta = [int(token) for token in value["target_text_delta_ids"]]
    safe = bool(any(value["safe_commit_mask"]))
    supported = safe and len(delta) >= int(minimum_delta_tokens)
    value["natural_action_target"] = "WRITE" if supported else "READ"
    value["commit_target"] = "COMMIT" if supported else "WAIT"
    value["commit_delta_tokens"] = len(delta) if supported else 0
    value["commit_label_reason"] = (
        "safe_nonempty_teacher_delta" if supported else "no_safe_speakable_delta"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta-tokens", type=int, default=2)
    parser.add_argument("--valid-modulus", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.minimum_delta_tokens < 1 or args.valid_modulus < 2:
        raise ValueError("invalid commit-label geometry")
    counts: collections.Counter[str] = collections.Counter()
    split_counts: dict[str, collections.Counter[str]] = {
        "train": collections.Counter(), "valid": collections.Counter()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = relabel(json.loads(line), args.minimum_delta_tokens)
            identity = f"{row['sample_id']}:{row['chunk_end_ms']}"
            partition = split_for(identity, args.valid_modulus)
            counts[str(row["commit_target"])] += 1
            split_counts[partition][str(row["commit_target"])] += 1
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if not all(counts[action] for action in ("WAIT", "COMMIT")):
        raise ValueError("commit relabeling collapsed an action class")
    report = {
        "schema_version": "uniss_content_gated_commit_events_v4",
        "status": "passed",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "minimum_delta_tokens": args.minimum_delta_tokens,
        "events": dict(counts),
        "splits": {key: dict(value) for key, value in split_counts.items()},
    }
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
