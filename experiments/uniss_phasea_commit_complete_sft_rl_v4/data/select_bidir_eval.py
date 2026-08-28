#!/usr/bin/env python3
"""Select a deterministic balanced long-form train-seen diagnostic subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-direction", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.per_direction < 1:
        raise ValueError("output must be new and per-direction positive")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for line in args.episodes.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            groups[str(row["direction"])].append(row)
    chosen = []
    for direction in ("cmn->eng", "eng->cmn"):
        rows = groups[direction]
        if len(rows) < args.per_direction:
            raise ValueError(f"not enough {direction} episodes")
        chosen.extend(rows[: args.per_direction])
    chosen.sort(key=lambda row: str(row["episode_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in chosen),
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "episodes": len(chosen), "directions": sorted(groups)}, sort_keys=True))


if __name__ == "__main__":
    main()
