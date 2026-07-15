"""Deterministically mix UniSS sample JSONL files by integer ratios.

Phase 2 in the paper mixes new S2ST data with Phase 1 alignment data at a 2:1
ratio. This utility performs that mix before sequence packing, preserving the
standard sample schema and adding a small ``mix_group`` audit field.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class MixGroup:
    name: str
    weight: int
    paths: tuple[Path, ...]


def parse_group_spec(spec: str) -> MixGroup:
    try:
        name_weight, paths_text = spec.split(":", 1)
        name, weight_text = name_weight.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            "Group spec must look like 'name=weight:path1[,path2...]'"
        ) from exc
    name = name.strip()
    if not name:
        raise ValueError("Group name must be non-empty")
    try:
        weight = int(weight_text)
    except ValueError as exc:
        raise ValueError(f"Group weight must be an integer: {weight_text!r}") from exc
    if weight <= 0:
        raise ValueError("Group weight must be positive")
    paths = tuple(Path(part) for part in paths_text.split(",") if part)
    if not paths:
        raise ValueError("Group must include at least one JSONL path")
    return MixGroup(name=name, weight=weight, paths=paths)


def iter_jsonl(paths: tuple[Path, ...]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL row must be an object at {path}:{line_no}")
                yield value


def mix_groups(
    groups: list[MixGroup],
    max_records: int | None = None,
    add_mix_group: bool = True,
) -> Iterator[dict[str, object]]:
    if not groups:
        raise ValueError("At least one mix group is required")
    iterators = {group.name: iter_jsonl(group.paths) for group in groups}
    active = list(groups)
    emitted = 0

    while active:
        next_active: list[MixGroup] = []
        for group in active:
            produced = 0
            for _ in range(group.weight):
                try:
                    row = next(iterators[group.name])
                except StopIteration:
                    break
                if add_mix_group:
                    row = dict(row)
                    row["mix_group"] = group.name
                yield row
                emitted += 1
                produced += 1
                if max_records is not None and emitted >= max_records:
                    return
            if produced == group.weight:
                next_active.append(group)
        active = next_active


def write_jsonl(samples: Iterator[Mapping[str, object]], output_path: Path) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            group = str(sample.get("mix_group", "unknown"))
            task = str(sample.get("task", "unknown"))
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            counts[group] += 1
            counts[f"task:{task}"] += 1
            counts["total"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Mix group spec: name=weight:path1[,path2...]",
    )
    parser.add_argument("--output", required=True, help="Mixed sample JSONL path")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-add-mix-group", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = [parse_group_spec(spec) for spec in args.group]
    mixed = mix_groups(
        groups,
        max_records=args.max_records,
        add_mix_group=not args.no_add_mix_group,
    )
    counts = write_jsonl(mixed, Path(args.output))
    print(json.dumps({"output": args.output, "counts": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
