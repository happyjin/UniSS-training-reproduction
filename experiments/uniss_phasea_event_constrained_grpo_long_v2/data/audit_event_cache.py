#!/usr/bin/env python3
"""Parallel, streaming audit of the immutable pilot15 trajectory cache.

The audit reads every JSONL row but copies neither source audio nor NPZ teacher
bundles.  It writes compact deterministic candidates for balanced action SFT
and retains every row whose sample belongs to the selected long episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable


SCHEMA = "uniss_event_cache_audit_v2"
CATEGORIES = (
    "cmn->eng:READ",
    "cmn->eng:WRITE",
    "eng->cmn:READ",
    "eng->cmn:WRITE",
)


def stable_key(sample_id: str, chunk_end_ms: int) -> int:
    payload = f"{sample_id}:{int(chunk_end_ms)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def episode_sample_ids(path: Path, episode_ids: set[str]) -> set[str]:
    values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["episode_id"]) not in episode_ids:
                continue
            values.update(str(component["sample_id"]) for component in row["components"])
    return values


def baseline_episode_ids(path: Path, maximum: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [str(row["episode_id"]) for row in payload["summaries"]]
    if len(values) < maximum:
        raise ValueError(f"baseline contains only {len(values)} episodes")
    return values[:maximum]


def compact(row: dict[str, object], *, targeted: bool) -> dict[str, object]:
    return {
        "sample_id": str(row["sample_id"]),
        "shard": int(row["shard"]),
        "row_index": int(row["row_index"]),
        "src_lang": str(row["src_lang"]),
        "tgt_lang": str(row["tgt_lang"]),
        "source_duration_ms": int(row["source_duration_ms"]),
        "chunk_end_ms": int(row["chunk_end_ms"]),
        "causal_source_glm": [int(value) for value in row["causal_source_glm"]],
        "translation_ids": [int(value) for value in row["translation_ids"]],
        "speaker_global": [int(value) for value in row["speaker_global"]],
        "previous_committed_length": int(row["previous_committed_length"]),
        "stable_target_length": int(row["stable_target_length"]),
        "new_supported_count": int(row["new_supported_count"]),
        "support_bucket": int(row["support_bucket"]),
        "safe_commit_mask": [bool(value) for value in row["safe_commit_mask"]],
        "natural_action_target": str(row["natural_action_target"]),
        "deadline_action_target": str(row["deadline_action_target"]),
        "deadline_forced_target": bool(row["deadline_forced_target"]),
        "deadline_loss_enabled": bool(row["deadline_loss_enabled"]),
        "target_text_delta_ids": [int(value) for value in row["target_text_delta_ids"]],
        "soft_deadline_ms": int(row["soft_deadline_ms"]),
        "hard_deadline_ms": int(row["hard_deadline_ms"]),
        "quality_flags": [str(value) for value in row.get("quality_flags", [])],
        "targeted_long_episode_component": bool(targeted),
        "selection_key": stable_key(str(row["sample_id"]), int(row["chunk_end_ms"])),
    }


def structural_violations(row: dict[str, object]) -> list[str]:
    failures: list[str] = []
    action = str(row.get("natural_action_target"))
    deadline = str(row.get("deadline_action_target"))
    delta = list(row.get("target_text_delta_ids", []))
    previous = int(row.get("previous_committed_length", -1))
    stable = int(row.get("stable_target_length", -1))
    supported = int(row.get("new_supported_count", -1))
    translation = list(row.get("translation_ids", []))
    safe = list(row.get("safe_commit_mask", []))
    chunk = int(row.get("chunk_end_ms", -1))
    soft = int(row.get("soft_deadline_ms", -1))
    hard = int(row.get("hard_deadline_ms", -1))
    if action not in {"READ", "WRITE"} or deadline not in {"READ", "WRITE"}:
        failures.append("invalid_action")
    if not 0 <= previous <= stable <= len(translation):
        failures.append("invalid_commit_lengths")
    if supported != stable - previous:
        failures.append("supported_count_mismatch")
    if len(safe) != len(translation):
        failures.append("safe_mask_geometry")
    if action == "WRITE" and (supported <= 0 or not delta):
        failures.append("natural_write_without_delta")
    if action == "READ" and (supported != 0 or delta):
        failures.append("natural_read_with_delta")
    if delta != translation[previous:stable]:
        failures.append("delta_slice_mismatch")
    if not 0 < chunk <= int(row.get("source_duration_ms", -1)):
        failures.append("invalid_chunk_time")
    if not 0 < soft <= hard:
        failures.append("invalid_deadline")
    forced = bool(row.get("deadline_forced_target", False))
    if forced and not (action == "READ" and deadline == "WRITE"):
        failures.append("invalid_forced_deadline")
    return failures


def audit_part(
    path_value: str,
    target_ids: set[str],
    candidate_root_value: str,
    per_category_quota: int,
) -> dict[str, object]:
    path = Path(path_value)
    candidate_root = Path(candidate_root_value)
    heaps: dict[str, list[tuple[int, int, dict[str, object]]]] = {
        category: [] for category in CATEGORIES
    }
    targets: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    target_samples: Counter[str] = Counter()
    last_chunk: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                violations["invalid_json"] += 1
                continue
            direction = f"{row.get('src_lang')}->{row.get('tgt_lang')}"
            action = str(row.get("natural_action_target"))
            category = f"{direction}:{action}"
            directions[direction] += 1
            actions[action] += 1
            categories[category] += 1
            counts["deadline_forced"] += int(bool(row.get("deadline_forced_target")))
            counts["deadline_loss_enabled"] += int(bool(row.get("deadline_loss_enabled")))
            for failure in structural_violations(row):
                violations[failure] += 1
            for flag in row.get("quality_flags", []):
                quality_flags[str(flag)] += 1
            sample_id = str(row.get("sample_id"))
            chunk = int(row.get("chunk_end_ms", -1))
            if sample_id in last_chunk and chunk <= last_chunk[sample_id]:
                violations["non_monotonic_sample_time"] += 1
            last_chunk[sample_id] = chunk
            targeted = sample_id in target_ids
            value = compact(row, targeted=targeted)
            if targeted:
                targets.append(value)
                target_samples[sample_id] += 1
            if category not in heaps or value["quality_flags"]:
                continue
            key = int(value["selection_key"])
            heap = heaps[category]
            entry = (-key, line_index, value)
            if len(heap) < per_category_quota:
                heapq.heappush(heap, entry)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    part_name = path.parent.name
    candidate_path = candidate_root / f"{part_name}.jsonl"
    selected: dict[tuple[str, int], dict[str, object]] = {}
    for heap in heaps.values():
        for _, _, row in heap:
            selected[(str(row["sample_id"]), int(row["chunk_end_ms"]))] = row
    for row in targets:
        selected[(str(row["sample_id"]), int(row["chunk_end_ms"]))] = row
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in sorted(
            selected.values(), key=lambda item: (int(item["selection_key"]), str(item["sample_id"]))
        ):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "path": str(path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "counts": dict(counts),
        "directions": dict(directions),
        "actions": dict(actions),
        "categories": dict(categories),
        "violations": dict(violations),
        "quality_flags": dict(quality_flags),
        "target_samples": dict(target_samples),
        "candidate_rows": len(selected),
    }


def add_counters(reports: Iterable[dict[str, object]], field: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for report in reports:
        total.update({str(key): int(value) for key, value in report[field].items()})  # type: ignore[union-attr]
    return dict(sorted(total.items()))


def merge_candidates(
    paths: list[Path],
    output: Path,
    per_category_quota: int,
) -> dict[str, object]:
    heaps: dict[str, list[tuple[int, str, dict[str, object]]]] = {
        category: [] for category in CATEGORIES
    }
    targeted: dict[tuple[str, int], dict[str, object]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = (str(row["sample_id"]), int(row["chunk_end_ms"]))
                if bool(row["targeted_long_episode_component"]):
                    targeted[identity] = row
                category = f"{row['src_lang']}->{row['tgt_lang']}:{row['natural_action_target']}"
                if category not in heaps or row["quality_flags"]:
                    continue
                key = int(row["selection_key"])
                heap = heaps[category]
                entry = (-key, f"{identity[0]}:{identity[1]}", row)
                if len(heap) < per_category_quota:
                    heapq.heappush(heap, entry)
                elif key < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    selected = dict(targeted)
    selected_categories: Counter[str] = Counter()
    for category, heap in heaps.items():
        for _, _, row in heap:
            identity = (str(row["sample_id"]), int(row["chunk_end_ms"]))
            selected[identity] = row
            selected_categories[category] += 1
    with output.open("w", encoding="utf-8") as handle:
        for row in sorted(
            selected.values(), key=lambda item: (int(item["selection_key"]), str(item["sample_id"]))
        ):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "rows": len(selected),
        "targeted_rows": len(targeted),
        "targeted_samples": len({identity[0] for identity in targeted}),
        "balanced_category_rows_before_target_union": dict(sorted(selected_categories.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-episodes", type=int, default=64)
    parser.add_argument("--per-category-quota", type=int, default=16_000)
    parser.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.per_category_quota <= 0 or args.workers <= 0:
        raise ValueError("quota and workers must be positive")
    parts = sorted(args.cache_root.glob("part-*/trajectory_cache.jsonl"))
    if len(parts) != 15:
        raise ValueError(f"expected 15 cache parts, found {len(parts)}")
    episode_ids = baseline_episode_ids(args.baseline_rollout, args.maximum_episodes)
    if len(set(episode_ids)) != args.maximum_episodes:
        raise ValueError("selected episode IDs are not unique")
    target_ids = episode_sample_ids(args.episodes, set(episode_ids))
    args.output.mkdir(parents=True)
    candidate_root = args.output / "part_candidates"
    candidate_root.mkdir()
    per_part = max(1_000, (args.per_category_quota + len(parts) - 1) // len(parts) * 3)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(parts))) as pool:
        reports = list(
            pool.map(
                audit_part,
                (str(path) for path in parts),
                (target_ids for _ in parts),
                (str(candidate_root) for _ in parts),
                (per_part for _ in parts),
            )
        )
    filtered = args.output / "filtered_events.jsonl"
    selection = merge_candidates(
        [Path(report["candidate_path"]) for report in reports],
        filtered,
        args.per_category_quota,
    )
    target_counts = add_counters(reports, "target_samples")
    missing_targets = sorted(target_ids - set(target_counts))
    report = {
        "schema_version": SCHEMA,
        "status": "passed" if not missing_targets else "failed",
        "cache_root": str(args.cache_root.resolve()),
        "source_parts": len(parts),
        "selected_episode_ids": episode_ids,
        "selected_episode_count": len(episode_ids),
        "target_component_samples": len(target_ids),
        "missing_target_component_samples": missing_targets,
        "full_cache": {
            "counts": add_counters(reports, "counts"),
            "directions": add_counters(reports, "directions"),
            "actions": add_counters(reports, "actions"),
            "categories": add_counters(reports, "categories"),
            "violations": add_counters(reports, "violations"),
            "quality_flags": add_counters(reports, "quality_flags"),
        },
        "selection": selection,
        "filtered_events": str(filtered.resolve()),
        "part_reports": reports,
        "claim_boundary": (
            "Structural causal-label invariants were audited. This does not prove that an "
            "external teacher implementation never accessed future information."
        ),
    }
    (args.output / "AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if missing_targets:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

