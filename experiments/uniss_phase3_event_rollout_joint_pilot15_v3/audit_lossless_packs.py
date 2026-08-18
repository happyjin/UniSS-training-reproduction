#!/usr/bin/env python3
"""Audit raw/canonical WRITE parity over indexed fixed15 pack manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MultiFilePackIndex,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    IndexedDenseTrajectoryDataset,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.event_rollout import (
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v3.training.dataset import (
    canonical_runtime_pack,
)


SCHEMA = "uniss_event_rollout_pilot15_v3_lossless_audit_v1"


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit_manifest(
    manifest: str | Path,
    *,
    expected_split: str,
    maximum_packs: int | None = None,
) -> dict[str, object]:
    namespace = MultiFilePackIndex(manifest, expected_split=expected_split)
    total = len(namespace) if maximum_packs is None else min(len(namespace), maximum_packs)
    readers = [
        IndexedDenseTrajectoryDataset(part.packed, seq_length=18_000)
        for part in namespace.parts
    ]
    counters: Counter[str] = Counter()
    semantic_lengths: list[int] = []
    maximum_semantic = 0
    for global_index in range(total):
        part_index, local_index = namespace.resolve(global_index)
        raw = readers[part_index]._read(local_index)
        canonical = canonical_runtime_pack(raw)
        raw_sessions = oracle_sessions_from_pack(raw)
        canonical_sessions = oracle_sessions_from_pack(canonical)
        if len(raw_sessions) != len(canonical_sessions):
            raise AssertionError("canonical session count changed")
        counters["packs"] += 1
        counters["sessions"] += len(raw_sessions)
        for before, after in zip(raw_sessions, canonical_sessions):
            if before.sample_id != after.sample_id or len(before.events) != len(after.events):
                raise AssertionError("canonical session identity/event count changed")
            for raw_event, canonical_event in zip(before.events, after.events):
                counters["events"] += 1
                if raw_event.action == "WRITE":
                    counters["raw_writes"] += 1
                    parsed = parse_write_outcome(raw_event.outcome_tokens)
                    semantic_lengths.append(len(parsed.semantic_codes))
                    maximum_semantic = max(maximum_semantic, len(parsed.semantic_codes))
                    if not parsed.text_ids:
                        counters["semantic_only_writes"] += 1
                else:
                    counters["raw_waits"] += 1
                if canonical_event.action == "WRITE":
                    counters["canonical_writes"] += 1
                else:
                    counters["canonical_waits"] += 1
                if (
                    raw_event.action != canonical_event.action
                    or raw_event.outcome_tokens != canonical_event.outcome_tokens
                ):
                    counters["event_mismatches"] += 1
                    raise AssertionError("canonical event differs from raw event")
    status = "pass" if (
        counters["raw_writes"] == counters["canonical_writes"]
        and counters["raw_waits"] == counters["canonical_waits"]
        and counters["event_mismatches"] == 0
        and maximum_semantic <= 24
    ) else "fail"
    return {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest": str(Path(manifest).resolve()),
        "split": expected_split,
        "sampled": maximum_packs is not None and total < len(namespace),
        "audited_packs": total,
        "available_packs": len(namespace),
        **dict(counters),
        "write_rate": counters["raw_writes"] / max(1, counters["events"]),
        "semantic_only_write_fraction": counters["semantic_only_writes"]
        / max(1, counters["raw_writes"]),
        "semantic_length": {
            "count": len(semantic_lengths),
            "mean": statistics.fmean(semantic_lengths) if semantic_lengths else None,
            "p50": percentile(semantic_lengths, 0.50),
            "p95": percentile(semantic_lengths, 0.95),
            "p99": percentile(semantic_lengths, 0.99),
            "max": maximum_semantic,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--maximum-packs", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_manifest(
        args.manifest,
        expected_split=args.split,
        maximum_packs=args.maximum_packs,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
