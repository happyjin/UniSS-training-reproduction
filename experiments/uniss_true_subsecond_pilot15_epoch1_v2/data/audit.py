#!/usr/bin/env python3
"""Audit repaired cache gates before any training launch."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.build_cache import CACHE_PART_SCHEMA
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import Action, TrajectoryRecord


AUDIT_SCHEMA = "uniss_true_subsecond_pilot15_data_audit_v2"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def audit(root: Path, shard_count: int = 15) -> dict[str, object]:
    if shard_count != 15:
        raise ValueError("pilot audit is frozen to shards 0..14")
    counts: Counter[str] = Counter()
    direction: dict[str, Counter[str]] = defaultdict(Counter)
    for shard in range(shard_count):
        part = root / f"part-{shard:03d}"
        marker = json.loads((part / "PART_COMPLETE.json").read_text(encoding="utf-8"))
        if marker.get("schema_version") != CACHE_PART_SCHEMA:
            raise ValueError(f"incomplete v2 cache shard {shard}")
        current_id: str | None = None
        records: list[TrajectoryRecord] = []

        def inspect_session(session: list[TrajectoryRecord]) -> None:
            if not session:
                return
            counts["sessions"] += 1
            times = [record.chunk_end_ms for record in session]
            if times != sorted(times) or len(times) != len(set(times)):
                counts["time_monotonic_violations"] += 1
            eligible = session[0].source_duration_ms >= 800
            exact = any(record.chunk_end_ms == 800 for record in session)
            if eligible:
                counts["deadline_eligible_sessions"] += 1
                counts["exact_800_sessions"] += int(exact)
            if any(record.deadline_loss_enabled for record in session) != (eligible and exact):
                counts["deadline_mask_violations"] += 1
            cursor = 0
            for record in session:
                counts["records"] += 1
                key = f"{record.src_lang}->{record.tgt_lang}"
                direction[key]["records"] += 1
                direction[key]["write"] += int(record.natural_action_target is Action.WRITE)
                counts["natural_write"] += int(record.natural_action_target is Action.WRITE)
                counts["deadline_forced"] += int(record.deadline_forced_target)
                safe_positive = sum(bool(value) for value in record.safe_commit_mask)
                counts["safe_positive"] += safe_positive
                counts["safe_total"] += len(record.safe_commit_mask)
                direction[key]["safe_positive"] += safe_positive
                direction[key]["safe_total"] += len(record.safe_commit_mask)
                if record.deadline_forced_target and record.chunk_end_ms != 800:
                    counts["forced_not_exact_800"] += 1
                if record.future_2_end_ms > record.source_duration_ms:
                    counts["future_leakage"] += 1
                if record.natural_action_target is Action.WRITE:
                    counts["natural_write_events"] += 1
                    if record.semantic_target_start != cursor:
                        counts["semantic_continuity_violations"] += 1
                    cursor = record.semantic_target_end

        with (part / "trajectory_cache.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = TrajectoryRecord.from_dict(json.loads(line))
                if current_id is not None and record.sample_id != current_id:
                    inspect_session(records)
                    records = []
                current_id = record.sample_id
                records.append(record)
        inspect_session(records)

    write_fraction = counts["natural_write"] / max(1, counts["records"])
    writes_per_session = counts["natural_write"] / max(1, counts["sessions"])
    forced_fraction = counts["deadline_forced"] / max(1, counts["records"])
    safe_positive_fraction = counts["safe_positive"] / max(1, counts["safe_total"])
    deadline_coverage = counts["exact_800_sessions"] / max(
        1, counts["deadline_eligible_sessions"]
    )
    direction_summary = {
        key: {
            "records": value["records"],
            "natural_write": value["write"],
            "natural_write_fraction": value["write"] / max(1, value["records"]),
            "safe_positive": value["safe_positive"],
            "safe_total": value["safe_total"],
            "safe_positive_fraction": value["safe_positive"] / max(1, value["safe_total"]),
        }
        for key, value in sorted(direction.items())
    }
    hard_failures = {
        "time_monotonic_violations": counts["time_monotonic_violations"],
        "deadline_mask_violations": counts["deadline_mask_violations"],
        "forced_not_exact_800": counts["forced_not_exact_800"],
        "future_leakage": counts["future_leakage"],
        "semantic_continuity_violations": counts["semantic_continuity_violations"],
    }
    gates = {
        "hard_failures_zero": all(value == 0 for value in hard_failures.values()),
        "exact_800_coverage_100pct": deadline_coverage == 1.0,
        # Five dense observations replace the old two-snapshot schedule. The
        # event-level fraction therefore falls even when WRITE/session is
        # unchanged; gate both quantities instead of reusing the v1 denominator.
        "natural_write_event_fraction_5_to_25pct": 0.05 <= write_fraction <= 0.25,
        "natural_write_per_session_0p25_to_2": 0.25 <= writes_per_session <= 2.0,
        "each_direction_write_at_least_3pct": all(
            value["natural_write_fraction"] >= 0.03 for value in direction_summary.values()
        ),
        "deadline_forced_at_most_35pct": forced_fraction <= 0.35,
        "safe_positive_nonzero_each_direction": all(
            value["safe_positive"] > 0 for value in direction_summary.values()
        ),
    }
    result = {
        "schema_version": AUDIT_SCHEMA,
        "shard_count": shard_count,
        "counts": dict(counts),
        "hard_failures": hard_failures,
        "deadline_coverage": deadline_coverage,
        "natural_write_fraction": write_fraction,
        "natural_writes_per_session": writes_per_session,
        "deadline_forced_fraction": forced_fraction,
        "safe_positive_fraction": safe_positive_fraction,
        "directions": direction_summary,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=15)
    args = parser.parse_args()
    result = audit(args.root, args.shard_count)
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
