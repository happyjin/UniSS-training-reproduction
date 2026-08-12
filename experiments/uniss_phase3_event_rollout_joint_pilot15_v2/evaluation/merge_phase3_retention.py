#!/usr/bin/env python3
"""Merge paired Phase3 retention shards and reject missing system outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from evaluation.io_utils import iter_jsonl, write_json, write_jsonl
from evaluation.sharding import row_key
from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.evaluate_phase3_retention import (
    SYSTEMS,
)


def merge(parts: Sequence[Path], output_root: Path) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite retention aggregate: {output_root}")
    rows = []
    keys = set()
    for part in parts:
        for row in iter_jsonl(part):
            key = row_key(row)
            if key in keys:
                raise ValueError(f"duplicate retention result: {key}")
            keys.add(key)
            rows.append(row)
    by_sample: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_sample[str(row["id"])].add(str(row["mode"]))
    expected_systems = set(SYSTEMS)
    incomplete = {
        sample_id: sorted(expected_systems - systems)
        for sample_id, systems in by_sample.items()
        if systems != expected_systems
    }
    if incomplete:
        raise ValueError(f"unpaired Phase3 retention samples: {dict(list(incomplete.items())[:8])}")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    metrics = {}
    for system, values in sorted(grouped.items()):
        metrics[system] = {
            "samples": len(values),
            "generated_text_rate": fmean(bool(str(row.get("generated_translation") or "").strip()) for row in values),
            "semantic_output_rate": fmean(int(row.get("semantic_token_count", 0) or 0) > 0 for row in values),
            "eos_rate": fmean(bool(row.get("has_eos")) for row in values),
            "playable_audio_rate": fmean(bool(row.get("audio_path")) and not row.get("error") for row in values),
            "finite_audio_rate": fmean(bool(row.get("audio_finite")) for row in values),
            "non_silent_audio_rate": fmean(float(row.get("audio_non_silent_fraction", 0.0) or 0.0) >= 0.01 for row in values),
            "mean_generation_seconds": fmean(float(row["generation_seconds"]) for row in values),
        }
    report = {
        "schema_version": "uniss_event_rollout_fixed15_phase3_retention_aggregate_v1",
        "samples": len(by_sample),
        "result_rows": len(rows),
        "systems": dict(sorted(Counter(str(row["mode"]) for row in rows).items())),
        "paired_complete": True,
        "groups": metrics,
        "selection_note": (
            "These paired offline Phase3 prompts measure replay retention. Streaming exact-runtime "
            "latency and quality are evaluated separately and must not be inferred from this report."
        ),
    }
    output_root.mkdir(parents=True)
    write_jsonl(output_root / "results.jsonl", rows)
    write_json(output_root / "aggregate.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.part, args.output_root), indent=2))


if __name__ == "__main__":
    main()
