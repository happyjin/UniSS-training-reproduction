#!/usr/bin/env python3
"""Compare cached/uncached and fused/unfused exact-runtime semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "uniss_event_rollout_fixed15_runtime_parity_v1"


def semantic_trace(row: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "event_index": int(event["event_index"]),
            "source_end_ms": int(event["source_end_ms"]),
            "source_finished": bool(event["source_finished"]),
            "new_source_codes": int(event["new_source_codes"]),
            "action": str(event["action"]),
            "text_ids": [int(value) for value in event.get("text_ids", [])],
            "semantic_codes": [int(value) for value in event.get("semantic_codes", [])],
            "continuation_choice": event.get("continuation_choice"),
        }
        for event in row.get("events", [])
    ]


def compare(summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if len(summaries) < 2:
        raise ValueError("runtime parity requires at least two modes")
    by_mode = {
        mode: {str(row["sample_id"]): row for row in summary.get("samples", [])}
        for mode, summary in summaries.items()
    }
    reference_mode = sorted(by_mode)[0]
    reference_ids = set(by_mode[reference_mode])
    failures: list[dict[str, object]] = []
    for mode, rows in sorted(by_mode.items()):
        if set(rows) != reference_ids:
            failures.append(
                {
                    "mode": mode,
                    "reason": "sample_id_set_mismatch",
                    "missing": sorted(reference_ids - set(rows)),
                    "extra": sorted(set(rows) - reference_ids),
                }
            )
            continue
        for sample_id in sorted(reference_ids):
            reference = by_mode[reference_mode][sample_id]
            candidate = rows[sample_id]
            checks = {
                "generated_text": candidate.get("generated_text") == reference.get("generated_text"),
                "semantic_trace": semantic_trace(candidate) == semantic_trace(reference),
                "natural_writes": candidate.get("natural_writes") == reference.get("natural_writes"),
                "natural_eos": candidate.get("natural_eos") == reference.get("natural_eos"),
                "forced_writes": candidate.get("forced_writes") == reference.get("forced_writes"),
            }
            if not all(checks.values()):
                failures.append(
                    {
                        "mode": mode,
                        "sample_id": sample_id,
                        "reason": "semantic_output_mismatch",
                        "checks": checks,
                    }
                )
    return {
        "schema_version": SCHEMA,
        "reference_mode": reference_mode,
        "modes": sorted(summaries),
        "samples": len(reference_ids),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "comparison_scope": (
            "Exact actions, committed text IDs, semantic IDs and EOS; wall-clock timing "
            "is intentionally excluded because cache/fusion changes compute latency."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-summary", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for value in args.mode_summary:
        mode, separator, raw_path = value.partition("=")
        if not separator or not mode or not raw_path:
            raise ValueError("--mode-summary must be MODE=/path/to/summary.json")
        if mode in summaries:
            raise ValueError(f"duplicate runtime parity mode: {mode}")
        summaries[mode] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    report = compare(summaries)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
