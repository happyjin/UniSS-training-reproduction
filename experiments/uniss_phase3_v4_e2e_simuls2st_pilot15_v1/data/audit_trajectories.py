#!/usr/bin/env python3
"""Stream and validate every serialized E2E trajectory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    validate_trajectory,
)


AUDIT_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_trajectory_audit_v1"


def audit_file(
    path: Path,
    *,
    require_audio_hash: bool,
    require_v1_rollout: bool,
    limit: int | None = None,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if limit is not None and line_index >= limit:
                break
            trajectory = E2ETrajectory.from_mapping(json.loads(line))
            if trajectory.sample_id in seen_ids:
                raise ValueError(f"duplicate sample ID: {trajectory.sample_id}")
            seen_ids.add(trajectory.sample_id)
            metrics = validate_trajectory(
                trajectory,
                require_audio_hash=require_audio_hash,
                require_v1_rollout=require_v1_rollout,
            )
            counts["records"] += 1
            counts["events"] += int(metrics["events"])
            counts["source_glm_tokens"] += int(metrics["source_glm_tokens"])
            counts["target_semantic_tokens"] += int(metrics["target_semantic_tokens"])
            counts["prefinal_target_writes"] += int(metrics["prefinal_target_writes"])
            counts[f"direction:{trajectory.src_lang}-{trajectory.tgt_lang}"] += 1
    if not counts["records"]:
        raise ValueError("trajectory audit selected no records")
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "path": str(path.resolve()),
        "require_audio_hash": require_audio_hash,
        "require_v1_rollout": require_v1_rollout,
        "counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-audio-hash", action="store_true")
    parser.add_argument("--require-v1-rollout", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = audit_file(
        args.input,
        require_audio_hash=args.require_audio_hash,
        require_v1_rollout=args.require_v1_rollout,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
