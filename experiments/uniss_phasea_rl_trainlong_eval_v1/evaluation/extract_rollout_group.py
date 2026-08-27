#!/usr/bin/env python3
"""Extract one deterministic pre-RL candidate group for a fixed protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def extract(
    rollout: dict[str, Any], protocol: dict[str, Any], group_index: int
) -> list[dict[str, Any]]:
    summaries = {str(row["episode_id"]): row for row in rollout["summaries"]}
    results: list[dict[str, Any]] = []
    for record in protocol["records"]:
        episode_id = str(record["episode_id"])
        if episode_id not in summaries:
            raise ValueError(f"protocol episode missing from rollout: {episode_id}")
        candidates = [
            value
            for value in summaries[episode_id]["candidates"]
            if int(value["group_index"]) == group_index
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one group-{group_index} candidate for {episode_id}, got {len(candidates)}"
            )
        result = copy.deepcopy(candidates[0]["result"])
        if Path(str(result["source_audio"])).resolve() != Path(
            str(record["source_audio"])
        ).resolve():
            raise ValueError(f"rollout/protocol source differs for {episode_id}")
        result["rollout_candidate_sample_id"] = str(result["sample_id"])
        result["sample_id"] = episode_id
        result["pre_rl_candidate_group"] = group_index
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    results = extract(rollout, protocol, args.group_index)
    payload = {
        "schema_version": "uniss_phasea_rl_prerl_rollout_group_extract_v1",
        "status": "complete",
        "run_id": f"phasea_prerl_rollout_group{args.group_index}_runtime_v2",
        "claim_boundary": (
            "This is a saved pre-RL Phase A rollout candidate, not a regenerated formal "
            "Phase A/iter15/iter30/iter45 comparison. It is provisional evidence only."
        ),
        "adapter_manifest": {
            "enabled": False,
            "source": str(args.rollout.resolve()),
            "candidate_group": args.group_index,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
