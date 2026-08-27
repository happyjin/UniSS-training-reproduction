#!/usr/bin/env python3
"""Freeze the existing 64-episode protocol and map audited local events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_events(path: Path):
    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row["targeted_long_episode_component"]):
                output[str(row["sample_id"])].append(row)
    for rows in output.values():
        rows.sort(key=lambda row: int(row["chunk_end_ms"]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--audited-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-episodes", type=int, default=64)
    parser.add_argument("--boundary-mask-ms", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.baseline_rollout.read_text(encoding="utf-8"))
    requested = [str(row["episode_id"]) for row in payload["summaries"]][
        : args.maximum_episodes
    ]
    requested_set = set(requested)
    episodes = {}
    with args.episodes.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["episode_id"]) in requested_set:
                episodes[str(row["episode_id"])] = row
    missing = sorted(requested_set - set(episodes))
    if missing:
        raise ValueError(f"episode manifest is missing {missing}")
    event_map = load_events(args.audited_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    direction_counts: dict[str, int] = defaultdict(int)
    mapped_total = 0
    component_total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for episode_id in requested:
            row = episodes[episode_id]
            cursor = 0
            mapped: list[dict[str, object]] = []
            components: list[dict[str, object]] = []
            for index, component in enumerate(row["components"]):
                duration = int(component["duration_ms"])
                start = cursor
                stop = start + duration
                component_copy = dict(component)
                component_copy.update(
                    {
                        "component_index": index,
                        "global_start_ms": start,
                        "global_end_ms": stop,
                        "boundary_mask_start_ms": max(0, start - args.boundary_mask_ms),
                        "boundary_mask_end_ms": min(
                            int(row["duration_ms"]), stop + args.boundary_mask_ms
                        ),
                    }
                )
                components.append(component_copy)
                sample_id = str(component["sample_id"])
                local_rows = event_map.get(sample_id, [])
                if not local_rows:
                    raise ValueError(f"audited events missing component {sample_id}")
                for event in local_rows:
                    global_ms = start + int(event["chunk_end_ms"])
                    mapped.append(
                        {
                            "component_index": index,
                            "sample_id": sample_id,
                            "local_chunk_end_ms": int(event["chunk_end_ms"]),
                            "global_chunk_end_ms": global_ms,
                            "natural_action_target": str(event["natural_action_target"]),
                            "deadline_action_target": str(event["deadline_action_target"]),
                            "deadline_forced_target": bool(event["deadline_forced_target"]),
                            "safe_commit": bool(any(event["safe_commit_mask"])),
                            "target_text_delta_ids": event["target_text_delta_ids"],
                            "boundary_masked": bool(
                                global_ms - start <= args.boundary_mask_ms
                                or stop - global_ms <= args.boundary_mask_ms
                            ),
                        }
                    )
                # ``gap_ms`` is the builder's boundary/cross-fade geometry;
                # it is not extra silence.  The audited long manifests have
                # duration_ms == sum(component.duration_ms), so adding it here
                # would drift every downstream READ/WRITE timestamp.
                cursor = stop
            if abs(cursor - int(row["duration_ms"])) > 1:
                raise ValueError(
                    f"component timeline differs for {episode_id}: {cursor} vs {row['duration_ms']}"
                )
            mapped.sort(key=lambda event: int(event["global_chunk_end_ms"]))
            output = dict(row)
            output["components"] = components
            output["mapped_action_events"] = mapped
            output["mapped_event_count"] = len(mapped)
            output["schema_version"] = "uniss_event_grpo_selected_episode_v2"
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            direction_counts[str(row["direction"])] += 1
            mapped_total += len(mapped)
            component_total += len(components)
    report = {
        "schema_version": "uniss_event_grpo_episode_selection_v2",
        "status": "passed",
        "episodes": len(requested),
        "directions": dict(sorted(direction_counts.items())),
        "components": component_total,
        "mapped_action_events": mapped_total,
        "boundary_mask_ms": args.boundary_mask_ms,
        "output": str(args.output.resolve()),
        "source_audio_copied": False,
        "claim_boundary": "All selected episodes are train-seen and do not establish generalization.",
    }
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
