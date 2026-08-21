#!/usr/bin/env python3
"""Freeze a small, balanced validation subset for free-running E2E gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    validate_trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    SELECTION_SCHEMA,
    sha256_file,
    write_new_json,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    iter_trajectories,
    selected_total,
)


DIRECTIONS = ("cmn->eng", "eng->cmn")
DURATION_BANDS = ("short", "medium", "long")


def duration_band(duration_ms: int) -> str:
    if int(duration_ms) < 5_000:
        return "short"
    if int(duration_ms) < 9_000:
        return "medium"
    return "long"


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{sample_id}".encode("utf-8")).hexdigest()


def eligible_record(
    record_index: int, trajectory: E2ETrajectory, *, seed: int
) -> dict[str, object] | None:
    validate_trajectory(
        trajectory,
        require_audio_hash=True,
        require_audio_audit=True,
    )
    direction = f"{trajectory.src_lang}->{trajectory.tgt_lang}"
    if direction not in DIRECTIONS:
        return None
    prefinal_text = sum(
        bool(event.target_text_delta) and not event.source_final
        for event in trajectory.events
    )
    prefinal_semantic = sum(
        bool(event.target_semantic_delta) and not event.source_final
        for event in trajectory.events
    )
    if prefinal_text <= 0 or prefinal_semantic <= 0:
        return None
    return {
        "record_index": int(record_index),
        "sample_id": trajectory.sample_id,
        "src_lang": trajectory.src_lang,
        "tgt_lang": trajectory.tgt_lang,
        "direction": direction,
        "duration_ms": int(trajectory.source_duration_ms),
        "duration_band": duration_band(trajectory.source_duration_ms),
        "events": len(trajectory.events),
        "prefinal_target_text_events": prefinal_text,
        "prefinal_target_semantic_events": prefinal_semantic,
        "source_audio": trajectory.source_audio,
        "source_audio_sha256": trajectory.source_audio_sha256,
        "rank": stable_rank(seed, trajectory.sample_id),
    }


def _round_robin_strata(
    pools: Mapping[tuple[str, str], Sequence[dict[str, object]]],
    direction: str,
    count: int,
) -> list[dict[str, object]]:
    queues = {
        band: list(pools.get((direction, band), ())) for band in DURATION_BANDS
    }
    selected: list[dict[str, object]] = []
    cursor = 0
    while len(selected) < count:
        progressed = False
        for offset in range(len(DURATION_BANDS)):
            band = DURATION_BANDS[(cursor + offset) % len(DURATION_BANDS)]
            if queues[band]:
                selected.append(queues[band].pop(0))
                cursor = (DURATION_BANDS.index(band) + 1) % len(DURATION_BANDS)
                progressed = True
                break
        if not progressed:
            raise ValueError(
                f"not enough eligible {direction} validation records: "
                f"requested={count} selected={len(selected)}"
            )
    return selected


def freeze_selection(
    input_path: Path,
    *,
    samples: int,
    e_s2s_samples: int,
    seed: int,
) -> dict[str, object]:
    if samples < 4 or samples % len(DIRECTIONS):
        raise ValueError("free-running sample count must be even and at least four")
    if not 4 <= e_s2s_samples <= samples or e_s2s_samples % len(DIRECTIONS):
        raise ValueError("E-S2S sample count must be even, >=4 and <= total samples")
    offsets, total = selected_total(input_path, None)
    pools: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    eligible = 0
    for record_index, trajectory in iter_trajectories(
        input_path, offsets, 0, total
    ):
        record = eligible_record(record_index, trajectory, seed=seed)
        if record is None:
            continue
        eligible += 1
        pools[(str(record["direction"]), str(record["duration_band"]))].append(
            record
        )
    for values in pools.values():
        values.sort(key=lambda value: (str(value["rank"]), str(value["sample_id"])))

    per_direction = samples // len(DIRECTIONS)
    chosen: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        chosen.extend(_round_robin_strata(pools, direction, per_direction))
    chosen.sort(key=lambda value: (str(value["direction"]), str(value["rank"])))

    s2s_per_direction = e_s2s_samples // len(DIRECTIONS)
    for direction in DIRECTIONS:
        candidates = [value for value in chosen if value["direction"] == direction]
        # Prefer shorter samples for the first executable smoke while retaining
        # both directions. Full E-ASR/E-MT still runs over the complete selection.
        candidates.sort(
            key=lambda value: (
                int(value["duration_ms"]),
                str(value["rank"]),
            )
        )
        selected_ids = {
            str(value["sample_id"]) for value in candidates[:s2s_per_direction]
        }
        for value in candidates:
            value["run_e_s2s"] = str(value["sample_id"]) in selected_ids

    counts: Counter[str] = Counter()
    for value in chosen:
        counts[f"direction:{value['direction']}"] += 1
        counts[f"duration_band:{value['duration_band']}"] += 1
        counts["e_s2s"] += int(bool(value["run_e_s2s"]))
        value.pop("rank", None)
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": "frozen",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "records": total,
        },
        "selection_seed": int(seed),
        "eligible_records": eligible,
        "samples": samples,
        "e_s2s_samples": e_s2s_samples,
        "counts": dict(sorted(counts.items())),
        "records": chosen,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--e-s2s-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = freeze_selection(
        args.input,
        samples=args.samples,
        e_s2s_samples=args.e_s2s_samples,
        seed=args.seed,
    )
    write_new_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

