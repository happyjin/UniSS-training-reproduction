#!/usr/bin/env python3
"""Freeze one trajectory-coverage epoch and its uniform Phase3 replay subset."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset import (
    build_subset,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.packed_epoch import (
    curriculum_group_counts,
)


MANIFEST_SCHEMA = "uniss_true_subsecond_pilot15_epoch_manifest_v2"


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


def build(
    *,
    trajectory_packed: Path,
    trajectory_offsets: Path,
    replay_packed: Path,
    replay_offsets: Path,
    audit_path: Path,
    output_root: Path,
    global_batch_size: int = 128,
    data_parallel_microbatch: int = 16,
    seed: int = 20260810,
) -> dict[str, object]:
    trajectory_meta = json.loads(
        trajectory_offsets.with_suffix(trajectory_offsets.suffix + ".json").read_text(
            encoding="utf-8"
        )
    )
    replay_meta = json.loads(
        replay_offsets.with_suffix(replay_offsets.suffix + ".json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("data audit did not pass; refusing to freeze an epoch")
    trajectory_count = int(trajectory_meta["records"])
    replay_available = int(replay_meta["records"])
    if global_batch_size % data_parallel_microbatch:
        raise ValueError("global batch must be divisible by DP microbatch")
    groups_per_step = global_batch_size // data_parallel_microbatch
    required_trajectory_groups = math.ceil(trajectory_count / data_parallel_microbatch)
    train_iters = 1
    while True:
        schedule_groups = train_iters * groups_per_step
        replay_groups, trajectory_groups = curriculum_group_counts(schedule_groups)
        if trajectory_groups >= required_trajectory_groups:
            break
        train_iters += 1
    replay_selected = replay_groups * data_parallel_microbatch
    trajectory_scheduled = trajectory_groups * data_parallel_microbatch
    if replay_selected > replay_available:
        raise ValueError("pilot replay source is too small for one trajectory epoch")

    output_root.mkdir(parents=True, exist_ok=True)
    subset_offsets = output_root / "replay_subset.offsets.u64"
    if subset_offsets.exists():
        subset_meta = json.loads(
            subset_offsets.with_suffix(subset_offsets.suffix + ".json").read_text(
                encoding="utf-8"
            )
        )
        if int(subset_meta.get("records", -1)) != replay_selected:
            raise ValueError("existing replay subset has different epoch geometry")
    else:
        subset_meta = build_subset(
            kind="replay",
            packed=replay_packed,
            source_offsets=replay_offsets,
            output_offsets=subset_offsets,
            records=replay_selected,
        )
        # The shared helper deliberately labels arbitrary diagnostic subsets as
        # incomplete. Here the subset is the complete replay source of a frozen
        # formal epoch, so promote it only after exact geometry is known and
        # bind that decision to this experiment's manifest schema.
        subset_meta = dict(subset_meta)
        subset_meta["complete"] = True
        subset_meta["max_records"] = None
        subset_meta["formal_subset_schema"] = MANIFEST_SCHEMA
        _atomic_json(
            subset_offsets.with_suffix(subset_offsets.suffix + ".json"), subset_meta
        )

    if not bool(subset_meta.get("complete")):
        raise ValueError("frozen replay subset is not marked complete")
    if subset_meta.get("formal_subset_schema") != MANIFEST_SCHEMA:
        raise ValueError("replay subset is not bound to the v2 epoch manifest")

    natural_write_fraction = float(audit["natural_write_fraction"])
    safe_positive_fraction = float(audit["safe_positive_fraction"])
    action_write_weight = min(
        3.0,
        max(1.0, math.sqrt((1.0 - natural_write_fraction) / max(1e-6, natural_write_fraction))),
    )
    safe_positive_alpha = min(0.85, max(0.60, 1.0 - safe_positive_fraction))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "seed": seed,
        "strict_global_shuffle": True,
        "trajectory_packed": str(trajectory_packed.resolve()),
        "trajectory_offsets": str(trajectory_offsets.resolve()),
        "trajectory_source_count": trajectory_count,
        "trajectory_scheduled": trajectory_scheduled,
        "trajectory_padding": trajectory_scheduled - trajectory_count,
        "replay_packed": str(replay_packed.resolve()),
        "replay_source_offsets": str(replay_offsets.resolve()),
        "replay_source_count": replay_available,
        "replay_subset_offsets": str(subset_offsets.resolve()),
        "replay_selected": replay_selected,
        "schedule_count": train_iters * global_batch_size,
        "train_iters": train_iters,
        "warmup_iters": min(20, max(1, math.ceil(train_iters * 0.10))),
        "global_batch_size": global_batch_size,
        "micro_batch_size": 2,
        "data_parallel_microbatch": data_parallel_microbatch,
        "curriculum_boundaries": [
            min(train_iters, max(1, math.ceil(train_iters * fraction)))
            for fraction in (0.083, 0.333, 0.75, 1.0)
        ],
        "action_write_weight": action_write_weight,
        "safe_positive_alpha": safe_positive_alpha,
        "audit": str(audit_path.resolve()),
        "replay_subset_metadata": subset_meta,
    }
    _atomic_json(output_root / "epoch_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-packed", required=True, type=Path)
    parser.add_argument("--trajectory-offsets", required=True, type=Path)
    parser.add_argument("--replay-packed", required=True, type=Path)
    parser.add_argument("--replay-offsets", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--data-parallel-microbatch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    print(json.dumps(build(
        trajectory_packed=args.trajectory_packed,
        trajectory_offsets=args.trajectory_offsets,
        replay_packed=args.replay_packed,
        replay_offsets=args.replay_offsets,
        audit_path=args.audit,
        output_root=args.output_root,
        global_batch_size=args.global_batch_size,
        data_parallel_microbatch=args.data_parallel_microbatch,
        seed=args.seed,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
