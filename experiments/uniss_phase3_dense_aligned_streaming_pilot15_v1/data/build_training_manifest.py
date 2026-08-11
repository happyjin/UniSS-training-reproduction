#!/usr/bin/env python3
"""Freeze the fixed15 replay subset and audit three independent shuffle epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from torch.utils.data import Dataset

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    ThreeEpochGlobalShuffleSchedule,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset import (
    build_subset,
)
from training.simul_uniss.jsonl_index import load_index


MANIFEST_SCHEMA = "uniss_dense_aligned_streaming_training_manifest_v1"
SHUFFLE_AUDIT_SCHEMA = "uniss_dense_aligned_streaming_shuffle_audit_v1"


class _CountDataset(Dataset):
    def __init__(self, kind: str, length: int) -> None:
        self.kind = kind
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {"sample_kind": self.kind, "source_index": index}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_replay_count(offsets: Path) -> int:
    metadata = json.loads(
        offsets.with_suffix(offsets.suffix + ".json").read_text(encoding="utf-8")
    )
    if metadata.get("schema_version") != "uniss_phase3_replay_offsets_v1":
        raise ValueError("unexpected fixed15 replay offset schema")
    if not bool(metadata.get("complete")):
        raise ValueError("fixed15 replay source is incomplete")
    return int(metadata["records"])


def _shuffle_audit(
    *,
    trajectory_count: int,
    replay_count: int,
    coverage_epochs: int,
    dp_group: int,
    global_batch_size: int,
    seed: int,
    replay_fraction: float,
) -> dict[str, object]:
    schedule = ThreeEpochGlobalShuffleSchedule(
        _CountDataset("trajectory", trajectory_count),
        _CountDataset("replay", replay_count),
        coverage_epochs=coverage_epochs,
        data_parallel_group_size=dp_group,
        global_batch_size=global_batch_size,
        shuffle_seed=seed,
        target_replay_fraction=replay_fraction,
    )
    repeated = ThreeEpochGlobalShuffleSchedule(
        _CountDataset("trajectory", trajectory_count),
        _CountDataset("replay", replay_count),
        coverage_epochs=coverage_epochs,
        data_parallel_group_size=dp_group,
        global_batch_size=global_batch_size,
        shuffle_seed=seed,
        target_replay_fraction=replay_fraction,
    )
    epoch_values: list[dict[str, object]] = []
    for epoch in range(coverage_epochs):
        start = epoch * schedule.epoch_samples
        seen = {"trajectory": set(), "replay": set()}
        counts = {"trajectory": 0, "replay": 0}
        digest = hashlib.sha256()
        first: list[list[object]] = []
        for local in range(schedule.epoch_samples):
            value = schedule.scheduled_index(start + local)
            seen[value.sample_kind].add(value.source_index)
            counts[value.sample_kind] += 1
            digest.update(
                f"{value.sample_kind}:{value.source_index}\n".encode("ascii")
            )
            if local < 64:
                first.append([value.sample_kind, value.source_index])
        if len(seen["trajectory"]) != trajectory_count:
            raise AssertionError(f"trajectory coverage failed in epoch {epoch}")
        if len(seen["replay"]) != replay_count:
            raise AssertionError(f"replay coverage failed in epoch {epoch}")
        epoch_values.append(
            {
                "epoch": epoch,
                "seed": seed + epoch * 1009,
                "samples": schedule.epoch_samples,
                "sha256": digest.hexdigest(),
                "trajectory": {
                    "source_records": trajectory_count,
                    "scheduled": counts["trajectory"],
                    "tail_padding": counts["trajectory"] - trajectory_count,
                    "coverage": 1.0,
                },
                "replay": {
                    "source_records": replay_count,
                    "scheduled": counts["replay"],
                    "tail_padding": counts["replay"] - replay_count,
                    "coverage": 1.0,
                },
                "first_64": first,
            }
        )
    independent = (
        len({value["sha256"] for value in epoch_values}) == coverage_epochs
    )
    has_permutable_geometry = (
        trajectory_count > dp_group or replay_count > dp_group
    )
    if has_permutable_geometry and not independent:
        raise AssertionError("coverage epochs did not receive independent permutations")
    for index in range(min(len(schedule), 4096)):
        if schedule.scheduled_index(index) != repeated.scheduled_index(index):
            raise AssertionError("restart changed the deterministic shuffle order")
    return {
        "schema_version": SHUFFLE_AUDIT_SCHEMA,
        "status": "pass",
        "shuffle_unit": "complete_dense_session_inside_18k_pack",
        "session_internal_event_order": "preserved",
        "coverage_epochs": coverage_epochs,
        "base_seed": seed,
        "epoch_seed_stride": 1009,
        "data_parallel_group_size": dp_group,
        "global_batch_size": global_batch_size,
        "epoch_groups": schedule.epoch_groups,
        "epoch_samples": schedule.epoch_samples,
        "total_samples": len(schedule),
        "restart_first_4096_exact": True,
        "independent_epoch_permutations": independent,
        "small_geometry_independence_exempt": not has_permutable_geometry,
        "epochs": epoch_values,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    trajectory = Path(args.trajectory_packed).resolve()
    replay_packed = Path(args.replay_packed).resolve()
    replay_offsets = Path(args.replay_offsets).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_offsets = load_index(trajectory)
    if trajectory_offsets is None:
        raise ValueError(f"missing dense trajectory index for {trajectory}")
    trajectory_count = len(trajectory_offsets)
    replay_available = _source_replay_count(replay_offsets)
    dp_group = args.data_parallel_size * args.micro_batch_size
    if args.global_batch_size % dp_group:
        raise ValueError("global batch size must be divisible by DP*MBS")
    trajectory_groups = math.ceil(trajectory_count / dp_group)
    replay_groups = max(
        1,
        round(
            trajectory_groups
            * args.replay_fraction
            / (1.0 - args.replay_fraction)
        ),
    )
    replay_selected = replay_groups * dp_group
    if replay_selected > replay_available:
        raise ValueError(
            f"fixed15 replay has {replay_available} records, needs {replay_selected}"
        )
    subset = output_root / "replay_subset.u64"
    if subset.is_file():
        subset_meta = json.loads(
            subset.with_suffix(subset.suffix + ".json").read_text(encoding="utf-8")
        )
        if int(subset_meta.get("records", -1)) != replay_selected:
            raise ValueError("existing replay subset has different geometry")
    else:
        subset_meta = build_subset(
            kind="replay",
            packed=replay_packed,
            source_offsets=replay_offsets,
            output_offsets=subset,
            records=replay_selected,
        )
        subset_meta = dict(subset_meta)
        subset_meta["complete"] = True
        subset_meta["max_records"] = None
        subset_meta["formal_subset_schema"] = MANIFEST_SCHEMA
        _atomic_json(
            subset.with_suffix(subset.suffix + ".json"), subset_meta
        )
    if subset_meta.get("formal_subset_schema") != MANIFEST_SCHEMA:
        raise ValueError("replay subset is not bound to this experiment")

    audit = _shuffle_audit(
        trajectory_count=trajectory_count,
        replay_count=replay_selected,
        coverage_epochs=args.coverage_epochs,
        dp_group=dp_group,
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        replay_fraction=args.replay_fraction,
    )
    _atomic_json(output_root / "shuffle_audit.json", audit)
    epoch_samples = int(audit["epoch_samples"])
    total_samples = int(audit["total_samples"])
    if total_samples % args.global_batch_size:
        raise AssertionError("three-epoch schedule does not end on a global batch")
    train_iters = total_samples // args.global_batch_size
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "trajectory_packed": str(trajectory),
        "trajectory_count": trajectory_count,
        "replay_packed": str(replay_packed),
        "replay_source_offsets": str(replay_offsets),
        "replay_available": replay_available,
        "replay_subset_offsets": str(subset.resolve()),
        "replay_selected": replay_selected,
        "replay_fraction_target": args.replay_fraction,
        "coverage_epochs": args.coverage_epochs,
        "epoch_samples": epoch_samples,
        "total_samples": total_samples,
        "train_iters": train_iters,
        "warmup_iters": min(
            max(0, train_iters - 1),
            min(1000, max(50, math.ceil(train_iters * 0.025))),
        ),
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "data_parallel_size": args.data_parallel_size,
        "data_parallel_group_size": dp_group,
        "seed": args.seed,
        "epoch_seed_stride": 1009,
        "strict_global_shuffle": True,
        "session_internal_event_order": "preserved",
        "shuffle_audit": str((output_root / "shuffle_audit.json").resolve()),
        "replay_subset_metadata": subset_meta,
    }
    _atomic_json(output_root / "training_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-packed", required=True)
    parser.add_argument("--replay-packed", required=True)
    parser.add_argument("--replay-offsets", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--coverage-epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--replay-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260811)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
