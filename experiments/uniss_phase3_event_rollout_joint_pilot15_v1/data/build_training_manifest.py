#!/usr/bin/env python3
"""Freeze pilot15 replay geometry and strict global pack shuffle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_training_manifest import (
    _atomic_json,
    _shuffle_audit,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.training.dataset import (
    MultiFilePackIndex,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset import (
    build_subset,
)


MANIFEST_SCHEMA = "uniss_event_rollout_pilot15_training_manifest_v1"


def _replay_count(path: Path) -> int:
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "uniss_phase3_replay_offsets_v1":
        raise ValueError("unexpected Phase3 replay index schema")
    if not bool(metadata.get("complete")):
        raise ValueError("Phase3 replay source index is incomplete")
    return int(metadata["records"])


def _require_data_audit(path: Path, *, allow_sampled: bool) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"pass", "sampled_pass"} if allow_sampled else {"pass"}
    if value.get("status") not in expected:
        raise ValueError(f"data audit status {value.get('status')!r} is not formal-pass")
    gates = dict(value.get("gates", {}))
    required = (
        "fixed_shards_only",
        "deterministic_split",
        "train_valid_intersection_zero",
        "phase3_replay_fixed15_only",
    )
    if any(not bool(gates.get(name)) for name in required):
        raise ValueError("fixed15 provenance gates did not pass")
    if not allow_sampled:
        complete = (
            "complete_160ms_sessions",
            "gap_free_text_and_semantic_coverage",
            "sessions_never_cross_packs",
            "runtime_parsers_accept_all_packs",
        )
        if any(not bool(gates.get(name)) for name in complete):
            raise ValueError("full session/runtime data gates did not pass")
    return value


def build(args: argparse.Namespace) -> dict[str, object]:
    trajectory_manifest = Path(args.trajectory_manifest).resolve()
    valid_manifest = Path(args.valid_trajectory_manifest).resolve()
    replay_packed = Path(args.replay_packed).resolve()
    replay_offsets = Path(args.replay_offsets).resolve()
    audit_path = Path(args.data_audit).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    audit = _require_data_audit(audit_path, allow_sampled=bool(args.allow_sampled_audit))
    trajectory_count = len(MultiFilePackIndex(trajectory_manifest, expected_split="train"))
    validation_count = len(MultiFilePackIndex(valid_manifest, expected_split="valid"))
    replay_available = _replay_count(replay_offsets)
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    if int(args.global_batch_size) % dp_group:
        raise ValueError("global batch size must be divisible by DP*MBS")
    trajectory_groups = math.ceil(trajectory_count / dp_group)
    replay_groups = max(
        1,
        round(trajectory_groups * args.replay_fraction / (1.0 - args.replay_fraction)),
    )
    replay_selected = replay_groups * dp_group
    if replay_selected > replay_available:
        raise ValueError("fixed15 Phase3 replay is too small for requested fraction")
    subset = output_root / "replay_subset.u64"
    if subset.is_file():
        subset_meta = json.loads(
            subset.with_suffix(subset.suffix + ".json").read_text(encoding="utf-8")
        )
        if int(subset_meta.get("records", -1)) != replay_selected:
            raise ValueError("existing replay subset has different training geometry")
    else:
        subset_meta = dict(
            build_subset(
                kind="replay",
                packed=replay_packed,
                source_offsets=replay_offsets,
                output_offsets=subset,
                records=replay_selected,
            )
        )
        subset_meta["complete"] = True
        subset_meta["formal_subset_schema"] = MANIFEST_SCHEMA
        _atomic_json(subset.with_suffix(subset.suffix + ".json"), subset_meta)
    if subset_meta.get("formal_subset_schema") != MANIFEST_SCHEMA:
        raise ValueError("replay subset is not bound to this experiment")

    shuffle = _shuffle_audit(
        trajectory_count=trajectory_count,
        replay_count=replay_selected,
        coverage_epochs=int(args.coverage_epochs),
        dp_group=dp_group,
        global_batch_size=int(args.global_batch_size),
        seed=int(args.seed),
        replay_fraction=float(args.replay_fraction),
    )
    shuffle.update(
        {
            "schema_version": "uniss_event_rollout_pilot15_shuffle_audit_v1",
            "trajectory_namespace": str(trajectory_manifest),
            "shuffle_unit": "global_complete_18000_token_pack_id",
            "multi_file_prefix_sum_namespace": True,
        }
    )
    shuffle_path = output_root / "shuffle_audit.json"
    _atomic_json(shuffle_path, shuffle)
    total_samples = int(shuffle["total_samples"])
    if total_samples % int(args.global_batch_size):
        raise AssertionError("coverage schedule does not end on a global batch")
    train_iters = total_samples // int(args.global_batch_size)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "scope": "fixed UniST train shards 00000-00014",
        "base_checkpoint": "checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075",
        "trajectory_manifest": str(trajectory_manifest),
        "trajectory_count": trajectory_count,
        "valid_trajectory_manifest": str(valid_manifest),
        "valid_trajectory_count": validation_count,
        "replay_packed": str(replay_packed),
        "replay_source_offsets": str(replay_offsets),
        "replay_available": replay_available,
        "replay_subset_offsets": str(subset.resolve()),
        "replay_selected": replay_selected,
        "replay_fraction_target": float(args.replay_fraction),
        "coverage_epochs": int(args.coverage_epochs),
        "epoch_samples": int(shuffle["epoch_samples"]),
        "total_samples": total_samples,
        "train_iters": train_iters,
        "warmup_iters": min(
            max(0, train_iters - 1),
            min(1000, max(2 if args.allow_sampled_audit else 50, math.ceil(train_iters * 0.025))),
        ),
        "micro_batch_size": int(args.micro_batch_size),
        "global_batch_size": int(args.global_batch_size),
        "data_parallel_size": int(args.data_parallel_size),
        "data_parallel_group_size": dp_group,
        "seq_length": 18_000,
        "seed": int(args.seed),
        "epoch_seed_stride": 1009,
        "strict_global_shuffle": True,
        "shuffle_unit": "global_complete_18000_token_pack_id",
        "multi_file_prefix_sum_namespace": True,
        "session_internal_event_order": "preserved",
        "data_audit": str(audit_path),
        "data_audit_status": audit["status"],
        "timing_provenance": audit["timing_provenance_truth"],
        "shuffle_audit": str(shuffle_path.resolve()),
        "replay_subset_metadata": subset_meta,
    }
    _atomic_json(output_root / "training_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--valid-trajectory-manifest", required=True)
    parser.add_argument("--replay-packed", required=True)
    parser.add_argument("--replay-offsets", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--coverage-epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--replay-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--allow-sampled-audit", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()

