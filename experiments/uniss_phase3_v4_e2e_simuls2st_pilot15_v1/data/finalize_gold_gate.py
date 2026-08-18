#!/usr/bin/env python3
"""Finalize the train/valid gold-trajectory gate from immutable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GATE_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_gold_gate_v1"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(
    frozen_split_path: Path,
    train_build_path: Path,
    valid_build_path: Path,
    train_audit_path: Path,
    valid_audit_path: Path,
) -> dict[str, object]:
    frozen = _load(frozen_split_path)
    train_build = _load(train_build_path)
    valid_build = _load(valid_build_path)
    train_audit = _load(train_audit_path)
    valid_audit = _load(valid_audit_path)
    if frozen.get("status") != "frozen" or int(frozen.get("train_valid_id_overlap", -1)) != 0:
        raise ValueError("frozen split did not pass")
    for split, build, audit in (
        ("train", train_build, train_audit),
        ("valid", valid_build, valid_audit),
    ):
        if build.get("status") != "complete":
            raise ValueError(f"{split} trajectory build is incomplete")
        if audit.get("status") != "passed":
            raise ValueError(f"{split} trajectory audit did not pass")
        if build.get("hash_audio") is not True or build.get("audit_audio") is not True:
            raise ValueError(f"{split} build did not hash and decode source audio")
        expected = int(frozen["splits"][split]["records"])  # type: ignore[index]
        if int(build["counts"]["records"]) != expected:  # type: ignore[index]
            raise ValueError(f"{split} build record count differs from frozen split")
        if int(audit["counts"]["records"]) != expected:  # type: ignore[index]
            raise ValueError(f"{split} audit record count differs from frozen split")
        if audit.get("require_audio_hash") is not True or audit.get("require_audio_audit") is not True:
            raise ValueError(f"{split} audit did not enforce audio hard gates")
    return {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "train_valid_id_overlap": 0,
        "v1_rollout_status": "pending",
        "formal_training_authorized": False,
        "reason_training_not_authorized": "V1 free-running rollout and teacher caches are not built yet",
        "train": {
            "records": train_audit["counts"]["records"],  # type: ignore[index]
            "events": train_audit["counts"]["events"],  # type: ignore[index]
            "prefinal_target_writes": train_audit["counts"]["prefinal_target_writes"],  # type: ignore[index]
        },
        "valid": {
            "records": valid_audit["counts"]["records"],  # type: ignore[index]
            "events": valid_audit["counts"]["events"],  # type: ignore[index]
            "prefinal_target_writes": valid_audit["counts"]["prefinal_target_writes"],  # type: ignore[index]
        },
        "inputs": {
            "frozen_split": str(frozen_split_path.resolve()),
            "train_build": str(train_build_path.resolve()),
            "valid_build": str(valid_build_path.resolve()),
            "train_audit": str(train_audit_path.resolve()),
            "valid_audit": str(valid_audit_path.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--train-build", type=Path, required=True)
    parser.add_argument("--valid-build", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--valid-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite gold gate: {args.output}")
    result = finalize(
        args.frozen_split,
        args.train_build,
        args.valid_build,
        args.train_audit,
        args.valid_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
