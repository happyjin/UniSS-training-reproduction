#!/usr/bin/env python3
"""Static audit of the native Phase3-to-V1 compound checkpoint transition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron import (
    _metadata_base_key,
    validate_v1_fingerprint_manifest,
)


NATIVE_PREFIXES = ("embedding.", "decoder.", "output_layer.")
STAGE_A_PREFIX = "stage_a_objective."


def _keys(path: Path) -> set[str]:
    metadata = FileSystemReader(str(path)).read_metadata().state_dict_metadata
    return {
        _metadata_base_key(key)
        for key in metadata
        if _metadata_base_key(key).startswith((*NATIVE_PREFIXES, STAGE_A_PREFIX))
    }


def _hash_keys(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def audit(
    *,
    v1_load_root: str | Path,
    phase3_checkpoint: str | Path,
    fingerprints: str | Path,
) -> dict[str, object]:
    identity = validate_v1_fingerprint_manifest(v1_load_root, fingerprints)
    v1_checkpoint = Path(str(identity["checkpoint"]))
    phase3_checkpoint = Path(phase3_checkpoint).resolve()
    if not phase3_checkpoint.is_dir():
        raise FileNotFoundError(phase3_checkpoint)
    v1 = _keys(v1_checkpoint)
    phase3 = _keys(phase3_checkpoint)
    v1_native = {key for key in v1 if key.startswith(NATIVE_PREFIXES)}
    phase3_native = {key for key in phase3 if key.startswith(NATIVE_PREFIXES)}
    stage_a = {key for key in v1 if key.startswith(STAGE_A_PREFIX)}
    unknown = v1 - v1_native - stage_a
    required = {
        "stage_a_objective.bridge_norm.weight",
        "stage_a_objective.bridge_projection.weight",
        "stage_a_objective.ctc_head.weight",
        "stage_a_objective.frontend.encoder.conv1.weight",
        "stage_a_objective.frontend.encoder.codebook.weight",
    }
    missing_required = sorted(required - stage_a)
    native_missing = sorted(phase3_native - v1_native)
    native_unexpected = sorted(v1_native - phase3_native)
    if (
        not phase3_native
        or not stage_a
        or unknown
        or missing_required
        or native_missing
        or native_unexpected
    ):
        raise RuntimeError(
            "V1 static compound audit failed: "
            f"native_missing={native_missing[:20]} "
            f"native_unexpected={native_unexpected[:20]} "
            f"unknown={sorted(unknown)[:20]} "
            f"missing_required={missing_required}"
        )
    return {
        "schema_version": "uniss_phase3_v4_e2e_v1_checkpoint_static_audit_v1",
        "status": "passed",
        "v1_checkpoint": str(v1_checkpoint),
        "v1_tree_sha256": identity["tree_sha256"],
        "phase3_checkpoint": str(phase3_checkpoint),
        "native_key_count": len(v1_native),
        "native_key_sha256": _hash_keys(v1_native),
        "stage_a_key_count": len(stage_a),
        "stage_a_key_sha256": _hash_keys(stage_a),
        "native_phase3_to_v1_exact": True,
        "required_stage_a_keys_present": sorted(required),
        "runtime_model_exact_key_audit_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-load-root", type=Path, required=True)
    parser.add_argument("--phase3-checkpoint", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V1 audit: {args.output}")
    value = audit(
        v1_load_root=args.v1_load_root,
        phase3_checkpoint=args.phase3_checkpoint,
        fingerprints=args.fingerprints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
