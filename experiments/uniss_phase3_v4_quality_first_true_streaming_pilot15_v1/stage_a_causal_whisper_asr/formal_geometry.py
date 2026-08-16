#!/usr/bin/env python3
"""Validate Stage A formal packs and materialize a non-overwriting run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "uniss_stage_a_formal_run_manifest_v1"
PACK_SCHEMA = "uniss_quality_first_stage_a_pack_build_v1"
GATE_SCHEMA = "uniss_stage_a_training_gate_v1"
PCM_GLM_GATE_SCHEMA = "uniss_stage_a_formal_pcm_glm_geometry_gate_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pack_report(report: Mapping[str, Any], *, split: str) -> int:
    if report.get("schema_version") != PACK_SCHEMA or report.get("status") != "complete":
        raise ValueError(f"Stage A {split} pack report is not complete")
    if int(report.get("seq_length", 0)) != 18_000:
        raise ValueError(f"Stage A {split} packs must use seq=18000")
    counts = report.get("counts")
    index = report.get("index")
    if not isinstance(counts, Mapping) or not isinstance(index, Mapping):
        raise ValueError(f"Stage A {split} pack report is malformed")
    packs = int(counts.get("packs", 0))
    if packs <= 0 or int(index.get("records", -1)) != packs:
        raise ValueError(f"Stage A {split} pack/index count mismatch")
    data_path = Path(str(index.get("data_path", "")))
    if not data_path.is_file() or data_path.stat().st_size != int(
        index.get("data_size_bytes", -1)
    ):
        raise ValueError(f"Stage A {split} packed data size mismatch")
    return packs


def formal_geometry(
    train_report: Mapping[str, Any],
    valid_report: Mapping[str, Any],
    *,
    global_batch_size: int = 128,
    coverage_epochs: int = 3,
    eval_global_batch_size: int = 8,
) -> dict[str, int]:
    if global_batch_size <= 0 or coverage_epochs <= 0 or eval_global_batch_size <= 0:
        raise ValueError("Stage A formal batch geometry must be positive")
    train_packs = _validate_pack_report(train_report, split="train")
    valid_packs = _validate_pack_report(valid_report, split="valid")
    steps_per_epoch = math.ceil(train_packs / global_batch_size)
    epoch_samples = steps_per_epoch * global_batch_size
    train_iters = steps_per_epoch * coverage_epochs
    eval_iters = math.ceil(valid_packs / eval_global_batch_size)
    return {
        "train_packs": train_packs,
        "valid_packs": valid_packs,
        "global_batch_size": global_batch_size,
        "coverage_epochs": coverage_epochs,
        "steps_per_epoch": steps_per_epoch,
        "epoch_samples": epoch_samples,
        "train_iters": train_iters,
        "train_samples": train_iters * global_batch_size,
        "eval_global_batch_size": eval_global_batch_size,
        "eval_iters": eval_iters,
        "eval_samples": eval_iters * eval_global_batch_size,
        "warmup_iters": min(200, math.ceil(0.03 * train_iters)),
    }


def write_formal_manifest(
    *,
    train_build: Path,
    valid_build: Path,
    training_gate: Path,
    pcm_glm_geometry_gate: Path,
    output: Path,
    run_id: str,
    git_head: str,
    initialization: str = "phase3_v4_iter_0009075_non_strict_handoff",
    resume_load: str | None = None,
) -> dict[str, Any]:
    train_report = json.loads(train_build.read_text(encoding="utf-8"))
    valid_report = json.loads(valid_build.read_text(encoding="utf-8"))
    gate = json.loads(training_gate.read_text(encoding="utf-8"))
    if gate.get("schema_version") != GATE_SCHEMA or not bool(gate.get("passed")):
        raise ValueError("Stage A training gate has not passed")
    pcm_glm_gate = json.loads(pcm_glm_geometry_gate.read_text(encoding="utf-8"))
    if (
        pcm_glm_gate.get("schema_version") != PCM_GLM_GATE_SCHEMA
        or not bool(pcm_glm_gate.get("passed"))
        or int(pcm_glm_gate.get("violation_count", -1)) != 0
    ):
        raise ValueError("Stage A formal PCM/GLM geometry gate has not passed")
    geometry = formal_geometry(train_report, valid_report)
    manifest = {
        "schema_version": SCHEMA,
        "status": "ready",
        "run_id": run_id,
        "git_head": git_head,
        "framework": "native_megatron",
        "initialization": initialization,
        "resume_load": resume_load,
        "sequence_length": 18_000,
        "micro_batch_size": 1,
        "precision": "bf16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_acoustics_per_pack": 2,
        "save_interval": 100,
        "eval_interval": 50,
        "geometry": geometry,
        "train_build_report": str(train_build.resolve()),
        "train_build_sha256": _sha256(train_build),
        "valid_build_report": str(valid_build.resolve()),
        "valid_build_sha256": _sha256(valid_build),
        "training_gate": str(training_gate.resolve()),
        "training_gate_sha256": _sha256(training_gate),
        "pcm_glm_geometry_gate": str(pcm_glm_geometry_gate.resolve()),
        "pcm_glm_geometry_gate_sha256": _sha256(pcm_glm_geometry_gate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-build", type=Path, required=True)
    parser.add_argument("--valid-build", type=Path, required=True)
    parser.add_argument("--training-gate", type=Path, required=True)
    parser.add_argument("--pcm-glm-geometry-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument(
        "--initialization",
        default="phase3_v4_iter_0009075_non_strict_handoff",
    )
    parser.add_argument("--resume-load")
    args = parser.parse_args()
    manifest = write_formal_manifest(
        train_build=args.train_build,
        valid_build=args.valid_build,
        training_gate=args.training_gate,
        pcm_glm_geometry_gate=args.pcm_glm_geometry_gate,
        output=args.output,
        run_id=args.run_id,
        git_head=args.git_head,
        initialization=args.initialization,
        resume_load=args.resume_load,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
