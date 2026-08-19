"""Preflight and finalize the post-task-pool E2E Megatron canary sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools import (
    BUILD_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_STREAMING_ASR,
    TASK_FAMILIES,
)


PREFLIGHT_SCHEMA = "uniss_e2e_post_task_pool_canary_preflight_v1"
REPORT_SCHEMA = "uniss_e2e_post_task_pool_canary_report_v1"
EXPECTED_RUNS = (
    ("structural", None, 2),
    *((family, family, 1) for family in TASK_FAMILIES),
)
FATAL_LOG_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"traceback \(most recent call last\)",
        r"cuda out of memory",
        r"floatingpointerror",
        r"childfailederror",
        r"non-finite gradient",
        r"runtimeerror:",
    )
)
REQUIRED_FAMILY_COUNTS = {
    FAMILY_STREAMING_ASR: (
        "loss:asr_ce",
        "loss:boundary_ce",
        "teacher:v1_asr:positions",
    ),
    FAMILY_INCREMENTAL_MT: (
        "loss:mt_ce",
        "loss:boundary_ce",
        "commit_consistency_positions",
        "teacher:phase3:positions",
    ),
    FAMILY_INTERLEAVED: (
        "loss:asr_ce",
        "loss:mt_ce",
        "loss:semantic_ce",
        "loss:boundary_ce",
        "loss:eos_ce",
        "teacher:phase3:positions",
    ),
    FAMILY_PHASE3_QUALITY: ("loss:phase3_replay_ce",),
    FAMILY_PHASE3_PERFORMANCE: ("loss:phase3_replay_ce",),
}


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _validate_quality_gate(path: str | Path) -> dict[str, object]:
    gate = _json(path)
    if gate.get("status") != "passed":
        raise ValueError(f"rollout quality gate did not pass: {path}")
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
    }


def validate_task_pool(path: str | Path, split: str) -> dict[str, object]:
    report = _json(path)
    if (
        report.get("schema_version") != BUILD_SCHEMA
        or report.get("status") != "passed"
        or report.get("split") != split
        or int(report.get("seq_length", -1)) != 18_000
        or int(report.get("records", 0)) <= 0
    ):
        raise ValueError(f"task-pool report is not a passed {split} 18k build: {path}")
    families = report.get("families")
    if not isinstance(families, dict) or set(families) != set(TASK_FAMILIES):
        raise ValueError(f"task-pool report does not contain exactly five families: {path}")
    family_summary: dict[str, object] = {}
    for family in TASK_FAMILIES:
        metadata = families[family]
        if not isinstance(metadata, dict) or metadata.get("family") != family:
            raise ValueError(f"malformed task-pool family metadata: {family}")
        counts = metadata.get("counts")
        if (
            not isinstance(counts, dict)
            or int(metadata.get("records", 0)) <= 0
            or int(counts.get("supervised_tokens", 0)) <= 0
        ):
            raise ValueError(f"empty task-pool supervision for family {family}")
        missing = [
            name
            for name in REQUIRED_FAMILY_COUNTS[family]
            if int(counts.get(name, 0)) <= 0
        ]
        if missing:
            raise ValueError(
                f"task-pool family {family} has zero/missing active counts: {missing}"
            )
        data_path = Path(str(metadata.get("path", ""))).resolve()
        if not data_path.is_file() or data_path.stat().st_size != int(
            metadata.get("bytes", -1)
        ):
            raise ValueError(f"task-pool family bytes changed: {family}")
        index = metadata.get("index")
        index_path = (
            Path(str(index.get("binary_path", ""))).resolve()
            if isinstance(index, dict)
            else Path()
        )
        if not index_path.is_file():
            raise FileNotFoundError(f"task-pool family index is missing: {family}")
        family_summary[family] = {
            "records": int(metadata["records"]),
            "supervised_tokens": int(counts["supervised_tokens"]),
            "required_counts": {
                name: int(counts[name]) for name in REQUIRED_FAMILY_COUNTS[family]
            },
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
        }
    quality_path = Path(str(report.get("quality_gate", ""))).resolve()
    quality = _validate_quality_gate(quality_path)
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "split": split,
        "records": int(report["records"]),
        "seq_length": 18_000,
        "quality_gate": quality,
        "families": family_summary,
    }


def validate_teacher_audit(path: str | Path, cache_kind: str, split: str) -> dict[str, object]:
    audit = _json(path)
    if audit.get("status") != "passed":
        raise ValueError(f"teacher cache audit did not pass: {path}")
    selection_start = int(audit.get("selection_start", -1))
    selection_stop = int(audit.get("selection_stop", -1))
    output = Path(str(audit.get("output", ""))).resolve()
    if not 0 <= selection_start < selection_stop or not output.is_file():
        raise ValueError(f"teacher cache audit coverage/output is invalid: {path}")
    if output.stat().st_size != int(audit.get("output_bytes", -1)):
        raise ValueError(f"teacher cache output bytes changed: {path}")
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "cache_kind": cache_kind,
        "split": split,
        "records": selection_stop - selection_start,
        "output": str(output),
        "output_bytes": output.stat().st_size,
    }


def build_preflight(
    *,
    data_run_id: str,
    task_pool_run_id: str,
    teacher_run_id: str,
    train_report: str | Path,
    valid_report: str | Path,
    v1_train_audit: str | Path,
    phase3_train_audit: str | Path,
    v1_valid_audit: str | Path,
    phase3_valid_audit: str | Path,
    gold_gate: str | Path,
) -> dict[str, object]:
    gate = _json(gold_gate)
    if gate.get("status") != "passed" or bool(gate.get("formal_training_authorized")):
        raise ValueError(
            "post-task-pool canaries require a passed but still unauthorized gold gate"
        )
    task_pools = {
        "train": validate_task_pool(train_report, "train"),
        "valid": validate_task_pool(valid_report, "valid"),
    }
    teacher_caches = {
        "v1_train": validate_teacher_audit(v1_train_audit, "v1_asr", "train"),
        "phase3_train": validate_teacher_audit(
            phase3_train_audit, "phase3", "train"
        ),
        "v1_valid": validate_teacher_audit(v1_valid_audit, "v1_asr", "valid"),
        "phase3_valid": validate_teacher_audit(
            phase3_valid_audit, "phase3", "valid"
        ),
    }
    for split in ("train", "valid"):
        records = int(task_pools[split]["records"])
        for cache in (f"v1_{split}", f"phase3_{split}"):
            if int(teacher_caches[cache]["records"]) != records:
                raise ValueError(
                    f"teacher cache {cache} does not cover the {split} task-pool records"
                )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed",
        "data_run_id": data_run_id,
        "task_pool_run_id": task_pool_run_id,
        "teacher_run_id": teacher_run_id,
        "task_pools": task_pools,
        "teacher_caches": teacher_caches,
        "gold_gate": {
            "path": str(Path(gold_gate).resolve()),
            "sha256": _sha256(gold_gate),
            "formal_training_authorized": False,
        },
        "canary_scope": [
            {"name": name, "family": family, "train_iters": train_iters}
            for name, family, train_iters in EXPECTED_RUNS
        ],
        "formal_training_authorized": False,
        "next_required_gates": [
            "free_running_e_asr_e_mt_e_s2s_validation",
            "frozen_parameter_bitwise_audit",
            "explicit_formal_training_authorization",
        ],
    }


def _gpu_summary(path: Path) -> dict[str, object]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    float(row["utilization_gpu_percent"]),
                    float(row["power_draw_w"]),
                )
            )
    if not rows:
        raise ValueError(f"empty canary GPU monitor: {path}")
    utilization = [value[0] for value in rows]
    power = [value[1] for value in rows]
    return {
        "samples": len(rows),
        "utilization_mean_percent": sum(utilization) / len(utilization),
        "utilization_max_percent": max(utilization),
        "power_mean_w": sum(power) / len(power),
        "power_max_w": max(power),
    }


def finalize_canaries(
    preflight_path: str | Path,
    results_path: str | Path,
    frozen_audit_path: str | Path,
) -> dict[str, object]:
    preflight = _json(preflight_path)
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA or preflight.get(
        "status"
    ) != "passed":
        raise ValueError("canary preflight report is not passed")
    gate_info = preflight["gold_gate"]
    if not isinstance(gate_info, dict):
        raise TypeError("canary preflight gold gate is malformed")
    gate_path = Path(str(gate_info["path"]))
    gate = _json(gate_path)
    if (
        _sha256(gate_path) != gate_info["sha256"]
        or bool(gate.get("formal_training_authorized"))
    ):
        raise RuntimeError("formal training gate changed during canary execution")
    values = [
        json.loads(line)
        for line in Path(results_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_name = {str(value["name"]): value for value in values}
    if len(by_name) != len(values):
        raise ValueError("duplicate canary result names")
    runs = []
    for name, family, train_iters in EXPECTED_RUNS:
        value = by_name.get(name)
        if value is None:
            raise ValueError(f"missing canary result: {name}")
        if value.get("family") != family or int(value.get("train_iters", -1)) != train_iters:
            raise ValueError(f"canary result scope differs: {name}")
        if int(value.get("exit_code", -1)) != 0:
            raise RuntimeError(f"canary run failed: {name}")
        log = Path(str(value["log"])).resolve()
        command = Path(f"{log}.command")
        gpu_csv = Path(str(value["gpu_csv"])).resolve()
        save_dir = Path(str(value["save_dir"])).resolve()
        latest = save_dir / "latest_checkpointed_iteration.txt"
        for required in (log, command, gpu_csv, latest):
            if not required.is_file():
                raise FileNotFoundError(required)
        text = log.read_text(encoding="utf-8", errors="replace")
        fatal = [pattern.pattern for pattern in FATAL_LOG_PATTERNS if pattern.search(text)]
        if fatal:
            raise RuntimeError(f"fatal pattern(s) in canary {name}: {fatal}")
        checkpoint_iteration = int(latest.read_text(encoding="utf-8").strip())
        if checkpoint_iteration != train_iters:
            raise ValueError(f"canary checkpoint iteration differs: {name}")
        runs.append(
            {
                **value,
                "log": str(log),
                "log_sha256": _sha256(log),
                "command_sha256": _sha256(command),
                "checkpoint_iteration": checkpoint_iteration,
                "gpu": _gpu_summary(gpu_csv),
            }
        )
    frozen_audit = _json(frozen_audit_path)
    if (
        frozen_audit.get("schema_version")
        != "uniss_e2e_frozen_stage_a_bitwise_audit_v1"
        or frozen_audit.get("status") != "passed"
        or not bool(frozen_audit.get("exact_bitwise_match"))
    ):
        raise RuntimeError("frozen Stage-A bitwise audit did not pass")
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed",
        "preflight": str(Path(preflight_path).resolve()),
        "preflight_sha256": _sha256(preflight_path),
        "runs": runs,
        "frozen_stage_a_bitwise_audit": {
            "path": str(Path(frozen_audit_path).resolve()),
            "sha256": _sha256(frozen_audit_path),
        },
        "formal_training_authorized": False,
        "next_required_gates": [
            value
            for value in preflight["next_required_gates"]
            if value != "frozen_parameter_bitwise_audit"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    for name in (
        "data_run_id",
        "task_pool_run_id",
        "teacher_run_id",
        "train_report",
        "valid_report",
        "v1_train_audit",
        "phase3_train_audit",
        "v1_valid_audit",
        "phase3_valid_audit",
        "gold_gate",
        "output",
    ):
        preflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--preflight", required=True)
    finalize.add_argument("--results", required=True)
    finalize.add_argument("--frozen-audit", required=True)
    finalize.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        value = build_preflight(
            data_run_id=args.data_run_id,
            task_pool_run_id=args.task_pool_run_id,
            teacher_run_id=args.teacher_run_id,
            train_report=args.train_report,
            valid_report=args.valid_report,
            v1_train_audit=args.v1_train_audit,
            phase3_train_audit=args.phase3_train_audit,
            v1_valid_audit=args.v1_valid_audit,
            phase3_valid_audit=args.phase3_valid_audit,
            gold_gate=args.gold_gate,
        )
    else:
        value = finalize_canaries(args.preflight, args.results, args.frozen_audit)
    _write_new_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
