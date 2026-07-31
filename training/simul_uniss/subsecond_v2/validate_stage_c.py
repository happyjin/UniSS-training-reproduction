"""Validate formal Stage-C calibration before Stage-D training is allowed."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


SCHEMA = "simul_uniss_subsecond_stage_c_quality_gate_v2"


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


def validate(
    calibration_path: Path,
    *,
    minimum_fast_recall: float,
    minimum_balanced_recall: float,
    minimum_quality_recall: float,
    maximum_ece: float,
) -> dict[str, object]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("scope") != "formal_target_microphrase_safe_commit_v2":
        raise ValueError("Stage-C calibration is not formal target safe-commit supervision")
    operating = calibration["operating_points"]
    recalls = {name: float(operating[name]["recall"]) for name in ("fast", "balanced", "quality")}
    thresholds = {
        "fast": minimum_fast_recall,
        "balanced": minimum_balanced_recall,
        "quality": minimum_quality_recall,
    }
    ece = float(calibration["calibrated_ece"])
    passed = all(recalls[name] >= threshold for name, threshold in thresholds.items()) and ece <= maximum_ece
    return {
        "schema_version": SCHEMA,
        "status": "passed" if passed else "failed",
        "scope": calibration["scope"],
        "calibration": str(calibration_path.resolve()),
        "records": int(calibration["records"]),
        "positive_rate": float(calibration["positive_rate"]),
        "calibrated_ece": ece,
        "maximum_ece": maximum_ece,
        "recall": recalls,
        "minimum_recall": thresholds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-fast-recall", type=float, default=0.01)
    parser.add_argument("--minimum-balanced-recall", type=float, default=0.01)
    parser.add_argument("--minimum-quality-recall", type=float, default=0.001)
    parser.add_argument("--maximum-ece", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(
        Path(args.calibration),
        minimum_fast_recall=args.minimum_fast_recall,
        minimum_balanced_recall=args.minimum_balanced_recall,
        minimum_quality_recall=args.minimum_quality_recall,
        maximum_ece=args.maximum_ece,
    )
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

