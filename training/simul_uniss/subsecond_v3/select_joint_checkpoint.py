"""Select Stage-B-v3 using balanced agreement and frozen-Phase3 BLEU."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


SCHEMA = "simul_uniss_stage_b_v3_joint_selection_v1"
DIRECTIONS = ("eng->cmn", "cmn->eng")


def _harmonic(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bleu_scores(report: dict[str, object], stream_name: str) -> dict[str, float]:
    text_bleu = report.get("text_bleu")
    if not isinstance(text_bleu, dict) or not isinstance(text_bleu.get("groups"), dict):
        raise ValueError("Phase3 report is missing text_bleu.groups")
    groups = text_bleu["groups"]
    scores: dict[str, float] = {}
    for direction in DIRECTIONS:
        group = groups.get(f"{stream_name}:{direction}")
        if not isinstance(group, dict) or "score" not in group:
            raise ValueError(f"missing BLEU group {stream_name}:{direction}")
        scores[direction] = float(group["score"])
    return scores


def select(args: argparse.Namespace) -> dict[str, object]:
    candidates_path = Path(args.candidates).resolve()
    candidates_value = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = candidates_value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("CANDIDATES.json has no candidates")
    if not math.isclose(args.agreement_weight + args.bleu_weight, 1.0):
        raise ValueError("agreement and BLEU weights must sum to one")

    result_dir = Path(args.phase3_result_dir).resolve()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("candidate entry must be an object")
        checkpoint = Path(str(candidate["checkpoint"])).resolve()
        stream_name = f"candidate_{checkpoint.stem}"
        report_path = result_dir / f"{checkpoint.stem}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(str(report.get("student_checkpoint"))).resolve() != checkpoint:
            raise ValueError(f"checkpoint mismatch in {report_path}")
        if report.get("student_stream_name") != stream_name:
            raise ValueError(f"stream mismatch in {report_path}")
        bleu = _bleu_scores(report, stream_name)
        bleu_hmean = _harmonic(bleu["eng->cmn"], bleu["cmn->eng"])
        agreement_score = float(candidate["score"])
        joint_score = (
            args.agreement_weight * agreement_score
            + args.bleu_weight * (bleu_hmean / 100.0)
        )
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "agreement_score": agreement_score,
                "agreement_metrics": candidate.get("metrics", {}),
                "bleu_eng_cmn": bleu["eng->cmn"],
                "bleu_cmn_eng": bleu["cmn->eng"],
                "bleu_direction_hmean": bleu_hmean,
                "joint_score": joint_score,
                "phase3_report": str(report_path),
            }
        )
    rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    winner = rows[0]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best = output_dir / "best.pt"
    if best.exists() and not args.allow_replace:
        raise FileExistsError(f"refusing to replace existing {best}")
    temporary = best.with_name(f".{best.name}.tmp.{os.getpid()}")
    shutil.copy2(str(winner["checkpoint"]), temporary)
    os.replace(temporary, best)
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "candidates": str(candidates_path),
        "agreement_weight": args.agreement_weight,
        "bleu_weight": args.bleu_weight,
        "ranking": rows,
        "selected_checkpoint": winner["checkpoint"],
        "exported_best": str(best),
    }
    _atomic_json(output_dir / "JOINT_SELECTION.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--phase3-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agreement-weight", type=float, default=0.5)
    parser.add_argument("--bleu-weight", type=float, default=0.5)
    parser.add_argument("--allow-replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    select(parse_args())
