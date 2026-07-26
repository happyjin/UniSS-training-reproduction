"""Compute paper-aligned Speech Length Compliance metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping, Sequence

import soundfile as sf

from evaluation.io_utils import iter_jsonl, write_json, write_jsonl


def duration_seconds(path: str | Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"invalid samplerate for {path}")
    return float(info.frames / info.samplerate)


def resolve_audio_path(value: object, *, results_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return results_path.parent / path


def compute_slc_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    results_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scored: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in rows:
        generated_value = row.get("audio_path")
        source_value = row.get("source_audio_path")
        if not generated_value or not source_value:
            skipped.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "missing_audio_path"})
            continue
        generated_path = resolve_audio_path(generated_value, results_path=results_path)
        source_path = resolve_audio_path(source_value, results_path=results_path)
        try:
            source_duration = float(row.get("source_audio_duration_seconds") or duration_seconds(source_path))
            generated_duration = float(row.get("audio_duration_seconds") or duration_seconds(generated_path))
        except Exception as exc:
            skipped.append(
                {"id": row.get("id"), "mode": row.get("mode"), "reason": f"duration_error:{type(exc).__name__}:{exc}"}
            )
            continue
        if source_duration <= 0 or not math.isfinite(source_duration) or not math.isfinite(generated_duration):
            skipped.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "invalid_duration"})
            continue
        ratio = generated_duration / source_duration
        scored.append(
            {
                "id": row.get("id"),
                "mode": row.get("mode"),
                "src_lang": row.get("src_lang"),
                "tgt_lang": row.get("tgt_lang"),
                "source_duration_seconds": source_duration,
                "generated_duration_seconds": generated_duration,
                "duration_ratio": ratio,
                "slc_0_2": abs(ratio - 1.0) <= 0.2,
                "slc_0_4": abs(ratio - 1.0) <= 0.4,
            }
        )
    return scored, skipped


def aggregate_slc(scored: Sequence[Mapping[str, object]], skipped: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in scored:
        groups[(str(row["mode"]), str(row["src_lang"]), str(row["tgt_lang"]))].append(row)
    aggregates: dict[str, object] = {}
    for (mode, src_lang, tgt_lang), group_rows in sorted(groups.items()):
        ratios = [float(row["duration_ratio"]) for row in group_rows]
        aggregates[f"{mode}:{src_lang}->{tgt_lang}"] = {
            "sample_count": len(group_rows),
            "slc_0_2": sum(bool(row["slc_0_2"]) for row in group_rows) / len(group_rows),
            "slc_0_4": sum(bool(row["slc_0_4"]) for row in group_rows) / len(group_rows),
            "duration_ratio_mean": mean(ratios),
            "duration_ratio_std": pstdev(ratios),
            "duration_ratio_min": min(ratios),
            "duration_ratio_max": max(ratios),
        }
    return {
        "groups": aggregates,
        "scored_count": len(scored),
        "skipped_count": len(skipped),
        "skipped": list(skipped),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = list(iter_jsonl(args.input))
    scored, skipped = compute_slc_rows(rows, results_path=args.input)
    report = aggregate_slc(scored, skipped)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_sample_slc.jsonl", scored)
    write_json(args.output_dir / "slc.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
