"""Validate decoded CVSS-T outputs before computing paper metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import soundfile as sf

from evaluation.io_utils import iter_jsonl, write_json


EXPECTED_SYNTHETIC_FLAGS = {
    "cmn->eng": (False, True),
    "eng->cmn": (True, False),
}


def resolve_path(value: object, *, input_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else input_path.parent / path


def readable_audio(path: Path) -> tuple[bool, str | None]:
    try:
        info = sf.info(path)
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    if info.frames <= 0 or info.samplerate <= 0:
        return False, f"invalid_audio_shape:frames={info.frames},sample_rate={info.samplerate}"
    return True, None


def validate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    input_path: Path,
    expected_pairs: int,
    expected_direction: str,
    modes: Sequence[str],
    allow_generated_failures: bool,
) -> dict[str, object]:
    if expected_pairs < 1:
        raise ValueError("expected_pairs must be positive")
    if expected_direction not in EXPECTED_SYNTHETIC_FLAGS:
        raise ValueError(f"unsupported CVSS-T direction: {expected_direction}")
    expected_modes = tuple(dict.fromkeys(modes))
    if not expected_modes:
        raise ValueError("at least one evaluation mode is required")

    expected_row_count = expected_pairs * len(expected_modes)
    keys: set[tuple[str, str]] = set()
    ids_by_mode: dict[str, set[str]] = defaultdict(set)
    direction_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    errors: list[dict[str, object]] = []
    audio_errors: list[dict[str, object]] = []
    generated_failures = 0

    expected_synthetic_source, expected_synthetic_reference = EXPECTED_SYNTHETIC_FLAGS[expected_direction]
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("id", ""))
        mode = str(row.get("mode", ""))
        direction = f"{row.get('src_lang')}->{row.get('tgt_lang')}"
        key = (sample_id, mode)
        if not sample_id:
            errors.append({"row_index": row_index, "reason": "missing_id"})
        if key in keys:
            errors.append({"id": sample_id, "mode": mode, "reason": "duplicate_id_mode"})
        keys.add(key)
        ids_by_mode[mode].add(sample_id)
        direction_counts[direction] += 1
        mode_counts[mode] += 1
        if mode not in expected_modes:
            errors.append({"id": sample_id, "mode": mode, "reason": "unexpected_mode"})
        if direction != expected_direction:
            errors.append(
                {
                    "id": sample_id,
                    "mode": mode,
                    "reason": "unexpected_direction",
                    "actual": direction,
                    "expected": expected_direction,
                }
            )
        if bool(row.get("synthetic_source")) != expected_synthetic_source:
            errors.append({"id": sample_id, "mode": mode, "reason": "synthetic_source_flag_mismatch"})
        if bool(row.get("synthetic_reference")) != expected_synthetic_reference:
            errors.append({"id": sample_id, "mode": mode, "reason": "synthetic_reference_flag_mismatch"})

        paths: dict[str, Path] = {}
        for field in ("source_audio_path", "reference_audio_path"):
            value = row.get(field)
            if not value:
                audio_errors.append({"id": sample_id, "mode": mode, "field": field, "reason": "missing_path"})
                continue
            path = resolve_path(value, input_path=input_path)
            paths[field] = path
            readable, reason = readable_audio(path)
            if not readable:
                audio_errors.append({"id": sample_id, "mode": mode, "field": field, "path": str(path), "reason": reason})

        generated_error = str(row.get("error") or "")
        generated_value = row.get("audio_path")
        if generated_error or not generated_value:
            generated_failures += 1
            if not allow_generated_failures:
                audio_errors.append(
                    {
                        "id": sample_id,
                        "mode": mode,
                        "field": "audio_path",
                        "reason": generated_error or "missing_generated_audio",
                    }
                )
        else:
            generated_path = resolve_path(generated_value, input_path=input_path)
            paths["audio_path"] = generated_path
            readable, reason = readable_audio(generated_path)
            if not readable:
                audio_errors.append(
                    {"id": sample_id, "mode": mode, "field": "audio_path", "path": str(generated_path), "reason": reason}
                )
            if generated_path in {paths.get("source_audio_path"), paths.get("reference_audio_path")}:
                audio_errors.append(
                    {"id": sample_id, "mode": mode, "field": "audio_path", "reason": "generated_audio_reuses_official_waveform"}
                )
        for field in ("source_audio_duration_seconds", "reference_audio_duration_seconds"):
            try:
                value = float(row.get(field, 0))
            except (TypeError, ValueError):
                value = 0.0
            if value <= 0 or not math.isfinite(value):
                errors.append({"id": sample_id, "mode": mode, "field": field, "reason": "invalid_duration"})

    missing_modes = sorted(set(expected_modes) - set(ids_by_mode))
    unexpected_modes = sorted(set(ids_by_mode) - set(expected_modes))
    if missing_modes:
        errors.append({"reason": "missing_modes", "modes": missing_modes})
    if unexpected_modes:
        errors.append({"reason": "unexpected_modes", "modes": unexpected_modes})
    if len(rows) != expected_row_count:
        errors.append({"reason": "row_count_mismatch", "actual": len(rows), "expected": expected_row_count})
    for mode in expected_modes:
        if len(ids_by_mode.get(mode, set())) != expected_pairs:
            errors.append(
                {
                    "reason": "mode_pair_count_mismatch",
                    "mode": mode,
                    "actual": len(ids_by_mode.get(mode, set())),
                    "expected": expected_pairs,
                }
            )
    if expected_modes:
        baseline = ids_by_mode.get(expected_modes[0], set())
        for mode in expected_modes[1:]:
            if ids_by_mode.get(mode, set()) != baseline:
                errors.append(
                    {
                        "reason": "mode_id_set_mismatch",
                        "mode": mode,
                        "missing_count": len(baseline - ids_by_mode.get(mode, set())),
                        "unexpected_count": len(ids_by_mode.get(mode, set()) - baseline),
                    }
                )

    report = {
        "input": str(input_path.resolve()),
        "valid": not errors and not audio_errors,
        "expected_pairs": expected_pairs,
        "expected_modes": list(expected_modes),
        "expected_direction": expected_direction,
        "row_count": len(rows),
        "unique_key_count": len(keys),
        "mode_counts": dict(sorted(mode_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "generated_failure_count": generated_failures,
        "structural_error_count": len(errors),
        "audio_error_count": len(audio_errors),
        "structural_error_examples": errors[:50],
        "audio_error_examples": audio_errors[:50],
    }
    if not report["valid"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=4897)
    parser.add_argument("--expected-direction", choices=tuple(EXPECTED_SYNTHETIC_FLAGS), required=True)
    parser.add_argument("--modes", nargs="+", default=["quality", "performance"])
    parser.add_argument("--allow-generated-failures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_rows(
        list(iter_jsonl(args.input)),
        input_path=args.input,
        expected_pairs=args.expected_pairs,
        expected_direction=args.expected_direction,
        modes=args.modes,
        allow_generated_failures=args.allow_generated_failures,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
