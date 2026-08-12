#!/usr/bin/env python3
"""Aggregate exact-runtime shards without turning missing quality evidence into pass."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from evaluation.runtime_parity_streaming.gates import percentile


SCHEMA = "uniss_event_rollout_fixed15_runtime_aggregate_v1"


def _values(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def _quantiles(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, float | None]:
    values = _values(rows, key)
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "mean": fmean(values) if values else None,
    }


def _inter_write_gaps(row: Mapping[str, object], key: str) -> list[float]:
    writes = [event for event in row.get("events", []) if event.get("action") == "WRITE"]
    values = [float(event[key]) for event in writes if event.get(key) is not None]
    return [later - earlier for earlier, later in zip(values, values[1:])]


def _group(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"samples": 0, "status": "not_evaluable"}
    all_wait = [int(row.get("natural_writes", 0) or 0) == 0 for row in rows]
    post_eos = [bool(row.get("source_finished_before_first_write", True)) for row in rows]
    premature = [
        row.get("first_write_source_ms") is not None
        and row.get("oracle_first_safe_write_ms") is not None
        and float(row["first_write_source_ms"]) < float(row["oracle_first_safe_write_ms"])
        for row in rows
    ]
    forced = sum(int(row.get("forced_writes", 0) or 0) for row in rows)
    revisions = sum(int(row.get("committed_revision_violations", 0) or 0) for row in rows)
    playable = [int(row.get("translation_audio_samples", 0) or 0) > 0 for row in rows]
    finite = [bool(row.get("translation_audio_finite", False)) for row in rows]
    collapse = [bool(row.get("severe_semantic_collapse", True)) for row in rows]
    eos = [bool(row.get("natural_eos", False)) for row in rows]
    quality = [bool(row.get("quality_passed", False)) for row in rows]
    source_gaps = [gap for row in rows for gap in _inter_write_gaps(row, "source_end_ms")]
    wall_gaps = [gap for row in rows for gap in _inter_write_gaps(row, "wall_end_ms")]
    writes = [int(row.get("natural_writes", 0) or 0) for row in rows]
    event_counts = [len(row.get("events", [])) for row in rows]
    return {
        "samples": len(rows),
        "quality_pass_rate": fmean(quality),
        "natural_write_sample_rate": 1.0 - fmean(all_wait),
        "all_wait_rate": fmean(all_wait),
        "post_source_eos_first_write_rate": fmean(post_eos),
        "first_write_premature_rate": fmean(premature),
        "granular_premature_write_rate": "not_evaluable",
        "forced_writes": forced,
        "committed_revision_violations": revisions,
        "playable_pcm_rate": fmean(playable),
        "finite_pcm_rate": fmean(finite),
        "severe_semantic_collapse_rate": fmean(collapse),
        "natural_eos_rate": fmean(eos),
        "natural_writes_per_sample": fmean(writes),
        "read_write_ratio": (
            sum(max(0, events - write_count) for events, write_count in zip(event_counts, writes))
            / max(1, sum(writes))
        ),
        "first_write_source_ms": _quantiles(rows, "first_write_source_ms"),
        "first_arbitrary_pcm_source_ms": _quantiles(rows, "first_audio_source_ms"),
        "first_write_wall_ms": _quantiles(rows, "first_write_wall_ms"),
        "first_arbitrary_pcm_wall_ms": _quantiles(rows, "first_audio_wall_ms"),
        "first_useful_audio_wall_ms": "not_evaluable_until_prefix_asr",
        "rtf": _quantiles(rows, "rtf"),
        "maximum_compute_backlog_ms": _quantiles(rows, "maximum_compute_backlog_ms"),
        "inter_write_source_gap_ms": {
            "count": len(source_gaps),
            "p50": percentile(source_gaps, 0.50),
            "p90": percentile(source_gaps, 0.90),
            "p95": percentile(source_gaps, 0.95),
        },
        "inter_write_wall_gap_ms": {
            "count": len(wall_gaps),
            "p50": percentile(wall_gaps, 0.50),
            "p90": percentile(wall_gaps, 0.90),
            "p95": percentile(wall_gaps, 0.95),
        },
        "source_to_target_average_lagging_ms": "not_evaluable_without_token_level_runtime_alignment",
    }


def _expected_samples(
    manifests: Sequence[Path],
) -> tuple[set[str], Counter[str]]:
    ids: set[str] = set()
    directions: Counter[str] = Counter()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["id"])
                if sample_id in ids:
                    raise ValueError(
                        f"duplicate expected sample ID {sample_id} in {manifest}:{line_number}"
                    )
                ids.add(sample_id)
                directions[f"{row.get('src_lang')}->{row.get('tgt_lang')}"] += 1
    return ids, directions


def aggregate(
    summaries: Sequence[Mapping[str, object]],
    *,
    expected_sample_ids: set[str] | None = None,
    expected_directions: Mapping[str, int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    ids: set[str] = set()
    for summary in summaries:
        provenance.append(
            {
                "checkpoint": summary.get("checkpoint"),
                "formal_manifest": summary.get("formal_manifest"),
                "schema_version": summary.get("schema_version"),
            }
        )
        for raw in summary.get("samples", []):
            row = dict(raw)
            sample_id = str(row["sample_id"])
            if sample_id in ids:
                raise ValueError(f"duplicate exact-runtime sample: {sample_id}")
            ids.add(sample_id)
            row.update(
                {
                    "id": sample_id,
                    "mode": "exact_runtime",
                    "translation_ref": str(row.get("target_text", "")),
                    "generated_translation": str(row.get("generated_text", "")),
                    "audio_duration_seconds": float(row.get("translation_audio_samples", 0))
                    / max(1, int(row.get("translation_audio_sample_rate", 16000))),
                    "source_audio_duration_seconds": float(row.get("source_duration_ms", 0)) / 1000.0,
                }
            )
            rows.append(row)
    if expected_sample_ids is not None:
        missing = sorted(expected_sample_ids - ids)
        extra = sorted(ids - expected_sample_ids)
        if missing or extra:
            raise ValueError(
                "exact-runtime sample coverage mismatch: "
                f"missing={len(missing)} {missing[:8]}, extra={len(extra)} {extra[:8]}"
            )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[f"{row.get('src_lang')}->{row.get('tgt_lang')}"] .append(row)
    actual_directions = Counter(
        f"{row.get('src_lang')}->{row.get('tgt_lang')}" for row in rows
    )
    if expected_directions is not None and actual_directions != Counter(expected_directions):
        raise ValueError(
            "exact-runtime direction coverage mismatch: "
            f"expected={dict(expected_directions)}, actual={dict(actual_directions)}"
        )
    report = {
        "schema_version": SCHEMA,
        "samples": len(rows),
        "unique_sample_ids": len(ids),
        "directions": dict(sorted(actual_directions.items())),
        "coverage": {
            "expected_samples": (
                len(expected_sample_ids) if expected_sample_ids is not None else "not_provided"
            ),
            "observed_samples": len(ids),
            "complete": expected_sample_ids is not None and ids == expected_sample_ids,
        },
        "provenance": provenance,
        "groups": {name: _group(values) for name, values in sorted(grouped.items())},
        "missing_evidence": [
            "first_useful_audio_prefix_asr",
            "target_language_asr_and_speech_bleu",
            "autopcp",
            "slc",
            "speaker_similarity",
            "cached_uncached_parity",
            "fused_unfused_parity",
            "phase3_replay_retention",
        ],
    }
    return report, rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Fixed15 exact-runtime aggregate",
        "",
        f"- Samples: {report['samples']}",
        f"- Directions: `{json.dumps(report['directions'], sort_keys=True)}`",
        "- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.",
        "",
        "| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in report["groups"].items():
        first_write = group["first_write_source_ms"]
        first_audio = group["first_arbitrary_pcm_wall_ms"]
        rtf = group["rtf"]
        lines.append(
            f"| {name} | {group['samples']} | {group['natural_write_sample_rate']:.4f} | "
            f"{group['all_wait_rate']:.4f} | {group['first_write_premature_rate']:.4f} | "
            f"{group['playable_pcm_rate']:.4f} | {group['severe_semantic_collapse_rate']:.4f} | "
            f"{group['natural_eos_rate']:.4f} | {first_write['p50']}/{first_write['p95']} | "
            f"{first_audio['p50']}/{first_audio['p95']} | {rtf['p50']}/{rtf['p95']} |"
        )
    lines.extend(["", "## Missing hard-gate evidence", ""])
    lines.extend(f"- `{value}`" for value in report["missing_evidence"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--expected-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {args.output_root}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    expected_ids = None
    expected_directions = None
    if args.expected_manifest:
        expected_ids, expected_directions = _expected_samples(args.expected_manifest)
    report, rows = aggregate(
        summaries,
        expected_sample_ids=expected_ids,
        expected_directions=expected_directions,
    )
    args.output_root.mkdir(parents=True)
    (args.output_root / "aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "aggregate.md").write_text(markdown(report), encoding="utf-8")
    _write_jsonl(args.output_root / "results.jsonl", rows)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
