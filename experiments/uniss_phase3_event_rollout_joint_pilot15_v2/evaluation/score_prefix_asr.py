#!/usr/bin/env python3
"""Identify first useful translated audio from target-language prefix ASR."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from evaluation.runtime_parity_streaming.gates import percentile


SCHEMA = "uniss_event_rollout_fixed15_useful_audio_asr_v1"


def normalize(text: str, language: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    if language == "cmn":
        return "".join(value.split())
    return " ".join(value.split())


def content_units(text: str, language: str) -> int:
    value = normalize(text, language)
    if language == "cmn":
        return len(value)
    return len(re.findall(r"[a-z0-9]+", value))


def prefix_similarity(asr_text: str, prefixes: Sequence[str], language: str) -> float:
    hypothesis = normalize(asr_text, language)
    if not hypothesis:
        return 0.0
    return max(
        (
            difflib.SequenceMatcher(
                None, hypothesis, normalize(prefix, language), autojunk=False
            ).ratio()
            for prefix in prefixes
            if normalize(prefix, language)
        ),
        default=0.0,
    )


def score(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_similarity: float,
    minimum_content_units: int,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        language = str(row["tgt_lang"])
        similarity = prefix_similarity(
            str(row.get("asr_text", "")),
            [str(value) for value in row.get("oracle_target_text_prefixes", [])],
            language,
        )
        units = content_units(str(row.get("asr_text", "")), language)
        row["prefix_similarity"] = similarity
        row["content_units"] = units
        row["useful_translated_audio"] = (
            units >= minimum_content_units and similarity >= minimum_similarity
        )
        grouped[str(row["parent_sample_id"])].append(row)

    samples = []
    for sample_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda value: (float(value["wall_end_ms"]), int(value["candidate_index"])))
        useful = next((value for value in candidates if value["useful_translated_audio"]), None)
        first = candidates[0]
        samples.append(
            {
                "sample_id": sample_id,
                "src_lang": first["src_lang"],
                "tgt_lang": first["tgt_lang"],
                "candidate_count": len(candidates),
                "first_useful_audio_found": useful is not None,
                "first_useful_audio_source_ms": None if useful is None else useful["source_end_ms"],
                "first_useful_audio_wall_ms": None if useful is None else useful["wall_end_ms"],
                "first_useful_audio_asr_text": None if useful is None else useful.get("asr_text"),
                "first_useful_audio_prefix_similarity": None if useful is None else useful["prefix_similarity"],
                "first_useful_audio_candidate_id": None if useful is None else useful["id"],
                "candidates": candidates,
            }
        )

    def group_report(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
        latency = [
            float(value["first_useful_audio_wall_ms"])
            for value in values
            if value["first_useful_audio_wall_ms"] is not None
        ]
        return {
            "samples": len(values),
            "useful_audio_recall": fmean(
                bool(value["first_useful_audio_found"]) for value in values
            ) if values else None,
            "first_useful_audio_wall_ms": {
                "count": len(latency),
                "p50": percentile(latency, 0.50),
                "p90": percentile(latency, 0.90),
                "p95": percentile(latency, 0.95),
            },
        }

    by_direction: dict[str, list[dict[str, object]]] = defaultdict(list)
    for value in samples:
        by_direction[f"{value['src_lang']}->{value['tgt_lang']}"] .append(value)
    groups = {"all": group_report(samples)}
    groups.update({name: group_report(values) for name, values in sorted(by_direction.items())})
    return {
        "schema_version": SCHEMA,
        "minimum_similarity": minimum_similarity,
        "minimum_content_units": minimum_content_units,
        "definition": (
            "Earliest cumulative natural-WRITE PCM prefix whose target-language ASR "
            "contains at least the minimum content units and matches an oracle translation prefix."
        ),
        "groups": groups,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-similarity", type=float, default=0.50)
    parser.add_argument("--minimum-content-units", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite useful-audio report: {args.output}")
    with args.asr_results.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = score(
        rows,
        minimum_similarity=args.minimum_similarity,
        minimum_content_units=args.minimum_content_units,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

