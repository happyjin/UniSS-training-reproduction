#!/usr/bin/env python3
"""Build and audit deterministic Stage A source-CTC target maps."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.events import (
    build_asr_event_session,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.ctc_targets import (
    UTF8ByteCTCMap,
    minimum_ctc_steps,
)
from training.phase3_whisper_streamspeech_joint.tokenizer_maps import CompactCTCMap
from training.simul_uniss.jsonl_index import load_index


SCHEMA = "uniss_quality_first_stage_a_ctc_maps_v2"
SHARD_PATTERN = re.compile(r"train-(\d{5})\.parquet$")


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A CTC report: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
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


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, total))
    return [
        (total * part // workers, total * (part + 1) // workers)
        for part in range(workers)
    ]


def _scan_worker(
    manifest: str,
    start: int,
    stop: int,
    model: str,
    allowed: dict[str, tuple[int, ...]] | None,
    target_kind: str,
) -> dict[str, Any]:
    path = Path(manifest)
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing validated offset index: {path}")
    tokenizer = None
    if target_kind == "qwen_train_compact":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=False,
        )
    token_counts = {"eng": Counter(), "cmn": Counter()}
    allowed_sets = None
    if allowed is not None:
        allowed_sets = {language: set(values) for language, values in allowed.items()}
    counters: Counter[str] = Counter()
    with path.open("rb") as handle:
        for index in range(start, stop):
            handle.seek(int(offsets[index]))
            record = json.loads(handle.readline())
            if not bool(record.get("formal_a68_pass")):
                raise ValueError(f"formal_a68_pass_false at record {index}")
            source = str(record.get("source_parquet") or "")
            match = SHARD_PATTERN.search(source)
            if match is None or not 0 <= int(match.group(1)) <= 14:
                raise ValueError(f"outside_train_00000_00014 at record {index}")
            session = build_asr_event_session(record)
            if target_kind == "utf8_byte":
                token_ids = list(session.normalized_transcript.encode("utf-8"))
                if bytes(token_ids).decode("utf-8") != session.normalized_transcript:
                    raise ValueError(f"utf8_roundtrip_failed at record {index}")
            else:
                token_ids = tokenizer.encode(  # type: ignore[union-attr]
                    session.normalized_transcript,
                    add_special_tokens=False,
                )
            if not token_ids:
                raise ValueError(f"empty_qwen_asr_target at record {index}")
            language = session.src_lang
            values = token_counts[language]
            values.update(int(token) for token in token_ids)
            counters["records"] += 1
            counters[f"records:{language}"] += 1
            counters["tokens"] += len(token_ids)
            counters[f"tokens:{language}"] += len(token_ids)
            minimum_steps = minimum_ctc_steps(token_ids)
            available_steps = max(1, (session.source_duration_ms + 19) // 20)
            counters["minimum_ctc_steps"] += minimum_steps
            counters["available_ctc_steps"] += available_steps
            counters["ctc_infeasible_records"] += int(minimum_steps > available_steps)
            counters[f"ctc_infeasible_records:{language}"] += int(
                minimum_steps > available_steps
            )
            counters["max_minimum_ctc_steps"] = max(
                counters["max_minimum_ctc_steps"], minimum_steps
            )
            if allowed_sets is not None:
                oov = sum(int(token) not in allowed_sets[language] for token in token_ids)
                counters["oov_tokens"] += oov
                counters[f"oov_tokens:{language}"] += oov
    return {
        "counters": dict(counters),
        "token_counts": {
            language: dict(sorted(values.items()))
            for language, values in token_counts.items()
        },
    }


def _scan(
    manifest: Path,
    model: Path,
    workers: int,
    allowed: dict[str, tuple[int, ...]] | None = None,
    target_kind: str = "utf8_byte",
) -> tuple[Counter[str], dict[str, Counter[int]]]:
    offsets = load_index(manifest)
    if offsets is None:
        raise ValueError(f"missing validated offset index: {manifest}")
    futures = []
    with ProcessPoolExecutor(max_workers=min(workers, len(offsets))) as pool:
        for start, stop in _ranges(len(offsets), workers):
            futures.append(
                pool.submit(
                    _scan_worker,
                    str(manifest),
                    start,
                    stop,
                    str(model),
                    allowed,
                    target_kind,
                )
            )
        parts = [future.result() for future in as_completed(futures)]
    counters: Counter[str] = Counter()
    token_counts = {"eng": Counter(), "cmn": Counter()}
    max_keys = {"max_minimum_ctc_steps"}
    for part in parts:
        for key, value in part["counters"].items():
            key = str(key)
            if key in max_keys:
                counters[key] = max(counters[key], int(value))
            else:
                counters[key] += int(value)
        for language in ("eng", "cmn"):
            token_counts[language].update(
                {int(key): int(value) for key, value in part["token_counts"][language].items()}
            )
    return counters, token_counts


def _token_details(
    counts: Counter[int],
    tokenizer: Any,
    limit: int = 50,
) -> list[dict[str, object]]:
    return [
        {
            "qwen_id": int(token),
            "count": int(count),
            "token": tokenizer.convert_ids_to_tokens(int(token)),
            "decoded": tokenizer.decode([int(token)], skip_special_tokens=False),
        }
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-map-dir", type=Path)
    parser.add_argument(
        "--target-kind",
        choices=("utf8_byte", "qwen_train_compact"),
        default="utf8_byte",
    )
    parser.add_argument("--train-workers", type=int, default=30)
    parser.add_argument("--valid-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage A CTC maps: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    train_counters, train_counts = _scan(
        args.train_manifest,
        args.model,
        args.train_workers,
        target_kind=args.target_kind,
    )
    if args.target_kind == "utf8_byte":
        maps = {language: UTF8ByteCTCMap(language) for language in ("eng", "cmn")}
        allowed = {language: tuple(range(256)) for language in ("eng", "cmn")}
    else:
        maps = {
            language: CompactCTCMap(
                language=language,
                qwen_to_compact={
                    token: index for index, token in enumerate(sorted(train_counts[language]))
                },
                compact_to_qwen=tuple(sorted(train_counts[language])),
            )
            for language in ("eng", "cmn")
        }
        allowed = {
            language: mapping.compact_to_qwen  # type: ignore[union-attr]
            for language, mapping in maps.items()
        }
    for language, mapping in maps.items():
        mapping.save(args.output_dir / f"ctc_qwen_{language}.json")

    valid_counters, valid_counts = _scan(
        args.valid_manifest,
        args.model,
        args.valid_workers,
        allowed,
        target_kind=args.target_kind,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    validation_oov: dict[str, Counter[int]] = {}
    reference_missing: dict[str, Counter[int]] = {}
    for language in ("eng", "cmn"):
        allowed_set = set(allowed[language])
        validation_oov[language] = Counter(
            {
                token: count
                for token, count in valid_counts[language].items()
                if token not in allowed_set
            }
        )
        if args.reference_map_dir is not None and args.target_kind == "qwen_train_compact":
            reference = CompactCTCMap.load(
                args.reference_map_dir / f"ctc_qwen_{language}.json"
            )
            reference_set = set(reference.compact_to_qwen)
            reference_missing[language] = Counter(
                {
                    token: count
                    for token, count in train_counts[language].items()
                    if token not in reference_set
                }
            )

    checks = {
        "train_records_nonzero": train_counters["records"] > 0,
        "train_vocab_nonempty_eng": maps["eng"].blank_id > 0,
        "train_vocab_nonempty_cmn": maps["cmn"].blank_id > 0,
        "valid_records_nonzero": valid_counters["records"] > 0,
        "valid_oov_zero": valid_counters["oov_tokens"] == 0,
        "train_ctc_feasible": train_counters["ctc_infeasible_records"] == 0,
        "valid_ctc_feasible": valid_counters["ctc_infeasible_records"] == 0,
    }
    report = {
        "schema_version": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "target_kind": args.target_kind,
        "provenance_policy": (
            "label-independent fixed 256-byte UTF-8 inventory; train and validation are audit-only"
            if args.target_kind == "utf8_byte"
            else "maps are derived only from Stage A train canonical transcripts; validation is audit-only"
        ),
        "train_manifest": str(args.train_manifest.resolve()),
        "valid_manifest": str(args.valid_manifest.resolve()),
        "model": str(args.model.resolve()),
        "train_workers": args.train_workers,
        "valid_workers": args.valid_workers,
        "train_counters": dict(sorted(train_counters.items())),
        "valid_counters": dict(sorted(valid_counters.items())),
        "maps": {
            language: {
                "path": str((args.output_dir / f"ctc_qwen_{language}.json").resolve()),
                "classes_without_blank": maps[language].blank_id,
                "blank_id": maps[language].blank_id,
                "validation_oov_unique": len(validation_oov[language]),
                "validation_oov_tokens": sum(validation_oov[language].values()),
                "validation_oov_top": _token_details(validation_oov[language], tokenizer),
                "reference_missing_unique": len(reference_missing.get(language, {})),
                "reference_missing_tokens": sum(reference_missing.get(language, {}).values()),
                "reference_missing_top": _token_details(
                    reference_missing.get(language, Counter()),
                    tokenizer,
                ),
            }
            for language in ("eng", "cmn")
        },
    }
    _atomic_json(args.output_dir / "ctc_map_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
