#!/usr/bin/env python3
"""Run dense train/validation audio through the true-input-causal runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_demo.streaming_s2st_r2_v1.audio_io import write_json
from web_demo.true_subsecond_pilot15_streaming_v1.engine import (
    TrueSubsecondStreamingEngine,
)

from .config import REPO_ROOT, load_config


MANIFESTS = {
    "train": REPO_ROOT
    / "data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/dense_train.jsonl",
    "valid": REPO_ROOT
    / "data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/dense_valid.jsonl",
}


def jsonl_record(path: Path, index: int) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"record {index} does not exist in {path}")


def source_record(dense: dict[str, object]) -> dict[str, object]:
    return jsonl_record(Path(str(dense["source_manifest"])), int(dense["source_index"]))


def direction_for(record: dict[str, object]) -> str:
    pair = (str(record["src_lang"]), str(record["tgt_lang"]))
    if pair == ("eng", "cmn"):
        return "英文 → 中文"
    if pair == ("cmn", "eng"):
        return "中文 → 英文"
    raise ValueError(f"unsupported language pair: {pair}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--valid-count", type=int, default=2)
    parser.add_argument("--chunk-ms", type=int, choices=(320, 480, 640), default=320)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "reports/uniss_phase3_dense_aligned_streaming_pilot15_v1/"
        "iter_0000500_true_streaming_samples.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_count < 0 or args.valid_count < 0:
        raise ValueError("sample counts must be non-negative")
    config = load_config()
    config.validate_assets(require_export=True)
    engine = TrueSubsecondStreamingEngine(config)
    engine.load()
    rows: list[dict[str, object]] = []
    for split, count in (("train", args.train_count), ("valid", args.valid_count)):
        for index in range(count):
            dense = jsonl_record(MANIFESTS[split], index)
            source = source_record(dense)
            result = None
            for update in engine.stream(
                Path(str(source["source_audio"])),
                direction=direction_for(source),
                decision_chunk_ms=args.chunk_ms,
            ):
                if update.result is not None:
                    result = update.result
            if result is None:
                raise RuntimeError(f"stream ended without result for {dense['sample_id']}")
            first_write = result.first_write_source_ms
            source_ms = float(source["source_duration_ms"])
            rows.append(
                {
                    "split": split,
                    "dense_index": index,
                    "sample_id": dense["sample_id"],
                    "source_audio": source["source_audio"],
                    "reference_transcription": source["transcription"],
                    "reference_translation": source["translation"],
                    "predicted_translation": result.committed_translation,
                    "source_duration_ms": source_ms,
                    "first_write_source_ms": first_write,
                    "first_write_before_source_end": (
                        first_write is not None and float(first_write) < source_ms
                    ),
                    "first_useful_audio_source_ms": result.first_useful_audio_source_ms,
                    "natural_writes": result.natural_writes,
                    "forced_writes": result.forced_writes,
                    "wait_events": result.wait_events,
                    "semantic_tokens": result.semantic_tokens,
                    "translation_coverage_ratio": result.translation_coverage_ratio,
                    "rtf": result.rtf,
                    "quality_passed": result.quality_passed,
                    "quality_failures": result.quality_failures,
                    "committed_revision_violations": result.committed_revision_violations,
                    "result_path": result.result_path,
                    "stereo_path": result.stereo_path,
                }
            )
            print(
                f"{split}[{index}] {dense['sample_id']} "
                f"first_write={first_write}ms natural={result.natural_writes} "
                f"forced={result.forced_writes} quality={result.quality_passed}",
                flush=True,
            )
    report = {
        "schema_version": "uniss_dense_aligned_true_streaming_sample_eval_v1",
        "checkpoint": str(config.checkpoint.resolve()),
        "selected_iteration": 500,
        "decision_chunk_ms": args.chunk_ms,
        "true_input_causal_runtime": True,
        "browser_webrtc_live": False,
        "samples": rows,
    }
    write_json(args.output, report)
    print(f"REPORT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

