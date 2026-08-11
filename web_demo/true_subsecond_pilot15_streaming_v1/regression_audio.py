#!/usr/bin/env python3
"""Run one fixed uploaded-audio regression through the public demo engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DemoConfig
from .engine import TrueSubsecondStreamingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--direction", default="英文 → 中文")
    parser.add_argument("--chunk-ms", type=int, default=640, choices=(320, 480, 640))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoConfig.from_env()
    engine = TrueSubsecondStreamingEngine(config)
    result = None
    for update in engine.stream(
        args.input, direction=args.direction, decision_chunk_ms=args.chunk_ms
    ):
        if update.event is not None:
            print(update.status, flush=True)
        if update.result is not None:
            result = update.result
    if result is None:
        raise RuntimeError("stream ended without a result")
    summary = {
        "request_dir": result.request_dir,
        "selected_iteration": result.selected_iteration,
        "direction": result.direction,
        "decision_chunk_ms": result.decision_chunk_ms,
        "translation": result.committed_translation,
        "source_seconds": result.source_duration_seconds,
        "translation_seconds": result.translation_duration_seconds,
        "coverage": result.translation_coverage_ratio,
        "natural_writes": result.natural_writes,
        "forced_writes": result.forced_writes,
        "semantic_tokens": result.semantic_tokens,
        "semantic_unique_ratio": result.semantic_unique_ratio,
        "first_audio_ms": result.first_useful_audio_source_ms,
        "rtf": result.rtf,
        "quality_passed": result.quality_passed,
        "quality_failures": result.quality_failures,
        "result_path": result.result_path,
    }
    print("REGRESSION_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
