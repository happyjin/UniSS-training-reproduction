#!/usr/bin/env python3
"""Reproducible CLI validation for bounded long-form inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.streaming_engine import (
    EngineConfig,
    PrefixStreamingEngine,
)

from .config import LongFormDemoConfig
from .engine import BoundedLongFormEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--direction", choices=["zh-en", "en-zh"], required=True)
    parser.add_argument("--chunk-ms", type=int, choices=[320, 480, 640], default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LongFormDemoConfig.from_env()
    config = replace(
        config,
        device=args.device,
        output_root=args.output_root or config.output_root,
    )
    config.validate_assets()
    base = PrefixStreamingEngine(
        EngineConfig(
            adapter_dir=config.adapter_dir,
            speech_tokenizer_dir=config.speech_tokenizer_dir,
            output_root=config.output_root / "window_runs",
            device=config.device,
            chunk_ms=args.chunk_ms,
            max_upload_bytes=config.max_upload_bytes,
            max_audio_seconds=config.maximum_window_seconds + 0.05,
        )
    )
    engine = BoundedLongFormEngine(config, base_engine=base)
    engine.load()
    final = None
    for update in engine.run(
        args.input, direction=args.direction, chunk_ms=args.chunk_ms
    ):
        print(
            json.dumps(
                {
                    "progress": update.progress,
                    "status": update.status,
                    "translation_chars": len(update.translation),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if update.result is not None:
            final = update.result
    if final is None:
        raise RuntimeError("validation completed without a final result")
    print(f"RESULT_JSON={final.result_path}", flush=True)
    print(f"STEREO_WAV={final.stereo_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
