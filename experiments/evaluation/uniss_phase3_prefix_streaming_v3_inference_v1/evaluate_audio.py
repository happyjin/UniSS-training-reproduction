#!/usr/bin/env python3
"""Run one audio through one or all supported streaming chunk sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .streaming_engine import EngineConfig, PrefixStreamingEngine


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--direction", choices=["zh-en", "en-zh"], required=True)
    parser.add_argument("--chunk-ms", choices=["320", "480", "640", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=root / "checkpoints/exported_adapters/uniss_phase3_prefix_streaming_full198_joint_v3_iter_0008000_lora_v1",
    )
    parser.add_argument("--speech-tokenizer", type=Path, default=root / "pretrained_models/UniSS")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "eval_outputs/uniss_phase3_prefix_streaming_v3_iter8000_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = [320, 480, 640] if args.chunk_ms == "all" else [int(args.chunk_ms)]
    summaries: list[dict[str, object]] = []
    for chunk in chunks:
        engine = PrefixStreamingEngine(
            EngineConfig(
                adapter_dir=args.adapter_dir,
                speech_tokenizer_dir=args.speech_tokenizer,
                output_root=args.output_root,
                device=args.device,
                chunk_ms=chunk,
            )
        )
        final = None
        for update in engine.stream(args.audio, direction=args.direction):
            print(update.status, flush=True)
            if update.result is not None:
                final = update.result
        if final is None:
            raise RuntimeError(f"chunk {chunk} produced no final result")
        summaries.append(final.to_dict())
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

