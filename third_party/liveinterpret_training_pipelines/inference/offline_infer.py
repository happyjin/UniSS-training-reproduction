#!/usr/bin/env python3
"""Offline qwen_omnispeak inference CLI.

Supports text/text, text/audio, speech/text and speech/audio test cases from a
single prompt/audio pair or a JSONL file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

try:
    from .common import (
        InferenceConfig,
        InferenceSample,
        append_jsonl,
        generate_audio,
        generate_text,
        load_model_and_processor,
        make_output_paths,
        prepare_inputs,
        print_metric_summary,
        read_jsonl_samples,
        save_audio,
        save_text,
        single_sample,
    )
except ImportError:  # pragma: no cover - allows `python inference/offline_infer.py`.
    from common import (  # type: ignore
        InferenceConfig,
        InferenceSample,
        append_jsonl,
        generate_audio,
        generate_text,
        load_model_and_processor,
        make_output_paths,
        prepare_inputs,
        print_metric_summary,
        read_jsonl_samples,
        save_audio,
        save_text,
        single_sample,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline qwen_omnispeak inference tester")
    parser.add_argument("--model", required=True, help="Path to exported qwen_omnispeak model directory")
    parser.add_argument("--prompt", help="Prompt for single-sample inference")
    parser.add_argument("--audio", help="Optional input audio path for speech tasks")
    parser.add_argument("--samples", help="Optional JSONL batch file")
    parser.add_argument("--sample-id", default="sample_0", help="Single-sample id")
    parser.add_argument("--task", default="auto", help="Task label, e.g. t2t/t2s/s2t/s2s")
    parser.add_argument("--output", choices=("text", "audio"), default="text", help="Prediction output modality")
    parser.add_argument("--stream", action="store_true", help="Stream generated tokens/audio chunks and report first latency")
    parser.add_argument("--target-lang", help="Target language for audio codec control prefix")
    parser.add_argument("--output-dir", default="inference_outputs/offline")
    parser.add_argument("--metrics-file", default="metrics.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-codec-frames", type=int, default=256)
    parser.add_argument("--codec-chunk-frames", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--silence-threshold", type=float, default=1e-4)
    parser.add_argument("--codec-tokenizer", help="Override Qwen3-TTS tokenizer id/path")
    parser.add_argument("--codec-tokenizer-device", help="Device for Qwen3-TTS tokenizer, e.g. cuda:0")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_codec_frames=args.max_codec_frames,
        codec_chunk_frames=args.codec_chunk_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        target_lang=args.target_lang,
        silence_threshold=args.silence_threshold,
        codec_tokenizer=args.codec_tokenizer,
        codec_tokenizer_device=args.codec_tokenizer_device,
    )


def load_samples(args: argparse.Namespace) -> List[InferenceSample]:
    if args.samples:
        return list(read_jsonl_samples(args.samples, default_task=args.task, default_output=args.output))
    if args.prompt is None:
        raise SystemExit("--prompt is required when --samples is not provided")
    return [single_sample(args.sample_id, args.prompt, args.audio, args.task, args.output, args.target_lang)]


def run_sample(runtime, sample: InferenceSample, config: InferenceConfig, *, stream: bool):
    model, processor, model_config, device = runtime
    inputs = prepare_inputs(processor, sample.prompt, sample.audio, device)
    text_path, audio_path = make_output_paths(config.output_dir, sample.sample_id)

    if sample.output == "audio":
        def on_audio(chunk, sr, elapsed_ms):
            print(f"[{sample.sample_id}] audio_chunk samples={len(chunk)} sr={sr} elapsed_ms={elapsed_ms:.2f}")

        result = generate_audio(
            model,
            processor,
            model_config,
            inputs,
            sample,
            config,
            stream=stream,
            realtime_schedule=False,
            callback=on_audio if stream else None,
        )
        result.metrics.output_audio_path = save_audio(audio_path, result.audio, result.sample_rate)
        return result.metrics

    def on_text(piece: str, token_id: int) -> None:
        print(piece, end="", flush=True)

    result = generate_text(
        model,
        processor,
        inputs,
        sample,
        config,
        stream=stream,
        callback=on_text if stream else None,
    )
    if stream:
        print()
    result.metrics.output_text_path = save_text(text_path, result.text)
    return result.metrics


def main() -> None:
    args = parse_args()
    config = build_config(args)
    samples = load_samples(args)
    output_dir = Path(config.output_dir)
    metrics_path = output_dir / args.metrics_file
    if metrics_path.exists():
        metrics_path.unlink()

    model, processor, model_config, device, _dtype = load_model_and_processor(config)
    runtime = (model, processor, model_config, device)

    rows = []
    for sample in samples:
        if sample.target_lang is None:
            sample.target_lang = args.target_lang
        metrics = run_sample(runtime, sample, config, stream=args.stream)
        print_metric_summary(metrics)
        rows.append(metrics.to_json())
        append_jsonl(metrics_path, [metrics.to_json()])
    print(f"Wrote {len(rows)} metric rows to {metrics_path}")


if __name__ == "__main__":
    main()
