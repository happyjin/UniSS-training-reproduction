#!/usr/bin/env python3
"""Realtime qwen_omnispeak inference tester.

`--source file` simulates streaming file input and measures first non-silent
output latency. `--source mic` records utterances from microphone, streams model
speech output, optionally plays it, and saves each utterance result.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from .common import (
        InferenceConfig,
        InferenceSample,
        append_jsonl,
        generate_audio,
        input_audio_duration_ms,
        iter_audio_chunks,
        load_model_and_processor,
        make_output_paths,
        prepare_inputs,
        print_metric_summary,
        read_jsonl_samples,
        require_sounddevice,
        save_audio,
        single_sample,
        write_temp_wav,
    )
except ImportError:  # pragma: no cover
    from common import (  # type: ignore
        InferenceConfig,
        InferenceSample,
        append_jsonl,
        generate_audio,
        input_audio_duration_ms,
        iter_audio_chunks,
        load_model_and_processor,
        make_output_paths,
        prepare_inputs,
        print_metric_summary,
        read_jsonl_samples,
        require_sounddevice,
        save_audio,
        single_sample,
        write_temp_wav,
    )


class PlaybackSink:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._sd = None
        self._stream = None
        self._sr = None

    def write(self, chunk: np.ndarray, sample_rate: int) -> None:
        if not self.enabled or chunk.size == 0:
            return
        if self._sd is None:
            self._sd = require_sounddevice()
        if self._stream is None or self._sr != sample_rate:
            self.close()
            self._sr = sample_rate
            self._stream = self._sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._stream.start()
        self._stream.write(chunk.astype(np.float32).reshape(-1, 1))

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime qwen_omnispeak inference tester")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", choices=("file", "mic"), required=True)
    parser.add_argument("--prompt", required=True, help="Prompt used for all samples/utterances")
    parser.add_argument("--audio", help="Input audio file for --source file")
    parser.add_argument("--samples", help="Optional JSONL file for --source file")
    parser.add_argument("--sample-id", default="sample_0")
    parser.add_argument("--task", default="s2s")
    parser.add_argument("--target-lang")
    parser.add_argument("--output-dir", default="inference_outputs/realtime")
    parser.add_argument("--metrics-file", default="metrics.jsonl")
    parser.add_argument("--play", action="store_true", help="Play generated audio chunks in realtime")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"))
    parser.add_argument("--max-codec-frames", type=int, default=256)
    parser.add_argument("--codec-chunk-frames", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--silence-threshold", type=float, default=1e-4)
    parser.add_argument("--streaming-extra-lag-ms", type=float, default=0.0)
    parser.add_argument("--codec-tokenizer")
    parser.add_argument("--codec-tokenizer-device")

    parser.add_argument("--chunk-ms", type=int, default=320, help="File/mic input chunk size in milliseconds")
    parser.add_argument("--infer-every-chunks", type=int, default=4, help="For file mode, try prefix inference every N chunks until first audio")
    parser.add_argument("--min-prefix-ms", type=int, default=800, help="For file mode, do not infer before this much source audio has arrived")
    parser.add_argument("--no-realtime-sleep", action="store_true", help="For file mode, do not sleep between chunks")

    parser.add_argument("--mic-sample-rate", type=int, default=16000)
    parser.add_argument("--mic-device", help="sounddevice input device name/id")
    parser.add_argument("--vad-threshold", type=float, default=1e-3)
    parser.add_argument("--utterance-silence-ms", type=int, default=800)
    parser.add_argument("--max-utterances", type=int, default=0, help="0 means run until Ctrl-C")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        max_codec_frames=args.max_codec_frames,
        codec_chunk_frames=args.codec_chunk_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        target_lang=args.target_lang,
        silence_threshold=args.silence_threshold,
        streaming_extra_lag_ms=args.streaming_extra_lag_ms,
        use_streaming_schedule=args.use_streaming_mask,
        codec_tokenizer=args.codec_tokenizer,
        codec_tokenizer_device=args.codec_tokenizer_device,
    )


def load_file_samples(args: argparse.Namespace) -> List[InferenceSample]:
    if args.samples:
        samples = list(read_jsonl_samples(args.samples, default_task=args.task, default_output="audio"))
        for sample in samples:
            if not sample.prompt:
                sample.prompt = args.prompt
            sample.output = "audio"
        return samples
    if not args.audio:
        raise SystemExit("--audio is required when --source file and --samples is not provided")
    return [single_sample(args.sample_id, args.prompt, args.audio, args.task, "audio", args.target_lang)]


def run_full_audio_generation(
    runtime,
    sample: InferenceSample,
    config: InferenceConfig,
    sink: PlaybackSink,
    *,
    save_result: bool = True,
):
    model, processor, model_config, device = runtime
    inputs = prepare_inputs(processor, sample.prompt, sample.audio, device)

    def on_audio(chunk, sr, elapsed_ms):
        print(f"[{sample.sample_id}] audio_chunk samples={len(chunk)} sr={sr} elapsed_ms={elapsed_ms:.2f}")
        sink.write(chunk, sr)

    result = generate_audio(
        model,
        processor,
        model_config,
        inputs,
        sample,
        config,
        stream=True,
        realtime_schedule=True,
        callback=on_audio,
    )
    if save_result:
        _, audio_path = make_output_paths(config.output_dir, sample.sample_id)
        result.metrics.output_audio_path = save_audio(audio_path, result.audio, result.sample_rate)
    return result.metrics


def run_file_sample(runtime, sample: InferenceSample, config: InferenceConfig, args: argparse.Namespace, sink: PlaybackSink):
    assert sample.audio is not None
    first_audio_latency_ms: Optional[float] = None
    stream_start = time.perf_counter()
    prefix_chunks: List[np.ndarray] = []

    with tempfile.TemporaryDirectory(prefix="qwen_omnispeak_prefix_") as tmpdir:
        for idx, (chunk, sr, arrived_ms) in enumerate(iter_audio_chunks(sample.audio, args.chunk_ms), start=1):
            prefix_chunks.append(chunk)
            if not args.no_realtime_sleep:
                time.sleep(len(chunk) / sr)
            should_probe = (
                arrived_ms >= args.min_prefix_ms
                and first_audio_latency_ms is None
                and idx % max(1, args.infer_every_chunks) == 0
            )
            if not should_probe:
                continue
            prefix_audio = np.concatenate(prefix_chunks, axis=0)
            prefix_path = write_temp_wav(prefix_audio, sr, tmpdir, name=f"prefix_{idx:04d}.wav")
            prefix_sample = InferenceSample(
                sample_id=f"{sample.sample_id}_prefix_{idx:04d}",
                prompt=sample.prompt,
                audio=prefix_path,
                task=sample.task,
                output="audio",
                target_lang=sample.target_lang,
            )
            metrics = run_full_audio_generation(runtime, prefix_sample, config, PlaybackSink(False), save_result=False)
            if metrics.first_audio_latency_ms is not None:
                first_audio_latency_ms = (time.perf_counter() - stream_start) * 1000.0
                print(f"[{sample.sample_id}] first_non_silent_audio_latency_ms={first_audio_latency_ms:.2f}")
                break
        if first_audio_latency_ms is None and prefix_chunks:
            prefix_audio = np.concatenate(prefix_chunks, axis=0)
            prefix_path = write_temp_wav(prefix_audio, sr, tmpdir, name="prefix_final.wav")
            prefix_sample = InferenceSample(
                sample_id=f"{sample.sample_id}_prefix_final",
                prompt=sample.prompt,
                audio=prefix_path,
                task=sample.task,
                output="audio",
                target_lang=sample.target_lang,
            )
            metrics = run_full_audio_generation(runtime, prefix_sample, config, PlaybackSink(False), save_result=False)
            if metrics.first_audio_latency_ms is not None:
                first_audio_latency_ms = (time.perf_counter() - stream_start) * 1000.0

    final_metrics = run_full_audio_generation(runtime, sample, config, sink)
    if first_audio_latency_ms is not None:
        final_metrics.first_audio_latency_ms = first_audio_latency_ms
    print_metric_summary(final_metrics)
    return final_metrics


def run_file_mode(runtime, samples: List[InferenceSample], config: InferenceConfig, args: argparse.Namespace) -> None:
    sink = PlaybackSink(args.play)
    metrics_path = Path(config.output_dir) / args.metrics_file
    if metrics_path.exists():
        metrics_path.unlink()
    try:
        for sample in samples:
            if sample.target_lang is None:
                sample.target_lang = args.target_lang
            metrics = run_file_sample(runtime, sample, config, args, sink)
            append_jsonl(metrics_path, [metrics.to_json()])
    finally:
        sink.close()
    print(f"Wrote realtime file metrics to {metrics_path}")


def record_mic_utterances(args: argparse.Namespace):
    sd = require_sounddevice()
    block = max(1, int(round(args.mic_sample_rate * args.chunk_ms / 1000.0)))
    silence_blocks_needed = max(1, int(round(args.utterance_silence_ms / args.chunk_ms)))
    utterance: List[np.ndarray] = []
    speech_started = False
    silence_blocks = 0
    count = 0

    with sd.InputStream(
        samplerate=args.mic_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block,
        device=args.mic_device,
    ) as stream:
        print("Listening from microphone. Press Ctrl-C to stop.")
        while args.max_utterances <= 0 or count < args.max_utterances:
            data, overflowed = stream.read(block)
            if overflowed:
                print("[warn] microphone input overflowed")
            chunk = data.reshape(-1).copy()
            level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if level >= args.vad_threshold:
                speech_started = True
                silence_blocks = 0
            elif speech_started:
                silence_blocks += 1
            if speech_started:
                utterance.append(chunk)
            if speech_started and silence_blocks >= silence_blocks_needed and utterance:
                count += 1
                wav = np.concatenate(utterance, axis=0)
                yield count, wav, args.mic_sample_rate
                utterance = []
                speech_started = False
                silence_blocks = 0


def run_mic_mode(runtime, config: InferenceConfig, args: argparse.Namespace) -> None:
    sink = PlaybackSink(args.play)
    metrics_path = Path(config.output_dir) / args.metrics_file
    if metrics_path.exists():
        metrics_path.unlink()
    try:
        with tempfile.TemporaryDirectory(prefix="qwen_omnispeak_mic_") as tmpdir:
            for idx, wav, sr in record_mic_utterances(args):
                audio_path = write_temp_wav(wav, sr, tmpdir, name=f"utterance_{idx:04d}.wav")
                sample = InferenceSample(
                    sample_id=f"mic_{idx:04d}",
                    prompt=args.prompt,
                    audio=audio_path,
                    task=args.task,
                    output="audio",
                    target_lang=args.target_lang,
                )
                started = time.perf_counter()
                metrics = run_full_audio_generation(runtime, sample, config, sink)
                # Preserve model's first output latency and also expose end-to-end
                # utterance turnaround from endpoint detection.
                metrics.extra["utterance_audio_duration_ms"] = input_audio_duration_ms(audio_path)
                metrics.extra["endpoint_to_complete_latency_ms"] = (time.perf_counter() - started) * 1000.0
                print_metric_summary(metrics)
                append_jsonl(metrics_path, [metrics.to_json()])
    except KeyboardInterrupt:
        print("Stopped microphone inference.")
    finally:
        sink.close()
    print(f"Wrote realtime microphone metrics to {metrics_path}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model, processor, model_config, device, _dtype = load_model_and_processor(config)
    runtime = (model, processor, model_config, device)

    if args.source == "file":
        run_file_mode(runtime, load_file_samples(args), config, args)
    else:
        run_mic_mode(runtime, config, args)


if __name__ == "__main__":
    main()
