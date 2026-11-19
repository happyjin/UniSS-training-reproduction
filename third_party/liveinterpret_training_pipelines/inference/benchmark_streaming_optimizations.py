#!/usr/bin/env python3
"""Reproducible single-configuration streaming inference benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.common import (
    InferenceConfig,
    generate_audio,
    generate_text,
    load_model_and_processor,
    prepare_messages_inputs,
)
from inference.online_s2s import generate_online_s2s
from inference.s2s_infer import (
    DEFAULT_CODEC_TOKENIZER,
    DEFAULT_SPEAKER_ENCODER,
    resolve_layout,
    resolve_sample,
)


DEFAULT_WDS_ROOT = "/fsx/zeyangsong/unist/converted/unist_qwen3_s2s_v2/speech"


def _parse_indices(value: str) -> list[int]:
    indices = []
    for part in value.replace(" ", ",").split(","):
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("indices must be non-negative")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("indices must contain at least one value")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one streaming optimization configuration"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--wds-audio-root", default=DEFAULT_WDS_ROOT)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--warmup-index",
        type=int,
        help="Validation index used for one excluded compile/runtime warmup",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("fp32", "float32", "bf16", "bfloat16", "fp16", "float16"),
    )
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-codec-frames", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--thinker-temperature", type=float)
    parser.add_argument("--thinker-top-k", type=int)
    parser.add_argument("--thinker-top-p", type=float)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--stream-emit-every", type=int, default=4)
    parser.add_argument("--stream-decode-window", type=int, default=80)
    parser.add_argument("--stream-overlap-samples", type=int, default=0)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--code-predictor-kv-cache", action="store_true")
    parser.add_argument("--compile-code-predictor", action="store_true")
    parser.add_argument(
        "--code-predictor-compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--thinker-kv-cache", action="store_true")
    parser.add_argument("--gpu-codec-buffer", action="store_true")
    parser.add_argument("--compile-streaming-decoder", action="store_true")
    parser.add_argument("--online-thinker-talker", action="store_true")
    parser.add_argument(
        "--online-talker-start-tokens", type=int, choices=(1, 2, 4), default=1
    )
    parser.add_argument("--compile-talker", action="store_true")
    parser.add_argument(
        "--predictor-frame-batch", type=int, choices=(1, 2, 4), default=1
    )
    parser.add_argument(
        "--latency-profiling", choices=("sync", "off"), default="sync"
    )
    parser.add_argument(
        "--speaker-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--speaker-encoder", default=str(DEFAULT_SPEAKER_ENCODER))
    parser.add_argument("--codec-tokenizer", default=str(DEFAULT_CODEC_TOKENIZER))
    parser.add_argument("--codec-tokenizer-device", default="cuda:0")
    parser.add_argument(
        "--talker-layout",
        choices=("auto", "reply_add", "full_add"),
        default="auto",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--target-lang")
    parser.add_argument("--audio")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-id", default="benchmark")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _run_one(
    *,
    args: argparse.Namespace,
    val_index: int,
    config: InferenceConfig,
    model: Any,
    processor: Any,
    model_config: Any,
    device: torch.device,
) -> Dict[str, Any]:
    _seed_everything(args.seed + val_index)
    sample, metadata, messages = resolve_sample(args, val_index=val_index)
    inputs = prepare_messages_inputs(
        processor, messages, device, require_audio=True
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    chunk_count = 0

    def _on_audio(chunk: np.ndarray, sample_rate: int, elapsed_ms: float) -> None:
        nonlocal chunk_count
        del chunk, sample_rate, elapsed_ms
        chunk_count += 1

    if config.online_thinker_talker:
        text_result, audio_result = generate_online_s2s(
            model,
            processor,
            model_config,
            inputs,
            sample,
            config,
            stream=config.streaming,
            callback=_on_audio if config.streaming else None,
            use_speaker=args.speaker_prompt,
        )
    else:
        text_result = generate_text(model, processor, inputs, sample, config)
        audio_result = generate_audio(
            model,
            processor,
            model_config,
            inputs,
            sample,
            config,
            stream=config.streaming,
            callback=_on_audio if config.streaming else None,
            text_token_ids=text_result.token_ids,
            use_speaker=args.speaker_prompt,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    codec_codes = np.asarray(audio_result.codec_codes, dtype=np.int64)
    return {
        "val_index": val_index,
        "sample_id": sample.sample_id,
        "source_lang": metadata.get("source_lang"),
        "target_lang": sample.target_lang,
        "wall_ms": (time.perf_counter() - started_at) * 1000.0,
        "text": text_result.text,
        "text_token_ids": text_result.token_ids,
        "text_token_hash": _array_hash(
            np.asarray(text_result.token_ids, dtype=np.int64)
        ),
        "codec_codes": codec_codes.tolist(),
        "codec_hash": _array_hash(codec_codes),
        "canonical_wav_hash": _array_hash(
            np.asarray(audio_result.audio, dtype=np.float32)
        ),
        "eos_triggered": (
            audio_result.metrics.generated_codec_frames
            < config.max_codec_frames
        ),
        "stream_chunks": chunk_count,
        "peak_cuda_memory_bytes": peak_memory,
        "text_metrics": text_result.metrics.to_json(),
        "audio_metrics": audio_result.metrics.to_json(),
    }


def _metric_values(samples: Iterable[Dict[str, Any]], *path: str) -> list[float]:
    values = []
    for sample in samples:
        value: Any = sample
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _aggregate(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    paths = {
        "wall_ms": ("wall_ms",),
        "text_total_ms": ("text_metrics", "total_latency_ms"),
        "first_token_ms": ("text_metrics", "first_token_latency_ms"),
        "audio_total_ms": ("audio_metrics", "total_latency_ms"),
        "first_audio_ms": ("audio_metrics", "first_audio_latency_ms"),
        "rtf": ("audio_metrics", "rtf"),
        "codec_frames": ("audio_metrics", "generated_codec_frames"),
    }
    for phase in (
        "thinker_generate_ms",
        "thinker_finalize_ms",
        "talker_prefill_ms",
        "talker_step_ms",
        "code_predictor_ms",
        "codec_transfer_ms",
        "stream_decoder_ms",
        "compile_warmup_ms",
    ):
        paths[phase] = ("audio_metrics", "extra", phase)
        if phase == "thinker_generate_ms":
            paths[phase] = ("text_metrics", "extra", phase)
    result: Dict[str, Any] = {"sample_count": len(samples)}
    for name, path in paths.items():
        values = _metric_values(samples, *path)
        if values:
            result[name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
            }
    result["eos_rate"] = (
        sum(bool(sample["eos_triggered"]) for sample in samples)
        / max(1, len(samples))
    )
    result["peak_cuda_memory_bytes"] = max(
        (int(sample["peak_cuda_memory_bytes"]) for sample in samples),
        default=0,
    )
    return result


def _correctness(
    samples: list[Dict[str, Any]], baseline_path: Optional[str]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "baseline_results": baseline_path,
        "eos_rate": (
            sum(bool(sample["eos_triggered"]) for sample in samples)
            / max(1, len(samples))
        ),
    }
    if not baseline_path:
        return result
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    by_index = {
        int(sample["val_index"]): sample for sample in baseline["samples"]
    }
    comparisons = []
    for sample in samples:
        reference = by_index.get(int(sample["val_index"]))
        if reference is None:
            continue
        codes = np.asarray(sample["codec_codes"], dtype=np.int64)
        reference_codes = np.asarray(reference["codec_codes"], dtype=np.int64)
        common_frames = min(len(codes), len(reference_codes))
        main_divergence = int(
            np.count_nonzero(
                codes[:common_frames, 0] != reference_codes[:common_frames, 0]
            )
        ) if common_frames else 0
        residual_divergence = int(
            np.count_nonzero(
                np.any(
                    codes[:common_frames, 1:]
                    != reference_codes[:common_frames, 1:],
                    axis=1,
                )
            )
        ) if common_frames and codes.shape[1] > 1 else 0
        comparisons.append(
            {
                "val_index": sample["val_index"],
                "text_tokens_equal": (
                    sample["text_token_ids"] == reference["text_token_ids"]
                ),
                "frame_count_equal": len(codes) == len(reference_codes),
                "main_divergent_frames": main_divergence,
                "residual_divergent_frames": residual_divergence,
                "canonical_wav_hash_equal": (
                    sample["canonical_wav_hash"]
                    == reference["canonical_wav_hash"]
                ),
            }
        )
    result["comparisons"] = comparisons
    result["all_exact"] = bool(comparisons) and all(
        comparison["text_tokens_equal"]
        and comparison["frame_count_equal"]
        and comparison["main_divergent_frames"] == 0
        and comparison["residual_divergent_frames"] == 0
        and comparison["canonical_wav_hash_equal"]
        for comparison in comparisons
    )
    return result


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    indices = _parse_indices(args.indices)
    warmup_index = args.warmup_index
    if warmup_index is None:
        warmup_index = indices[0]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    os.environ["QWEN3_TTS_TOKENIZER"] = str(
        Path(args.codec_tokenizer).expanduser().resolve()
    )
    if args.speaker_prompt:
        os.environ["QWEN3_TTS_SPEAKER_ENCODER"] = str(
            Path(args.speaker_encoder).expanduser().resolve()
        )
    config = InferenceConfig(
        model=str(checkpoint),
        model_type=args.model_type,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_codec_frames=args.max_codec_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        thinker_temperature=args.thinker_temperature,
        thinker_top_k=args.thinker_top_k,
        thinker_top_p=args.thinker_top_p,
        codec_tokenizer=args.codec_tokenizer,
        codec_tokenizer_device=args.codec_tokenizer_device,
        streaming=args.streaming,
        stream_emit_every=args.stream_emit_every,
        stream_decode_window=args.stream_decode_window,
        stream_overlap_samples=args.stream_overlap_samples,
        use_kv_cache=args.kv_cache,
        use_code_predictor_kv_cache=args.code_predictor_kv_cache,
        compile_code_predictor=args.compile_code_predictor,
        code_predictor_compile_mode=args.code_predictor_compile_mode,
        use_thinker_kv_cache=args.thinker_kv_cache,
        use_gpu_codec_buffer=args.gpu_codec_buffer,
        compile_streaming_decoder=args.compile_streaming_decoder,
        online_thinker_talker=args.online_thinker_talker,
        online_talker_start_tokens=args.online_talker_start_tokens,
        compile_talker=args.compile_talker,
        predictor_frame_batch=args.predictor_frame_batch,
        latency_profiling=args.latency_profiling,
    )
    config.validate_static_optimizations()
    model, processor, model_config, device, resolved_dtype = (
        load_model_and_processor(config)
    )
    layout = resolve_layout(args.talker_layout, str(checkpoint), model_config)
    os.environ["QWEN_OMNISPEAK_FULL_ADD_LAYOUT"] = (
        "true" if layout == "full_add" else "false"
    )
    os.environ["QWEN_OMNISPEAK_USE_SPEAKER_PROMPT"] = (
        "true" if args.speaker_prompt else "false"
    )

    warmup_sample = _run_one(
        args=args,
        val_index=warmup_index,
        config=config,
        model=model,
        processor=processor,
        model_config=model_config,
        device=device,
    )
    samples = [
        _run_one(
            args=args,
            val_index=index,
            config=config,
            model=model,
            processor=processor,
            model_config=model_config,
            device=device,
        )
        for index in indices
    ]
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype": str(resolved_dtype),
        "attention_implementation": args.attn_impl,
        "seed": args.seed,
    }
    payload = {
        "config": {
            **vars(args),
            "checkpoint": str(checkpoint),
            "indices": indices,
            "warmup_index": warmup_index,
            "talker_layout": layout,
        },
        "environment": environment,
        "warmup_sample": warmup_sample,
        "samples": samples,
        "aggregate": _aggregate(samples),
        "correctness": _correctness(samples, args.baseline_results),
    }
    output_path = Path(args.output_dir).expanduser() / "results.json"
    _write_json_atomic(output_path, payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
