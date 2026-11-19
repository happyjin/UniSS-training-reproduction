#!/usr/bin/env python3
"""End-to-end autoregressive speech-to-speech translation inference.

The checkpoint is loaded through the same explicit ms-swift ``model_type``
registration used during training.  Inference first generates the translated
Thinker reply, then passes those reply token ids to the Talker so its semantic
layout matches S2S training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom.task_prompts import s2s_task_prompt

try:
    from .common import (
        InferenceConfig,
        InferenceSample,
        build_e2e_latency_metrics,
        decode_text,
        generate_audio,
        generate_text,
        load_model_and_processor,
        prepare_messages_inputs,
        save_audio,
        save_text,
    )
    from .online_s2s import generate_online_s2s
except ImportError:  # pragma: no cover - standalone script execution
    from common import (  # type: ignore
        InferenceConfig,
        InferenceSample,
        build_e2e_latency_metrics,
        decode_text,
        generate_audio,
        generate_text,
        load_model_and_processor,
        prepare_messages_inputs,
        save_audio,
        save_text,
    )
    from inference.online_s2s import generate_online_s2s


DEFAULT_CODEC_TOKENIZER = REPO / "models" / "Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_SPEAKER_ENCODER = REPO / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_WDS_AUDIO_ROOT = REPO.parents[1] / "data" / "speech_bidirectional" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S2S inference with explicit training-time model_type selection")
    parser.add_argument(
        "--model-type",
        required=True,
        help="ms-swift model type used in training, e.g. qwen2_5_3b_omnispeak_a",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Source speech file for a single inference sample")
    source.add_argument("--val-jsonl", help="Use one sample from a training val_dataset.jsonl")
    parser.add_argument("--index", type=int, default=0, help="Zero-based JSONL sample index")
    parser.add_argument(
        "--indices",
        help="Comma-separated val indices; loads the checkpoint once and runs all samples",
    )
    parser.add_argument("--sample-id", default="sample_0")
    parser.add_argument("--prompt", help="Override the S2S task prompt")
    parser.add_argument("--target-lang", help="Target language/code, e.g. zh or en")

    parser.add_argument(
        "--talker-layout",
        choices=("auto", "reply_add", "full_add"),
        default="auto",
        help="Talker input layout; auto infers it from the checkpoint path/config",
    )
    parser.add_argument(
        "--speaker-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the source speaker with the training-time x-vector path",
    )
    parser.add_argument(
        "--speaker-encoder",
        default=str(DEFAULT_SPEAKER_ENCODER),
        help="Local Qwen3-TTS base model used by the speaker encoder path",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", default="bf16",
        choices=("auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"),
    )
    parser.add_argument("--attn-impl", help="Optional attention backend, e.g. flash_attention_2 or sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-codec-frames", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold (used when temperature>0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalty on repeated main-codec tokens (1.0=off; Qwen3-TTS uses 1.05)")
    parser.add_argument(
        "--thinker-temperature", type=float,
        help="Thinker-only temperature; unset preserves the legacy shared value",
    )
    parser.add_argument(
        "--thinker-top-k", type=int,
        help="Thinker-only top-k; unset preserves the legacy shared value",
    )
    parser.add_argument(
        "--thinker-top-p", type=float,
        help="Thinker-only top-p; unset preserves the historical value 1.0",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Optional reproducible per-sample torch seed (seed + val index)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable windowed streaming talker decode (decode only the recent "
             "window each chunk instead of the whole sequence). Audio output is "
             "identical; this changes how PCM is emitted, not the final wav.",
    )
    parser.add_argument("--stream-emit-every", type=int, default=4, help="Emit a PCM chunk every N codec frames (streaming only)")
    parser.add_argument("--stream-decode-window", type=int, default=80, help="Decode window size in codec frames (streaming only; larger=better quality)")
    parser.add_argument("--stream-overlap-samples", type=int, default=0, help="Crossfade overlap samples between streamed chunks (streaming only)")
    parser.add_argument(
        "--kv-cache",
        action="store_true",
        help="Use the inference-only KV-cached talker session (O(1) attention "
             "per frame instead of O(T^2) full recompute). Upstream reply_add/"
             "full_add path only.",
    )
    parser.add_argument(
        "--code-predictor-kv-cache",
        action="store_true",
        help="Cache code-predictor attention keys/values across residual groups.",
    )
    parser.add_argument(
        "--compile-code-predictor",
        action="store_true",
        help="Compile the selected code-predictor inference path.",
    )
    parser.add_argument(
        "--code-predictor-compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
        help="torch.compile mode for the code predictor; default is the safe mode.",
    )
    parser.add_argument(
        "--thinker-kv-cache",
        action="store_true",
        help="Use the Thinker's native generate() KV cache for text decoding.",
    )
    parser.add_argument(
        "--gpu-codec-buffer",
        action="store_true",
        help="Keep generated codec frames in a preallocated CUDA buffer.",
    )
    parser.add_argument(
        "--compile-streaming-decoder",
        action="store_true",
        help="Compile fixed-window streaming callback decode (requires --streaming).",
    )
    parser.add_argument(
        "--online-thinker-talker", action="store_true",
        help="Interleave incremental Thinker tokens with Talker codec frames.",
    )
    parser.add_argument(
        "--online-talker-start-tokens", type=int, choices=(1, 2, 4), default=1,
        help="Committed Thinker conditions required before the Talker starts.",
    )
    parser.add_argument(
        "--compile-talker", action="store_true",
        help="Compile the DynamicCache Talker step with safe mode=default.",
    )
    parser.add_argument(
        "--predictor-frame-batch", type=int, choices=(1, 2, 4), default=1,
        help="Micro-batch independent residual predictor frames.",
    )
    parser.add_argument(
        "--latency-profiling", choices=("sync", "off"), default="sync",
        help="Phase timing policy; off avoids per-phase CUDA synchronization.",
    )
    parser.add_argument("--codec-tokenizer", default=str(DEFAULT_CODEC_TOKENIZER))
    parser.add_argument("--codec-tokenizer-device", default="cpu")
    parser.add_argument(
        "--wds-audio-root",
        default=str(DEFAULT_WDS_AUDIO_ROOT),
        help="Local shard root used to resolve wds:// validation audio",
    )
    parser.add_argument("--output-dir", default=str(REPO / "inference_outputs" / "s2s"))
    return parser.parse_args()


def _parse_audio_field(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, list):
            return str(decoded[0]) if decoded else None
        return str(decoded)
    return None


def load_val_row(path: str, index: int) -> Dict[str, Any]:
    if index < 0:
        raise ValueError("--index must be non-negative")
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"JSONL index {index} is out of range: {path}")


def _parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.replace(" ", ",").split(","):
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("--indices values must be non-negative")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("--indices must contain at least one index")
    return indices


def _default_prompt(target_lang: str) -> str:
    return s2s_task_prompt(target_lang)


def _materialize_source_audio(audio: str, args: argparse.Namespace) -> str:
    """Return a local source-audio path for legacy paths or indexed WDS URIs."""
    from custom.wds_audio import is_wds_uri, parse_wds_uri, read_wds_audio_bytes

    if not is_wds_uri(audio):
        path = Path(audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Source audio does not exist: {audio}")
        return str(path)

    reference = parse_wds_uri(audio)
    payload = read_wds_audio_bytes(audio, root=args.wds_audio_root)
    digest = hashlib.sha256(audio.encode("utf-8")).hexdigest()[:20]
    suffix = f".{reference.audio_format or 'wav'}"
    cache_dir = Path(args.output_dir).expanduser() / ".source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}{suffix}"
    if not path.is_file() or path.stat().st_size != len(payload):
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return str(path)


def _training_messages(prompt: str, audio: str) -> list[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]


def _val_messages(row: Dict[str, Any], audio: str) -> list[Dict[str, Any]]:
    """Restore the stored training conversation, dropping the GT assistant."""
    messages = []
    for raw_message in row.get("messages") or []:
        message = json.loads(raw_message) if isinstance(raw_message, str) else dict(raw_message)
        if message.get("role") == "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            blocks = []
            if "<audio>" in content:
                blocks.append({"type": "audio", "audio": audio})
                remaining = content.replace("<audio>", "").strip()
                if remaining:
                    blocks.append({"type": "text", "text": remaining})
            else:
                blocks.append({"type": "text", "text": content})
            message["content"] = blocks
        elif isinstance(content, list):
            normalized = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "audio":
                    block = {**block, "audio": block.get("audio") or audio}
                normalized.append(block)
            message["content"] = normalized
        messages.append(message)
    return messages


def _set_inference_system_prompt(
    messages: list[Dict[str, Any]], prompt: str,
) -> list[Dict[str, Any]]:
    """Set the exact training-time prompt while preserving multimodal blocks."""
    system = {"role": "system", "content": [{"type": "text", "text": prompt}]}
    result = [dict(message) for message in messages]
    for index, message in enumerate(result):
        if message.get("role") == "system":
            result[index] = system
            return result
    result.insert(0, system)
    return result


def resolve_sample(
    args: argparse.Namespace,
    *,
    val_index: int | None = None,
) -> tuple[InferenceSample, Dict[str, Any], list[Dict[str, Any]]]:
    metadata: Dict[str, Any] = {}
    if args.val_jsonl:
        selected_index = args.index if val_index is None else val_index
        row = load_val_row(args.val_jsonl, selected_index)
        audio = _parse_audio_field(row.get("audios") or row.get("audio"))
        target_lang = args.target_lang or row.get("target_lang")
        prompt = args.prompt
        sample_id = str(row.get("sample_id") or row.get("id") or f"val_{selected_index}")
        metadata = {
            "val_jsonl": str(Path(args.val_jsonl).expanduser()),
            "val_index": selected_index,
            "source_lang": row.get("source_lang"),
            "target_text": row.get("target_text"),
            "_source_text_ids": row.get("source_text_ids"),
        }
    else:
        row = None
        audio = args.audio
        target_lang = args.target_lang
        prompt = args.prompt
        sample_id = args.sample_id

    if not audio:
        raise FileNotFoundError(f"Source audio does not exist: {audio}")
    source_audio_original = str(audio)
    audio = _materialize_source_audio(source_audio_original, args)
    if not target_lang:
        raise ValueError("Target language is required; pass --target-lang or use a val row containing target_lang")
    prompt = prompt or _default_prompt(str(target_lang))
    sample = InferenceSample(
        sample_id=sample_id,
        prompt=str(prompt),
        audio=str(Path(audio).expanduser()),
        task="s2s",
        output="audio",
        target_lang=str(target_lang),
    )
    metadata["source_audio_original"] = source_audio_original
    if row is not None and args.prompt is None and row.get("messages"):
        messages = _set_inference_system_prompt(
            _val_messages(row, sample.audio), sample.prompt)
    else:
        messages = _training_messages(sample.prompt, sample.audio)
    return sample, metadata, messages


def _safe_sample_id(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text"))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )
    return ""


def _system_prompt(messages: list[Dict[str, Any]]) -> str:
    return "\n".join(
        text for message in messages
        if message.get("role") == "system"
        for text in [_message_text(message.get("content"))]
        if text
    )


def resolve_layout(requested: str, checkpoint: str, model_config: Any) -> str:
    if requested != "auto":
        return requested
    checkpoint_name = str(checkpoint).lower()
    if "full_add" in checkpoint_name:
        return "full_add"
    if "reply_add" in checkpoint_name:
        return "reply_add"
    return "full_add" if bool(getattr(model_config, "full_add_layout", False)) else "reply_add"


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    if args.indices and not args.val_jsonl:
        raise ValueError("--indices can only be used together with --val-jsonl")
    val_indices = _parse_indices(args.indices) if args.indices else [args.index]
    if args.val_jsonl:
        resolved_samples = [resolve_sample(args, val_index=index) for index in val_indices]
    else:
        resolved_samples = [resolve_sample(args)]

    first_sample = resolved_samples[0][0]
    os.environ.setdefault("QWEN3_TTS_TOKENIZER", args.codec_tokenizer)
    if args.speaker_prompt:
        os.environ.setdefault(
            "QWEN3_TTS_SPEAKER_ENCODER",
            str(Path(args.speaker_encoder).expanduser().resolve()),
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
        target_lang=first_sample.target_lang,
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
    model, processor, model_config, device, _ = load_model_and_processor(config)

    layout = resolve_layout(args.talker_layout, str(checkpoint), model_config)
    os.environ["QWEN_OMNISPEAK_FULL_ADD_LAYOUT"] = "true" if layout == "full_add" else "false"
    os.environ["QWEN_OMNISPEAK_USE_SPEAKER_PROMPT"] = "true" if args.speaker_prompt else "false"

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema_version": "s2s-eval-v3",
        "latency_standard": {
            "origin": "processor_inputs_ready",
            "first_pcm": "first_callback_pcm_or_complete_wav_for_offline",
            "complete": "final_waveform_ready_before_file_io",
            "cuda_sync": "once_at_request_start_and_once_at_request_end",
        },
        "model_type": args.model_type,
        "checkpoint": str(checkpoint),
        "talker_layout": layout,
        "speaker_prompt": args.speaker_prompt,
        "configuration": {
            "streaming": config.streaming,
            "talker_kv_cache": config.use_kv_cache,
            "code_predictor_kv_cache": config.use_code_predictor_kv_cache,
            "compile_code_predictor": config.compile_code_predictor,
            "code_predictor_compile_mode": config.code_predictor_compile_mode,
            "thinker_kv_cache": config.use_thinker_kv_cache,
            "gpu_codec_buffer": config.use_gpu_codec_buffer,
            "compile_streaming_decoder": config.compile_streaming_decoder,
            "online_thinker_talker": config.online_thinker_talker,
            "online_talker_start_tokens": config.online_talker_start_tokens,
            "compile_talker": config.compile_talker,
            "predictor_frame_batch": config.predictor_frame_batch,
            "latency_profiling": config.latency_profiling,
            "thinker_temperature": config.thinker_temperature,
            "thinker_top_k": config.thinker_top_k,
            "thinker_top_p": config.thinker_top_p,
            "talker_temperature": config.temperature,
            "talker_top_k": config.top_k,
            "talker_top_p": config.top_p,
            "talker_repetition_penalty": config.repetition_penalty,
            "seed": args.seed,
        },
        "samples": [],
    }
    summary_path = output_dir / "summary.json"

    for sample, raw_metadata, messages in resolved_samples:
        metadata = dict(raw_metadata)
        source_text_ids = metadata.pop("_source_text_ids", None)
        source_text = (
            decode_text(processor, source_text_ids).strip()
            if source_text_ids else ""
        )

        inputs = prepare_messages_inputs(
            processor, messages, device, require_audio=True)
        val_index = int(metadata.get("val_index", 0))
        if args.seed is not None:
            sample_seed = int(args.seed) + val_index
            torch.manual_seed(sample_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(sample_seed)
        stream_chunks: list = []
        first_chunk_ms: list = []

        def _on_audio(chunk, sr, elapsed_ms):
            if not first_chunk_ms:
                first_chunk_ms.append(elapsed_ms)
                print(f"[stream] {sample.sample_id} first_chunk_ms={elapsed_ms:.1f} sr={sr}")
            stream_chunks.append(chunk)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        request_started_at = time.perf_counter()
        generation_started_at = time.perf_counter()
        if config.online_thinker_talker:
            text_result, audio_result = generate_online_s2s(
                model,
                processor,
                model_config,
                inputs,
                sample,
                config,
                stream=bool(args.streaming),
                callback=_on_audio if args.streaming else None,
                use_speaker=args.speaker_prompt,
            )
        else:
            text_result = generate_text(
                model, processor, inputs, sample, config, stream=False
            )
            audio_result = generate_audio(
                model,
                processor,
                model_config,
                inputs,
                sample,
                config,
                stream=bool(args.streaming),
                realtime_schedule=False,
                callback=_on_audio if args.streaming else None,
                text_token_ids=text_result.token_ids,
                use_speaker=args.speaker_prompt,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        request_finished_at = time.perf_counter()
        origin_offset_ms = (
            generation_started_at - request_started_at
        ) * 1000.0
        observed_first_pcm_ms = (
            origin_offset_ms + first_chunk_ms[0]
            if first_chunk_ms else None
        )
        e2e_metrics = build_e2e_latency_metrics(
            complete_wav_ms=(request_finished_at - request_started_at) * 1000.0,
            first_text_ms=(
                origin_offset_ms + text_result.metrics.first_token_latency_ms
                if text_result.metrics.first_token_latency_ms is not None
                else None
            ),
            first_pcm_ms=observed_first_pcm_ms,
            compile_warmup_ms=float(
                audio_result.metrics.extra.get("compile_warmup_ms", 0.0)
            ),
        )
        if args.streaming:
            n_chunks = len(stream_chunks)
            ttfc = first_chunk_ms[0] if first_chunk_ms else None
            print(f"[stream] {sample.sample_id} chunks={n_chunks} "
                  f"first_chunk_ms={ttfc if ttfc is None else round(ttfc, 1)}")

        sample_dir = output_dir / _safe_sample_id(sample.sample_id)
        source_audio_path = sample_dir / "source.wav"
        generated_audio_path = sample_dir / "generated.wav"
        prediction_path = sample_dir / "prediction.txt"
        sample_json_path = sample_dir / "sample.json"
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.audio, source_audio_path)
        save_audio(generated_audio_path, audio_result.audio, audio_result.sample_rate)
        save_text(prediction_path, text_result.text)

        sample_summary = {
            "sample_id": sample.sample_id,
            **metadata,
            "source_lang": metadata.get("source_lang"),
            "target_lang": sample.target_lang,
            "source_text": source_text,
            "target_text": metadata.get("target_text"),
            "predicted_text": text_result.text,
            "task_prompt": sample.prompt,
            "effective_system_prompt": _system_prompt(messages),
            "prompt_overridden": bool(args.prompt),
            "source_audio_original": metadata.get("source_audio_original", sample.audio),
            "source_audio": str(source_audio_path.resolve()),
            "generated_audio": str(generated_audio_path.resolve()),
            "prediction_text": str(prediction_path.resolve()),
            "pred_text_tokens": len(text_result.token_ids),
            "codec_frames": int(audio_result.metrics.generated_codec_frames),
            "audio_ms": audio_result.metrics.output_audio_duration_ms,
            "rtf": audio_result.metrics.rtf,
            "e2e_metrics": e2e_metrics,
            "text_metrics": text_result.metrics.to_json(),
            "audio_metrics": audio_result.metrics.to_json(),
        }
        sample_json_path.write_text(
            json.dumps(sample_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["samples"].append(sample_summary)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(sample_summary, ensure_ascii=False, indent=2))

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
