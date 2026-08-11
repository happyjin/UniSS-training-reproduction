#!/usr/bin/env python3
"""Export a Megatron checkpoint and run strict natural-WRITE PCM evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.audio_cached_frontend import (
    SAMPLE_RATE,
    StreamingCachedWhisperVQFrontend,
)
from training.simul_uniss.jsonl_index import load_index
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)
from web_demo.runtime_parity_streaming_v2.inference import (
    NaturalRuntimeParityGenerator,
    evaluate_waveform,
)
from web_demo.true_subsecond_pilot15_streaming_v1.checkpoint_export import (
    export_runtime,
)
from web_demo.true_subsecond_pilot15_streaming_v1.model_loader import (
    load_runtime_models,
)


def _row(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def _waveform(path: Path) -> np.ndarray:
    raw, rate = sf.read(path, dtype="float32", always_2d=False)
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"runtime-exact evaluator requires 16kHz PCM, found {rate}")
    return value.reshape(-1)


def _stereo(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    length = max(len(source), len(target))
    value = np.zeros((length, 2), dtype=np.float32)
    value[: len(source), 0] = source
    value[: len(target), 1] = target
    return value


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint).resolve()
    formal_path = Path(args.formal_manifest).resolve()
    speaker_path = Path(args.speaker_formal_manifest).resolve()
    output = Path(args.output).resolve()
    export = Path(args.export).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True)

    formal_offsets = load_index(formal_path)
    speaker_offsets = load_index(speaker_path)
    if formal_offsets is None or speaker_offsets is None:
        raise ValueError("formal or fixed-speaker manifest is missing its index")
    with speaker_path.open("rb") as handle:
        speaker = _row(handle, speaker_offsets[args.speaker_source_index])
    speaker_global = [int(value) for value in speaker["bicodec_global"]]
    if len(speaker_global) != 32:
        raise ValueError("fixed runtime speaker has invalid token geometry")

    export_manifest = export_runtime(
        checkpoint, Path(args.base_model).resolve(), export
    )
    device = torch.device(args.device)
    whisper_encoder = load_quantize_encoder(args.whispervq_model).to(device).eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.whispervq_model
    )
    frontend = StreamingCachedWhisperVQFrontend(
        whisper_encoder, feature_extractor.mel_filters, device=device
    )
    model, tokenizer, objective, _, _ = load_runtime_models(
        export,
        codebook_weight=whisper_encoder.codebook.weight,
        device=device,
    )
    bicodec = BiCodecTokenizer(
        model_dir=Path(args.speech_tokenizer).resolve() / "bicodec",
        device=device,
    )

    results: list[dict[str, object]] = []
    with formal_path.open("rb") as handle:
        for row_index in range(min(args.samples, len(formal_offsets))):
            record = _row(handle, formal_offsets[row_index])
            sample_id = str(record["id"])
            waveform = _waveform(Path(str(record["source_audio"])).resolve())
            generator = NaturalRuntimeParityGenerator(
                model,
                tokenizer,
                objective,
                target_lang=str(record["tgt_lang"]),
                speaker_global=speaker_global,
                device=device,
                maximum_text_tokens=args.maximum_text_tokens,
                maximum_semantic_tokens=args.maximum_semantic_tokens,
                fuse_ticks=args.fuse_ticks,
                use_static_cache=args.static_cache,
                maximum_cache_tokens=args.maximum_cache_tokens,
            )
            codec = StreamingBiCodecDecoder(
                bicodec_decode_function(bicodec),
                sample_rate=SAMPLE_RATE,
                semantic_rate=50.0,
                left_context_tokens=50,
                holdback_tokens=5,
                overlap_ms=80,
            )
            codec.set_speaker_tokens(speaker_global)
            result = evaluate_waveform(
                sample_id=sample_id,
                waveform=waveform,
                target_text=str(record["translation"]),
                target_lang=str(record["tgt_lang"]),
                speaker_global=speaker_global,
                frontend=frontend,
                generator=generator,
                codec=codec,
                maximum_drain_ticks=args.maximum_drain_ticks,
                minimum_text_similarity=args.minimum_text_similarity,
                maximum_rtf=args.maximum_rtf,
                maximum_first_audio_wall_ms=args.maximum_first_audio_wall_ms,
            )
            sample_root = output / f"{row_index:04d}_{sample_id}"
            sample_root.mkdir()
            sf.write(sample_root / "source.wav", waveform, SAMPLE_RATE, subtype="PCM_16")
            sf.write(
                sample_root / "translation.wav",
                result.translation_audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            sf.write(
                sample_root / "translation_timeline.wav",
                result.timeline_audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            sf.write(
                sample_root / "stereo_left_source_right_translation.wav",
                _stereo(waveform, result.timeline_audio),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            metadata = result.metadata()
            (sample_root / "result.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(metadata)
            print(json.dumps(metadata, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": "uniss_runtime_parity_pcm_evaluation_v1",
        "checkpoint": str(checkpoint),
        "runtime_export": export_manifest,
        "formal_manifest": str(formal_path),
        "fixed_speaker_manifest": str(speaker_path),
        "fixed_speaker_source_index": args.speaker_source_index,
        "fixed_speaker_sample_id": str(speaker["id"]),
        "frontend": {
            "schema": "uniss_cached_block_causal_whispervq_v2",
            "pcm_tick_ms": 160,
            "token_hop_ms": 80,
            "right_context_ms": 0,
            "stateful_encoder_kv": True,
            "committed_revision_violations": 0,
        },
        "samples": results,
        "quality_passed": bool(results) and all(
            bool(value["quality_passed"]) for value in results
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--speaker-formal-manifest", required=True)
    parser.add_argument("--speaker-source-index", type=int, default=0)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--speech-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--maximum-text-tokens", type=int, default=16)
    parser.add_argument("--maximum-semantic-tokens", type=int, default=80)
    parser.add_argument("--fuse-ticks", action="store_true")
    parser.add_argument("--static-cache", action="store_true")
    parser.add_argument("--maximum-cache-tokens", type=int, default=32_768)
    parser.add_argument("--maximum-drain-ticks", type=int, default=32)
    parser.add_argument("--minimum-text-similarity", type=float, default=0.50)
    parser.add_argument("--maximum-rtf", type=float, default=1.0)
    parser.add_argument("--maximum-first-audio-wall-ms", type=float, default=1000.0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
