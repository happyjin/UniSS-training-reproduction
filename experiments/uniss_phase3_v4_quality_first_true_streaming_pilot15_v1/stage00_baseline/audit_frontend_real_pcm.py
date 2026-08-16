#!/usr/bin/env python3
"""Audit full-mask versus cached WhisperVQ on a real pilot15 PCM sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_MS,
    BLOCK_SAMPLES,
    FRONTEND_SCHEMA,
    SAMPLE_RATE,
    SharedCausalWhisperVQFrontend,
)
from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_row(path: Path, row_index: int) -> dict[str, Any]:
    if row_index < 0:
        raise ValueError("manifest row index must be non-negative")
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"manifest has no row {row_index}: {path}")


def _waveform(path: Path) -> tuple[np.ndarray, str]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(values, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.reshape(-1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz PCM, found {sample_rate}: {path}")
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"empty or non-finite PCM: {path}")
    digest = hashlib.sha256(waveform.astype("<f4", copy=False).tobytes()).hexdigest()
    return waveform, digest


def _stream(
    frontend: SharedCausalWhisperVQFrontend, waveform: np.ndarray
) -> dict[str, Any]:
    state = None
    hidden: list[torch.Tensor] = []
    quantized: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    reset_blocks: list[int] = []
    step_ms: list[float] = []
    for block_index, start in enumerate(range(0, len(waveform), BLOCK_SAMPLES)):
        end = min(len(waveform), start + BLOCK_SAMPLES)
        began = time.perf_counter()
        step = frontend.push(
            waveform[start:end], state, is_final=end == len(waveform)
        )
        torch.cuda.synchronize(frontend.device) if frontend.device.type == "cuda" else None
        step_ms.append((time.perf_counter() - began) * 1000.0)
        state = step.state
        hidden.append(step.pre_vq_hidden.float().cpu())
        quantized.append(step.quantized_hidden.float().cpu())
        tokens.append(step.token_ids.cpu())
        if step.encoder_reset_before_block:
            reset_blocks.append(block_index)
    return {
        "hidden": torch.cat(hidden, dim=1),
        "quantized": torch.cat(quantized, dim=1),
        "tokens": torch.cat(tokens, dim=1),
        "state": state,
        "reset_blocks": reset_blocks,
        "step_ms": step_ms,
    }


def _hidden_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "allclose": False,
        }
    reference = reference.float()
    actual = actual.float()
    absolute = (reference - actual).abs()
    relative = absolute / reference.abs().clamp_min(1e-12)
    return {
        "shape_equal": True,
        "shape": list(reference.shape),
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "maximum_relative_error": float(relative.max().item()),
        "allclose": bool(torch.allclose(reference, actual, rtol=2e-5, atol=2e-6)),
        "rtol": 2e-5,
        "atol": 2e-6,
    }


def _token_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    shape_equal = reference.shape == actual.shape
    matches = int((reference == actual).sum().item()) if shape_equal else 0
    total = int(reference.numel()) if shape_equal else max(reference.numel(), actual.numel())
    return {
        "shape_equal": shape_equal,
        "reference_tokens": int(reference.numel()),
        "actual_tokens": int(actual.numel()),
        "matching_tokens": matches,
        "match_ratio": float(matches / total) if total else 1.0,
        "exact": bool(shape_equal and matches == total),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    manifest = Path(args.manifest).resolve()
    model_path = Path(args.whispervq_model).resolve()
    record = _manifest_row(manifest, args.row_index)
    audio_path = Path(str(record["source_audio"])).resolve()
    waveform, pcm_sha256 = _waveform(audio_path)
    duration_ms = int(round(len(waveform) * 1000 / SAMPLE_RATE))
    if abs(duration_ms - int(record["source_duration_ms"])) > args.duration_tolerance_ms:
        raise ValueError("manifest and decoded PCM durations differ")

    encoder = load_quantize_encoder(str(model_path)).to(args.device).float().eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
    frontend = SharedCausalWhisperVQFrontend(
        encoder, feature_extractor.mel_filters, device=args.device
    ).float().eval()

    full_started = time.perf_counter()
    full = frontend.forward_recomputed_reference(waveform)
    torch.cuda.synchronize(frontend.device) if frontend.device.type == "cuda" else None
    full_seconds = time.perf_counter() - full_started
    single_mask = frontend.forward_full_reference(waveform)
    streamed = _stream(frontend, waveform)
    full_hidden = full.pre_vq_hidden.float().cpu()
    full_quantized = full.quantized_hidden.float().cpu()
    full_tokens = full.token_ids.cpu()

    hidden_parity = _hidden_metrics(full_hidden, streamed["hidden"])
    quantized_parity = _hidden_metrics(full_quantized, streamed["quantized"])
    token_parity = _token_metrics(full_tokens, streamed["tokens"])
    single_mask_hidden = _hidden_metrics(
        single_mask.pre_vq_hidden.float().cpu(), streamed["hidden"]
    )
    single_mask_tokens = _token_metrics(
        single_mask.token_ids.cpu(), streamed["tokens"]
    )

    future_block = min(max(1, args.future_block), max(1, math.ceil(len(waveform) / BLOCK_SAMPLES) - 1))
    changed = waveform.copy()
    mutation_start = future_block * BLOCK_SAMPLES
    generator = np.random.default_rng(args.seed)
    changed[mutation_start:] = generator.normal(
        0.0, max(float(np.std(waveform)), 0.01), len(changed) - mutation_start
    ).astype(np.float32)
    changed_full = frontend.forward_recomputed_reference(changed)
    committed_tokens = future_block * 2
    future_hidden = _hidden_metrics(
        full_hidden[:, :committed_tokens],
        changed_full.pre_vq_hidden.float().cpu()[:, :committed_tokens],
    )
    future_tokens = _token_metrics(
        full_tokens[:, :committed_tokens],
        changed_full.token_ids.cpu()[:, :committed_tokens],
    )

    reset_samples = int(round(args.reset_audit_seconds * SAMPLE_RATE))
    repeats = math.ceil(reset_samples / len(waveform))
    reset_waveform = np.tile(waveform, repeats)[:reset_samples].copy()
    reset_full = frontend.forward_recomputed_reference(reset_waveform)
    reset_stream = _stream(frontend, reset_waveform)
    reset_hidden = _hidden_metrics(
        reset_full.pre_vq_hidden.float().cpu(), reset_stream["hidden"]
    )
    reset_tokens = _token_metrics(reset_full.token_ids.cpu(), reset_stream["tokens"])
    expected_tokens = math.ceil(len(waveform) / (SAMPLE_RATE * 0.08))
    expected_reset_tokens = math.ceil(len(reset_waveform) / (SAMPLE_RATE * 0.08))

    checks = {
        "real_pcm_hidden_recomputed_cached": bool(hidden_parity["allclose"]),
        "real_pcm_quantized_recomputed_cached": bool(quantized_parity["allclose"]),
        "real_pcm_token_recomputed_cached_100pct": bool(token_parity["exact"]),
        "real_pcm_token_coverage": int(full_tokens.numel()) == expected_tokens,
        "future_hidden_exact_before_changed_block": bool(
            future_hidden.get("maximum_absolute_error", 1.0) == 0.0
        ),
        "future_tokens_exact_before_changed_block": bool(future_tokens["exact"]),
        "partial_final_block_exercised": len(waveform) % BLOCK_SAMPLES != 0,
        "reset_boundary_exercised": bool(reset_stream["state"].encoder_resets >= 1),
        "reset_hidden_recomputed_cached": bool(reset_hidden["allclose"]),
        "reset_token_recomputed_cached_100pct": bool(reset_tokens["exact"]),
        "reset_token_coverage": int(reset_full.token_ids.numel()) == expected_reset_tokens,
    }
    passed = all(checks.values())
    step_ms = streamed["step_ms"]
    result: dict[str, Any] = {
        "schema_version": "uniss_stage00_frontend_real_pcm_audit_v1",
        "frontend_schema": FRONTEND_SCHEMA,
        "passed": passed,
        "checks": checks,
        "input": {
            "manifest": str(manifest),
            "manifest_row": args.row_index,
            "sample_id": record.get("id"),
            "src_lang": record.get("src_lang"),
            "tgt_lang": record.get("tgt_lang"),
            "audio": str(audio_path),
            "pcm_f32le_sha256": pcm_sha256,
            "samples": len(waveform),
            "duration_ms": duration_ms,
        },
        "frontend": {
            "whispervq_model": str(model_path),
            "dtype": str(next(encoder.parameters()).dtype),
            "device": str(frontend.device),
            "block_ms": BLOCK_MS,
            "stft_center": False,
            "normalization": "arrived_block_local_peak_v1",
            "right_context_ms": 0,
            "position_embeddings": int(frontend.cached_encoder.position_embedding.num_embeddings),
            "maximum_segment_ms": frontend.maximum_segment_ms,
        },
        "real_pcm_parity": {
            "strict_reference": "block_recomputed_without_persistent_kv",
            "hidden": hidden_parity,
            "quantized": quantized_parity,
            "tokens": token_parity,
            "full_seconds": full_seconds,
            "cached_step_ms_p50": float(np.percentile(step_ms, 50)),
            "cached_step_ms_p95": float(np.percentile(step_ms, 95)),
            "cached_step_ms_max": float(max(step_ms)),
            "cached_rtf": float(sum(step_ms) / duration_ms),
        },
        "single_mask_numerical_diagnostic": {
            "gate": False,
            "reason": "different CUDA GEMM reduction geometry",
            "hidden": single_mask_hidden,
            "tokens": single_mask_tokens,
        },
        "future_perturbation": {
            "changed_block_index": future_block,
            "changed_from_ms": future_block * BLOCK_MS,
            "committed_tokens_compared": committed_tokens,
            "hidden": future_hidden,
            "tokens": future_tokens,
        },
        "reset_boundary": {
            "audio_seconds": len(reset_waveform) / SAMPLE_RATE,
            "full_segments": reset_full.encoder_segments,
            "cached_resets": reset_stream["state"].encoder_resets,
            "reset_before_blocks": reset_stream["reset_blocks"],
            "hidden": reset_hidden,
            "tokens": reset_tokens,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(frontend.device)
            if frontend.device.type == "cuda"
            else None,
            "seed": args.seed,
        },
        "elapsed_seconds": time.time() - started,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--passed-marker", required=True)
    parser.add_argument("--future-block", type=int, default=3)
    parser.add_argument("--reset-audit-seconds", type=float, default=30.4)
    parser.add_argument("--duration-tolerance-ms", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).resolve()
    marker = Path(args.passed_marker).resolve()
    if output.exists() or marker.exists():
        raise FileExistsError("refusing to overwrite an existing Stage 00 audit")
    result = audit(args)
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    _atomic_json(
        marker,
        {
            "schema_version": "uniss_stage00_frontend_gate_v1",
            "passed": True,
            "audit_json": str(output),
            "frontend_schema": FRONTEND_SCHEMA,
            "checks": result["checks"],
        },
    )


if __name__ == "__main__":
    main()
