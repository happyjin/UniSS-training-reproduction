#!/usr/bin/env python3
"""Checkpoint-level cached/full and future-PCM causality gate for Stage A v2."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_MS,
    BLOCK_SAMPLES,
    SAMPLE_RATE,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    cache_growth_is_valid,
    hidden_metrics,
    load_trained_objective,
    make_cached_frontend,
    run_cached_frontend,
    token_metrics,
)


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint gate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def manifest_row(path: Path, row_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"manifest has no row {row_index}: {path}")


def load_waveform(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"checkpoint gate requires 16 kHz PCM: {path}")
    waveform = np.asarray(values, dtype=np.float32).mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"invalid checkpoint-gate PCM: {path}")
    return waveform


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    row = manifest_row(manifest, args.row_index)
    audio = Path(str(row["source_audio"])).resolve()
    waveform = load_waveform(audio)
    device = torch.device(args.device)
    objective, checkpoint = load_trained_objective(
        args.checkpoint,
        args.whispervq_model,
        device,
        dtype=torch.float32,
    )
    frontend = make_cached_frontend(objective, device)

    recomputed = frontend.forward_recomputed_reference(waveform)
    cached = run_cached_frontend(frontend, waveform)
    hidden_parity = hidden_metrics(recomputed.pre_vq_hidden.cpu(), cached.hidden)
    quantized_parity = hidden_metrics(recomputed.quantized_hidden.cpu(), cached.quantized)
    token_parity = token_metrics(recomputed.token_ids.cpu(), cached.tokens)
    recomputed_residual = objective.bridge_projection(
        objective.bridge_norm(recomputed.pre_vq_hidden.to(device))
    ).float().cpu()
    cached_residual = objective.bridge_projection(
        objective.bridge_norm(cached.hidden.to(device))
    ).float().cpu()
    residual_parity = hidden_metrics(recomputed_residual, cached_residual)

    values = torch.from_numpy(waveform.copy()).to(device).unsqueeze(0)
    lengths = torch.tensor([len(waveform)], dtype=torch.long, device=device)
    train_full = objective.frontend(values, lengths, chunk_ms=160)
    keep = int(train_full.pooled_lengths[0].item())
    train_full_diagnostic = hidden_metrics(
        recomputed.pre_vq_hidden.cpu(),
        train_full.pooled_hidden[:, :keep].float().cpu(),
    )

    blocks = math.ceil(len(waveform) / BLOCK_SAMPLES)
    future_block = min(max(1, int(args.future_block)), max(1, blocks - 1))
    mutation_start = future_block * BLOCK_SAMPLES
    changed = waveform.copy()
    generator = np.random.default_rng(args.seed)
    changed[mutation_start:] = generator.normal(
        0.0,
        max(float(np.std(waveform)), 0.01),
        len(changed) - mutation_start,
    ).astype(np.float32)
    changed_recomputed = frontend.forward_recomputed_reference(changed)
    changed_cached = run_cached_frontend(frontend, changed)
    committed_tokens = future_block * 2
    future_recomputed_hidden = hidden_metrics(
        recomputed.pre_vq_hidden.cpu()[:, :committed_tokens],
        changed_recomputed.pre_vq_hidden.cpu()[:, :committed_tokens],
    )
    future_recomputed_tokens = token_metrics(
        recomputed.token_ids.cpu()[:, :committed_tokens],
        changed_recomputed.token_ids.cpu()[:, :committed_tokens],
    )
    future_cached_hidden = hidden_metrics(
        cached.hidden[:, :committed_tokens],
        changed_cached.hidden[:, :committed_tokens],
    )
    future_cached_tokens = token_metrics(
        cached.tokens[:, :committed_tokens],
        changed_cached.tokens[:, :committed_tokens],
    )

    checks = {
        "trained_hidden_recomputed_cached": bool(hidden_parity["allclose"]),
        "trained_quantized_recomputed_cached": bool(quantized_parity["allclose"]),
        "trained_tokens_recomputed_cached_100pct": bool(token_parity["exact"]),
        "trained_bridge_residual_recomputed_cached": bool(residual_parity["allclose"]),
        "future_recomputed_hidden_exact": float(
            future_recomputed_hidden.get("maximum_absolute_error", 1.0)
        ) == 0.0,
        "future_recomputed_tokens_exact": bool(future_recomputed_tokens["exact"]),
        "future_cached_hidden_exact": float(
            future_cached_hidden.get("maximum_absolute_error", 1.0)
        ) == 0.0,
        "future_cached_tokens_exact": bool(future_cached_tokens["exact"]),
        "cached_state_growth_valid": cache_growth_is_valid(
            cached.frames_seen, cached.reset_blocks
        ),
        "partial_final_block_exercised": len(waveform) % BLOCK_SAMPLES != 0,
    }
    return {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_frontend_gate_v2",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(checkpoint),
        "input": {
            "manifest": str(manifest),
            "row_index": int(args.row_index),
            "sample_id": row.get("id", row.get("sample_id")),
            "source_audio": str(audio),
            "samples": len(waveform),
            "duration_ms": len(waveform) * 1000 / SAMPLE_RATE,
        },
        "runtime": {
            "device": str(device),
            "dtype": "torch.float32",
            "pcm_block_ms": BLOCK_MS,
            "right_context_ms": 0,
            "frames_seen": cached.frames_seen,
            "reset_blocks": cached.reset_blocks,
        },
        "recomputed_cached_parity": {
            "hidden": hidden_parity,
            "quantized": quantized_parity,
            "tokens": token_parity,
            "bridge_residual": residual_parity,
        },
        "training_single_mask_diagnostic": {
            "gate": False,
            "reason": "single-mask and cached GEMM reduction geometry can differ",
            "hidden": train_full_diagnostic,
        },
        "future_perturbation": {
            "changed_block": future_block,
            "changed_from_ms": future_block * BLOCK_MS,
            "committed_tokens_compared": committed_tokens,
            "recomputed_hidden": future_recomputed_hidden,
            "recomputed_tokens": future_recomputed_tokens,
            "cached_hidden": future_cached_hidden,
            "cached_tokens": future_cached_tokens,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--future-block", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--passed-marker", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_json.exists() or args.passed_marker.exists():
        raise FileExistsError("refusing to overwrite Stage A v2 checkpoint gate")
    result = audit(args)
    atomic_json(args.output_json.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    atomic_json(
        args.passed_marker.resolve(),
        {
            "schema_version": "uniss_quality_first_stage_a_checkpoint_frontend_passed_v2",
            "passed": True,
            "checkpoint": result["checkpoint"],
            "audit_json": str(args.output_json.resolve()),
            "checks": result["checks"],
        },
    )


if __name__ == "__main__":
    main()
