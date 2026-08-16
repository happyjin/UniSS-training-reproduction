#!/usr/bin/env python3
"""Real-checkpoint parity and gradient gate for the Stage A frontend."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_SAMPLES,
    SAMPLE_RATE,
    SharedCausalWhisperVQFrontend,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A parity artifact: {path}")
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
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"manifest has no row {row_index}: {path}")


def _waveform(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Stage A parity requires 16 kHz PCM: {path}")
    waveform = np.asarray(values, dtype=np.float32).mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"empty or non-finite Stage A PCM: {path}")
    return waveform


def _hidden_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    shape_equal = reference.shape == actual.shape
    if not shape_equal:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "allclose": False,
        }
    absolute = (reference.float() - actual.float()).abs()
    return {
        "shape_equal": True,
        "shape": list(reference.shape),
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "allclose": bool(torch.allclose(reference.float(), actual.float(), rtol=2e-5, atol=2e-6)),
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


def _run_trainable(
    model: TrainableSharedCausalWhisperVQ,
    waveform: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    values = torch.from_numpy(waveform).to(device=device).unsqueeze(0)
    lengths = torch.tensor([len(waveform)], dtype=torch.long, device=device)
    output = model(values, lengths, chunk_ms=160)
    keep = int(output.pooled_lengths[0].item())
    pooled = output.pooled_hidden[:, :keep]
    codebook = model.codebook.to(device=pooled.device, dtype=pooled.dtype)
    flat = pooled.reshape(-1, pooled.shape[-1])
    distances = (
        flat.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * flat @ codebook.t()
    )
    tokens = distances.argmin(dim=1).reshape(pooled.shape[:-1])
    return output.frame_hidden, pooled, tokens


def _gradient_audit(
    model: TrainableSharedCausalWhisperVQ,
    waveform: np.ndarray,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    short = waveform[: 2 * BLOCK_SAMPLES]
    device = next(model.parameters()).device
    values = torch.from_numpy(short.copy()).to(device=device).unsqueeze(0)
    lengths = torch.tensor([len(short)], dtype=torch.long, device=device)
    output = model(values, lengths, chunk_ms=160)
    loss = output.frame_hidden.float().square().mean() + output.pooled_hidden.float().square().mean()
    loss.backward()
    trainable = []
    nonzero = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable.append(name)
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if bool(parameter.grad.detach().abs().max() > 0):
            nonzero.append(name)
    post_vq_trainable = [
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
        and name.startswith("layers.")
        and int(name.split(".")[1]) >= int(model.encoder.config.pooling_position)
    ]
    result = {
        "loss": float(loss.detach().item()),
        "trainable_parameter_tensors": len(trainable),
        "nonzero_gradient_tensors": len(nonzero),
        "nonfinite_gradient_tensors": nonfinite,
        "conv1_has_nonzero_gradient": any(name.startswith("encoder.conv1.") for name in nonzero),
        "pre_vq_layer_has_nonzero_gradient": any(name.startswith("encoder.layers.") for name in nonzero),
        "codebook_frozen": not model.encoder.codebook.weight.requires_grad,
        "codebook_gradient_absent": model.encoder.codebook.weight.grad is None,
        "pooling_frozen": all(
            not parameter.requires_grad for parameter in model.encoder.pooling_layer.parameters()
        ),
        "post_vq_trainable_parameters": post_vq_trainable,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    record = _manifest_row(manifest, args.row_index)
    audio = Path(str(record["source_audio"])).resolve()
    waveform = _waveform(audio)
    device = torch.device(args.device)
    model = TrainableSharedCausalWhisperVQ(
        args.whispervq_model,
        gradient_checkpointing=False,
    ).to(device).float().eval()
    reference = SharedCausalWhisperVQFrontend(
        model.encoder,
        model.mel_filters,
        device=device,
    ).float().eval()

    with torch.inference_mode():
        frame_hidden, pooled, tokens = _run_trainable(model, waveform)
        full = reference.forward_full_reference(waveform)
        reference_hidden = full.pre_vq_hidden
        reference_tokens = full.token_ids

        future_block = min(
            max(1, int(args.future_block)),
            max(1, (len(waveform) + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES - 1),
        )
        changed = waveform.copy()
        mutation_start = future_block * BLOCK_SAMPLES
        generator = np.random.default_rng(args.seed)
        changed[mutation_start:] = generator.normal(
            0.0,
            max(float(np.std(waveform)), 0.01),
            len(changed) - mutation_start,
        ).astype(np.float32)
        changed_frames, changed_pooled, changed_tokens = _run_trainable(model, changed)

    pooled_parity = _hidden_metrics(reference_hidden, pooled)
    token_parity = _token_metrics(reference_tokens, tokens)
    committed_frames = future_block * 8
    committed_tokens = future_block * 2
    future_frames = _hidden_metrics(
        frame_hidden[:, :committed_frames], changed_frames[:, :committed_frames]
    )
    future_hidden = _hidden_metrics(
        pooled[:, :committed_tokens], changed_pooled[:, :committed_tokens]
    )
    future_tokens = _token_metrics(
        tokens[:, :committed_tokens], changed_tokens[:, :committed_tokens]
    )
    source_glm = torch.tensor(record["source_glm"], device=tokens.device).reshape_as(tokens)
    source_glm_diagnostic = _token_metrics(source_glm, tokens)
    gradients = _gradient_audit(model, waveform)
    checks = {
        "training_full_hidden_matches_stage00_single_mask": bool(pooled_parity["allclose"]),
        "training_glm_tokens_match_stage00_100pct": bool(token_parity["exact"]),
        "future_frames_exact": float(future_frames.get("maximum_absolute_error", 1.0)) == 0.0,
        "future_pooled_hidden_exact": float(future_hidden.get("maximum_absolute_error", 1.0)) == 0.0,
        "future_tokens_exact": bool(future_tokens["exact"]),
        "trainable_gradients_finite": not gradients["nonfinite_gradient_tensors"],
        "conv1_gradient_nonzero": bool(gradients["conv1_has_nonzero_gradient"]),
        "pre_vq_gradient_nonzero": bool(gradients["pre_vq_layer_has_nonzero_gradient"]),
        "codebook_frozen": bool(gradients["codebook_frozen"]),
        "codebook_gradient_absent": bool(gradients["codebook_gradient_absent"]),
        "pooling_frozen": bool(gradients["pooling_frozen"]),
        "post_vq_frozen": not gradients["post_vq_trainable_parameters"],
    }
    return {
        "schema_version": "uniss_stage_a_trainable_frontend_parity_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "input": {
            "manifest": str(manifest),
            "row_index": int(args.row_index),
            "sample_id": record.get("id"),
            "source_audio": str(audio),
            "source_duration_ms": record.get("source_duration_ms"),
            "source_glm_tokens": len(record["source_glm"]),
        },
        "frontend": {
            "whispervq_model": str(Path(args.whispervq_model).resolve()),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "chunk_ms": 160,
            "stft_center": False,
            "block_local_normalization": True,
        },
        "stage00_single_mask_parity": {
            "hidden": pooled_parity,
            "tokens": token_parity,
        },
        "future_perturbation": {
            "changed_block": future_block,
            "changed_from_ms": future_block * 160,
            "frames": future_frames,
            "pooled_hidden": future_hidden,
            "tokens": future_tokens,
        },
        "offline_source_glm_agreement_diagnostic": {
            **source_glm_diagnostic,
            "gate": False,
            "reason": "offline bidirectional GLM identity is not the causal cached/full parity gate",
        },
        "gradient_audit": gradients,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--future-block", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--passed-marker", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).resolve()
    marker = Path(args.passed_marker).resolve()
    if output.exists() or marker.exists():
        raise FileExistsError("refusing to overwrite Stage A frontend parity output")
    result = audit(args)
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    _atomic_json(
        marker,
        {
            "schema_version": "uniss_stage_a_trainable_frontend_gate_v1",
            "passed": True,
            "audit_json": str(output),
            "checks": result["checks"],
        },
    )


if __name__ == "__main__":
    main()
