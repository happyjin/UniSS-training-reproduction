"""Validate Stage-B-v2 against its causal sidecar target and streaming gates."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torchaudio

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v1.validate_stage_b import _edit_distance
from training.simul_uniss.subsecond_v2.validate_stage_b_latent import (
    _tokenize,
    load_model,
    percentile,
    stable_prefix_tokens,
    synthetic_tests,
)


SCHEMA = "simul_uniss_stage_b_v2_validation_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def validate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint = load_model(args.checkpoint, device, None, None)
    structural = synthetic_tests(model, device)
    sidecar_path = Path(args.sidecar_manifest).resolve()
    source_path = Path(args.source_manifest).resolve()
    offsets = load_index(sidecar_path)
    if offsets is None or not offsets:
        raise ValueError(f"missing sidecar index for {sidecar_path}")
    stride = max(1, len(offsets) // max(1, args.samples))
    selected = list(range(0, len(offsets), stride))[: args.samples]
    shard_cache: dict[str, dict[str, object]] = {}
    target_exact = target_aligned = target_edit = target_reference = 0
    full_exact = full_aligned = full_edit = full_reference = 0
    audio_seconds = compute_seconds = 0.0
    first_self: list[float] = []
    first_correct: list[float] = []
    committed_tokens = committed_target_exact = committed_final_exact = 0
    with sidecar_path.open("rb") as sidecar_handle, source_path.open("rb") as source_handle:
        for tested, offset_index in enumerate(selected):
            sidecar_handle.seek(offsets[offset_index])
            row = json.loads(sidecar_handle.readline())
            source_handle.seek(int(row["source_manifest_offset"]))
            source = json.loads(source_handle.readline())
            path = str(row["shard_path"])
            shard = shard_cache.get(path)
            if shard is None:
                shard = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                shard_cache[path] = shard
            target = shard["target_tokens"][int(row["target_start"]) : int(row["target_end"])].tolist()  # type: ignore[index]
            reference = shard["full_reference_tokens"][int(row["reference_start"]) : int(row["reference_end"])].tolist()  # type: ignore[index]
            waveform, sample_rate = torchaudio.load(str(source["source_audio"]))
            waveform = waveform[:1]
            if sample_rate != model.config.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, model.config.sample_rate
                )
            waveform = waveform[..., : model.config.sample_rate * args.max_audio_seconds].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predicted = _tokenize(model, waveform)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds += time.perf_counter() - started
            audio_seconds += waveform.shape[-1] / model.config.sample_rate

            common = min(len(predicted), len(target))
            target_exact += sum(predicted[index] == target[index] for index in range(common))
            target_aligned += common
            target_edit += _edit_distance(predicted, target)
            target_reference += max(1, len(target))
            full_common = min(len(predicted), len(reference))
            full_exact += sum(predicted[index] == reference[index] for index in range(full_common))
            full_aligned += full_common
            full_edit += _edit_distance(predicted, reference)
            full_reference += max(1, len(reference))

            if tested < args.latency_samples:
                committed, times = stable_prefix_tokens(
                    model,
                    waveform,
                    sample_rate=model.config.sample_rate,
                    chunk_ms=args.chunk_ms,
                    right_context_ms=args.right_context_ms,
                    stability_ticks=args.stability_ticks,
                )
                if times:
                    first_self.append(times[0])
                    if target and committed and target[0] == committed[0]:
                        first_correct.append(times[0])
                committed_tokens += len(committed)
                committed_target_exact += sum(
                    value == target[index]
                    for index, value in enumerate(committed[: len(target)])
                )
                committed_final_exact += sum(
                    value == predicted[index]
                    for index, value in enumerate(committed[: len(predicted)])
                )
    target_position = target_exact / max(1, target_aligned)
    target_edit_agreement = 1.0 - target_edit / max(1, target_reference)
    coverage = len(first_correct) / max(1, min(len(selected), args.latency_samples))
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "sidecar_manifest": str(sidecar_path),
        "source_manifest": str(source_path),
        "samples": len(selected),
        "target_position_agreement": target_position,
        "target_edit_agreement": target_edit_agreement,
        "full_teacher_position_agreement": full_exact / max(1, full_aligned),
        "full_teacher_edit_agreement": 1.0 - full_edit / max(1, full_reference),
        "active_rtf": compute_seconds / max(1e-9, audio_seconds),
        "audio_seconds": audio_seconds,
        "compute_seconds": compute_seconds,
        "first_self_stable_coverage": len(first_self)
        / max(1, min(len(selected), args.latency_samples)),
        "first_self_stable_p50_ms": percentile(first_self, 0.50) if first_self else None,
        "first_self_stable_p95_ms": percentile(first_self, 0.95) if first_self else None,
        "first_correct_stable_coverage": coverage,
        "first_correct_stable_p50_ms": percentile(first_correct, 0.50)
        if first_correct
        else None,
        "first_correct_stable_p95_ms": percentile(first_correct, 0.95)
        if first_correct
        else None,
        "committed_target_accuracy": committed_target_exact / max(1, committed_tokens),
        "committed_final_parity": committed_final_exact / max(1, committed_tokens),
        **structural,
    }
    structural_pass = (
        structural["cache_max_abs"] <= args.cache_tolerance
        and structural["future_perturbation_max_abs"] <= args.future_tolerance
        and structural["long_session_output_frames"] == structural["long_session_input_frames"]
    )
    quality_pass = (
        target_edit_agreement >= args.minimum_target_agreement
        and result["active_rtf"] <= args.maximum_rtf
        and coverage >= args.minimum_correct_stable_coverage
        and result["first_correct_stable_p50_ms"] is not None
        and result["first_correct_stable_p50_ms"] <= args.maximum_first_p50_ms
        and result["first_correct_stable_p95_ms"] is not None
        and result["first_correct_stable_p95_ms"] <= args.maximum_first_p95_ms
    )
    result["structural_pass"] = structural_pass
    result["quality_pass"] = quality_pass
    result["gate_status"] = "passed" if structural_pass and quality_pass else "failed"
    _atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sidecar-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--latency-samples", type=int, default=16)
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--right-context-ms", type=int, default=80)
    parser.add_argument("--stability-ticks", type=int, default=2)
    parser.add_argument("--cache-tolerance", type=float, default=1e-4)
    parser.add_argument("--future-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-target-agreement", type=float, default=0.70)
    parser.add_argument("--maximum-rtf", type=float, default=0.25)
    parser.add_argument("--minimum-correct-stable-coverage", type=float, default=0.90)
    parser.add_argument("--maximum-first-p50-ms", type=float, default=700.0)
    parser.add_argument("--maximum-first-p95-ms", type=float, default=1_000.0)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
