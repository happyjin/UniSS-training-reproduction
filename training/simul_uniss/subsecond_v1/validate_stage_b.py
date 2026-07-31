"""Validate Stage-B cache parity, future causality, and audio token compatibility."""

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
from training.simul_uniss.subsecond_v1.model import (
    CausalAudioStudentV2,
    StageBModelConfig,
    greedy_ctc_tokens,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


@torch.inference_mode()
def synthetic_tests(model: CausalAudioStudentV2, device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(20260730)
    projected = torch.randn(1, 16, model.config.hidden_size, generator=generator, device=device)
    full, _ = model.forward_projected(projected, torch.tensor([16], device=device))
    cached = model.infer_projected(projected)
    cache_max_abs = float((full - cached).abs().max())
    cache_mean_abs = float((full - cached).abs().mean())

    changed = projected.clone()
    changed[:, 12:] = torch.randn(
        changed[:, 12:].shape, generator=generator, device=device, dtype=changed.dtype
    )
    original_stream = model.infer_projected(projected)
    changed_stream = model.infer_projected(changed)
    # The whole segment beginning at frame 8 may attend frames 12--13 as its
    # configured right context. Outputs from earlier completed segments must
    # remain invariant.
    safe_frames = 12 - model.config.segment_frames
    future_max_abs = float(
        (original_stream[:, :safe_frames] - changed_stream[:, :safe_frames]).abs().max()
    )
    return {
        "cache_max_abs": cache_max_abs,
        "cache_mean_abs": cache_mean_abs,
        "future_perturbation_max_abs": future_max_abs,
    }


@torch.inference_mode()
def audio_tests(
    model: CausalAudioStudentV2,
    manifest: Path,
    device: torch.device,
    samples: int,
    *,
    reference_field: str = "source_glm",
    compatibility_reference_field: str | None = None,
) -> dict[str, float]:
    offsets = load_index(manifest)
    if offsets is None or not offsets:
        raise ValueError(f"missing or empty index for {manifest}")
    total_distance = 0
    total_reference = 0
    total_audio_seconds = 0.0
    total_compute_seconds = 0.0
    compatibility_distance = 0
    compatibility_reference = 0
    tested = 0
    with manifest.open("rb") as handle:
        step = max(1, len(offsets) // max(1, samples))
        for index in range(0, len(offsets), step):
            handle.seek(offsets[index])
            item = json.loads(handle.readline())
            waveform, sample_rate = torchaudio.load(item["source_audio"])
            waveform = waveform[:1]
            if sample_rate != model.config.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, model.config.sample_rate
                )
            waveform = waveform.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model.infer_waveform(waveform)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_compute_seconds += time.perf_counter() - started
            predicted = [value - 1 for value in greedy_ctc_tokens(output["teacher_glm_logits"])]
            if reference_field not in item:
                raise KeyError(f"Stage-B validation reference is missing: {reference_field}")
            reference = [int(value) for value in item[reference_field]]
            total_distance += _edit_distance(predicted, reference)
            total_reference += max(1, len(reference))
            if compatibility_reference_field:
                if compatibility_reference_field not in item:
                    raise KeyError(
                        f"Stage-B compatibility reference is missing: {compatibility_reference_field}"
                    )
                compatibility = [int(value) for value in item[compatibility_reference_field]]
                compatibility_distance += _edit_distance(reference, compatibility)
                compatibility_reference += max(1, len(compatibility))
            total_audio_seconds += waveform.shape[-1] / model.config.sample_rate
            tested += 1
            if tested >= samples:
                break
    result = {
        "audio_samples": float(tested),
        "glm_token_agreement": 1.0 - total_distance / max(1, total_reference),
        "active_rtf": total_compute_seconds / max(1e-9, total_audio_seconds),
        "audio_seconds": total_audio_seconds,
        "compute_seconds": total_compute_seconds,
    }
    if compatibility_reference_field:
        result["reference_compatibility_agreement"] = 1.0 - compatibility_distance / max(
            1, compatibility_reference
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mark-complete", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-tolerance", type=float, default=1e-4)
    parser.add_argument("--future-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-agreement", type=float, default=0.90)
    parser.add_argument("--maximum-rtf", type=float, default=0.25)
    parser.add_argument("--reference-field", default="source_glm")
    parser.add_argument("--compatibility-reference-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = StageBModelConfig.from_dict(checkpoint["model_config"])
    model = CausalAudioStudentV2(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    synthetic = synthetic_tests(model, device)
    audio = audio_tests(
        model,
        Path(args.manifest),
        device,
        args.samples,
        reference_field=args.reference_field,
        compatibility_reference_field=args.compatibility_reference_field,
    )
    structural_pass = (
        synthetic["cache_max_abs"] <= args.cache_tolerance
        and synthetic["future_perturbation_max_abs"] <= args.future_tolerance
    )
    quality_pass = (
        audio["glm_token_agreement"] >= args.minimum_agreement
        and audio["active_rtf"] <= args.maximum_rtf
    )
    passed = structural_pass and (args.smoke or quality_pass)
    result = {
        "schema_version": "simul_uniss_subsecond_stage_b_validation_v2",
        "status": "passed" if passed else "failed",
        "smoke": args.smoke,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "reference_field": args.reference_field,
        "compatibility_reference_field": args.compatibility_reference_field,
        "structural_pass": structural_pass,
        "quality_pass": quality_pass,
        **synthetic,
        **audio,
    }
    output = Path(args.output)
    _atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    if args.mark_complete:
        marker = output.parent / (
            "STAGE_B_SMOKE_COMPLETE.json" if args.smoke else "STAGE_B_PILOT_COMPLETE.json"
        )
        _atomic_json(marker, result)


if __name__ == "__main__":
    main()
