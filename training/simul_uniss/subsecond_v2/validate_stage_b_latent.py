"""Validate corrected latent Stage-B quality, causality, and streaming latency."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import median

import torch
import torchaudio

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v1.validate_stage_b import _edit_distance
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    DEFAULT_CODEBOOK_KEY,
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
    load_whispervq_codebook,
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


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _tokenize(
    model: LatentCausalAudioStudent,
    waveform: torch.Tensor,
    *,
    utterance_samples: int | None = None,
) -> list[int]:
    output = model.infer_waveform(
        waveform, utterance_sample_length=utterance_samples
    )
    length = int(output["token_lengths"][0])
    return model.quantize(output["glm_latent"][:, :length]).reshape(-1).tolist()


def stable_prefix_tokens(
    model: LatentCausalAudioStudent,
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    chunk_ms: int,
    right_context_ms: int,
    stability_ticks: int,
) -> tuple[list[int], list[float]]:
    """Commit only token prefixes repeated across consecutive causal ticks."""

    chunk_samples = round(sample_rate * chunk_ms / 1_000)
    right_samples = round(sample_rate * right_context_ms / 1_000)
    committed: list[int] = []
    commit_times_ms: list[float] = []
    previous: list[int] = []
    persistence: list[int] = []
    for end in range(chunk_samples, waveform.shape[-1] + chunk_samples, chunk_samples):
        utterance_end = min(end, waveform.shape[-1])
        visible_end = min(waveform.shape[-1], utterance_end + right_samples)
        current = _tokenize(
            model,
            waveform[..., :visible_end],
            utterance_samples=utterance_end,
        )
        width = max(len(previous), len(current))
        if len(persistence) < width:
            persistence.extend([0] * (width - len(persistence)))
        for index in range(width):
            same = index < len(previous) and index < len(current) and previous[index] == current[index]
            persistence[index] = persistence[index] + 1 if same else 1
        while (
            len(committed) < len(current)
            and persistence[len(committed)] >= stability_ticks
        ):
            committed.append(current[len(committed)])
            commit_times_ms.append(utterance_end / sample_rate * 1_000.0)
        previous = current
        if utterance_end >= waveform.shape[-1]:
            break
    return committed, commit_times_ms


@torch.inference_mode()
def synthetic_tests(
    model: LatentCausalAudioStudent,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(20_260_802)
    projected = torch.randn(1, 16, model.config.hidden_size, generator=generator, device=device)
    full, _ = model.forward_projected(projected, torch.tensor([16], device=device))
    cached = model.infer_projected(projected)
    cache_max_abs = float((full - cached).abs().max())
    cache_mean_abs = float((full - cached).abs().mean())
    changed = projected.clone()
    changed[:, 12:] = torch.randn(
        changed[:, 12:].shape,
        generator=generator,
        device=device,
        dtype=changed.dtype,
    )
    original_stream = model.infer_projected(projected)
    changed_stream = model.infer_projected(changed)
    safe_frames = 12 - model.config.segment_frames
    future_max_abs = float(
        (original_stream[:, :safe_frames] - changed_stream[:, :safe_frames]).abs().max()
    )
    # A one-minute projected stream verifies output/cache work stays bounded by
    # the Emformer left-context configuration rather than session duration.
    long_frames = 60 * 25
    long_projected = torch.randn(
        1, long_frames, model.config.hidden_size, generator=generator, device=device
    )
    started = time.perf_counter()
    long_output = model.infer_projected(long_projected)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    long_elapsed = time.perf_counter() - started
    return {
        "cache_max_abs": cache_max_abs,
        "cache_mean_abs": cache_mean_abs,
        "future_perturbation_max_abs": future_max_abs,
        "long_session_input_frames": float(long_frames),
        "long_session_output_frames": float(long_output.shape[1]),
        "long_session_active_rtf": long_elapsed / 60.0,
    }


@torch.inference_mode()
def audio_tests(
    model: LatentCausalAudioStudent,
    manifest: Path,
    device: torch.device,
    samples: int,
    *,
    reference_field: str,
    latency_samples: int,
    right_context_ms: int,
    stability_ticks: int,
) -> dict[str, float]:
    offsets = load_index(manifest)
    if offsets is None or not offsets:
        raise ValueError(f"missing or empty index for {manifest}")
    exact = 0
    aligned = 0
    edit_distance = 0
    reference_tokens = 0
    total_audio_seconds = 0.0
    total_compute_seconds = 0.0
    first_stable: list[float] = []
    first_correct_stable: list[float] = []
    chunk_invariance: list[float] = []
    committed_tokens = 0
    committed_teacher_exact = 0
    committed_final_exact = 0
    tested = 0
    with manifest.open("rb") as handle:
        stride = max(1, len(offsets) // max(1, samples))
        for offset_index in range(0, len(offsets), stride):
            handle.seek(offsets[offset_index])
            item = json.loads(handle.readline())
            waveform, sample_rate = torchaudio.load(item["source_audio"])
            waveform = waveform[:1]
            if sample_rate != model.config.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, model.config.sample_rate
                )
            waveform = waveform[..., : model.config.sample_rate * 8].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predicted = _tokenize(model, waveform)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_compute_seconds += time.perf_counter() - started
            reference = [int(value) for value in item[reference_field]]
            common = min(len(predicted), len(reference))
            exact += sum(predicted[index] == reference[index] for index in range(common))
            aligned += common
            edit_distance += _edit_distance(predicted, reference)
            reference_tokens += max(1, len(reference))
            total_audio_seconds += waveform.shape[-1] / model.config.sample_rate
            if tested < latency_samples:
                variants: dict[int, list[int]] = {}
                commit_times: dict[int, list[float]] = {}
                for chunk_ms in (160, 240, 320):
                    committed, times = stable_prefix_tokens(
                        model,
                        waveform,
                        sample_rate=model.config.sample_rate,
                        chunk_ms=chunk_ms,
                        right_context_ms=right_context_ms,
                        stability_ticks=stability_ticks,
                    )
                    variants[chunk_ms] = committed
                    commit_times[chunk_ms] = times
                baseline = variants[160]
                baseline_times = commit_times[160]
                if baseline_times:
                    first_stable.append(baseline_times[0])
                    if reference and baseline and baseline[0] == reference[0]:
                        first_correct_stable.append(baseline_times[0])
                committed_tokens += len(baseline)
                committed_teacher_exact += sum(
                    value == reference[index]
                    for index, value in enumerate(baseline[: len(reference)])
                )
                committed_final_exact += sum(
                    value == predicted[index]
                    for index, value in enumerate(baseline[: len(predicted)])
                )
                for chunk_ms in (240, 320):
                    denominator = max(1, len(baseline))
                    chunk_invariance.append(
                        1.0 - _edit_distance(baseline, variants[chunk_ms]) / denominator
                    )
            tested += 1
            if tested >= samples:
                break
    return {
        "audio_samples": float(tested),
        "position_token_agreement": exact / max(1, aligned),
        "edit_token_agreement": 1.0 - edit_distance / max(1, reference_tokens),
        "aligned_tokens": float(aligned),
        "reference_tokens": float(reference_tokens),
        "active_rtf": total_compute_seconds / max(1e-9, total_audio_seconds),
        "audio_seconds": total_audio_seconds,
        "compute_seconds": total_compute_seconds,
        "first_stable_glm_coverage": len(first_stable) / max(1, min(tested, latency_samples)),
        "first_stable_glm_p50_ms": percentile(first_stable, 0.50),
        "first_stable_glm_p95_ms": percentile(first_stable, 0.95),
        "first_correct_stable_glm_coverage": len(first_correct_stable)
        / max(1, min(tested, latency_samples)),
        "first_correct_stable_glm_p50_ms": percentile(first_correct_stable, 0.50)
        if first_correct_stable
        else None,
        "first_correct_stable_glm_p95_ms": percentile(first_correct_stable, 0.95)
        if first_correct_stable
        else None,
        "committed_teacher_token_accuracy": committed_teacher_exact
        / max(1, committed_tokens),
        "committed_final_token_parity": committed_final_exact
        / max(1, committed_tokens),
        "chunk_polling_invariance": median(chunk_invariance) if chunk_invariance else 0.0,
    }


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    codebook_model: str | None,
    codebook_key: str | None,
) -> tuple[LatentCausalAudioStudent, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LatentStageBModelConfig.from_dict(checkpoint["model_config"])
    model_path = codebook_model or str(checkpoint["codebook_model"])
    key = codebook_key or str(checkpoint.get("codebook_key", DEFAULT_CODEBOOK_KEY))
    codebook = load_whispervq_codebook(model_path, key=key)
    model = LatentCausalAudioStudent(config, codebook).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--latency-samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook-model")
    parser.add_argument("--codebook-key")
    parser.add_argument("--reference-field", default="teacher_source_glm")
    parser.add_argument("--right-context-ms", type=int, default=80)
    parser.add_argument("--stability-ticks", type=int, default=2)
    parser.add_argument("--cache-tolerance", type=float, default=1e-4)
    parser.add_argument("--future-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-agreement", type=float, default=0.70)
    parser.add_argument("--goal-agreement", type=float, default=0.90)
    parser.add_argument("--maximum-rtf", type=float, default=0.25)
    parser.add_argument("--maximum-first-stable-p50-ms", type=float, default=700.0)
    parser.add_argument("--maximum-first-stable-p95-ms", type=float, default=1_000.0)
    parser.add_argument("--minimum-chunk-invariance", type=float, default=0.95)
    parser.add_argument("--minimum-correct-stable-coverage", type=float, default=0.90)
    parser.add_argument("--mark-complete", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, checkpoint = load_model(
        args.checkpoint, device, args.codebook_model, args.codebook_key
    )
    synthetic = synthetic_tests(model, device)
    audio = audio_tests(
        model,
        Path(args.manifest),
        device,
        args.samples,
        reference_field=args.reference_field,
        latency_samples=args.latency_samples,
        right_context_ms=args.right_context_ms,
        stability_ticks=args.stability_ticks,
    )
    structural_pass = (
        synthetic["cache_max_abs"] <= args.cache_tolerance
        and synthetic["future_perturbation_max_abs"] <= args.future_tolerance
        and synthetic["long_session_output_frames"] == synthetic["long_session_input_frames"]
    )
    quality_pass = (
        audio["edit_token_agreement"] >= args.minimum_agreement
        and audio["active_rtf"] <= args.maximum_rtf
        and audio["first_correct_stable_glm_coverage"]
        >= args.minimum_correct_stable_coverage
        and audio["first_correct_stable_glm_p50_ms"] is not None
        and audio["first_correct_stable_glm_p50_ms"]
        <= args.maximum_first_stable_p50_ms
        and audio["first_correct_stable_glm_p95_ms"] is not None
        and audio["first_correct_stable_glm_p95_ms"]
        <= args.maximum_first_stable_p95_ms
        and audio["chunk_polling_invariance"] >= args.minimum_chunk_invariance
    )
    passed = structural_pass and (args.smoke or quality_pass)
    result = {
        "schema_version": "simul_uniss_stage_b_latent_validation_v2",
        "status": "passed" if passed else "failed",
        "smoke": args.smoke,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "manifest": str(Path(args.manifest).resolve()),
        "reference_field": args.reference_field,
        "minimum_agreement": args.minimum_agreement,
        "goal_agreement": args.goal_agreement,
        "minimum_correct_stable_coverage": args.minimum_correct_stable_coverage,
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
            "STAGE_B_LATENT_SMOKE_COMPLETE.json"
            if args.smoke
            else "STAGE_B_LATENT_COMPLETE.json"
        )
        _atomic_json(marker, result)


if __name__ == "__main__":
    main()
