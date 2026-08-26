#!/usr/bin/env python3
"""Complete long-audio bounded-window evaluation for one routed adapter.

Every source window invokes the strict-causal 160 ms PCM runtime, but model and
decoder state reset at the window boundary.  The resulting complete-file mode
is therefore explicitly reported as bounded-window pseudo-streaming, never as
a cached causal long-form encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    CHUNKS_MS,
    _load_models,
    _load_runtime,
    _route_for_prompt,
)
from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.windowing import (
    WindowSpan,
    place_target_without_overlap,
    plan_bounded_windows,
    render_target_timeline,
    stereo_waveform,
)


SAMPLE_RATE = 16_000


def _equal_partition_fallback(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    maximum_seconds: float,
) -> list[WindowSpan]:
    """Return complete <=maximum windows when silence planning has no solution.

    The shared silence-seeking planner intentionally requires every window to
    be at least ``minimum_seconds``.  A remaining interval in
    ``(maximum, 2 * minimum)`` cannot satisfy both bounds, even though the
    acoustic frontend can safely process the shorter window.  Complete-file
    evaluation must not drop that tail, so the local evaluator evenly
    repartitions the file while preserving the hard 30-second encoder cap.
    """

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    maximum = max(1, int(round(maximum_seconds * sample_rate)))
    count = max(1, int(math.ceil(len(values) / maximum)))
    base, remainder = divmod(len(values), count)
    if base <= 0:
        raise ValueError("cannot partition empty waveform")
    spans: list[WindowSpan] = []
    start = 0
    radius = max(1, int(round(0.10 * sample_rate)))
    for index in range(count):
        end = start + base + (1 if index < remainder else 0)
        chunk = np.asarray(
            values[max(0, end - radius) : min(len(values), end + radius)],
            dtype=np.float64,
        )
        boundary_rms = (
            float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
            if chunk.size
            else 0.0
        )
        spans.append(WindowSpan(index, start, end, boundary_rms))
        start = end
    if start != len(values) or any(span.samples > maximum for span in spans):
        raise AssertionError("equal-partition fallback violated complete coverage")
    return spans


def _plan_complete_windows(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_seconds: float,
    minimum_seconds: float,
    maximum_seconds: float,
) -> tuple[list[WindowSpan], str]:
    """Prefer silence boundaries, but never reject a valid short final tail."""

    try:
        return (
            plan_bounded_windows(
                waveform,
                sample_rate,
                target_seconds=target_seconds,
                minimum_seconds=minimum_seconds,
                maximum_seconds=maximum_seconds,
            ),
            "silence_seeking",
        )
    except ValueError as exc:
        if "window is shorter than minimum size" not in str(exc):
            raise
        return (
            _equal_partition_fallback(
                waveform,
                sample_rate,
                maximum_seconds=maximum_seconds,
            ),
            "equal_partition_relaxed_minimum",
        )


def _speaker_sha256(values: Sequence[int]) -> str:
    raw = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _silence_metrics(waveform: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    block = SAMPLE_RATE // 10
    active: list[bool] = []
    for start in range(0, len(values), block):
        piece = values[start : start + block]
        rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) if len(piece) else 0.0
        active.append(rms >= 1e-4)
    indices = [index for index, value in enumerate(active) if value]
    maximum_internal_gap = 0
    if indices:
        run = 0
        for value in active[indices[0] : indices[-1] + 1]:
            if value:
                run = 0
            else:
                run += 1
                maximum_internal_gap = max(maximum_internal_gap, run)
    return {
        "non_silent_fraction": float(np.mean(active)) if active else 0.0,
        "first_non_silent_ms": (indices[0] * 100.0 if indices else None),
        "last_non_silent_ms": ((indices[-1] + 1) * 100.0 if indices else None),
        "maximum_internal_silence_ms": maximum_internal_gap * 100.0,
    }


def _target_waveform_status(waveform: np.ndarray, sample_rate: int) -> str:
    """Classify an evaluated target without turning model silence into a crash."""

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError("window target audio has the wrong sample rate")
    if not np.isfinite(values).all():
        raise ValueError("window target audio contains non-finite samples")
    return "silent" if values.size == 0 else "non_silent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audio-protocol", type=Path, required=True)
    parser.add_argument("--decision-chunk-ms", type=int, choices=CHUNKS_MS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--strict-runtime", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--target-window-seconds", type=float, default=25.0)
    parser.add_argument("--minimum-window-seconds", type=float, default=18.0)
    parser.add_argument("--maximum-window-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = json.loads(args.audio_protocol.read_text(encoding="utf-8"))
    records = list(protocol["records"])
    requested = set(str(value) for value in args.sample_id)
    if requested:
        records = [row for row in records if str(row["sample_id"]) in requested]
        observed = {str(row["sample_id"]) for row in records}
        if observed != requested:
            raise ValueError(f"audio protocol is missing IDs: {sorted(requested-observed)}")
    if not records:
        raise ValueError("long-form audio protocol is empty")
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [
        int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]
    ]
    if len(fixed_speaker) != 32:
        raise ValueError("Stage-A fixed speaker must contain 32 tokens")

    runtime = _load_runtime(args.strict_runtime)
    model, tokenizer, controller, adapter_manifest, objective, codec = _load_models(args)
    original_generate = runtime.generate

    def routed_generate(*call_args, **call_kwargs):
        prompt = call_kwargs.get("prompt_ids")
        if prompt is None:
            raise ValueError("bounded long-form generation requires prompt_ids")
        enabled = args.adapter_checkpoint is not None and _route_for_prompt(prompt)
        with controller.route(enabled):
            return original_generate(*call_args, **call_kwargs)

    runtime.generate = routed_generate
    args.output.mkdir(parents=True)
    completed: list[dict[str, object]] = []
    try:
        for record in records:
            sample_started = time.perf_counter()
            sample_id = str(record["sample_id"])
            source_path = Path(str(record["source_audio"]))
            source, rate = sf.read(source_path, dtype="float32", always_2d=True)
            if int(rate) != SAMPLE_RATE:
                raise ValueError(f"long audio is not 16 kHz: {source_path}")
            source = np.asarray(source.mean(axis=1), dtype=np.float32)
            if not len(source) or not np.isfinite(source).all():
                raise ValueError(f"long audio is empty/non-finite: {source_path}")
            spans, window_plan_mode = _plan_complete_windows(
                source,
                SAMPLE_RATE,
                target_seconds=args.target_window_seconds,
                minimum_seconds=args.minimum_window_seconds,
                maximum_seconds=args.maximum_window_seconds,
            )
            sample_root = args.output / sample_id
            inputs_root = sample_root / "window_inputs"
            windows_root = sample_root / "windows"
            inputs_root.mkdir(parents=True)
            windows_root.mkdir()
            normalized_source = sample_root / "source.wav"
            sf.write(normalized_source, source, SAMPLE_RATE, subtype="PCM_16")
            placements: list[tuple[int, np.ndarray]] = []
            continuous_parts: list[np.ndarray] = []
            target_cursor = 0
            window_rows: list[dict[str, object]] = []
            failed = 0
            for position, span in enumerate(spans):
                print(
                    f"run={args.run_id} sample={sample_id} window={position+1}/{len(spans)}",
                    flush=True,
                )
                window_id = f"{sample_id}_window_{position:04d}"
                window_input = inputs_root / f"{window_id}.wav"
                sf.write(
                    window_input,
                    source[span.start_sample : span.end_sample],
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
                row = {
                    "id": window_id,
                    "src_lang": str(record["src_lang"]),
                    "tgt_lang": str(record["tgt_lang"]),
                    "source_audio": str(window_input.resolve()),
                    "transcription": "unreferenced external audio",
                    "translation": "unreferenced external audio",
                    "bicodec_global": fixed_speaker,
                    "_stage_a_fixed_speaker_global": fixed_speaker,
                }
                try:
                    value = runtime.evaluate_sample(
                        row,
                        decision_chunk_ms=args.decision_chunk_ms,
                        model=model,
                        tokenizer=tokenizer,
                        objective=objective,
                        bicodec=codec,
                        output=windows_root,
                        seed=20260825 + args.decision_chunk_ms * 100 + position * 1_000_000,
                    )
                    target_path = Path(str(value["continuous_audio_path"]))
                    target, target_rate = sf.read(
                        target_path, dtype="float32", always_2d=True
                    )
                    target = np.asarray(target.mean(axis=1), dtype=np.float32)
                    target_status = _target_waveform_status(target, int(target_rate))
                    local_first_ms = value.get("first_audio_source_ms")
                    if target_status == "silent":
                        window_rows.append(
                            {
                                "window_index": position,
                                "source_start_ms": span.start_sample * 1000.0 / SAMPLE_RATE,
                                "source_end_ms": span.end_sample * 1000.0 / SAMPLE_RATE,
                                "source_duration_ms": span.samples * 1000.0 / SAMPLE_RATE,
                                "boundary_rms": span.boundary_rms,
                                "first_audio_local_ms": None,
                                "first_audio_global_ms": None,
                                "target_end_global_ms": None,
                                "generated_asr": value["generated_streaming_transcription"],
                                "generated_translation": value[
                                    "generated_streaming_translation"
                                ],
                                "audio_writes": value["audio_writes"],
                                "semantic_tokens": value["semantic_tokens"],
                                "audio_healthy": False,
                                "window_result_path": str(target_path.resolve()),
                                "status": "silent",
                            }
                        )
                        continue
                    if local_first_ms is None:
                        local_first_ms = span.samples * 1000.0 / SAMPLE_RATE
                    available = span.start_sample + int(
                        round(float(local_first_ms) * SAMPLE_RATE / 1000.0)
                    )
                    target_start, target_end = place_target_without_overlap(
                        placements,
                        target,
                        available_sample=available,
                        cursor=target_cursor,
                    )
                    target_cursor = target_end
                    continuous_parts.append(target)
                    window_rows.append(
                        {
                            "window_index": position,
                            "source_start_ms": span.start_sample * 1000.0 / SAMPLE_RATE,
                            "source_end_ms": span.end_sample * 1000.0 / SAMPLE_RATE,
                            "source_duration_ms": span.samples * 1000.0 / SAMPLE_RATE,
                            "boundary_rms": span.boundary_rms,
                            "first_audio_local_ms": local_first_ms,
                            "first_audio_global_ms": target_start * 1000.0 / SAMPLE_RATE,
                            "target_end_global_ms": target_end * 1000.0 / SAMPLE_RATE,
                            "generated_asr": value["generated_streaming_transcription"],
                            "generated_translation": value["generated_streaming_translation"],
                            "audio_writes": value["audio_writes"],
                            "semantic_tokens": value["semantic_tokens"],
                            "audio_healthy": value["audio_audit"]["healthy"],
                            "window_result_path": str(
                                (windows_root / window_id / "translation_continuous.wav").resolve()
                            ),
                            "status": "complete",
                        }
                    )
                except Exception as exc:
                    failed += 1
                    window_rows.append(
                        {
                            "window_index": position,
                            "source_start_ms": span.start_sample * 1000.0 / SAMPLE_RATE,
                            "source_end_ms": span.end_sample * 1000.0 / SAMPLE_RATE,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if failed or not continuous_parts:
                raise RuntimeError(
                    f"{sample_id}: {failed}/{len(spans)} bounded windows failed"
                )
            continuous = np.concatenate(continuous_parts).astype(np.float32, copy=False)
            timeline = render_target_timeline(placements, len(source))
            stereo = stereo_waveform(source, timeline)
            continuous_path = sample_root / "translation_continuous.wav"
            timeline_path = sample_root / "translation_global_timeline.wav"
            stereo_path = sample_root / "stereo_left_source_right_translation.wav"
            sf.write(continuous_path, continuous, SAMPLE_RATE, subtype="PCM_16")
            sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
            sf.write(stereo_path, stereo, SAMPLE_RATE, subtype="PCM_16")
            elapsed = time.perf_counter() - sample_started
            completed.append(
                {
                    "sample_id": sample_id,
                    "src_lang": str(record["src_lang"]),
                    "tgt_lang": str(record["tgt_lang"]),
                    "source_path": str(normalized_source.resolve()),
                    "source_duration_seconds": len(source) / SAMPLE_RATE,
                    "translation_duration_seconds": len(continuous) / SAMPLE_RATE,
                    "timeline_duration_seconds": len(timeline) / SAMPLE_RATE,
                    "processing_seconds": elapsed,
                    "rtf": elapsed / (len(source) / SAMPLE_RATE),
                    "planned_windows": len(spans),
                    "window_plan_mode": window_plan_mode,
                    "completed_windows": len(spans),
                    "failed_windows": 0,
                    "silent_windows": sum(
                        row["status"] == "silent" for row in window_rows
                    ),
                    "first_audio_global_ms": min(
                        (
                            float(row["first_audio_global_ms"])
                            for row in window_rows
                            if row["status"] == "complete"
                        ),
                        default=None,
                    ),
                    "translation_path": str(continuous_path.resolve()),
                    "timeline_path": str(timeline_path.resolve()),
                    "stereo_path": str(stereo_path.resolve()),
                    "timeline_silence": _silence_metrics(timeline),
                    "speaker_condition_sha256": _speaker_sha256(fixed_speaker),
                    "speaker_condition_changes": 0,
                    "windows": window_rows,
                }
            )
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "windows": len(spans),
                        "rtf": completed[-1]["rtf"],
                        "first_audio_global_ms": completed[-1]["first_audio_global_ms"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        runtime.generate = original_generate
        controller.close()
    payload = {
        "schema_version": "uniss_stagea_joint_grpo_bounded_longform_v1",
        "status": "complete",
        "run_id": args.run_id,
        "mode": "complete bounded-window pseudo-streaming",
        "strict_causality_inside_each_window": True,
        "cross_window_model_state": "reset",
        "claim_boundary": (
            "Complete-file online-visibility simulation with bounded recomputation; "
            "not a causal cached long-form encoder."
        ),
        "decision_chunk_ms": args.decision_chunk_ms,
        "window_geometry_seconds": {
            "minimum": args.minimum_window_seconds,
            "target": args.target_window_seconds,
            "maximum": args.maximum_window_seconds,
        },
        "adapter_manifest": adapter_manifest,
        "results": completed,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
