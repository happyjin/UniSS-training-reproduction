"""Audit how much full-context WhisperVQ supervision is causally recoverable."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torchaudio

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v1.validate_stage_b import _edit_distance
from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer


SCHEMA = "simul_uniss_teacher_prefix_ceiling_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_prefix_sequences(
    reference: Sequence[int],
    token_end_ms: Sequence[int],
    commit_end_ms: Sequence[int],
    visible_end_ms: Sequence[int],
    predictions: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Score each token once, exactly when it first becomes committable."""

    if not (
        len(commit_end_ms) == len(visible_end_ms) == len(predictions)
    ):
        raise ValueError("prefix timelines and predictions must have equal length")
    if len(reference) != len(token_end_ms):
        raise ValueError("reference token/time lengths differ")
    previous_count = 0
    immediate_exact = 0
    immediate_total = 0
    prefix_edit_distance = 0
    prefix_reference_tokens = 0
    introduced_at: dict[int, int] = {}
    for tick, (committed_ms, predicted) in enumerate(zip(commit_end_ms, predictions)):
        count = min(len(reference), bisect.bisect_right(token_end_ms, committed_ms))
        prefix_reference = list(reference[:count])
        prefix_predicted = list(predicted[:count])
        prefix_edit_distance += _edit_distance(prefix_predicted, prefix_reference)
        prefix_reference_tokens += max(1, len(prefix_reference))
        for index in range(previous_count, count):
            introduced_at[index] = tick
            immediate_total += 1
            immediate_exact += int(index < len(predicted) and predicted[index] == reference[index])
        previous_count = max(previous_count, count)

    revision_160 = revision_320 = revision_final = revision_total = 0
    for index, tick in introduced_at.items():
        if index >= len(predictions[tick]):
            revision_160 += 1
            revision_320 += 1
            revision_final += 1
            revision_total += 1
            continue
        initial = predictions[tick][index]
        at_160 = predictions[min(len(predictions) - 1, tick + 1)]
        at_320 = predictions[min(len(predictions) - 1, tick + 2)]
        final = predictions[-1]
        revision_160 += int(index >= len(at_160) or at_160[index] != initial)
        revision_320 += int(index >= len(at_320) or at_320[index] != initial)
        revision_final += int(index >= len(final) or final[index] != initial)
        revision_total += 1

    first_correct_stable_visible_ms = None
    first_correct_stable_commit_ms = None
    if reference:
        for tick in range(max(0, len(predictions) - 2)):
            values = [
                predictions[current][0] if predictions[current] else None
                for current in (tick, tick + 1, tick + 2)
            ]
            if values[0] == values[1] == values[2] == reference[0]:
                first_correct_stable_commit_ms = float(commit_end_ms[tick])
                first_correct_stable_visible_ms = float(visible_end_ms[tick])
                break

    final_predicted = list(predictions[-1]) if predictions else []
    final_common = min(len(reference), len(final_predicted))
    return {
        "immediate_exact": immediate_exact,
        "immediate_total": immediate_total,
        "prefix_edit_distance": prefix_edit_distance,
        "prefix_reference_tokens": prefix_reference_tokens,
        "revision_160": revision_160,
        "revision_320": revision_320,
        "revision_final": revision_final,
        "revision_total": revision_total,
        "first_correct_stable_commit_ms": first_correct_stable_commit_ms,
        "first_correct_stable_visible_ms": first_correct_stable_visible_ms,
        "full_reencode_exact": sum(
            final_predicted[index] == reference[index] for index in range(final_common)
        ),
        "full_reencode_aligned": final_common,
        "full_reencode_edit_distance": _edit_distance(final_predicted, list(reference)),
        "full_reencode_reference_tokens": len(reference),
    }


def _load_record(path: Path, offset: int) -> dict[str, object]:
    with path.open("rb") as handle:
        handle.seek(offset)
        value = json.loads(handle.readline())
    waveform, sample_rate = torchaudio.load(str(value["source_audio"]))
    waveform = waveform[:1]
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    value["_waveform"] = waveform
    return value


def _sample_records(
    manifest: Path, samples: int, audio_workers: int
) -> list[dict[str, object]]:
    offsets = load_index(manifest)
    if offsets is None or not offsets:
        raise ValueError(f"missing or empty index for {manifest}")
    stride = max(1, len(offsets) // max(1, samples))
    selected = list(offsets[::stride][:samples])
    with concurrent.futures.ThreadPoolExecutor(max_workers=audio_workers) as executor:
        return list(executor.map(lambda offset: _load_record(manifest, offset), selected))


def _aggregate(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    sums = {
        key: sum(int(value[key]) for value in values)
        for key in (
            "immediate_exact",
            "immediate_total",
            "prefix_edit_distance",
            "prefix_reference_tokens",
            "revision_160",
            "revision_320",
            "revision_final",
            "revision_total",
            "full_reencode_exact",
            "full_reencode_aligned",
            "full_reencode_edit_distance",
            "full_reencode_reference_tokens",
        )
    }
    visible = [
        float(value["first_correct_stable_visible_ms"])
        for value in values
        if value["first_correct_stable_visible_ms"] is not None
    ]
    committed = [
        float(value["first_correct_stable_commit_ms"])
        for value in values
        if value["first_correct_stable_commit_ms"] is not None
    ]
    return {
        "samples": len(values),
        "causal_immediate_position_agreement": sums["immediate_exact"]
        / max(1, sums["immediate_total"]),
        "prefix_edit_agreement": 1.0
        - sums["prefix_edit_distance"] / max(1, sums["prefix_reference_tokens"]),
        "revision_rate_after_160ms": sums["revision_160"]
        / max(1, sums["revision_total"]),
        "revision_rate_after_320ms": sums["revision_320"]
        / max(1, sums["revision_total"]),
        "revision_rate_vs_full": sums["revision_final"]
        / max(1, sums["revision_total"]),
        "first_correct_stable_coverage": len(visible) / max(1, len(values)),
        "first_correct_stable_visible_p50_ms": _percentile(visible, 0.50),
        "first_correct_stable_visible_p95_ms": _percentile(visible, 0.95),
        "first_correct_stable_commit_p50_ms": _percentile(committed, 0.50),
        "full_reencode_position_agreement": sums["full_reencode_exact"]
        / max(1, sums["full_reencode_aligned"]),
        "full_reencode_edit_agreement": 1.0
        - sums["full_reencode_edit_distance"]
        / max(1, sums["full_reencode_reference_tokens"]),
        **sums,
    }


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = Path(args.manifest).resolve()
    records = _sample_records(manifest, args.samples, args.audio_workers)
    teacher = Glm4Tokenizer(args.whispervq_model, device=args.device)
    lookaheads = sorted(set(args.lookahead_ms))
    per_lookahead: dict[int, list[dict[str, object]]] = {
        value: [] for value in lookaheads
    }
    for sample_index, record in enumerate(records, start=1):
        waveform = record.pop("_waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("loaded waveform is not a tensor")
        waveform = waveform[..., : args.max_audio_seconds * 16_000]
        duration_ms = int(round(waveform.shape[-1] / 16))
        reference = [int(value) for value in record[args.reference_field]]  # type: ignore[index]
        ends = [int(value) for value in record[args.reference_end_field]]  # type: ignore[index]
        reference_count = bisect.bisect_right(ends, duration_ms)
        reference = reference[:reference_count]
        ends = ends[:reference_count]
        commit_ends = list(range(args.chunk_ms, duration_ms + args.chunk_ms, args.chunk_ms))
        commit_ends = [min(value, duration_ms) for value in commit_ends]
        commit_ends = list(dict.fromkeys(commit_ends))
        requests: list[tuple[int, int, int]] = []
        audio: list[tuple[torch.Tensor, int]] = []
        for lookahead in lookaheads:
            for tick, committed_ms in enumerate(commit_ends):
                visible_ms = min(duration_ms, committed_ms + lookahead)
                visible_samples = max(400, min(waveform.shape[-1], visible_ms * 16))
                requests.append((lookahead, tick, visible_ms))
                audio.append((waveform[..., :visible_samples], 16_000))
        outputs = teacher.bacth_tokenize(audio)
        grouped: dict[int, list[list[int]]] = {value: [] for value in lookaheads}
        visible: dict[int, list[int]] = {value: [] for value in lookaheads}
        for (lookahead, _, visible_ms), tokens in zip(requests, outputs):
            grouped[lookahead].append([int(value) for value in tokens])
            visible[lookahead].append(visible_ms)
        for lookahead in lookaheads:
            per_lookahead[lookahead].append(
                score_prefix_sequences(
                    reference,
                    ends,
                    commit_ends,
                    visible[lookahead],
                    grouped[lookahead],
                )
            )
        print(
            json.dumps(
                {"processed": sample_index, "samples": len(records)}, sort_keys=True
            ),
            flush=True,
        )
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "manifest": str(manifest),
        "whispervq_model": str(Path(args.whispervq_model).resolve()),
        "reference_field": args.reference_field,
        "samples": len(records),
        "chunk_ms": args.chunk_ms,
        "max_audio_seconds": args.max_audio_seconds,
        "audio_workers": args.audio_workers,
        "lookahead": {
            str(value): _aggregate(per_lookahead[value]) for value in lookaheads
        },
    }
    _atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--audio-workers", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, nargs="+", default=[80, 160, 320, 640])
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    parser.add_argument("--reference-field", default="teacher_source_glm")
    parser.add_argument("--reference-end-field", default="teacher_source_glm_end_ms")
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
