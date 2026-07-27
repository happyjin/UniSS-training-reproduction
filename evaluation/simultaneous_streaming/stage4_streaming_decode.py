"""Decode Stage4 semantic chunks through the real streaming BiCodec path."""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from evaluation.decode_audio import decode_token_batch
from evaluation.io_utils import iter_jsonl, write_json
from evaluation.simultaneous_streaming.stage4_metrics import per_sample_metrics
from training.generate_unist_eval_audio import audio_duration_seconds, safe_sample_name, write_jsonl_row
from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder, bicodec_decode_function


def batched(values: Iterable[object], size: int) -> Iterator[list[object]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def boundary_metrics(previous: np.ndarray, current: np.ndarray, sample_rate: int) -> dict[str, float]:
    if previous.size == 0 or current.size == 0:
        return {
            "amplitude_jump": 0.0,
            "rms_jump": 0.0,
            "spectral_distance": 0.0,
            "click": 0.0,
        }
    window = max(16, int(round(sample_rate * 0.02)))
    left = previous[-window:].astype(np.float32, copy=False)
    right = current[:window].astype(np.float32, copy=False)
    size = min(len(left), len(right))
    left = left[-size:]
    right = right[:size]
    amplitude_jump = abs(float(current[0]) - float(previous[-1]))
    left_rms = math.sqrt(float(np.mean(np.square(left))) + 1e-12)
    right_rms = math.sqrt(float(np.mean(np.square(right))) + 1e-12)
    hann = np.hanning(size).astype(np.float32)
    left_spectrum = np.log1p(np.abs(np.fft.rfft(left * hann)))
    right_spectrum = np.log1p(np.abs(np.fft.rfft(right * hann)))
    spectral_distance = float(np.mean(np.abs(left_spectrum - right_spectrum)))
    return {
        "amplitude_jump": amplitude_jump,
        "rms_jump": abs(left_rms - right_rms),
        "spectral_distance": spectral_distance,
        "click": float(amplitude_jump >= 0.2),
    }


def decode_streaming_row(
    row: Mapping[str, object],
    *,
    decode,
    sample_rate: int,
    semantic_rate: float,
    left_context_tokens: int,
    holdback_tokens: int,
    overlap_ms: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    codec = StreamingBiCodecDecoder(
        decode,
        sample_rate=sample_rate,
        semantic_rate=semantic_rate,
        left_context_tokens=left_context_tokens,
        holdback_tokens=holdback_tokens,
        overlap_ms=overlap_ms,
    )
    speaker = [int(value) for value in row["speaker_tokens"]]  # type: ignore[index]
    traces: list[dict[str, object]] = []
    chunks: list[np.ndarray] = []
    boundaries: list[dict[str, float]] = []
    for raw_event in row["event_trace"]:  # type: ignore[index]
        event = dict(raw_event)
        if event.get("action") != "write":
            event["codec_seconds"] = 0.0
            event["audio_samples"] = 0
            traces.append(event)
            continue
        semantic = [int(value) for value in event.get("generated_semantic_values") or []]
        started = time.perf_counter()
        waveform = codec.push(
            semantic,
            speaker_tokens=speaker,
            is_final=bool(event["source_is_final"]),
        )
        if bool(event["source_is_final"]) and waveform.size == 0 and codec.semantic_history:
            waveform = codec.push([], is_final=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        codec_seconds = time.perf_counter() - started
        event["codec_seconds"] = codec_seconds
        event["audio_samples"] = int(waveform.size)
        event["audio_seconds"] = float(waveform.size / sample_rate)
        if waveform.size:
            if chunks:
                boundary = boundary_metrics(chunks[-1], waveform, sample_rate)
                boundary["event_index"] = float(event["event_index"])
                boundary["sample_index"] = float(sum(len(chunk) for chunk in chunks))
                boundaries.append(boundary)
            chunks.append(waveform)
        traces.append(event)
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    summary = {
        "audio_chunks": len(chunks),
        "output_samples": int(waveform.size),
        "output_seconds": float(waveform.size / sample_rate),
        "boundary_count": len(boundaries),
        "boundary_amplitude_jump_mean": float(np.mean([x["amplitude_jump"] for x in boundaries])) if boundaries else 0.0,
        "boundary_amplitude_jump_p95": float(np.percentile([x["amplitude_jump"] for x in boundaries], 95)) if boundaries else 0.0,
        "boundary_amplitude_jump_max": max((x["amplitude_jump"] for x in boundaries), default=0.0),
        "boundary_rms_jump_mean": float(np.mean([x["rms_jump"] for x in boundaries])) if boundaries else 0.0,
        "boundary_spectral_distance_mean": float(np.mean([x["spectral_distance"] for x in boundaries])) if boundaries else 0.0,
        "boundary_click_rate": float(np.mean([x["click"] for x in boundaries])) if boundaries else 0.0,
        "boundaries": boundaries,
    }
    return waveform, traces, summary


def prepare_output(output_dir: Path, rank: int, resume: bool) -> tuple[Path, set[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results.rank{rank:03d}.jsonl"
    marker = output_dir / f"DECODE_COMPLETE.rank{rank:03d}"
    if marker.is_file() and resume:
        return result_path, {int(row["index"]) for row in iter_jsonl(result_path)}
    if result_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite decode output: {result_path}")
    result_path.touch(exist_ok=True)
    return result_path, {int(row["index"]) for row in iter_jsonl(result_path)}


def run_decode(args: argparse.Namespace) -> dict[str, object]:
    from uniss import UniSSTokenizer

    rank = args.rank
    world_size = args.world_size
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    output_dir = Path(args.output_dir)
    result_path, completed = prepare_output(output_dir, rank, args.resume)
    wav_dir = output_dir / "wav"
    source_dir = output_dir / "source_wav"
    reference_dir = output_dir / "reference_wav"
    for directory in (wav_dir, source_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in iter_jsonl(Path(args.input))
        if int(row["index"]) % world_size == rank and int(row["index"]) not in completed
    ]
    rows.sort(
        key=lambda row: max(
            len(row.get("semantic_values") or []),
            len(row.get("source_bicodec_values") or []),
            len(row.get("reference_semantic_values") or []),
        )
    )
    device = torch.device(args.device)
    tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    decode = bicodec_decode_function(tokenizer.bicodec)
    started = time.time()
    decoded = 0
    failed = 0
    total_audio_seconds = 0.0
    for raw_batch in batched(rows, args.batch_size):
        batch = [dict(value) for value in raw_batch if isinstance(value, Mapping)]
        streaming: dict[int, tuple[str | None, str | None, list[dict[str, object]], dict[str, object]]] = {}
        for row in batch:
            index = int(row["index"])
            name = safe_sample_name(index, row["id"], args.artifact_prefix)
            output_path = wav_dir / f"{name}.wav"
            try:
                waveform, traces, stream_summary = decode_streaming_row(
                    row,
                    decode=decode,
                    sample_rate=args.sample_rate,
                    semantic_rate=args.semantic_rate,
                    left_context_tokens=args.left_context_tokens,
                    holdback_tokens=args.holdback_tokens,
                    overlap_ms=args.overlap_ms,
                )
                if waveform.size == 0:
                    raise ValueError("streaming decode produced empty waveform")
                sf.write(output_path, waveform, args.sample_rate)
                streaming[index] = (str(output_path.resolve()), None, traces, stream_summary)
            except Exception as exc:
                streaming[index] = (
                    None,
                    f"streaming_decode_error:{type(exc).__name__}:{exc}",
                    [dict(value) for value in row["event_trace"]],  # type: ignore[index]
                    {},
                )

        source_items = []
        reference_items = []
        for slot, row in enumerate(batch):
            name = safe_sample_name(int(row["index"]), row["id"], args.artifact_prefix)
            source_items.append(
                {
                    "index": slot,
                    "global_values": row["speaker_tokens"],
                    "semantic_values": row["source_bicodec_values"],
                    "output_path": source_dir / f"{name}.wav",
                }
            )
            reference_items.append(
                {
                    "index": slot,
                    "global_values": row["speaker_tokens"],
                    "semantic_values": row["reference_semantic_values"],
                    "output_path": reference_dir / f"{name}.wav",
                }
            )
        source_decoded = decode_token_batch(
            speech_tokenizer=tokenizer,
            items=source_items,
            device=device,
        )
        reference_decoded = decode_token_batch(
            speech_tokenizer=tokenizer,
            items=reference_items,
            device=device,
        )
        for slot, row in enumerate(batch):
            index = int(row["index"])
            audio_path, error, traces, stream_summary = streaming[index]
            source_path, source_error = source_decoded[slot]
            reference_path, reference_error = reference_decoded[slot]
            if source_path is not None:
                source_path = str(Path(source_path).resolve())
            if reference_path is not None:
                reference_path = str(Path(reference_path).resolve())
            enriched = {
                **row,
                "event_trace": traces,
                "audio_path": audio_path,
                "audio_duration_seconds": audio_duration_seconds(audio_path),
                "source_audio_path": source_path,
                "source_audio_duration_seconds": audio_duration_seconds(source_path),
                "reference_audio_path": reference_path,
                "reference_audio_duration_seconds": audio_duration_seconds(reference_path),
                "source_audio_error": source_error,
                "reference_audio_error": reference_error,
                "error": error,
                "streaming_audio": stream_summary,
                "streaming_decode_config": {
                    "sample_rate": args.sample_rate,
                    "semantic_rate": args.semantic_rate,
                    "left_context_tokens": args.left_context_tokens,
                    "holdback_tokens": args.holdback_tokens,
                    "overlap_ms": args.overlap_ms,
                },
            }
            enriched["streaming_metrics"] = per_sample_metrics(enriched)
            write_jsonl_row(result_path, enriched)
            decoded += 1
            failed += int(bool(error or source_error or reference_error))
            total_audio_seconds += float(enriched["audio_duration_seconds"] or 0.0)
        summary = {
            "schema_version": "simul_uniss_stage4_decode_summary_v1",
            "rank": rank,
            "world_size": world_size,
            "completed_before_resume": len(completed),
            "decoded": decoded,
            "failed": failed,
            "generated_audio_seconds": total_audio_seconds,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_dir / f"decode_summary.rank{rank:03d}.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (output_dir / f"DECODE_COMPLETE.rank{rank:03d}").write_text("complete\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--speech-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--semantic-rate", type=float, default=50.0)
    parser.add_argument("--left-context-tokens", type=int, default=50)
    parser.add_argument("--holdback-tokens", type=int, default=5)
    parser.add_argument("--overlap-ms", type=float, default=80.0)
    parser.add_argument("--artifact-prefix", default="streaming_stage4")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_decode(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
