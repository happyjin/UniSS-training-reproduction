"""Decode generated UniSS semantic tokens and paired UniST reference audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from evaluation.io_utils import iter_jsonl, write_json
from training.generate_unist_eval_audio import (
    audio_duration_seconds,
    iter_manifest_records,
    maybe_decode_audio,
    safe_sample_name,
    write_jsonl_row,
)


def record_map(manifest: Path) -> dict[str, dict[str, object]]:
    records = {}
    for record in iter_manifest_records(manifest, limit_records=None):
        sample_id = str(record["id"])
        if sample_id in records:
            raise ValueError(f"Duplicate manifest id: {sample_id}")
        records[sample_id] = record
    return records


def decode_results(args: argparse.Namespace) -> dict[str, int]:
    from uniss import UniSSTokenizer

    source_results = Path(args.input)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to reuse audio output without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wav"
    source_dir = output_dir / "source_wav"
    reference_dir = output_dir / "reference_wav"
    wav_dir.mkdir(exist_ok=True)
    if args.save_source_audio:
        source_dir.mkdir(exist_ok=True)
    if args.save_reference_audio:
        reference_dir.mkdir(exist_ok=True)
    destination = output_dir / "results.jsonl"

    completed: set[tuple[str, str]] = set()
    if args.resume and destination.exists():
        completed = {(str(row["id"]), str(row["mode"])) for row in iter_jsonl(destination)}

    records = record_map(Path(args.manifest))
    device = torch.device(args.device)
    tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    counts = {
        "decoded": 0,
        "failed": 0,
        "source_audio": 0,
        "reference_audio": 0,
        "no_semantic_tokens": 0,
    }
    for index, generated in enumerate(iter_jsonl(source_results)):
        key = (str(generated["id"]), str(generated["mode"]))
        if key in completed:
            continue
        record = records.get(key[0])
        if record is None:
            raise KeyError(f"Generated id is not present in manifest: {key[0]}")
        semantic_values = generated.get("semantic_values") or []
        if not semantic_values:
            counts["no_semantic_tokens"] += 1
        name = safe_sample_name(int(generated.get("index", index)), key[0], key[1])
        audio_path, error = maybe_decode_audio(
            speech_tokenizer=tokenizer,
            global_values=record["bicodec_global"],  # type: ignore[arg-type]
            semantic_values=semantic_values,  # type: ignore[arg-type]
            output_path=wav_dir / f"{name}.wav",
            device=device,
        )
        source_path = None
        source_error = None
        if args.save_source_audio:
            source_path, source_error = maybe_decode_audio(
                speech_tokenizer=tokenizer,
                global_values=record["bicodec_global"],  # type: ignore[arg-type]
                semantic_values=record["source_bicodec"],  # type: ignore[arg-type]
                output_path=source_dir / f"{name}.wav",
                device=device,
            )
            if source_path:
                counts["source_audio"] += 1
        reference_path = None
        reference_error = None
        if args.save_reference_audio:
            reference_path, reference_error = maybe_decode_audio(
                speech_tokenizer=tokenizer,
                global_values=record["bicodec_global"],  # type: ignore[arg-type]
                semantic_values=record["target_bicodec"],  # type: ignore[arg-type]
                output_path=reference_dir / f"{name}.wav",
                device=device,
            )
            if reference_path:
                counts["reference_audio"] += 1
        row = {
            **generated,
            "audio_path": audio_path,
            "audio_duration_seconds": audio_duration_seconds(audio_path),
            "source_audio_path": source_path,
            "source_audio_duration_seconds": audio_duration_seconds(source_path),
            "source_audio_error": source_error,
            "reference_audio_path": reference_path,
            "reference_audio_duration_seconds": audio_duration_seconds(reference_path),
            "reference_audio_error": reference_error,
            "error": error,
        }
        write_jsonl_row(destination, row)
        counts["decoded"] += 1
        if error or source_error or reference_error:
            counts["failed"] += 1
        write_json(output_dir / "summary.json", counts)
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="vLLM generation_results.jsonl")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-source-audio", action="store_true")
    parser.add_argument("--save-reference-audio", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(decode_results(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
