"""Decode generated UniSS semantic tokens and paired UniST reference audio."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

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


def batched(values: Iterable[object], size: int) -> Iterator[list[object]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def decode_token_batch(
    *,
    speech_tokenizer,
    items: Sequence[Mapping[str, object]],
    device: torch.device,
) -> dict[int, tuple[str | None, str | None]]:
    """Decode a length-bucketed BiCodec batch with per-item fallback."""

    results: dict[int, tuple[str | None, str | None]] = {}
    valid = []
    for item in items:
        index = int(item["index"])
        semantic_values = item.get("semantic_values") or []
        if not semantic_values:
            results[index] = (None, "no_semantic_tokens")
            continue
        valid.append(item)
    if not valid:
        return results

    def decode_one(item: Mapping[str, object]) -> tuple[str | None, str | None]:
        return maybe_decode_audio(
            speech_tokenizer=speech_tokenizer,
            global_values=item["global_values"],  # type: ignore[arg-type]
            semantic_values=item["semantic_values"],  # type: ignore[arg-type]
            output_path=Path(str(item["output_path"])),
            device=device,
        )

    if len(valid) == 1:
        item = valid[0]
        results[int(item["index"])] = decode_one(item)
        return results

    codec_items = [
        {
            "index": int(item["index"]),
            "global_tokens": torch.tensor(item["global_values"], dtype=torch.long),
            "semantic_tokens": torch.tensor(item["semantic_values"], dtype=torch.long),
        }
        for item in valid
    ]
    try:
        decoded = speech_tokenizer.bicodec.batch_decode(codec_items)
        waves = list(decoded["wavs"])
        indices = [int(index) for index in decoded["indices"]]
        if len(waves) != len(valid) or len(indices) != len(valid):
            raise RuntimeError(
                f"BiCodec batch returned waves={len(waves)} indices={len(indices)} expected={len(valid)}"
            )
        item_by_index = {int(item["index"]): item for item in valid}
        for index, wave in zip(indices, waves):
            item = item_by_index[index]
            output_path = Path(str(item["output_path"]))
            speech_tokenizer.save_audio(wave, output_path, sample_rate=16000)
            results[index] = (str(output_path), None)
    except Exception as exc:
        # Preserve full-corpus progress if one unusually long/corrupt sample
        # cannot participate in a padded batch; isolate it with the established
        # single-item decoder and record its exact error.
        print(
            f"BiCodec batch decode fell back to single-item decoding: {type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        for item in valid:
            results[int(item["index"])] = decode_one(item)
    return results


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
    pending = [
        (index, generated)
        for index, generated in enumerate(iter_jsonl(source_results))
        if (str(generated["id"]), str(generated["mode"])) not in completed
    ]
    pending.sort(
        key=lambda pair: max(
            len(pair[1].get("semantic_values") or []),
            len(records[str(pair[1]["id"])]["source_bicodec"]),  # type: ignore[arg-type]
            len(records[str(pair[1]["id"])]["target_bicodec"]),  # type: ignore[arg-type]
        )
    )
    for pending_batch in batched(pending, args.batch_size):
        jobs = []
        for slot, (index, generated) in enumerate(pending_batch):
            key = (str(generated["id"]), str(generated["mode"]))
            record = records.get(key[0])
            if record is None:
                raise KeyError(f"Generated id is not present in manifest: {key[0]}")
            semantic_values = generated.get("semantic_values") or []
            if not semantic_values:
                counts["no_semantic_tokens"] += 1
            name = safe_sample_name(int(generated.get("index", index)), key[0], key[1])
            jobs.append(
                {
                    "slot": slot,
                    "key": key,
                    "generated": generated,
                    "record": record,
                    "semantic_values": semantic_values,
                    "generated_path": wav_dir / f"{name}.wav",
                    "source_path": source_dir / f"{name}.wav",
                    "reference_path": reference_dir / f"{name}.wav",
                }
            )

        generated_decoded = decode_token_batch(
            speech_tokenizer=tokenizer,
            items=[
                {
                    "index": job["slot"],
                    "global_values": job["record"]["bicodec_global"],  # type: ignore[index]
                    "semantic_values": job["semantic_values"],
                    "output_path": job["generated_path"],
                }
                for job in jobs
            ],
            device=device,
        )
        source_decoded = {}
        if args.save_source_audio:
            source_decoded = decode_token_batch(
                speech_tokenizer=tokenizer,
                items=[
                    {
                        "index": job["slot"],
                        "global_values": job["record"]["bicodec_global"],  # type: ignore[index]
                        "semantic_values": job["record"]["source_bicodec"],  # type: ignore[index]
                        "output_path": job["source_path"],
                    }
                    for job in jobs
                ],
                device=device,
            )
        reference_decoded = {}
        if args.save_reference_audio:
            reference_decoded = decode_token_batch(
                speech_tokenizer=tokenizer,
                items=[
                    {
                        "index": job["slot"],
                        "global_values": job["record"]["bicodec_global"],  # type: ignore[index]
                        "semantic_values": job["record"]["target_bicodec"],  # type: ignore[index]
                        "output_path": job["reference_path"],
                    }
                    for job in jobs
                ],
                device=device,
            )

        for job in jobs:
            slot = int(job["slot"])
            generated = job["generated"]
            audio_path, error = generated_decoded[slot]
            source_path, source_error = source_decoded.get(slot, (None, None))
            reference_path, reference_error = reference_decoded.get(slot, (None, None))
            counts["source_audio"] += int(bool(source_path))
            counts["reference_audio"] += int(bool(reference_path))
            row = {
                **generated,  # type: ignore[arg-type]
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(decode_results(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
