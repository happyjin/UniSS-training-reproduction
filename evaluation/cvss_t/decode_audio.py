"""Decode UniSS outputs while retaining official CVSS source/reference audio."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from evaluation.decode_audio import decode_token_batch
from evaluation.io_utils import iter_jsonl, write_json
from evaluation.cvss_t.records import cvss_record_map
from training.generate_unist_eval_audio import audio_duration_seconds, safe_sample_name, write_jsonl_row


def batched(values: Iterable[object], size: int) -> Iterator[list[object]]:
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def decode_results(args: argparse.Namespace) -> dict[str, int]:
    from uniss import UniSSTokenizer

    generation_path = Path(args.input)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    results_path = output_dir / "results.jsonl"
    wav_dir = output_dir / "wav"
    if output_dir.exists() and not args.resume:
        protected = (results_path, output_dir / "summary.json", wav_dir)
        if any(path.exists() for path in protected):
            raise FileExistsError(f"Refusing to reuse CVSS output without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(exist_ok=True)

    completed: set[tuple[str, str]] = set()
    if args.resume and results_path.is_file():
        completed = {(str(row["id"]), str(row["mode"])) for row in iter_jsonl(results_path)}
    records = cvss_record_map(manifest_path)
    pending = [
        row
        for row in iter_jsonl(generation_path)
        if (str(row["id"]), str(row["mode"])) not in completed
    ]
    pending.sort(key=lambda row: len(row.get("semantic_values") or []))
    device = torch.device(args.device)
    tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    counts = {
        "decoded": len(completed),
        "failed": 0,
        "source_audio": len(completed),
        "reference_audio": len(completed),
        "no_semantic_tokens": 0,
    }
    for batch in batched(pending, args.batch_size):
        jobs: list[dict[str, object]] = []
        for slot, generated in enumerate(batch):
            sample_id = str(generated["id"])
            mode = str(generated["mode"])
            record = records.get(sample_id)
            if record is None:
                raise KeyError(f"Generated CVSS ID is absent from manifest: {sample_id}")
            output_path = wav_dir / f"{safe_sample_name(int(generated.get('index', slot)), sample_id, mode)}.wav"
            jobs.append(
                {
                    "slot": slot,
                    "generated": generated,
                    "record": record,
                    "output_path": output_path,
                }
            )
        decoded = decode_token_batch(
            speech_tokenizer=tokenizer,
            items=[
                {
                    "index": job["slot"],
                    "global_values": job["record"]["bicodec_global"],  # type: ignore[index]
                    "semantic_values": job["generated"].get("semantic_values") or [],  # type: ignore[union-attr]
                    "output_path": job["output_path"],
                }
                for job in jobs
            ],
            device=device,
        )
        for job in jobs:
            slot = int(job["slot"])
            generated = job["generated"]
            record = job["record"]
            audio_path, error = decoded[slot]
            source_path = str(record["source_audio_path"])
            reference_path = str(record["reference_audio_path"])
            row = {
                **generated,  # type: ignore[arg-type]
                "audio_path": audio_path,
                "audio_duration_seconds": audio_duration_seconds(audio_path),
                "source_audio_path": source_path,
                "source_audio_duration_seconds": audio_duration_seconds(source_path),
                "source_audio_error": None,
                "reference_audio_path": reference_path,
                "reference_audio_duration_seconds": audio_duration_seconds(reference_path),
                "reference_audio_error": None,
                "synthetic_source": bool(record.get("synthetic_source")),  # type: ignore[union-attr]
                "synthetic_reference": bool(record.get("synthetic_reference")),  # type: ignore[union-attr]
                "error": error,
            }
            write_jsonl_row(results_path, row)
            counts["decoded"] += 1
            counts["source_audio"] += 1
            counts["reference_audio"] += 1
            counts["no_semantic_tokens"] += int(not (generated.get("semantic_values") or []))  # type: ignore[union-attr]
            counts["failed"] += int(bool(error))
        write_json(output_dir / "summary.json", counts)
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--speech-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(decode_results(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
