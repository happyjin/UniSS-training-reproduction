"""Prepare pseudo-streaming Simul-UniSS schedules and weighted samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.simul_uniss.sample_builders import build_interleaved_sample
from training.simul_uniss.schedule import build_pseudo_schedule
from training.simul_uniss.schema import normalize_record, sha256_file


def load_text_encoder(tokenizer_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def iter_records(paths: Sequence[Path], batch_size: int = 256) -> Iterator[dict[str, object]]:
    for raw in iter_raw_records(paths, batch_size=batch_size):
        yield normalize_record(raw)


def iter_raw_records(paths: Sequence[Path], batch_size: int = 256) -> Iterator[dict[str, object]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def build_manifest(paths: Sequence[Path], args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "simul_uniss_manifest_v1",
        "created_at_unix": time.time(),
        "alignment_kind": "pseudo_proportional_token_alignment",
        "warning": "Bootstrap only: public UniST parquet has no word timestamps.",
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "chunk_ms": args.chunk_ms,
        "wait_k_chunks": args.wait_k_chunks,
        "max_phrase_tokens": args.max_phrase_tokens,
        "limit_records": args.limit_records,
        "skip_invalid_records": args.skip_invalid_records,
        "shards": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": None if args.skip_sha256 else sha256_file(path),
            }
            for path in paths
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chunk-ms", type=int, default=640)
    parser.add_argument("--wait-k-chunks", type=int, default=2)
    parser.add_argument("--max-phrase-tokens", type=int, default=16)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument(
        "--skip-invalid-records",
        action="store_true",
        help="Skip and count malformed rows instead of changing the historical fail-fast default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in args.input]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input shards: {missing}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedules_path = output_dir / "schedules.jsonl"
    samples_path = output_dir / "samples.jsonl"
    manifest_path = output_dir / "manifest.json"
    stats_path = output_dir / "stats.json"

    text_encoder = load_text_encoder(args.tokenizer)
    counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    total_events = 0
    started = time.time()
    with schedules_path.open("w", encoding="utf-8") as schedule_handle, samples_path.open(
        "w", encoding="utf-8"
    ) as sample_handle:
        for input_index, raw in enumerate(iter_raw_records(paths), start=1):
            counts["input_records"] += 1
            try:
                record = normalize_record(raw)
                schedule = build_pseudo_schedule(
                    record,
                    text_encoder,
                    chunk_ms=args.chunk_ms,
                    wait_k_chunks=args.wait_k_chunks,
                    max_phrase_tokens=args.max_phrase_tokens,
                )
                sample = build_interleaved_sample(schedule)
            except (KeyError, TypeError, ValueError) as exc:
                if not args.skip_invalid_records:
                    raise
                counts["skipped_invalid_records"] += 1
                invalid_reasons[f"{type(exc).__name__}: {exc}"] += 1
                if counts["skipped_invalid_records"] <= 10:
                    print(
                        json.dumps(
                            {
                                "skipped_invalid_record": counts["skipped_invalid_records"],
                                "input_index": input_index,
                                "id": str(raw.get("id", "")),
                                "reason": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                continue
            schedule_handle.write(json.dumps(schedule, ensure_ascii=False, separators=(",", ":")) + "\n")
            sample_handle.write(json.dumps(sample.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["records"] += 1
            events = schedule["events"]
            total_events += len(events)  # type: ignore[arg-type]
            for event in events:  # type: ignore[assignment]
                counts[str(event["action"])] += 1
            if args.progress_interval and input_index % args.progress_interval == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "input_records": input_index,
                            "records": counts["records"],
                            "skipped_invalid_records": counts["skipped_invalid_records"],
                            "input_records_per_second": input_index / elapsed,
                        }
                    ),
                    flush=True,
                )
            if args.limit_records is not None and counts["records"] >= args.limit_records:
                break

    manifest = build_manifest(paths, args)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = {
        "records": counts["records"],
        "input_records": counts["input_records"],
        "skipped_invalid_records": counts["skipped_invalid_records"],
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "events": total_events,
        "wait_events": counts["wait"],
        "write_events": counts["write"],
        "elapsed_seconds": time.time() - started,
        "schedules": str(schedules_path),
        "samples": str(samples_path),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
