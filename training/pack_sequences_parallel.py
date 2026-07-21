"""Pack UniSS JSONL samples with ordered multiprocessing.

The logical concatenation of all input files is divided into contiguous byte
ranges. Each worker aligns its range to JSONL line boundaries, packs its own
range, and writes a part file. Part files are concatenated in range order, so
sample order is deterministic and every input record is emitted exactly once.

Workers intentionally start a fresh packed sequence at range boundaries. This
can add at most ``workers - 1`` padded records compared with the single-process
packer, but it does not change sample contents, loss masks, or sample order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pack_sequences import (
    make_shifted_sample,
    pack_shifted_samples,
    write_packed_jsonl,
)


@dataclass(frozen=True)
class Segment:
    path: Path
    start: int
    end: int


@dataclass(frozen=True)
class WorkItem:
    index: int
    segments: tuple[Segment, ...]
    output: Path
    seq_length: int
    drop_overlong: bool


def build_ordered_chunks(paths: Sequence[Path], workers: int) -> list[tuple[Segment, ...]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    spans: list[tuple[Path, int, int]] = []
    cursor = 0
    for path in paths:
        size = path.stat().st_size
        if size:
            spans.append((path, cursor, cursor + size))
            cursor += size
    if cursor == 0:
        raise ValueError("all input files are empty")

    chunk_count = min(workers, cursor)
    chunks: list[tuple[Segment, ...]] = []
    for index in range(chunk_count):
        global_start = cursor * index // chunk_count
        global_end = cursor * (index + 1) // chunk_count
        segments: list[Segment] = []
        for path, file_start, file_end in spans:
            overlap_start = max(global_start, file_start)
            overlap_end = min(global_end, file_end)
            if overlap_start < overlap_end:
                segments.append(
                    Segment(
                        path=path,
                        start=overlap_start - file_start,
                        end=overlap_end - file_start,
                    )
                )
        chunks.append(tuple(segments))
    return chunks


def iter_segment_samples(segment: Segment) -> Iterator[dict[str, object]]:
    with segment.path.open("rb") as handle:
        if segment.start:
            handle.seek(segment.start - 1)
            previous = handle.read(1)
            handle.seek(segment.start)
            if previous != b"\n":
                handle.readline()
        else:
            handle.seek(0)

        while True:
            line_start = handle.tell()
            if line_start >= segment.end:
                break
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {segment.path} byte offset {line_start}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"JSON record at {segment.path} byte offset {line_start} must be an object"
                )
            yield item


def _pack_work_item(item: WorkItem) -> dict[str, object]:
    raw_samples = (
        sample for segment in item.segments for sample in iter_segment_samples(segment)
    )
    shifted = (make_shifted_sample(sample) for sample in raw_samples)
    packed = pack_shifted_samples(
        shifted,
        seq_length=item.seq_length,
        drop_overlong=item.drop_overlong,
    )
    count = write_packed_jsonl(packed, item.output)
    return {
        "index": item.index,
        "output": str(item.output),
        "packed_sequences": count,
        "bytes": item.output.stat().st_size,
    }


def parallel_pack(
    paths: Sequence[Path],
    output: Path,
    seq_length: int,
    workers: int,
    drop_overlong: bool = False,
    work_dir: Path | None = None,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one input path is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    chunks = build_ordered_chunks(paths, workers=workers)
    parts_dir = work_dir or output.with_name(f"{output.name}.parts")
    if parts_dir.exists():
        raise FileExistsError(f"parallel packing work directory already exists: {parts_dir}")
    parts_dir.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = [
        WorkItem(
            index=index,
            segments=segments,
            output=parts_dir / f"part-{index:05d}.jsonl",
            seq_length=seq_length,
            drop_overlong=drop_overlong,
        )
        for index, segments in enumerate(chunks)
    ]

    results: list[dict[str, object]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=len(items),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_pack_work_item, item) for item in items]
            for future in futures:
                result = future.result()
                results.append(result)
                print(
                    json.dumps({"parallel_chunk_complete": result}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )

        with output.open("wb") as destination:
            for item in items:
                with item.output.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

        total_count = sum(int(result["packed_sequences"]) for result in results)
        report = {
            "output": str(output),
            "packed_sequences": total_count,
            "workers": len(items),
            "part_bytes": sum(int(result["bytes"]) for result in results),
            "output_bytes": output.stat().st_size,
            "boundary_padding_records_at_most": max(0, len(items) - 1),
        }

        for item in items:
            item.output.unlink()
        parts_dir.rmdir()
        return report
    except BaseException:
        print(
            f"Parallel packing failed; preserving work directory: {parts_dir}",
            file=sys.stderr,
            flush=True,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--drop-overlong", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = parallel_pack(
        paths=args.input,
        output=args.output,
        seq_length=args.seq_length,
        workers=args.workers,
        drop_overlong=args.drop_overlong,
        work_dir=args.work_dir,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
