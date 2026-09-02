"""Build the prefix-to-prefix packed pools from the gold trajectories.

What this reads and what it does not
------------------------------------
Only the 15-shard gold trajectories and the source audio paths inside them.
Nothing here needs the V1 rollouts, the teacher caches, the strata manifest or
the quality gate that ``build_task_pools`` requires, because this pool is pure
cross-entropy: no teacher KL, no commit-consistency pairs, no replay.  That is
also why it is cheap to train on -- the interleaved families' teacher KL runs a
full 180407-vocabulary log_softmax, which is what made those batches take
22-35 s against the replay family's 7 s.

It also needs no GPU.  The one number that looked like it required a frontend
pass -- how many GLM tokens a cut of the audio yields -- turned out to be a
closed form, ``ceil(samples / 1280)``, verified against the frontend on 201
event boundaries.  A full 15-shard frontend pass measured 2.1 trajectories per
second per card, so this closed form is what removes about 22 GPU-hours from
the build.

Packing
-------
``pack_task_samples`` from the base experiment does the packing unchanged; it
groups one family per packed row and already emits every acoustic field except
the audio cut.  Rather than fork 180 audited lines, this module wraps it and
injects ``source_pcm_end`` into each acoustic row, resolving the owning sample
through the row's own ``sequence_ids`` and ``sample_ordinal``.  The base
``packing.py`` is under a frozen audit and is not touched.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    pack_task_samples,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    POOL_FAMILIES,
    P2STTaskSample,
    SOURCE_PREFIX_GOLD,
    build_p2st_incremental_mt_tasks,
    build_p2st_phase3_replay_tasks,
    build_p2st_streaming_asr_tasks,
    build_p2st_streaming_tts_tasks,
)
from training.simul_uniss.jsonl_index import write_index

POOL_SCHEMA = "uniss_streaming_p2st_pool_v1"


def pack_p2st_samples(
    samples: Iterable[P2STTaskSample], *, seq_length: int
) -> Iterator[dict[str, object]]:
    """Pack with the established packer, then add the audio cut.

    ``pack_task_samples`` emits ``sequence_ids`` for the packed row and a
    ``sample_ordinal`` on every acoustic row, which together identify the
    sample an acoustic row came from, so the cut can be attached without
    re-deriving the packing.
    """
    materialised = list(samples)
    by_sequence = {sample.sequence_id: sample for sample in materialised}
    if len(by_sequence) != len(materialised):
        raise ValueError("p2st sequence ids are not unique within a pack group")
    for row in pack_task_samples(materialised, seq_length=seq_length):
        sequence_ids = row["sequence_ids"]
        if not isinstance(sequence_ids, list):
            raise TypeError("packed row is missing its sequence ids")
        for acoustic in row["acoustic_rows"]:  # type: ignore[union-attr]
            if not isinstance(acoustic, dict):
                raise TypeError("packed acoustic row is not an object")
            ordinal = int(acoustic["sample_ordinal"])
            owner = by_sequence[str(sequence_ids[ordinal])]
            if owner.source_pcm_end <= 0:
                raise ValueError(
                    f"acoustic sample {owner.sequence_id} carries no audio cut"
                )
            acoustic["source_pcm_end"] = int(owner.source_pcm_end)
        yield row


def build_trajectory_samples(
    trajectory: E2ETrajectory,
    *,
    encode_text,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
) -> dict[str, list[P2STTaskSample]]:
    """Every pool family for one trajectory, keyed by family.

    The two phase3 replay families come last and are whole-utterance rather
    than per-event, so one trajectory yields exactly one sample for each.
    """
    replay = build_p2st_phase3_replay_tasks(trajectory, encode_text=encode_text)
    by_family: dict[str, list[P2STTaskSample]] = {
        FAMILY_PHASE3_QUALITY: [],
        FAMILY_PHASE3_PERFORMANCE: [],
    }
    for sample in replay:
        by_family[sample.family].append(sample)
    return {
        FAMILY_P2ST_ASR: build_p2st_streaming_asr_tasks(
            trajectory, encode_text=encode_text
        ),
        FAMILY_P2ST_MT: build_p2st_incremental_mt_tasks(
            trajectory,
            encode_text=encode_text,
            source_prefix_kind=source_prefix_kind,
        ),
        FAMILY_P2ST_TTS: build_p2st_streaming_tts_tasks(
            trajectory, encode_text=encode_text
        ),
        FAMILY_PHASE3_QUALITY: by_family[FAMILY_PHASE3_QUALITY],
        FAMILY_PHASE3_PERFORMANCE: by_family[FAMILY_PHASE3_PERFORMANCE],
    }


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    if workers <= 0:
        raise ValueError("worker count must be positive")
    workers = min(workers, max(1, total))
    step, extra = divmod(total, workers)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(workers):
        stop = start + step + (1 if index < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


class _PartWriter:
    """Buffers samples of one family and flushes packed rows to a part file."""

    def __init__(self, path: Path, seq_length: int, flush_rows: int = 32) -> None:
        self.path = path
        self.seq_length = int(seq_length)
        # Flush by token budget, not sample count.  Every flush ends with one
        # partially filled packed row, so flushing per 512 samples left the
        # short incremental-MT sequences at 29.7% fill; a budget of 32 rows
        # caps that waste near 1/32 instead.
        self.flush_tokens = int(flush_rows) * int(seq_length)
        self.pending: list[P2STTaskSample] = []
        self.pending_tokens = 0
        self.rows = 0
        self.samples = 0
        self.supervised = 0
        self.used = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def add(self, samples: Sequence[P2STTaskSample]) -> None:
        self.pending.extend(samples)
        self.samples += len(samples)
        self.pending_tokens += sum(sample.shifted_length for sample in samples)
        if self.pending_tokens >= self.flush_tokens:
            self._flush(final=False)

    def _flush(self, *, final: bool) -> None:
        if not self.pending:
            return
        rows = list(pack_p2st_samples(self.pending, seq_length=self.seq_length))
        self.pending = []
        self.pending_tokens = 0
        for row in rows:
            self.handle.write(json.dumps(row, sort_keys=True) + "\n")
            self.rows += 1
            self.supervised += int(row.get("supervised_tokens", 0))
            self.used += int(row.get("used_tokens", 0))
        if final:
            self.handle.flush()

    def close(self) -> dict[str, object]:
        self._flush(final=True)
        self.handle.close()
        return {
            "path": str(self.path),
            "rows": self.rows,
            "samples": self.samples,
            "supervised_tokens": self.supervised,
            "used_tokens": self.used,
        }


def _worker(task: tuple[object, ...]) -> dict[str, object]:
    (gold, start, stop, tokenizer_path, parts_root, split, seq_length, index) = task
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    writers = {
        family: _PartWriter(
            Path(parts_root) / f"{split}_{family}.part{index:04d}.jsonl",
            int(seq_length),
        )
        for family in POOL_FAMILIES
    }
    trajectories = 0
    skipped: list[str] = []
    with Path(gold).open() as handle:
        for ordinal, line in enumerate(handle):
            if ordinal < int(start):
                continue
            if ordinal >= int(stop):
                break
            trajectory = E2ETrajectory.from_mapping(json.loads(line))
            try:
                built = build_trajectory_samples(trajectory, encode_text=encode)
            except ValueError as error:
                # The TTS builder raises when a reconstructed semantic prefix
                # disagrees with the recorded offset.  Record it rather than
                # dropping it silently: a rate above zero means the alignment
                # this pool rests on is not what it was measured to be.
                skipped.append(f"{trajectory.sample_id}: {error}")
                continue
            for family, samples in built.items():
                if samples:
                    writers[family].add(samples)
            trajectories += 1
    return {
        "worker": int(index),
        "range": [int(start), int(stop)],
        "trajectories": trajectories,
        "skipped": skipped[:32],
        "skipped_count": len(skipped),
        "families": {
            family: writer.close() for family, writer in writers.items()
        },
    }


def _merge_family(
    parts: Sequence[Path], output: Path, *, chunk: int = 1 << 20
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    position = 0
    with output.open("wb") as sink:
        for part in parts:
            if not part.exists():
                continue
            with part.open("rb") as source:
                trailing = b""
                while True:
                    block = source.read(chunk)
                    if not block:
                        break
                    data = trailing + block
                    trailing = b""
                    if not data.endswith(b"\n"):
                        cut = data.rfind(b"\n") + 1
                        trailing = data[cut:]
                        data = data[:cut]
                    start = 0
                    while True:
                        end = data.find(b"\n", start)
                        if end < 0:
                            break
                        offsets.append(position + start)
                        start = end + 1
                    sink.write(data)
                    position += len(data)
                if trailing:
                    raise ValueError(f"part {part} does not end on a newline")
    index = write_index(output, offsets)
    return {
        "path": str(output),
        "rows": len(offsets),
        "bytes": position,
        "index": index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="build only the first N trajectories; 0 builds all of them",
    )
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(
            f"refusing to overwrite a non-empty pool root: {args.output_root}"
        )
    total = 0
    with args.gold.open() as handle:
        for _ in handle:
            total += 1
    if args.limit:
        total = min(total, args.limit)
    bounds = _ranges(total, args.workers)
    parts_root = args.output_root / "parts"
    packed_root = args.output_root / "packed"
    started = time.time()
    print(f"trajectories={total} workers={len(bounds)} seq_length={args.seq_length}")

    tasks = [
        (
            str(args.gold),
            start,
            stop,
            str(args.tokenizer),
            str(parts_root),
            args.split,
            int(args.seq_length),
            index,
        )
        for index, (start, stop) in enumerate(bounds)
    ]
    if len(tasks) == 1:
        results = [_worker(tasks[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(tasks)) as pool:
            results = pool.map(_worker, tasks)

    manifest: dict[str, object] = {
        "schema_version": POOL_SCHEMA,
        "gold": str(args.gold.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "split": args.split,
        "seq_length": int(args.seq_length),
        "trajectories": sum(int(r["trajectories"]) for r in results),
        "skipped_count": sum(int(r["skipped_count"]) for r in results),
        "skipped_examples": [
            item for r in results for item in r["skipped"]  # type: ignore[union-attr]
        ][:32],
        "workers": results,
        "families": {},
    }
    for family in POOL_FAMILIES:
        parts = [
            parts_root / f"{args.split}_{family}.part{index:04d}.jsonl"
            for index in range(len(bounds))
        ]
        merged = _merge_family(parts, packed_root / f"{args.split}_{family}.jsonl")
        merged["samples"] = sum(
            int(r["families"][family]["samples"]) for r in results  # type: ignore[index]
        )
        merged["supervised_tokens"] = sum(
            int(r["families"][family]["supervised_tokens"]) for r in results  # type: ignore[index]
        )
        merged["used_tokens"] = sum(
            int(r["families"][family]["used_tokens"]) for r in results  # type: ignore[index]
        )
        manifest["families"][family] = merged  # type: ignore[index]
        if not args.keep_parts:
            for part in parts:
                part.unlink(missing_ok=True)
    if not args.keep_parts:
        try:
            parts_root.rmdir()
        except OSError:
            pass

    manifest["elapsed_seconds"] = round(time.time() - started, 2)
    target = args.output_root / "POOL_MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(
        f"{'family':<22s}{'rows':>8s}{'samples':>10s}"
        f"{'supervised':>13s}{'used':>13s}{'fill':>7s}"
    )
    for family, value in manifest["families"].items():  # type: ignore[union-attr]
        fill = int(value["used_tokens"]) / max(
            1, int(value["rows"]) * int(args.seq_length)
        )
        print(
            f"  {family:<20s}{value['rows']:>8d}{value['samples']:>10d}"
            f"{value['supervised_tokens']:>13d}{value['used_tokens']:>13d}"
            f"{fill:>6.1%}"
        )
    print(
        f"skipped={manifest['skipped_count']} "
        f"elapsed={manifest['elapsed_seconds']}s -> {target}"
    )


if __name__ == "__main__":
    main()
