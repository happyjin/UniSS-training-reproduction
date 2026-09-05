"""Build a fixed-chunk pool: the same trajectories, read on a clock.

This is ``build_p2st_pools`` with one thing changed -- which builder makes the
three streaming families -- and everything else imported from it rather than
copied.  The packer, the part writer, the shard merge and the jsonl index are
under audit and have produced every pool in this lineage, so re-typing them
here would be the way to introduce a difference nobody asked for.

What differs from the pure-CE pool
----------------------------------
* ``FAMILY_P2ST_ASR``, ``FAMILY_P2ST_MT`` and ``FAMILY_P2ST_TTS`` come from
  ``uniform_chunk_tasks``: one sample per tick of a fixed ``--chunk-ms``
  clock, with the silent ticks supervised as "emit the terminator now".
* ``FAMILY_PHASE3_QUALITY`` and ``FAMILY_PHASE3_PERFORMANCE`` are taken
  unchanged from ``task_samples_p2st.build_p2st_phase3_replay_tasks``.  They
  are whole-utterance and carry no read schedule, so the clock does not apply
  to them, and holding them fixed is what makes the two pools comparable: any
  measured difference belongs to the streaming families.

The manifest records ``chunk_ms``, ``idle_ratio`` and ``tts_idle`` beside the
usual fields, and additionally counts how many samples of each family are
IDLE, so the pool's own composition can be checked against the 0.508/0.299
rates the chunk size was chosen from rather than assumed to match.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.build_p2st_pools import (
    POOL_SCHEMA,
    _merge_family,
    _PartWriter,
    _ranges,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    POOL_FAMILIES,
    SOURCE_PREFIX_GOLD,
    build_p2st_phase3_replay_tasks,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_NONE,
)
from experiments.uniss_streaming_p2st_traj_v1.data.uniform_chunk_tasks import (
    DEFAULT_CHUNK_MS,
    DEFAULT_IDLE_RATIO,
    build_uniform_chunk_samples,
)

TRAJ_POOL_SCHEMA = "uniss_streaming_p2st_traj_pool_v1"


def _is_idle(sample) -> bool:
    """True when the sample's whole target is its terminator.

    The IDLE form carries no content loss at all, so its supervised span is
    exactly ``[terminator, EOS]``; anything else has at least one ASR, MT or
    semantic position.
    """
    return all(
        kind in (LOSS_NONE, LOSS_BOUNDARY, LOSS_EOS) for kind in sample.loss_kinds
    )


def build_trajectory_samples(
    trajectory: E2ETrajectory,
    *,
    encode_text,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    idle_ratio: float = DEFAULT_IDLE_RATIO,
    tts_idle: bool = False,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
) -> dict[str, list]:
    """Every pool family for one trajectory, keyed by family."""
    built = build_uniform_chunk_samples(
        trajectory,
        encode_text=encode_text,
        chunk_ms=chunk_ms,
        idle_ratio=idle_ratio,
        source_prefix_kind=source_prefix_kind,
        tts_idle=tts_idle,
    )
    replay = build_p2st_phase3_replay_tasks(trajectory, encode_text=encode_text)
    built[FAMILY_PHASE3_QUALITY] = [
        s for s in replay if s.family == FAMILY_PHASE3_QUALITY
    ]
    built[FAMILY_PHASE3_PERFORMANCE] = [
        s for s in replay if s.family == FAMILY_PHASE3_PERFORMANCE
    ]
    return built


def _worker(task: tuple[object, ...]) -> dict[str, object]:
    (
        gold,
        start,
        stop,
        tokenizer_path,
        parts_root,
        split,
        seq_length,
        index,
        chunk_ms,
        idle_ratio,
        tts_idle,
    ) = task
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
    idle_counts = {family: 0 for family in POOL_FAMILIES}
    skipped: list[str] = []
    with Path(gold).open() as handle:
        for ordinal, line in enumerate(handle):
            if ordinal < int(start):
                continue
            if ordinal >= int(stop):
                break
            trajectory = E2ETrajectory.from_mapping(json.loads(line))
            try:
                built = build_trajectory_samples(
                    trajectory,
                    encode_text=encode,
                    chunk_ms=int(chunk_ms),
                    idle_ratio=float(idle_ratio),
                    tts_idle=bool(tts_idle),
                )
            except ValueError as error:
                # The TTS builder raises when a reconstructed semantic prefix
                # disagrees with the recorded offset, or when a merged span is
                # not contiguous.  Record it rather than dropping it silently:
                # a rate above zero means the alignment this pool rests on is
                # not what it was measured to be.
                skipped.append(f"{trajectory.sample_id}: {error}")
                continue
            for family, samples in built.items():
                if samples:
                    writers[family].add(samples)
                    idle_counts[family] += sum(1 for s in samples if _is_idle(s))
            trajectories += 1
    return {
        "worker": int(index),
        "range": [int(start), int(stop)],
        "trajectories": trajectories,
        "skipped": skipped[:32],
        "skipped_count": len(skipped),
        "idle_samples": idle_counts,
        "families": {
            family: writer.close() for family, writer in writers.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seq-length", type=int, default=18_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2)
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="fixed read clock in milliseconds; 640 is four frontend blocks",
    )
    parser.add_argument(
        "--idle-ratio",
        type=float,
        default=DEFAULT_IDLE_RATIO,
        help="cap on IDLE samples per content sample, per family, per utterance",
    )
    parser.add_argument(
        "--tts-idle",
        action="store_true",
        help="also teach the TTS family to speak nothing on an empty chunk",
    )
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
    print(
        f"trajectories={total} workers={len(bounds)} "
        f"seq_length={args.seq_length} chunk_ms={args.chunk_ms} "
        f"idle_ratio={args.idle_ratio} tts_idle={args.tts_idle}"
    )

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
            int(args.chunk_ms),
            float(args.idle_ratio),
            bool(args.tts_idle),
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
        "schema_version": TRAJ_POOL_SCHEMA,
        "base_schema_version": POOL_SCHEMA,
        "gold": str(args.gold.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "split": args.split,
        "seq_length": int(args.seq_length),
        "chunk_ms": int(args.chunk_ms),
        "idle_ratio": float(args.idle_ratio),
        "tts_idle": bool(args.tts_idle),
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
        merged["idle_samples"] = sum(
            int(r["idle_samples"][family]) for r in results  # type: ignore[index]
        )
        merged["idle_fraction"] = round(
            int(merged["idle_samples"]) / max(1, int(merged["samples"])), 4
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
        f"{'supervised':>13s}{'used':>13s}{'fill':>7s}{'idle':>7s}"
    )
    for family, value in manifest["families"].items():  # type: ignore[union-attr]
        fill = int(value["used_tokens"]) / max(
            1, int(value["rows"]) * int(args.seq_length)
        )
        print(
            f"  {family:<20s}{value['rows']:>8d}{value['samples']:>10d}"
            f"{value['supervised_tokens']:>13d}{value['used_tokens']:>13d}"
            f"{fill:>6.1%}{value['idle_fraction']:>7.3f}"
        )
    print(
        f"skipped={manifest['skipped_count']} "
        f"elapsed={manifest['elapsed_seconds']}s -> {target}"
    )


if __name__ == "__main__":
    main()
