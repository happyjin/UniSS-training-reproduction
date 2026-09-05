#!/usr/bin/env python3
"""Score every gold trajectory by the paper's normalized inversion rate.

SimulS2ST-Omni's ablation (paper section 4.5) makes this the single strongest
lever it reports: *"Without monotonicity filtering, low-latency (m1) performance
collapses entirely, plummeting to 4.59 and 3.56 BLEU in the two directions...
Introducing NIR-based filtering immediately recovers this gap, proving that
high-quality, difficulty-controlled trajectory supervision is the primary driver
of low-latency robustness."*  It then adds that *"curating a stable, filtered
data pool is far more critical than the specific multiplier sampling schedule."*

Appendix A.3 defines the statistic.  Let the source positions aligned to target
words form ``A = [a_1, a_2, ...]`` and let ``I`` be that sequence's inversion
number; then

    NIR = 2I / (|A| (|A| - 1)) * 100%

Lower means weaker reordering and therefore more stable read/write supervision.

WHY THIS IS COMPUTABLE WITHOUT RE-ALIGNING ANYTHING
---------------------------------------------------
The raw cross-lingual alignment is already on disk.  Stage-A's
``formal_accepted_manifest.jsonl`` carries, per target word, a ``target_support``
entry holding ``source_links[].source_index`` produced by the project's own
``neural_mutual_nearest`` aligner, alongside ``raw_support_end_ms`` (before
monotonisation) and ``support_end_ms`` (after).  The gold trajectory carries
``source_manifest`` and ``source_manifest_record``, so the join is a seek by
record index through the existing offset sidecar rather than a scan -- and no
new alignment run, and no new package, is needed.

The gold trajectory's own event list cannot be used for this: its events are
already monotone by construction (``source_glm_start`` is non-decreasing), so
the inversions the statistic is about have been destroyed by the time the
trajectory exists.  That is precisely why the Stage-A join is required.

Measured on a 3000-record sample before writing this: our pool has mean NIR
11.64%, median 8.89%, p90 25.49%, with 15.0% perfectly monotone -- between the
paper's random selection at 9.66% and its NIR-stratified pool at 13.79%.  So the
difficulty level is not itself wrong; what is missing is any control over its
distribution, which is what ``nir_stratify`` then applies.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index


def inversion_count(values: list[int]) -> int:
    """Inversions of ``values`` in O(n log n) via merge sort.

    A quadratic count would be tolerable at the ~20 target words a sentence
    usually holds, but the pool also contains long-form rows and this runs over
    1.3M trajectories, so the merge sort keeps the worst case bounded.
    """
    buffer = list(values)
    work = [0] * len(values)
    total = 0
    width = 1
    while width < len(buffer):
        for start in range(0, len(buffer), 2 * width):
            middle = min(start + width, len(buffer))
            end = min(start + 2 * width, len(buffer))
            left, right, out = start, middle, start
            while left < middle and right < end:
                if buffer[left] <= buffer[right]:
                    work[out] = buffer[left]
                    left += 1
                else:
                    # every remaining element of the left run is an inversion
                    # with this right-hand element
                    total += middle - left
                    work[out] = buffer[right]
                    right += 1
                out += 1
            while left < middle:
                work[out] = buffer[left]
                left += 1
                out += 1
            while right < end:
                work[out] = buffer[right]
                right += 1
                out += 1
        buffer, work = work, buffer
        width *= 2
    return total


def normalized_inversion_rate(positions: list[int]) -> float | None:
    """Appendix A.3's NIR, as a percentage.  ``None`` when it is undefined."""
    n = len(positions)
    if n < 2:
        return None
    return 200.0 * inversion_count(positions) / (n * (n - 1))


def score_record(gold: dict, stage_a: dict) -> dict:
    """One scored row.  Raises if the join is wrong rather than scoring garbage."""
    sample_id = str(gold["sample_id"])
    if str(stage_a.get("id")) != sample_id:
        raise ValueError(
            f"stage-A join mismatch: trajectory {sample_id} resolved to "
            f"{stage_a.get('id')!r} at record {gold.get('source_manifest_record')}"
        )
    support = stage_a.get("target_support") or []
    positions: list[int] = []
    confidences: list[float] = []
    shifts: list[int] = []
    for entry in support:
        links = entry.get("source_links") or []
        if not links:
            continue
        positions.append(int(links[0]["source_index"]))
        confidences.append(float(links[0].get("confidence") or 0.0))
        raw = entry.get("raw_support_end_ms")
        if raw is not None:
            shifts.append(abs(int(entry.get("support_end_ms", raw)) - int(raw)))
    nir = normalized_inversion_rate(positions)
    return {
        "sample_id": sample_id,
        "src_lang": str(gold["src_lang"]),
        "tgt_lang": str(gold["tgt_lang"]),
        "direction": "en2zh" if str(gold["src_lang"]) == "eng" else "zh2en",
        "source_duration_ms": int(gold["source_duration_ms"]),
        "target_words": len(positions),
        "aligned_words": len(support),
        # None when fewer than two aligned words leave the statistic undefined;
        # nir_stratify buckets those separately rather than assuming 0.
        "nir": nir,
        "alignment_confidence_mean": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        # How far monotonisation had to move the boundaries.  Large values mark
        # rows whose read/write schedule was reshaped the most, which is the
        # reordering the paper says hurts the low-latency tier.
        "monotonisation_shift_max_ms": max(shifts) if shifts else 0,
        "monotonisation_shift_mean_ms": (
            sum(shifts) / len(shifts) if shifts else 0.0
        ),
    }


def _shard(task: tuple[str, str, int, int, str]) -> tuple[str, int, int]:
    gold_path, output_path, index, total, manifest_hint = task
    gold = Path(gold_path)
    gold_offsets = load_index(gold)
    if gold_offsets is None:
        raise SystemExit(f"missing offset sidecar for {gold}")
    manifests: dict[str, tuple[Path, list[int]]] = {}
    written = 0
    skipped = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gold.open("rb") as gold_handle, out.open("w", encoding="utf-8") as sink:
        for position in range(index, len(gold_offsets), total):
            gold_handle.seek(int(gold_offsets[position]))
            record = json.loads(gold_handle.readline())
            manifest = str(record.get("source_manifest") or manifest_hint)
            if manifest not in manifests:
                path = Path(manifest)
                offsets = load_index(path)
                if offsets is None:
                    raise SystemExit(f"missing offset sidecar for {path}")
                manifests[manifest] = (path, list(offsets))
            path, offsets = manifests[manifest]
            row_index = int(record["source_manifest_record"])
            if not 0 <= row_index < len(offsets):
                raise ValueError(
                    f"{record['sample_id']}: source_manifest_record {row_index} "
                    f"is outside {path} ({len(offsets)} records)"
                )
            with path.open("rb") as handle:
                handle.seek(int(offsets[row_index]))
                stage_a = json.loads(handle.readline())
            scored = score_record(record, stage_a)
            if scored["nir"] is None:
                skipped += 1
            sink.write(json.dumps(scored, ensure_ascii=False) + "\n")
            written += 1
    return output_path, written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="gold trajectories jsonl")
    parser.add_argument("--output", required=True, help="NIR_SCORES.jsonl to write")
    parser.add_argument(
        "--stage-a-manifest",
        default="",
        help="fallback Stage-A manifest when a trajectory omits source_manifest",
    )
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    parts = [
        (
            args.gold,
            str(output.with_suffix(f".part{index:03d}")),
            index,
            workers,
            args.stage_a_manifest,
        )
        for index in range(workers)
    ]
    written = skipped = 0
    with Pool(workers) as pool:
        for path, count, undefined in pool.imap_unordered(_shard, parts):
            written += count
            skipped += undefined
    with output.open("w", encoding="utf-8") as sink:
        for index in range(workers):
            part = output.with_suffix(f".part{index:03d}")
            with part.open(encoding="utf-8") as handle:
                for line in handle:
                    sink.write(line)
            part.unlink()
    print(
        f"scored {written} trajectories -> {output} "
        f"({skipped} had fewer than two aligned target words, NIR undefined)"
    )


if __name__ == "__main__":
    main()
