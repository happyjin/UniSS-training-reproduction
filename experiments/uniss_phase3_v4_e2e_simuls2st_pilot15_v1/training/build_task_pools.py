#!/usr/bin/env python3
"""Parallel, immutable construction of all five 18k E2E task pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.audit_rollouts import (
    _audit_pair,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    atomic_json,
    file_sha256,
    selected_total,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.stratify_rollouts import (
    STRATUM_CLEAN,
    STRATUM_NOISY,
    STRATUM_QUARANTINE,
    validate_stratum_row,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    PACKED_TASK_SCHEMA,
    pack_task_samples,
    validate_packed_task,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    E2ETaskSample,
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_STREAMING_ASR,
    LOSS_KIND_NAMES,
    TASK_FAMILIES,
    build_incremental_mt_tasks,
    build_interleaved_task,
    build_phase3_replay_tasks,
    build_streaming_asr_task,
)
from training.simul_uniss.jsonl_index import load_index, write_index


PART_SCHEMA = "uniss_phase3_v4_e2e_task_pool_part_v1"
BUILD_SCHEMA = "uniss_phase3_v4_e2e_task_pools_v1"


def runtime_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("task_samples.py"),
        Path(__file__).with_name("packing.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(total)))
    return [
        (total * rank // workers, total * (rank + 1) // workers)
        for rank in range(workers)
    ]


class _PackedWriter:
    def __init__(self, path: Path, family: str, seq_length: int) -> None:
        self.path = path
        self.family = family
        self.seq_length = int(seq_length)
        self.handle = path.open("xb")
        self.offsets = array("Q")
        self.byte_offset = 0
        self.pending: list[E2ETaskSample] = []
        self.pending_length = 0
        self.counts: Counter[str] = Counter(
            {f"loss:{name}": 0 for name in LOSS_KIND_NAMES.values()}
        )

    def _write_pack(self) -> None:
        if not self.pending:
            return
        values = list(
            pack_task_samples(self.pending, seq_length=self.seq_length)
        )
        if len(values) != 1:
            raise AssertionError("worker-local E2E pack flush produced multiple packs")
        value = values[0]
        validate_packed_task(value, seq_length=self.seq_length)
        encoded = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.offsets.append(self.byte_offset)
        self.handle.write(encoded)
        self.byte_offset += len(encoded)
        self.counts["packed_records"] += 1
        self.counts["used_tokens"] += int(value["used_tokens"])
        self.counts["supervised_tokens"] += int(value["supervised_tokens"])
        self.counts["commit_consistency_pairs"] += len(
            value["commit_consistency"]
        )
        self.counts["commit_consistency_positions"] += sum(
            int(binding["positions"])
            for binding in value["commit_consistency"]
        )
        self.counts["teacher_bindings"] += len(value["teacher_bindings"])
        self.counts["teacher_positions"] += sum(
            int(binding["packed_stop"]) - int(binding["packed_start"])
            for binding in value["teacher_bindings"]
        )
        for binding in value["teacher_bindings"]:
            self.counts[f"teacher:{binding['cache_kind']}:bindings"] += 1
            self.counts[f"teacher:{binding['cache_kind']}:positions"] += int(
                binding["packed_stop"]
            ) - int(binding["packed_start"])
        used_tokens = int(value["used_tokens"])
        for kind in value["loss_kinds"][:used_tokens]:
            self.counts[f"loss:{LOSS_KIND_NAMES[int(kind)]}"] += 1
        self.pending = []
        self.pending_length = 0

    def add(self, sample: E2ETaskSample) -> None:
        if sample.family != self.family:
            raise ValueError("task sample was routed to the wrong family writer")
        if sample.shifted_length > self.seq_length:
            raise ValueError(
                f"E2E sample exceeds 18k pack: {sample.sequence_id}: "
                f"{sample.shifted_length}>{self.seq_length}"
            )
        if self.pending and self.pending_length + sample.shifted_length > self.seq_length:
            self._write_pack()
        self.pending.append(sample)
        self.pending_length += sample.shifted_length
        self.counts["raw_samples"] += 1

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def close(self) -> dict[str, object]:
        self._write_pack()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.counts["packed_records"] <= 0:
            raise ValueError(f"E2E task family produced no packs: {self.family}")
        index = write_index(self.path, self.offsets)
        return {
            "family": self.family,
            "path": str(self.path.resolve()),
            "records": len(self.offsets),
            "bytes": self.path.stat().st_size,
            "sha256": file_sha256(self.path),
            "counts": dict(sorted(self.counts.items())),
            "index": index,
        }


def _worker(task: tuple[object, ...]) -> dict[str, Any]:
    (
        rank,
        gold_value,
        rollout_value,
        strata_value,
        start,
        stop,
        selection_start,
        gold_total,
        tokenizer_value,
        output_value,
        seq_length,
    ) = task
    rank = int(rank)
    gold = Path(str(gold_value))
    rollouts = Path(str(rollout_value))
    strata = Path(str(strata_value))
    output_root = Path(str(output_value)) / f"part_{rank:03d}"
    output_root.mkdir(parents=True)
    gold_offsets = load_index(gold)
    rollout_offsets = load_index(rollouts)
    strata_offsets = load_index(strata)
    if gold_offsets is None or rollout_offsets is None or strata_offsets is None:
        raise ValueError("task pool worker is missing an input offset index")
    if len(strata_offsets) != len(rollout_offsets):
        raise ValueError("task pool strata/rollout record counts differ")
    rollout_is_full = len(rollout_offsets) == int(gold_total)
    rollout_base = 0 if rollout_is_full else int(selection_start)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_value), local_files_only=True
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    writers = {
        family: _PackedWriter(
            output_root / f"{family}.jsonl", family, int(seq_length)
        )
        for family in TASK_FAMILIES
    }
    records = 0
    strata_counts: Counter[str] = Counter()
    task_origins: dict[str, Counter[str]] = {
        stratum: Counter() for stratum in (STRATUM_CLEAN, STRATUM_NOISY, STRATUM_QUARANTINE)
    }
    excluded: Counter[str] = Counter()

    def add_sample(stratum: str, sample: E2ETaskSample) -> None:
        writers[sample.family].add(sample)
        values = task_origins[stratum]
        values[f"family:{sample.family}:raw_samples"] += 1
        values[f"family:{sample.family}:shifted_tokens"] += sample.shifted_length
        values[f"family:{sample.family}:supervised_tokens"] += sum(
            value != 0 for value in sample.loss_kinds
        )

    try:
        with (
            gold.open("rb") as gold_handle,
            rollouts.open("rb") as rollout_handle,
            strata.open("rb") as strata_handle,
        ):
            for record_index in range(int(start), int(stop)):
                gold_handle.seek(int(gold_offsets[record_index]))
                trajectory = E2ETrajectory.from_mapping(
                    json.loads(gold_handle.readline())
                )
                rollout_ordinal = record_index - rollout_base
                if not 0 <= rollout_ordinal < len(rollout_offsets):
                    raise ValueError("task pool rollout selection does not cover gold")
                rollout_handle.seek(int(rollout_offsets[rollout_ordinal]))
                rollout = V1Rollout.from_mapping(
                    json.loads(rollout_handle.readline())
                )
                strata_handle.seek(int(strata_offsets[rollout_ordinal]))
                stratum_row = json.loads(strata_handle.readline())
                validate_stratum_row(stratum_row)
                if (
                    int(stratum_row["rollout_ordinal"]) != rollout_ordinal
                    or int(stratum_row["source_manifest_record"]) != record_index
                    or str(stratum_row["sample_id"]) != trajectory.sample_id
                    or str(stratum_row["sample_id"]) != rollout.sample_id
                ):
                    raise ValueError("task pool stratum identity differs from gold/rollout")
                stratum = str(stratum_row["stratum"])
                _audit_pair(trajectory, rollout)
                if trajectory.source_manifest_record != record_index:
                    raise ValueError("gold source manifest record differs from JSONL index")
                strata_counts[stratum] += 1
                if stratum != STRATUM_QUARANTINE:
                    add_sample(
                        stratum,
                        build_streaming_asr_task(
                            trajectory, rollout, encode_text=encode
                        ),
                    )
                else:
                    excluded["quarantine:streaming_asr_event"] += 1
                for sample in build_incremental_mt_tasks(
                    trajectory, rollout, encode_text=encode
                ):
                    if stratum == STRATUM_QUARANTINE and not sample.sequence_id.endswith(
                        ":gold_source"
                    ):
                        excluded["quarantine:incremental_mt_event:v1_source"] += 1
                        continue
                    add_sample(stratum, sample)
                if stratum != STRATUM_QUARANTINE:
                    add_sample(
                        stratum,
                        build_interleaved_task(
                            trajectory,
                            encode_text=encode,
                            rollout=rollout,
                        ),
                    )
                else:
                    excluded["quarantine:interleaved_e2e_s2st"] += 1
                quality, performance = build_phase3_replay_tasks(
                    trajectory, encode_text=encode
                )
                add_sample(stratum, quality)
                add_sample(stratum, performance)
                records += 1
    except BaseException:
        # On an exception the incomplete part remains intentionally visible;
        # no completion marker is written and no future run may mistake it for
        # an immutable input.
        for writer in writers.values():
            writer.abort()
        raise
    family_reports = {
        family: writers[family].close() for family in TASK_FAMILIES
    }
    report = {
        "schema_version": PART_SCHEMA,
        "status": "complete",
        "rank": rank,
        "assigned_start": int(start),
        "assigned_stop": int(stop),
        "records": records,
        "seq_length": int(seq_length),
        "gold": str(gold.resolve()),
        "rollouts": str(rollouts.resolve()),
        "strata": str(strata.resolve()),
        "tokenizer": str(Path(str(tokenizer_value)).resolve()),
        "runtime_sha256": runtime_sha256(),
        "families": family_reports,
        "strata_counts": dict(sorted(strata_counts.items())),
        "task_origins": {
            name: dict(sorted(values.items()))
            for name, values in sorted(task_origins.items())
        },
        "excluded": dict(sorted(excluded.items())),
    }
    atomic_json(output_root / "PART_COMPLETE.json", report)
    return report


def _merge_family(
    family: str,
    parts: list[dict[str, object]],
    output_path: Path,
) -> dict[str, object]:
    offsets = array("Q")
    byte_base = 0
    counts: Counter[str] = Counter()
    with output_path.open("xb") as destination:
        for part in parts:
            family_report = part["families"][family]
            path = Path(str(family_report["path"]))
            if path.stat().st_size != int(family_report["bytes"]):
                raise ValueError("E2E task part byte count changed")
            if file_sha256(path) != family_report["sha256"]:
                raise ValueError("E2E task part digest changed")
            part_offsets = load_index(path)
            if part_offsets is None or len(part_offsets) != int(
                family_report["records"]
            ):
                raise ValueError("E2E task part offset index differs")
            offsets.extend(byte_base + int(value) for value in part_offsets)
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            byte_base += int(family_report["bytes"])
            counts.update(
                {
                    str(key): int(value)
                    for key, value in family_report["counts"].items()
                }
            )
        destination.flush()
        os.fsync(destination.fileno())
    index = write_index(output_path, offsets)
    return {
        "family": family,
        "schema_version": PACKED_TASK_SCHEMA,
        "path": str(output_path.resolve()),
        "records": len(offsets),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "counts": dict(sorted(counts.items())),
        "index": index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--strata-manifest", type=Path, required=True)
    parser.add_argument("--quality-gate", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--seq-length", type=int, default=18_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite E2E task pools: {output_root}")
    if args.workers <= 0 or args.seq_length != 18_000:
        raise ValueError("formal task pools require positive workers and seq-length 18000")
    gold_offsets, gold_total = selected_total(args.gold, None)
    rollout_offsets = load_index(args.rollouts)
    strata_offsets = load_index(args.strata_manifest)
    if rollout_offsets is None or strata_offsets is None:
        raise ValueError("task pool rollout or strata manifest is missing its offset index")
    if len(strata_offsets) != len(rollout_offsets):
        raise ValueError("task pool rollout and strata manifest counts differ")
    quality_gate = json.loads(args.quality_gate.read_text(encoding="utf-8"))
    if quality_gate.get("status") != "passed":
        raise ValueError("rollout quality gate did not pass")
    manifest = quality_gate.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("rollout quality gate has no manifest metadata")
    if Path(str(manifest.get("path"))).resolve() != args.strata_manifest.resolve():
        raise ValueError("rollout quality gate references a different strata manifest")
    if int(manifest.get("records", -1)) != len(strata_offsets):
        raise ValueError("rollout quality gate strata record count differs")
    if file_sha256(args.strata_manifest) != str(manifest.get("sha256")):
        raise ValueError("rollout quality strata digest differs from its gate")
    selection_start = int(args.start_index)
    total = gold_total - selection_start
    if args.limit is not None:
        total = min(total, max(0, int(args.limit)))
    rollout_is_full = len(rollout_offsets) == gold_total
    if total <= 0 or (
        rollout_is_full and len(rollout_offsets) < selection_start + total
    ) or (not rollout_is_full and len(rollout_offsets) < total):
        raise ValueError("task pool selection is empty or outside rollouts")
    output_root.mkdir(parents=True)
    parts_root = output_root / "parts"
    packed_root = output_root / "packed"
    parts_root.mkdir()
    packed_root.mkdir()
    tasks = [
        (
            rank,
            str(args.gold.resolve()),
            str(args.rollouts.resolve()),
            str(args.strata_manifest.resolve()),
            selection_start + start,
            selection_start + stop,
            selection_start,
            gold_total,
            str(args.tokenizer.resolve()),
            str(parts_root),
            args.seq_length,
        )
        for rank, (start, stop) in enumerate(_ranges(total, args.workers))
    ]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        parts = list(pool.map(_worker, tasks))
    parts.sort(key=lambda value: int(value["rank"]))
    cursor = selection_start
    invariant_keys = (
        "seq_length",
        "gold",
        "rollouts",
        "strata",
        "tokenizer",
        "runtime_sha256",
    )
    for expected_rank, part in enumerate(parts):
        if part.get("schema_version") != PART_SCHEMA or part.get("status") != "complete":
            raise ValueError("E2E task pool part is incomplete")
        if int(part["rank"]) != expected_rank or int(part["assigned_start"]) != cursor:
            raise ValueError("E2E task pool parts contain a rank/range gap")
        if parts:
            for key in invariant_keys:
                if part.get(key) != parts[0].get(key):
                    raise ValueError(f"E2E task pool part invariant differs: {key}")
        cursor = int(part["assigned_stop"])
    if cursor != selection_start + total:
        raise ValueError("E2E task pool parts do not cover the selection")
    families = {
        family: _merge_family(
            family, parts, packed_root / f"{args.split}_{family}.jsonl"
        )
        for family in TASK_FAMILIES
    }
    strata_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    task_origins: dict[str, Counter[str]] = {
        stratum: Counter() for stratum in (STRATUM_CLEAN, STRATUM_NOISY, STRATUM_QUARANTINE)
    }
    for part in parts:
        strata_counts.update(
            {str(key): int(value) for key, value in part["strata_counts"].items()}
        )
        excluded.update(
            {str(key): int(value) for key, value in part["excluded"].items()}
        )
        for stratum, raw in part["task_origins"].items():
            task_origins[str(stratum)].update(
                {str(key): int(value) for key, value in raw.items()}
            )
    if sum(strata_counts.values()) != total:
        raise ValueError("task pool strata counts do not cover the selection")
    report = {
        "schema_version": BUILD_SCHEMA,
        "status": "passed",
        "split": args.split,
        "selection_start": selection_start,
        "selection_stop": selection_start + total,
        "records": total,
        "workers": len(parts),
        "seq_length": args.seq_length,
        "gold": str(args.gold.resolve()),
        "gold_bytes": args.gold.stat().st_size,
        "rollouts": str(args.rollouts.resolve()),
        "rollout_bytes": args.rollouts.stat().st_size,
        "strata_manifest": str(args.strata_manifest.resolve()),
        "strata_bytes": args.strata_manifest.stat().st_size,
        "quality_gate": str(args.quality_gate.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "runtime_sha256": runtime_sha256(),
        "families": families,
        "strata_counts": dict(sorted(strata_counts.items())),
        "task_origins": {
            name: dict(sorted(values.items()))
            for name, values in sorted(task_origins.items())
        },
        "excluded": dict(sorted(excluded.items())),
    }
    atomic_json(output_root / "BUILD_COMPLETE.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
