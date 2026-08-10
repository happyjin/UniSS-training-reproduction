"""Indexed replay/trajectory data for the isolated full198 joint run.

The trajectory JSONL intentionally keeps cache references in compact sidecars.
This module resolves those references in DataLoader workers through a bounded
NPZ LRU, then flattens variable trajectory annotations during collation.  A
single microbatch is always homogeneous (all replay or all trajectory), which
keeps the Megatron data-parallel execution path deterministic and restartable.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs import (
    OFFSET_SCHEMA,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    PACKED_TRAJECTORY_SCHEMA,
    ROLE_ACTION,
    ROLE_KD,
    ROLE_TEXT,
)
from training import constants_uniss as c
from training.megatron_uniss_dataset import packed_json_to_megatron_item
from training.phase3_whisper_streamspeech_joint.dataset import IndexedPhase3ReplayDataset


CACHE_SCHEMA = "uniss_true_subsecond_trajectory_cache_part_v4"
TEACHER_VIEWS = ("prefix", "future_1", "future_2", "full")


def _source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _parse_cache_reference(reference: str, namespace: str) -> tuple[Path, int]:
    try:
        path_value, suffix = reference.rsplit("::", 1)
        actual_namespace, index_value = suffix.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"malformed cache reference: {reference}") from exc
    if actual_namespace != namespace:
        raise ValueError(
            f"cache reference namespace mismatch: expected {namespace}, got {actual_namespace}"
        )
    return Path(path_value), int(index_value)


class NpzBundleLRU:
    """Small per-worker cache of immutable NPZ bundle arrays."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity <= 0:
            raise ValueError("NPZ LRU capacity must be positive")
        self.capacity = int(capacity)
        self._values: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def load(self, path: Path) -> dict[str, np.ndarray]:
        path = path.resolve()
        cached = self._values.pop(path, None)
        if cached is None:
            with np.load(path, allow_pickle=False) as bundle:
                cached = {name: bundle[name].copy() for name in bundle.files}
            schema = str(cached["bundle_schema"].reshape(-1)[0])
            if schema != CACHE_SCHEMA:
                raise ValueError(f"unexpected cache schema {schema!r} in {path}")
        self._values[path] = cached
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return cached


@dataclass(frozen=True)
class TeacherTopK:
    indices: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor


def _teacher_from_bundle(bundle: Mapping[str, np.ndarray], index: int) -> TeacherTopK:
    prefix = f"request_{index}"
    indices = torch.from_numpy(bundle[f"{prefix}_indices"].astype(np.int64, copy=False))
    probabilities = torch.from_numpy(
        bundle[f"{prefix}_probabilities"].astype(np.float32, copy=False)
    )
    confidence = torch.from_numpy(
        bundle[f"{prefix}_confidence"].astype(np.float32, copy=False)
    )
    if indices.ndim != 2 or probabilities.shape != indices.shape:
        raise ValueError(f"malformed teacher top-k arrays for {prefix}")
    if confidence.shape != indices.shape[:1]:
        raise ValueError(f"malformed teacher confidence for {prefix}")
    return TeacherTopK(indices, probabilities, confidence)


def _causal_tokens_from_bundle(bundle: Mapping[str, np.ndarray], index: int) -> torch.Tensor:
    values = bundle["causal_tokens"]
    offsets = bundle["causal_token_offsets"]
    if not 0 <= index < len(offsets) - 1:
        raise IndexError(f"causal row {index} is outside bundle")
    start, end = int(offsets[index]), int(offsets[index + 1])
    result = torch.from_numpy(values[start:end].astype(np.int64, copy=False))
    if result.numel() <= 0:
        raise ValueError("causal token row is empty")
    if bool(((result < 0) | (result >= c.GLM_SEMANTIC_SIZE)).any()):
        raise ValueError("causal cache contains an invalid WhisperVQ code")
    return result


class IndexedTrajectoryDataset(Dataset[dict[str, object]]):
    """Random access to assembled 18k trajectory packs and their cache sidecars."""

    def __init__(
        self,
        path: str | Path,
        offsets: str | Path,
        *,
        seq_length: int,
        npz_lru_capacity: int = 8,
        require_complete: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        self.offset_path = Path(offsets).resolve()
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        metadata_path = self.offset_path.with_suffix(self.offset_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != OFFSET_SCHEMA:
            raise ValueError("unexpected trajectory offset schema")
        source = metadata.get("source")
        if not isinstance(source, dict) or source != _source_fingerprint(self.path):
            raise ValueError("trajectory packed source changed after indexing")
        if require_complete and int(metadata.get("records", 0)) <= 0:
            raise ValueError("formal trajectory index is incomplete")
        self.offsets = np.memmap(self.offset_path, mode="r", dtype="<u8")
        if len(self.offsets) != int(metadata["records"]):
            raise ValueError("trajectory offset count does not match metadata")
        self.seq_length = int(seq_length)
        self.npz_lru_capacity = int(npz_lru_capacity)
        self._bundle_lru: NpzBundleLRU | None = None

    @property
    def bundle_lru(self) -> NpzBundleLRU:
        if self._bundle_lru is None:
            self._bundle_lru = NpzBundleLRU(self.npz_lru_capacity)
        return self._bundle_lru

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_bundle_lru"] = None
        return value

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("schema_version") != PACKED_TRAJECTORY_SCHEMA:
            raise ValueError(f"unexpected trajectory schema at record {index}")
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        value = self._read(index)
        # The shared Phase3 adapter historically validates the JSON list as
        # integral before converting it to float32.  Trajectory packing writes
        # the equivalent 0.0/1.0 representation, so normalize only this local
        # view rather than changing the shared historical dataset code.
        megatron_value = dict(value)
        loss_mask = [float(item) for item in value["loss_mask"]]
        if any(item not in {0.0, 1.0} for item in loss_mask):
            raise ValueError("trajectory loss_mask must be binary")
        megatron_value["loss_mask"] = [int(item) for item in loss_mask]
        result: dict[str, object] = dict(
            packed_json_to_megatron_item(megatron_value, seq_length=self.seq_length)
        )
        result["sample_kind"] = "trajectory"
        result["token_roles"] = torch.tensor(value["token_roles"], dtype=torch.int64)
        result["source_ids"] = [str(item) for item in value["source_ids"]]
        result["sample_boundaries"] = torch.tensor(
            value["sample_boundaries"], dtype=torch.int64
        )

        sidecars = value.get("trajectory_sidecars")
        if not isinstance(sidecars, list) or not sidecars:
            raise ValueError("trajectory pack has no sidecars")
        annotations: list[dict[str, object]] = []
        boundaries = value["sample_boundaries"]
        token_roles = value["token_roles"]
        tokens = value["tokens"]
        if len(boundaries) != len(sidecars):
            raise ValueError("trajectory boundary/sidecar count mismatch")
        for boundary, raw_sidecar in zip(boundaries, sidecars):
            sidecar = dict(raw_sidecar)
            start, end = (int(boundary[0]), int(boundary[1]))
            action_positions = [
                position
                for position in range(start, end)
                if int(token_roles[position]) == ROLE_ACTION
            ]
            if len(action_positions) != 1:
                raise ValueError("each trajectory must have exactly one action label")

            causal_path, causal_index = _parse_cache_reference(
                str(sidecar["frontend_token_cache"]), "causal"
            )
            causal_bundle = self.bundle_lru.load(causal_path)
            causal_ids = _causal_tokens_from_bundle(causal_bundle, causal_index)
            glm_positions = [
                position
                for position in range(start, end)
                if c.GLM_SEMANTIC_OFFSET
                <= int(tokens[position])
                < c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
            ]
            if len(glm_positions) > len(causal_ids):
                raise ValueError(
                    "packed GLM prefix exceeds the cached causal row: "
                    f"{len(glm_positions)} > {len(causal_ids)}"
                )
            causal_ids = causal_ids[: len(glm_positions)]
            packed_codes = torch.tensor(
                [int(tokens[position]) - c.GLM_SEMANTIC_OFFSET for position in glm_positions],
                dtype=torch.int64,
            )
            if not torch.equal(packed_codes, causal_ids):
                raise ValueError("packed GLM prefix differs from its causal cache row")

            teachers: dict[str, TeacherTopK] = {}
            for view in TEACHER_VIEWS:
                key = "teacher_prefix_topk_path" if view == "prefix" else f"teacher_{view}_topk_path"
                teacher_path, teacher_index = _parse_cache_reference(
                    str(sidecar[key]), "teacher"
                )
                teachers[view] = _teacher_from_bundle(
                    self.bundle_lru.load(teacher_path), teacher_index
                )

            previous = int(sidecar["previous_committed_length"])
            supervised_positions = [
                position
                for position in range(start, end)
                if int(token_roles[position]) in {ROLE_TEXT, ROLE_KD}
            ]
            annotations.append(
                {
                    "sample_id": str(sidecar["sample_id"]),
                    "action_position": action_positions[0],
                    "glm_positions": torch.tensor(glm_positions, dtype=torch.int64),
                    "causal_ids": causal_ids,
                    "translation_ids": torch.tensor(
                        sidecar["translation_ids"], dtype=torch.int64
                    ),
                    "safe_commit_mask": torch.tensor(
                        sidecar["safe_commit_mask"], dtype=torch.bool
                    ),
                    "teacher": teachers,
                    "teacher_lm_positions": torch.tensor(
                        supervised_positions, dtype=torch.int64
                    ),
                    "teacher_target_indices": torch.arange(
                        previous, previous + len(supervised_positions), dtype=torch.int64
                    ),
                    "support_bucket": int(sidecar["support_bucket"]),
                    "natural_action": 1 if sidecar["natural_action_target"] == "WRITE" else 0,
                    "deadline_action": 1 if sidecar["deadline_action_target"] == "WRITE" else 0,
                    "deadline_forced": bool(sidecar["deadline_forced_target"]),
                    "chunk_end_ms": int(sidecar["chunk_end_ms"]),
                    "soft_deadline_ms": int(sidecar["soft_deadline_ms"]),
                    "hard_deadline_ms": int(sidecar["hard_deadline_ms"]),
                    "previous_committed_length": previous,
                    "stable_target_length": int(sidecar["stable_target_length"]),
                }
            )
        result["annotations"] = annotations
        return result


@dataclass(frozen=True)
class ScheduledIndex:
    sample_kind: str
    source_index: int


class DeterministicReplayTrajectorySchedule(Dataset[dict[str, object]]):
    """Restart-stable homogeneous DP groups with curriculum-aware task ratios."""

    PHASES = (
        (0.083, 0.45),
        (0.333, 0.40),
        (0.750, 0.35),
        (1.000, 0.40),
    )

    def __init__(
        self,
        trajectory: Dataset,
        replay: Dataset,
        *,
        total_samples: int,
        data_parallel_group_size: int,
    ) -> None:
        if not len(trajectory) or not len(replay):
            raise ValueError("trajectory and replay datasets must be non-empty")
        if total_samples <= 0 or data_parallel_group_size <= 0:
            raise ValueError("schedule geometry must be positive")
        self.trajectory = trajectory
        self.replay = replay
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.group_count = int(total_samples) // self.data_parallel_group_size
        if self.group_count <= 0:
            raise ValueError("schedule is smaller than one global microbatch")
        self.total_samples = self.group_count * self.data_parallel_group_size
        self.synchronize_sample_kind = True
        self.phase_group_boundaries = tuple(
            min(self.group_count, round(end * self.group_count)) for end, _ in self.PHASES
        )
        self._trajectory_cursor = np.empty(self.group_count, dtype=np.int64)
        self._replay_cursor = np.empty(self.group_count, dtype=np.int64)
        trajectory_count = replay_count = 0
        phase_start = 0
        for (phase_end_fraction, replay_fraction), phase_end in zip(
            self.PHASES, self.phase_group_boundaries
        ):
            del phase_end_fraction
            phase_groups = max(0, phase_end - phase_start)
            for local in range(phase_groups):
                group = phase_start + local
                # Midpoint thresholding is deterministic and differs by at most
                # one group from the requested replay fraction per phase.
                before = int(local * replay_fraction)
                after = int((local + 1) * replay_fraction)
                is_replay = after > before
                if is_replay:
                    self._replay_cursor[group] = replay_count
                    self._trajectory_cursor[group] = -1
                    replay_count += 1
                else:
                    self._trajectory_cursor[group] = trajectory_count
                    self._replay_cursor[group] = -1
                    trajectory_count += 1
            phase_start = phase_end
        self.replay_groups = replay_count
        self.trajectory_groups = trajectory_count

    def __len__(self) -> int:
        return self.total_samples

    def scheduled_index(self, index: int) -> ScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, lane = divmod(index, self.data_parallel_group_size)
        replay_cursor = int(self._replay_cursor[group])
        if replay_cursor >= 0:
            source = (replay_cursor * self.data_parallel_group_size + lane) % len(self.replay)
            return ScheduledIndex("replay", source)
        trajectory_cursor = int(self._trajectory_cursor[group])
        source = (
            trajectory_cursor * self.data_parallel_group_size + lane
        ) % len(self.trajectory)
        return ScheduledIndex("trajectory", source)

    def __getitem__(self, index: int) -> dict[str, object]:
        scheduled = self.scheduled_index(index)
        source = self.replay if scheduled.sample_kind == "replay" else self.trajectory
        value = dict(source[scheduled.source_index])
        if value.get("sample_kind") != scheduled.sample_kind:
            raise ValueError("scheduled kind disagrees with source dataset")
        return value


class CurriculumKindRandomSampler:
    """Shuffle DP groups within, but never across, curriculum phases."""

    def __init__(
        self,
        dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        data_sharding: bool,
    ) -> None:
        del data_sharding
        if not getattr(dataset, "synchronize_sample_kind", False):
            raise ValueError("dataset does not expose synchronized task groups")
        self.dataset = dataset
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_size = int(data_parallel_size)
        self.micro_batch_size = int(micro_batch_size)
        self.global_microbatch_size = self.data_parallel_size * self.micro_batch_size
        if self.global_microbatch_size != int(dataset.data_parallel_group_size):
            raise ValueError("dataset group size does not match DP microbatch geometry")
        self.active_total_samples = min(int(total_samples), len(dataset))
        self.active_total_samples -= self.active_total_samples % self.global_microbatch_size
        self.consumed_samples = int(consumed_samples)
        if self.consumed_samples % self.global_microbatch_size:
            raise ValueError("consumed samples must end on a DP group boundary")

    def __len__(self) -> int:
        return self.active_total_samples

    def __iter__(self):
        active_groups = self.active_total_samples // self.global_microbatch_size
        consumed_groups = (self.consumed_samples % self.active_total_samples) // self.global_microbatch_size
        epoch = self.consumed_samples // self.active_total_samples
        boundaries = [
            min(active_groups, int(value)) for value in self.dataset.phase_group_boundaries
        ]
        ordered: list[int] = []
        start = 0
        for phase_index, end in enumerate(boundaries):
            if end <= start:
                continue
            generator = torch.Generator().manual_seed(epoch * 1009 + phase_index)
            permutation = torch.randperm(end - start, generator=generator).tolist()
            ordered.extend(start + value for value in permutation)
            start = end
        for group in ordered[consumed_groups:]:
            group_start = group * self.global_microbatch_size
            rank_start = group_start + self.data_parallel_rank * self.micro_batch_size
            self.consumed_samples += self.global_microbatch_size
            yield list(range(rank_start, rank_start + self.micro_batch_size))


def _stable_group_id(sample_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), "little")


def collate_trajectory(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    if not batch or any(value.get("sample_kind") != "trajectory" for value in batch):
        raise ValueError("collate_trajectory accepts only trajectory records")
    tensor_keys = (
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "cu_seqlens",
        "max_seqlen",
        "token_roles",
    )
    result: dict[str, object] = {
        key: default_collate([value[key] for value in batch]) for key in tensor_keys
    }
    result["sample_kind"] = "trajectory"

    annotations = [
        (batch_row, annotation)
        for batch_row, value in enumerate(batch)
        for annotation in value["annotations"]  # type: ignore[index]
    ]
    count = len(annotations)
    if count <= 0:
        raise ValueError("trajectory microbatch contains no annotations")
    max_translation = max(len(value["translation_ids"]) for _, value in annotations)
    max_teacher = max(
        value["teacher"][view].indices.shape[0]  # type: ignore[index]
        for _, value in annotations
        for view in TEACHER_VIEWS
    )
    topk = max(
        value["teacher"][view].indices.shape[1]  # type: ignore[index]
        for _, value in annotations
        for view in TEACHER_VIEWS
    )
    teacher_indices = torch.zeros(count, len(TEACHER_VIEWS), max_teacher, topk, dtype=torch.long)
    teacher_probabilities = torch.zeros(
        count, len(TEACHER_VIEWS), max_teacher, topk, dtype=torch.float32
    )
    teacher_confidence = torch.zeros(count, len(TEACHER_VIEWS), max_teacher)
    teacher_mask = torch.zeros(count, len(TEACHER_VIEWS), max_teacher, dtype=torch.bool)
    translation_ids = torch.full((count, max_translation), c.TOKEN_PAD, dtype=torch.long)
    translation_mask = torch.zeros(count, max_translation, dtype=torch.bool)
    safe_targets = torch.zeros(count, max_translation, dtype=torch.float32)

    action_batch = torch.empty(count, dtype=torch.long)
    action_position = torch.empty(count, dtype=torch.long)
    support_bucket = torch.empty(count, dtype=torch.long)
    natural_action = torch.empty(count, dtype=torch.long)
    deadline_action = torch.empty(count, dtype=torch.long)
    deadline_forced = torch.empty(count, dtype=torch.bool)
    chunk_end_ms = torch.empty(count, dtype=torch.long)
    soft_deadline_ms = torch.empty(count, dtype=torch.long)
    hard_deadline_ms = torch.empty(count, dtype=torch.long)
    previous_committed = torch.empty(count, dtype=torch.long)
    stable_target = torch.empty(count, dtype=torch.long)
    sample_group = torch.empty(count, dtype=torch.long)
    frontend_batch: list[int] = []
    frontend_position: list[int] = []
    frontend_code: list[int] = []
    kd_batch: list[int] = []
    kd_position: list[int] = []
    kd_annotation: list[int] = []
    kd_target_index: list[int] = []

    for annotation_index, (batch_row, value) in enumerate(annotations):
        action_batch[annotation_index] = batch_row
        action_position[annotation_index] = int(value["action_position"])
        support_bucket[annotation_index] = int(value["support_bucket"])
        natural_action[annotation_index] = int(value["natural_action"])
        deadline_action[annotation_index] = int(value["deadline_action"])
        deadline_forced[annotation_index] = bool(value["deadline_forced"])
        chunk_end_ms[annotation_index] = int(value["chunk_end_ms"])
        soft_deadline_ms[annotation_index] = int(value["soft_deadline_ms"])
        hard_deadline_ms[annotation_index] = int(value["hard_deadline_ms"])
        previous_committed[annotation_index] = int(value["previous_committed_length"])
        stable_target[annotation_index] = int(value["stable_target_length"])
        sample_group[annotation_index] = _stable_group_id(str(value["sample_id"])) & ((1 << 63) - 1)

        ids = value["translation_ids"]
        safe = value["safe_commit_mask"]
        translation_ids[annotation_index, : len(ids)] = ids
        translation_mask[annotation_index, : len(ids)] = True
        safe_targets[annotation_index, : len(safe)] = safe.float()
        for view_index, view in enumerate(TEACHER_VIEWS):
            teacher = value["teacher"][view]
            length = teacher.indices.shape[0]
            width = teacher.indices.shape[1]
            teacher_indices[annotation_index, view_index, :length, :width] = teacher.indices
            teacher_probabilities[annotation_index, view_index, :length, :width] = teacher.probabilities
            teacher_confidence[annotation_index, view_index, :length] = teacher.confidence
            teacher_mask[annotation_index, view_index, :length] = True

        positions = value["glm_positions"].tolist()
        codes = value["causal_ids"].tolist()
        frontend_batch.extend([batch_row] * len(positions))
        frontend_position.extend(positions)
        frontend_code.extend(codes)
        lm_positions = value["teacher_lm_positions"].tolist()
        target_indices = value["teacher_target_indices"].tolist()
        kd_batch.extend([batch_row] * len(lm_positions))
        kd_position.extend(lm_positions)
        kd_annotation.extend([annotation_index] * len(lm_positions))
        kd_target_index.extend(target_indices)

    result.update(
        {
            "action_batch": action_batch,
            "action_position": action_position,
            "support_bucket": support_bucket,
            "natural_action": natural_action,
            "deadline_action": deadline_action,
            "deadline_forced": deadline_forced,
            "chunk_end_ms": chunk_end_ms,
            "soft_deadline_ms": soft_deadline_ms,
            "hard_deadline_ms": hard_deadline_ms,
            "previous_committed_length": previous_committed,
            "stable_target_length": stable_target,
            "sample_group": sample_group,
            "translation_ids": translation_ids,
            "translation_mask": translation_mask,
            "safe_commit_targets": safe_targets,
            "teacher_indices": teacher_indices,
            "teacher_probabilities": teacher_probabilities,
            "teacher_confidence": teacher_confidence,
            "teacher_mask": teacher_mask,
            "frontend_batch": torch.tensor(frontend_batch, dtype=torch.long),
            "frontend_position": torch.tensor(frontend_position, dtype=torch.long),
            "frontend_code": torch.tensor(frontend_code, dtype=torch.long),
            "kd_batch": torch.tensor(kd_batch, dtype=torch.long),
            "kd_position": torch.tensor(kd_position, dtype=torch.long),
            "kd_annotation": torch.tensor(kd_annotation, dtype=torch.long),
            "kd_target_index": torch.tensor(kd_target_index, dtype=torch.long),
        }
    )
    return result


def collate_replay_or_trajectory(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    kinds = {str(value.get("sample_kind")) for value in batch}
    if kinds == {"trajectory"}:
        return collate_trajectory(batch)
    if kinds == {"replay"}:
        return default_collate(batch)
    raise ValueError(f"one microbatch cannot mix sample kinds: {sorted(kinds)}")


__all__ = [
    "CurriculumKindRandomSampler",
    "DeterministicReplayTrajectorySchedule",
    "IndexedPhase3ReplayDataset",
    "IndexedTrajectoryDataset",
    "NpzBundleLRU",
    "TeacherTopK",
    "collate_replay_or_trajectory",
    "collate_trajectory",
]
