"""Indexed dense packs and restart-exact three-coverage-epoch global shuffle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
    ROLE_ACTION,
)
from training import constants_uniss as c
from training.megatron_uniss_dataset import packed_json_to_megatron_item
from training.phase3_whisper_streamspeech_joint.dataset import (
    IndexedPhase3ReplayDataset,
)
from training.simul_uniss.jsonl_index import load_index


class IndexedDenseTrajectoryDataset(Dataset[dict[str, object]]):
    """Random access to complete-session dense packs with inline sidecars."""

    def __init__(self, path: str | Path, *, seq_length: int) -> None:
        self.path = Path(path).resolve()
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing dense pack index for {self.path}")
        self.offsets = offsets
        self.seq_length = int(seq_length)
        if self.seq_length <= 0 or not self.offsets:
            raise ValueError("dense packed dataset geometry must be positive")

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("schema_version") != PACK_SCHEMA:
            raise ValueError(f"unexpected dense pack schema at record {index}")
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        value = self._read(index)
        result: dict[str, object] = dict(
            packed_json_to_megatron_item(value, seq_length=self.seq_length)
        )
        result["sample_kind"] = "trajectory"
        roles = torch.tensor(value["token_roles"], dtype=torch.int64)
        result["token_roles"] = roles
        result["source_ids"] = [str(item) for item in value["source_ids"]]
        result["sample_boundaries"] = torch.tensor(
            value["sample_boundaries"], dtype=torch.int64
        )
        sessions = value.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            raise ValueError("dense pack contains no sessions")
        annotations: list[dict[str, object]] = []
        tokens = value["tokens"]
        for session in sessions:
            session = dict(session)
            translation_ids = torch.tensor(
                session["translation_ids"], dtype=torch.int64
            )
            if translation_ids.numel() <= 0:
                raise ValueError("dense session has no target text tokens")
            for raw in session["annotations"]:
                annotation = dict(raw)
                action_position = int(annotation["action_position"])
                if int(roles[action_position]) != ROLE_ACTION:
                    raise ValueError("dense action annotation does not point to ACTION")
                positions = [int(item) for item in annotation["frontend_positions"]]
                ids = [int(item) for item in annotation["frontend_ids"]]
                if len(positions) != len(ids):
                    raise ValueError("frontend position/code lengths differ")
                packed_ids = [
                    int(tokens[position]) - c.GLM_SEMANTIC_OFFSET
                    for position in positions
                ]
                if packed_ids != ids:
                    raise ValueError("inline frontend codes differ from packed GLM tokens")
                previous = int(annotation["previous_committed_length"])
                stable = int(annotation["stable_target_length"])
                if not 0 <= previous <= stable <= translation_ids.numel():
                    raise ValueError("invalid committed target prefix lengths")
                annotations.append(
                    {
                        **annotation,
                        "translation_ids": translation_ids,
                        "frontend_positions": torch.tensor(positions, dtype=torch.int64),
                        "causal_ids": torch.tensor(ids, dtype=torch.int64),
                    }
                )
        if not annotations:
            raise ValueError("dense pack contains no action annotations")
        result["annotations"] = annotations
        return result


def _stable_group_id(sample_id: str) -> int:
    return (
        int.from_bytes(
            hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), "little"
        )
        & ((1 << 63) - 1)
    )


def collate_dense_trajectory(
    batch: Sequence[dict[str, object]]
) -> dict[str, object]:
    if not batch or any(value.get("sample_kind") != "trajectory" for value in batch):
        raise ValueError("dense trajectory collate received a non-trajectory record")
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
    max_translation = max(
        len(value["translation_ids"]) for _, value in annotations
    )
    max_frontend = max(
        1, max(len(value["causal_ids"]) for _, value in annotations)
    )
    translation_ids = torch.full(
        (count, max_translation), c.TOKEN_PAD, dtype=torch.long
    )
    translation_mask = torch.zeros(count, max_translation, dtype=torch.bool)
    safe_targets = torch.zeros(count, max_translation, dtype=torch.float32)
    frontend_ids = torch.zeros(count, max_frontend, dtype=torch.long)
    frontend_positions = torch.zeros(count, max_frontend, dtype=torch.long)
    frontend_mask = torch.zeros(count, max_frontend, dtype=torch.bool)

    action_batch = torch.empty(count, dtype=torch.long)
    action_position = torch.empty(count, dtype=torch.long)
    support_bucket = torch.empty(count, dtype=torch.long)
    natural_action = torch.empty(count, dtype=torch.long)
    deadline_action = torch.empty(count, dtype=torch.long)
    deadline_forced = torch.empty(count, dtype=torch.bool)
    deadline_loss_enabled = torch.empty(count, dtype=torch.bool)
    chunk_end_ms = torch.empty(count, dtype=torch.long)
    soft_deadline_ms = torch.empty(count, dtype=torch.long)
    hard_deadline_ms = torch.empty(count, dtype=torch.long)
    previous_committed = torch.empty(count, dtype=torch.long)
    stable_target = torch.empty(count, dtype=torch.long)
    sample_group = torch.empty(count, dtype=torch.long)
    playback_buffer_ms = torch.empty(count, dtype=torch.long)

    for annotation_index, (batch_row, value) in enumerate(annotations):
        action_batch[annotation_index] = batch_row
        action_position[annotation_index] = int(value["action_position"])
        support_bucket[annotation_index] = int(value["support_bucket"])
        natural_action[annotation_index] = int(value["natural_action"])
        deadline_action[annotation_index] = int(value["deadline_action"])
        deadline_forced[annotation_index] = bool(value["deadline_forced"])
        deadline_loss_enabled[annotation_index] = bool(
            value["deadline_loss_enabled"]
        )
        chunk_end_ms[annotation_index] = int(value["chunk_end_ms"])
        soft_deadline_ms[annotation_index] = int(value["soft_deadline_ms"])
        hard_deadline_ms[annotation_index] = int(value["hard_deadline_ms"])
        previous = int(value["previous_committed_length"])
        stable = int(value["stable_target_length"])
        previous_committed[annotation_index] = previous
        stable_target[annotation_index] = stable
        sample_group[annotation_index] = _stable_group_id(str(value["sample_id"]))
        playback_buffer_ms[annotation_index] = int(value["playback_buffer_ms"])

        ids = value["translation_ids"]
        translation_ids[annotation_index, : len(ids)] = ids
        translation_mask[annotation_index, : len(ids)] = True
        safe_targets[annotation_index, :stable] = 1.0
        codes = value["causal_ids"]
        positions = value["frontend_positions"]
        frontend_ids[annotation_index, : len(codes)] = codes
        frontend_positions[annotation_index, : len(positions)] = positions
        frontend_mask[annotation_index, : len(codes)] = True

    # Dense v1 uses Phase3 replay plus exact aligned CE as its quality anchor.
    # The tensors remain structurally compatible with the shared objective;
    # no fake teacher probabilities are manufactured.
    result.update(
        {
            "action_batch": action_batch,
            "action_position": action_position,
            "support_bucket": support_bucket,
            "natural_action": natural_action,
            "deadline_action": deadline_action,
            "deadline_forced": deadline_forced,
            "deadline_loss_enabled": deadline_loss_enabled,
            "chunk_end_ms": chunk_end_ms,
            "soft_deadline_ms": soft_deadline_ms,
            "hard_deadline_ms": hard_deadline_ms,
            "previous_committed_length": previous_committed,
            "stable_target_length": stable_target,
            "sample_group": sample_group,
            "playback_buffer_ms": playback_buffer_ms,
            "translation_ids": translation_ids,
            "translation_mask": translation_mask,
            "safe_commit_targets": safe_targets,
            "frontend_ids": frontend_ids,
            "frontend_positions": frontend_positions,
            "frontend_mask": frontend_mask,
            "teacher_indices": torch.empty(
                count, 4, 0, 1, dtype=torch.long
            ),
            "teacher_probabilities": torch.empty(
                count, 4, 0, 1, dtype=torch.float32
            ),
            "teacher_confidence": torch.empty(
                count, 4, 0, dtype=torch.float32
            ),
            "teacher_mask": torch.empty(count, 4, 0, dtype=torch.bool),
            "kd_batch": torch.empty(0, dtype=torch.long),
            "kd_position": torch.empty(0, dtype=torch.long),
            "kd_annotation": torch.empty(0, dtype=torch.long),
            "kd_target_index": torch.empty(0, dtype=torch.long),
        }
    )
    return result


def collate_replay_or_dense(
    batch: Sequence[dict[str, object]]
) -> dict[str, object]:
    kinds = {str(value.get("sample_kind")) for value in batch}
    if kinds == {"trajectory"}:
        return collate_dense_trajectory(batch)
    if kinds == {"replay"}:
        return default_collate(batch)
    raise ValueError(f"one microbatch cannot mix sample kinds: {sorted(kinds)}")


@dataclass(frozen=True)
class ScheduledIndex:
    epoch: int
    sample_kind: str
    source_index: int


class ThreeEpochGlobalShuffleSchedule(Dataset[dict[str, object]]):
    """Three complete, independently shuffled, restart-exact coverage epochs."""

    def __init__(
        self,
        trajectory: Dataset,
        replay: Dataset,
        *,
        coverage_epochs: int,
        data_parallel_group_size: int,
        global_batch_size: int,
        shuffle_seed: int,
        target_replay_fraction: float = 0.35,
    ) -> None:
        if not len(trajectory) or not len(replay):
            raise ValueError("trajectory and replay datasets must be non-empty")
        if coverage_epochs <= 0 or data_parallel_group_size <= 0:
            raise ValueError("coverage schedule geometry must be positive")
        if global_batch_size % data_parallel_group_size:
            raise ValueError("global batch must be divisible by the DP group")
        if not 0.0 < target_replay_fraction < 1.0:
            raise ValueError("target replay fraction must be in (0,1)")
        self.trajectory = trajectory
        self.replay = replay
        self.coverage_epochs = int(coverage_epochs)
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.global_batch_size = int(global_batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.target_replay_fraction = float(target_replay_fraction)
        self.required_trajectory_groups = math.ceil(
            len(trajectory) / self.data_parallel_group_size
        )
        self.required_replay_groups = math.ceil(
            len(replay) / self.data_parallel_group_size
        )
        groups_per_global = self.global_batch_size // self.data_parallel_group_size
        total_groups = self.required_trajectory_groups + self.required_replay_groups
        self.epoch_groups = math.ceil(total_groups / groups_per_global) * groups_per_global
        extra = self.epoch_groups - total_groups
        replay_extra = 0
        for _ in range(extra):
            current = (
                self.required_replay_groups + replay_extra
            ) / self.epoch_groups
            if current < self.target_replay_fraction:
                replay_extra += 1
        self.replay_groups = self.required_replay_groups + replay_extra
        self.trajectory_groups = self.epoch_groups - self.replay_groups
        if self.trajectory_groups < self.required_trajectory_groups:
            raise AssertionError("trajectory coverage was lost to padding allocation")
        self.epoch_samples = self.epoch_groups * self.data_parallel_group_size
        self.total_samples = self.coverage_epochs * self.epoch_samples
        self.synchronize_sample_kind = True
        self.collate_fn = collate_replay_or_dense
        self.split = "train"
        self._kinds: list[torch.Tensor] = []
        self._replay_cursors: list[torch.Tensor] = []
        self._trajectory_cursors: list[torch.Tensor] = []
        self._trajectory_permutations: list[torch.Tensor] = []
        self._replay_permutations: list[torch.Tensor] = []
        for epoch in range(self.coverage_epochs):
            base = self.shuffle_seed + epoch * 1009
            kind_generator = torch.Generator().manual_seed(base + 3)
            kinds = torch.cat(
                (
                    torch.ones(self.replay_groups, dtype=torch.bool),
                    torch.zeros(self.trajectory_groups, dtype=torch.bool),
                )
            )
            kinds = kinds[torch.randperm(self.epoch_groups, generator=kind_generator)]
            self._kinds.append(kinds)
            replay_before = torch.cumsum(kinds.to(torch.int64), dim=0) - kinds.to(
                torch.int64
            )
            self._replay_cursors.append(replay_before)
            self._trajectory_cursors.append(
                torch.arange(self.epoch_groups, dtype=torch.int64) - replay_before
            )
            # Shuffle every complete 18k pack ID, rather than shuffling blocks
            # of adjacent IDs. Padding is applied only after the full source
            # permutation, so each epoch sees every source record once before
            # any repeated tail item.
            self._trajectory_permutations.append(
                torch.randperm(
                    len(self.trajectory),
                    generator=torch.Generator().manual_seed(base + 17),
                )
            )
            self._replay_permutations.append(
                torch.randperm(
                    len(self.replay),
                    generator=torch.Generator().manual_seed(base + 29),
                )
            )

    def __len__(self) -> int:
        return self.total_samples

    def scheduled_index(self, index: int) -> ScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch, epoch_sample = divmod(index, self.epoch_samples)
        group, lane = divmod(epoch_sample, self.data_parallel_group_size)
        kinds = self._kinds[epoch]
        is_replay = bool(kinds[group])
        before = int(self._replay_cursors[epoch][group])
        if is_replay:
            cursor = before
            flat = cursor * self.data_parallel_group_size + lane
            permutation = self._replay_permutations[epoch]
            source = int(permutation[flat % len(permutation)])
            return ScheduledIndex(epoch, "replay", source)
        cursor = int(self._trajectory_cursors[epoch][group])
        flat = cursor * self.data_parallel_group_size + lane
        permutation = self._trajectory_permutations[epoch]
        source = int(permutation[flat % len(permutation)])
        return ScheduledIndex(epoch, "trajectory", source)

    def __getitem__(self, index: int) -> dict[str, object]:
        scheduled = self.scheduled_index(index)
        source = (
            self.replay
            if scheduled.sample_kind == "replay"
            else self.trajectory
        )
        value = dict(source[scheduled.source_index])
        if value.get("sample_kind") != scheduled.sample_kind:
            raise ValueError("scheduled kind disagrees with source dataset")
        return value


class CoverageEpochSampler:
    """Yield the schedule's already-shuffled DP groups without a second shuffle."""

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
        self.global_microbatch_size = (
            self.data_parallel_size * self.micro_batch_size
        )
        if self.global_microbatch_size != dataset.data_parallel_group_size:
            raise ValueError("dataset group size does not match Megatron geometry")
        self.active_total_samples = min(int(total_samples), len(dataset))
        self.active_total_samples -= (
            self.active_total_samples % self.global_microbatch_size
        )
        self.consumed_samples = int(consumed_samples)
        if self.consumed_samples % self.global_microbatch_size:
            raise ValueError("consumed samples must end on a DP group boundary")

    def __len__(self) -> int:
        return self.active_total_samples

    def __iter__(self):
        start_group = (
            self.consumed_samples % self.active_total_samples
        ) // self.global_microbatch_size
        total_groups = self.active_total_samples // self.global_microbatch_size
        for group in range(start_group, total_groups):
            group_start = group * self.global_microbatch_size
            rank_start = (
                group_start
                + self.data_parallel_rank * self.micro_batch_size
            )
            self.consumed_samples += self.global_microbatch_size
            yield list(range(rank_start, rank_start + self.micro_batch_size))


__all__ = [
    "CoverageEpochSampler",
    "IndexedDenseTrajectoryDataset",
    "IndexedPhase3ReplayDataset",
    "ScheduledIndex",
    "ThreeEpochGlobalShuffleSchedule",
    "collate_replay_or_dense",
]
