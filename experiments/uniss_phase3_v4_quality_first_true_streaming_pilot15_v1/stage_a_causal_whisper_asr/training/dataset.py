"""Indexed Stage A packs, bounded audio loading, and strict epoch shuffle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.dataset import (
    CoverageEpochSampler,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_NONE,
    PACK_SCHEMA,
)
from training import constants_uniss as c
from training.megatron_uniss_dataset import packed_json_to_megatron_item
from training.simul_uniss.jsonl_index import load_index


SAMPLE_RATE = 16_000


def _load_mono(path: str) -> torch.Tensor:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Stage A audio must be 16 kHz, got {sample_rate}: {path}")
    waveform = torch.from_numpy(values.mean(axis=1).copy())
    if not waveform.numel() or not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"Stage A audio is empty or non-finite: {path}")
    return waveform


def rotated_acoustic_indices(
    length: int, count: int, epoch: int, index: int
) -> list[int]:
    if length < 0:
        raise ValueError("Stage A acoustic count cannot be negative")
    if count <= 0 or length <= count:
        return list(range(length))
    start = (epoch * 104729 + index * 1009) % length
    return [(start + offset) % length for offset in range(count)]


class IndexedStageAPackDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        path: str | Path,
        *,
        seq_length: int,
        max_acoustics_per_pack: int,
        load_audio: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing Stage A pack index: {self.path}")
        self.offsets = offsets
        self.seq_length = int(seq_length)
        self.max_acoustics_per_pack = int(max_acoustics_per_pack)
        self.load_audio = bool(load_audio)
        if self.seq_length <= 0 or self.max_acoustics_per_pack <= 0 or not len(offsets):
            raise ValueError("Stage A dataset geometry must be positive")

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("schema_version") != PACK_SCHEMA:
            raise ValueError(f"unexpected Stage A pack schema at record {index}")
        return value

    def get_for_epoch(self, index: int, epoch: int) -> dict[str, object]:
        value = self._read(index)
        result: dict[str, object] = dict(
            packed_json_to_megatron_item(value, seq_length=self.seq_length)
        )
        result["sample_kind"] = "stage_a"
        result["coverage_epoch"] = int(epoch)
        result["source_pack_index"] = int(index)
        result["loss_kinds"] = torch.tensor(value["loss_kinds"], dtype=torch.long)
        if result["loss_kinds"].shape != result["loss_mask"].shape:
            raise ValueError("Stage A loss kind/mask geometry differs")
        raw_acoustics = list(value.get("acoustics", []))
        selected_indices = rotated_acoustic_indices(
            len(raw_acoustics), self.max_acoustics_per_pack, epoch, index
        )
        selected_index_set = set(selected_indices)
        boundaries = value.get("sample_boundaries", [])
        disabled_boundaries: set[int] = set()
        for acoustic_index, raw in enumerate(raw_acoustics):
            if acoustic_index in selected_index_set:
                continue
            boundary_index = int(raw["batch_boundary_index"])
            if not 0 <= boundary_index < len(boundaries):
                raise ValueError("Stage A acoustic boundary index is malformed")
            disabled_boundaries.add(boundary_index)
        for boundary_index in disabled_boundaries:
            start, end = (int(item) for item in boundaries[boundary_index])
            if not 0 <= start < end <= self.seq_length:
                raise ValueError("Stage A sample boundary is malformed")
            result["loss_mask"][start:end] = 0
            result["loss_kinds"][start:end] = LOSS_NONE

        selected = [raw_acoustics[position] for position in selected_indices]
        acoustics: list[dict[str, object]] = []
        tokens = value["tokens"]
        for raw in selected:
            acoustic = dict(raw)
            positions = [int(item) for item in acoustic["glm_positions"]]
            ids = [int(item) for item in acoustic["source_glm"]]
            if len(positions) != len(ids) or not positions:
                raise ValueError("Stage A acoustic GLM sidecar is malformed")
            packed = [int(tokens[position]) - c.GLM_SEMANTIC_OFFSET for position in positions]
            if packed != ids:
                raise ValueError("Stage A acoustic GLM IDs differ from packed tokens")
            ctc = [int(item) for item in acoustic["ctc_ids"]]
            if not ctc or any(not 0 <= item < 256 for item in ctc):
                raise ValueError("Stage A byte CTC target is malformed")
            if self.load_audio:
                waveform = _load_mono(str(acoustic["source_audio"]))
                expected = int(acoustic["source_duration_ms"])
                actual = round(1000 * waveform.numel() / SAMPLE_RATE)
                if abs(actual - expected) > 20:
                    raise ValueError(
                        f"Stage A audio duration differs from sidecar: {actual} vs {expected}"
                    )
                acoustic["waveform"] = waveform
            acoustics.append(acoustic)
        result["acoustics"] = acoustics
        result["selected_acoustics"] = len(acoustics)
        result["available_acoustics"] = len(raw_acoustics)
        result["disabled_acoustics"] = len(raw_acoustics) - len(acoustics)
        return result

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.get_for_epoch(index, 0)


class ThreeEpochStageASchedule(Dataset[dict[str, object]]):
    """Complete independently shuffled pack coverage with restart-exact padding."""

    def __init__(
        self,
        dataset: IndexedStageAPackDataset,
        *,
        coverage_epochs: int,
        data_parallel_group_size: int,
        global_batch_size: int,
        shuffle_seed: int,
    ) -> None:
        if coverage_epochs <= 0 or data_parallel_group_size <= 0:
            raise ValueError("Stage A coverage geometry must be positive")
        if global_batch_size % data_parallel_group_size:
            raise ValueError("global batch must divide into Stage A DP groups")
        self.dataset = dataset
        self.coverage_epochs = int(coverage_epochs)
        self.data_parallel_group_size = int(data_parallel_group_size)
        self.global_batch_size = int(global_batch_size)
        self.shuffle_seed = int(shuffle_seed)
        groups_per_global = global_batch_size // data_parallel_group_size
        source_groups = math.ceil(len(dataset) / data_parallel_group_size)
        self.epoch_groups = math.ceil(source_groups / groups_per_global) * groups_per_global
        self.epoch_samples = self.epoch_groups * data_parallel_group_size
        self.total_samples = self.coverage_epochs * self.epoch_samples
        self.synchronize_sample_kind = True
        self.split = "train"
        self.collate_fn = collate_stage_a
        self._permutations = [
            torch.randperm(
                len(dataset),
                generator=torch.Generator().manual_seed(shuffle_seed + epoch * 1009 + 17),
            )
            for epoch in range(self.coverage_epochs)
        ]

    def __len__(self) -> int:
        return self.total_samples

    def source_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch, epoch_index = divmod(index, self.epoch_samples)
        permutation = self._permutations[epoch]
        return epoch, int(permutation[epoch_index % len(permutation)])

    def __getitem__(self, index: int) -> dict[str, object]:
        epoch, source = self.source_index(index)
        return self.dataset.get_for_epoch(source, epoch)


class PaddedStageAValidationDataset(Dataset[dict[str, object]]):
    """Repeat validation packs to a complete Megatron DP microbatch.

    Megatron's random sampler drops an incomplete final DP microbatch.  When
    the complete validation set itself is smaller than that microbatch, its
    active sample count becomes zero.  Cyclic padding keeps every rank inside
    the same collectives and preserves all source packs at least once.
    """

    def __init__(
        self,
        dataset: IndexedStageAPackDataset,
        *,
        minimum_samples: int,
        data_parallel_group_size: int,
    ) -> None:
        if minimum_samples < 0 or data_parallel_group_size <= 0:
            raise ValueError("invalid Stage A validation padding geometry")
        self.dataset = dataset
        self.unpadded_length = len(dataset)
        requested = max(self.unpadded_length, int(minimum_samples))
        self.padded_length = (
            math.ceil(requested / data_parallel_group_size)
            * data_parallel_group_size
        )
        self.split = "valid"
        self.collate_fn = collate_stage_a

    def __len__(self) -> int:
        return self.padded_length

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset[index % self.unpadded_length]


def collate_stage_a(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    if not batch or any(value.get("sample_kind") != "stage_a" for value in batch):
        raise ValueError("Stage A collate received a foreign sample")
    tensor_keys = (
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "cu_seqlens",
        "max_seqlen",
        "loss_kinds",
    )
    result: dict[str, object] = {
        key: default_collate([value[key] for value in batch]) for key in tensor_keys
    }
    result["sample_kind"] = "stage_a"
    result["coverage_epoch"] = torch.tensor(
        [int(value["coverage_epoch"]) for value in batch], dtype=torch.long
    )
    result["source_pack_index"] = torch.tensor(
        [int(value["source_pack_index"]) for value in batch], dtype=torch.long
    )
    result["selected_acoustics"] = torch.tensor(
        [int(value["selected_acoustics"]) for value in batch], dtype=torch.long
    )
    result["available_acoustics"] = torch.tensor(
        [int(value["available_acoustics"]) for value in batch], dtype=torch.long
    )
    result["disabled_acoustics"] = torch.tensor(
        [int(value["disabled_acoustics"]) for value in batch], dtype=torch.long
    )
    flattened = [
        (batch_row, dict(acoustic))
        for batch_row, value in enumerate(batch)
        for acoustic in value["acoustics"]  # type: ignore[index]
    ]
    if not flattened:
        raise ValueError("Stage A training pack selected no acoustic supervision")
    max_waveform = max(len(value["waveform"]) for _, value in flattened)
    max_ctc = max(len(value["ctc_ids"]) for _, value in flattened)
    max_glm = max(len(value["source_glm"]) for _, value in flattened)
    waveform = torch.zeros(len(flattened), max_waveform, dtype=torch.float32)
    waveform_lengths = torch.empty(len(flattened), dtype=torch.long)
    ctc_ids = torch.zeros(len(flattened), max_ctc, dtype=torch.long)
    ctc_lengths = torch.empty(len(flattened), dtype=torch.long)
    glm_ids = torch.zeros(len(flattened), max_glm, dtype=torch.long)
    glm_positions = torch.zeros(len(flattened), max_glm, dtype=torch.long)
    glm_lengths = torch.empty(len(flattened), dtype=torch.long)
    acoustic_batch = torch.empty(len(flattened), dtype=torch.long)
    language_ids = torch.empty(len(flattened), dtype=torch.long)
    for row, (batch_row, value) in enumerate(flattened):
        samples = value["waveform"]
        ctc = value["ctc_ids"]
        glm = value["source_glm"]
        positions = value["glm_positions"]
        waveform[row, : len(samples)] = samples
        waveform_lengths[row] = len(samples)
        ctc_ids[row, : len(ctc)] = torch.tensor(ctc, dtype=torch.long)
        ctc_lengths[row] = len(ctc)
        glm_ids[row, : len(glm)] = torch.tensor(glm, dtype=torch.long)
        glm_positions[row, : len(positions)] = torch.tensor(positions, dtype=torch.long)
        glm_lengths[row] = len(glm)
        acoustic_batch[row] = batch_row
        language_ids[row] = 0 if str(value["src_lang"]) == "eng" else 1
    result.update(
        {
            "waveform": waveform,
            "waveform_lengths": waveform_lengths,
            "ctc_ids": ctc_ids,
            "ctc_lengths": ctc_lengths,
            "glm_ids": glm_ids,
            "glm_positions": glm_positions,
            "glm_lengths": glm_lengths,
            "acoustic_batch": acoustic_batch,
            "language_ids": language_ids,
            "acoustic_sample_ids": [
                str(value["sample_id"]) for _, value in flattened
            ],
            "source_audio_paths": [
                str(value["source_audio"]) for _, value in flattened
            ],
            "acoustic_source_duration_ms": torch.tensor(
                [int(value["source_duration_ms"]) for _, value in flattened],
                dtype=torch.long,
            ),
        }
    )
    return result


__all__ = [
    "CoverageEpochSampler",
    "IndexedStageAPackDataset",
    "PaddedStageAValidationDataset",
    "ThreeEpochStageASchedule",
    "collate_stage_a",
    "rotated_acoustic_indices",
]
