"""Audio dataset joining Stage02 targets to the immutable Stage-A manifest."""

from __future__ import annotations

import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


DIRECTION_TO_ID = {"eng->cmn": 0, "cmn->eng": 1}


class EndpointCTCAudioDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        dataset_index: str | Path,
        split: str,
        source_manifest: str | Path,
        source_offsets: str | Path,
    ) -> None:
        index = json.loads(Path(dataset_index).read_text(encoding="utf-8"))
        self.parts = []
        self.cumulative = []
        total = 0
        for entry in index["parts"][split]:
            records = int(entry["records"])
            if not records:
                continue
            offsets = np.memmap(entry["offsets"], mode="r", dtype=np.uint64)
            self.parts.append((Path(entry["manifest"]), offsets, records))
            total += records
            self.cumulative.append(total)
        self.source_manifest = Path(source_manifest)
        self.source_offsets = np.memmap(source_offsets, mode="r", dtype=np.uint64)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _target_row(self, index: int) -> dict[str, object]:
        part_index = bisect.bisect_right(self.cumulative, index)
        before = self.cumulative[part_index - 1] if part_index else 0
        local_index = index - before
        path, offsets, _ = self.parts[part_index]
        with path.open("rb") as handle:
            handle.seek(int(offsets[local_index]))
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        target = self._target_row(index)
        source_index = int(target["source_manifest_index"])
        with self.source_manifest.open("rb") as handle:
            handle.seek(int(self.source_offsets[source_index]))
            source = json.loads(handle.readline())
        if str(source["id"]) != str(target["id"]):
            raise ValueError(f"audio/CTC target mismatch: {source['id']} != {target['id']}")
        waveform, sample_rate = torchaudio.load(str(source["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        waveform = waveform.squeeze(0)
        return {
            "id": target["id"],
            "direction_id": DIRECTION_TO_ID[str(target["direction"])],
            "waveform": waveform,
            "source_token_ids": torch.tensor(target["source_token_ids"], dtype=torch.long),
            "target_token_ids": torch.tensor(target["target_token_ids"], dtype=torch.long),
        }


def collate_audio(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    waveform_lengths = torch.tensor(
        [len(row["waveform"]) for row in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    padded_samples = int(waveform_lengths.max())
    waveform = torch.zeros(len(batch), padded_samples, dtype=torch.float32)
    for index, row in enumerate(batch):
        value = row["waveform"]
        waveform[index, : len(value)] = value  # type: ignore[arg-type]
    source_lengths = torch.tensor(
        [len(row["source_token_ids"]) for row in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    target_lengths = torch.tensor(
        [len(row["target_token_ids"]) for row in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    return {
        "ids": [str(row["id"]) for row in batch],
        "direction_ids": torch.tensor(
            [int(row["direction_id"]) for row in batch], dtype=torch.long
        ),
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "source_targets": torch.cat(
            [row["source_token_ids"] for row in batch]  # type: ignore[list-item]
        ),
        "source_lengths": source_lengths,
        "target_targets": torch.cat(
            [row["target_token_ids"] for row in batch]  # type: ignore[list-item]
        ),
        "target_lengths": target_lengths,
    }


class DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    """Give all ranks similarly sized utterances at every DDP step."""

    def __init__(
        self,
        length_index: str | Path,
        batch_size: int,
        rank: int,
        world_size: int,
        *,
        seed: int = 20260803,
    ) -> None:
        self.lengths = np.memmap(length_index, mode="r", dtype=np.uint32)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.sorted_indices = np.argsort(self.lengths, kind="stable")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        group_size = self.batch_size * self.world_size
        usable = len(self.sorted_indices) - len(self.sorted_indices) % group_size
        groups = self.sorted_indices[:usable].reshape(-1, group_size).copy()
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(groups)))
        rng.shuffle(order)
        for group_index in order:
            group = groups[group_index].tolist()
            rng.shuffle(group)
            start = self.rank * self.batch_size
            yield group[start : start + self.batch_size]

    def __len__(self) -> int:
        return len(self.lengths) // (self.batch_size * self.world_size)
