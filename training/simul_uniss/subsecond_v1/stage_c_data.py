"""Prefix-state dataset for the Stage-C source safe-commit proxy pilot."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset

from training.simul_uniss.jsonl_index import load_index


class StageCSourceCommitDataset(Dataset):
    """Return causal audio prefixes and reference support available at each tick.

    This pilot labels *source-token commit safety*.  It deliberately does not
    claim target micro-phrase safety because the current Stage-A source
    manifest has no bilingual support alignment yet.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        max_audio_seconds: float = 8.0,
        min_prefix_ms: int = 320,
        chunk_ms: int = 160,
        right_context_ms: int = 80,
        safety_margin_ms: int = 80,
        prefixes_per_record: int = 1,
        random_prefix: bool = True,
    ) -> None:
        self.path = Path(manifest)
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing JSONL offset index for {self.path}")
        if prefixes_per_record <= 0:
            raise ValueError("prefixes_per_record must be positive")
        self.offsets = offsets
        self.max_samples = round(max_audio_seconds * 16000)
        self.min_prefix_samples = round(min_prefix_ms * 16)
        self.chunk_samples = round(chunk_ms * 16)
        self.right_context_samples = round(right_context_ms * 16)
        self.safety_margin_ms = safety_margin_ms
        self.prefixes_per_record = prefixes_per_record
        self.random_prefix = random_prefix

    @property
    def base_records(self) -> int:
        return len(self.offsets)

    def __len__(self) -> int:
        return self.base_records * self.prefixes_per_record

    def _read(self, record_index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[record_index])
            value = json.loads(handle.readline())
        if not isinstance(value, dict):
            raise TypeError(f"expected object at record {record_index}")
        return value

    def _prefix_samples(self, capped_samples: int, slot: int) -> int:
        minimum_ticks = max(1, math.ceil(self.min_prefix_samples / self.chunk_samples))
        maximum_ticks = max(minimum_ticks, math.ceil(capped_samples / self.chunk_samples))
        if self.random_prefix:
            ticks = int(torch.randint(minimum_ticks, maximum_ticks + 1, (1,)).item())
        elif self.prefixes_per_record == 1:
            ticks = maximum_ticks
        else:
            fraction = (slot + 1) / (self.prefixes_per_record + 1)
            ticks = round(minimum_ticks + fraction * (maximum_ticks - minimum_ticks))
            ticks = min(maximum_ticks, max(minimum_ticks, ticks))
        return min(capped_samples, max(400, ticks * self.chunk_samples))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, slot = divmod(index, self.prefixes_per_record)
        item = self._read(record_index)
        waveform, sample_rate = torchaudio.load(str(item["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.squeeze(0)
        full_samples = waveform.numel()
        capped_samples = min(full_samples, self.max_samples)
        prefix_samples = self._prefix_samples(capped_samples, slot)
        required_samples = prefix_samples + self.right_context_samples
        visible = waveform[:required_samples]
        if visible.numel() < required_samples:
            visible = F.pad(visible, (0, required_samples - visible.numel()))

        reference = torch.tensor([int(value) for value in item["source_glm"]], dtype=torch.long)
        end_times = [int(value) for value in item["source_glm_end_ms"]]
        prefix_ms = round(prefix_samples / 16)
        support_count = bisect.bisect_right(
            end_times, max(0, prefix_ms - self.safety_margin_ms)
        )
        direction = 0 if str(item["src_lang"]) == "cmn" else 1
        return {
            "waveform": visible,
            "utterance_samples": torch.tensor(prefix_samples, dtype=torch.long),
            "full_samples": torch.tensor(full_samples, dtype=torch.long),
            "reference_glm": reference,
            "support_count": torch.tensor(support_count, dtype=torch.long),
            "direction": torch.tensor(direction, dtype=torch.long),
            "record_index": torch.tensor(record_index, dtype=torch.long),
        }


def collate_stage_c(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    waveform_lengths = torch.tensor([len(item["waveform"]) for item in batch], dtype=torch.long)
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()), dtype=torch.float32)
    reference_lengths = torch.tensor(
        [len(item["reference_glm"]) for item in batch], dtype=torch.long
    )
    references = torch.full(
        (len(batch), int(reference_lengths.max())), -1, dtype=torch.long
    )
    for row, item in enumerate(batch):
        waveform[row, : len(item["waveform"])] = item["waveform"]
        references[row, : len(item["reference_glm"])] = item["reference_glm"]
    return {
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "utterance_sample_lengths": torch.stack([item["utterance_samples"] for item in batch]),
        "full_samples": torch.stack([item["full_samples"] for item in batch]),
        "reference_glm": references,
        "reference_glm_lengths": reference_lengths,
        "support_count": torch.stack([item["support_count"] for item in batch]),
        "direction": torch.stack([item["direction"] for item in batch]),
        "record_index": torch.stack([item["record_index"] for item in batch]),
    }
