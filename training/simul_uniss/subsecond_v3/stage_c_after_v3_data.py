"""Packed formal Stage-C data that decodes each source audio only once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v2.formal_supervision import safe_label
from training.simul_uniss.subsecond_v2.stage_c_data import collate_stage_c_formal


class StageCAfterV3PackedDataset(Dataset):
    """Return multiple safe/unsafe prefixes after one manifest/audio read."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        max_audio_seconds: float = 8.0,
        min_prefix_ms: int = 320,
        chunk_ms: int = 160,
        right_context_ms: int = 80,
        prefixes_per_record: int = 4,
        random_prefix: bool = True,
    ) -> None:
        self.path = Path(manifest)
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing JSONL offset index for {self.path}")
        if prefixes_per_record <= 0:
            raise ValueError("prefixes_per_record must be positive")
        self.offsets = offsets
        self.max_samples = round(max_audio_seconds * 16_000)
        self.min_prefix_ms = min_prefix_ms
        self.chunk_ms = chunk_ms
        self.right_context_samples = round(right_context_ms * 16)
        self.prefixes_per_record = prefixes_per_record
        self.random_prefix = random_prefix

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            value = json.loads(handle.readline())
        if not bool(value.get("formal_a68_pass")):
            raise ValueError("formal Stage-C manifest contains a rejected record")
        return value

    def _event_and_prefix(
        self, events: list[dict[str, object]], slot: int, capped_ms: int
    ) -> tuple[dict[str, object], int]:
        if self.random_prefix:
            event = events[int(torch.randint(0, len(events), (1,)).item())]
            want_positive = bool(int(torch.randint(0, 2, (1,)).item()))
        else:
            event = events[(slot // 2) % len(events)]
            want_positive = bool(slot % 2)
        threshold = int(event["safe_if_source_ms_gte"])
        if want_positive and threshold <= capped_ms:
            prefix_ms = threshold
        else:
            prefix_ms = max(
                self.min_prefix_ms,
                min(capped_ms, threshold - self.chunk_ms),
            )
        prefix_ms = max(self.min_prefix_ms, min(capped_ms, prefix_ms))
        return event, max(self.min_prefix_ms, prefix_ms // self.chunk_ms * self.chunk_ms)

    def __getitem__(self, index: int) -> list[dict[str, torch.Tensor]]:
        item = self._read(index)
        waveform, sample_rate = torchaudio.load(str(item["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        waveform = waveform.squeeze(0)
        full_samples = waveform.numel()
        capped_samples = min(full_samples, self.max_samples)
        capped_ms = max(self.min_prefix_ms, round(capped_samples / 16))
        events = [dict(value) for value in item["micro_write_events"]]  # type: ignore[index]
        if not events:
            raise ValueError("formal Stage-C record has no Micro-WRITE events")
        reference = torch.tensor(
            [int(value) for value in item["teacher_source_glm"]], dtype=torch.long
        )
        direction = 0 if str(item["src_lang"]) == "cmn" else 1
        rows: list[dict[str, torch.Tensor]] = []
        for slot in range(self.prefixes_per_record):
            event, prefix_ms = self._event_and_prefix(events, slot, capped_ms)
            prefix_samples = min(capped_samples, max(400, prefix_ms * 16))
            required_samples = prefix_samples + self.right_context_samples
            visible = waveform[:required_samples]
            if visible.numel() < required_samples:
                visible = F.pad(visible, (0, required_samples - visible.numel()))
            rows.append(
                {
                    "waveform": visible,
                    "utterance_samples": torch.tensor(prefix_samples, dtype=torch.long),
                    "full_samples": torch.tensor(full_samples, dtype=torch.long),
                    "reference_glm": reference,
                    "support_count": torch.tensor(0, dtype=torch.long),
                    "direction": torch.tensor(direction, dtype=torch.long),
                    "record_index": torch.tensor(index, dtype=torch.long),
                    "safe_label": torch.tensor(
                        safe_label(event, round(prefix_samples / 16)), dtype=torch.float32
                    ),
                    "event_support_end_ms": torch.tensor(
                        int(event["support_end_ms"]), dtype=torch.long
                    ),
                }
            )
        return rows


def collate_stage_c_after_v3(
    batch: list[list[Mapping[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    flattened = [row for package in batch for row in package]
    return collate_stage_c_formal(flattened)
