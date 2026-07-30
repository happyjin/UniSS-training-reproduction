"""Indexed Stage-A audio dataset for Stage-B causal student training."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torchaudio
from torch.utils.data import Dataset

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.policy_tokenizer import PolicyTokenizer


class StageBAudioDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        policy_tokenizer: PolicyTokenizer,
        *,
        max_audio_seconds: float = 8.0,
        min_prefix_ms: int = 640,
        chunk_ms: int = 160,
        right_context_ms: int = 80,
        prefix_training: bool = True,
    ) -> None:
        self.path = Path(manifest)
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing JSONL offset index for {self.path}")
        self.offsets = offsets
        self.policy_tokenizer = policy_tokenizer
        self.max_samples = int(round(max_audio_seconds * 16000))
        self.min_prefix_samples = int(round(min_prefix_ms * 16))
        self.chunk_samples = int(round(chunk_ms * 16))
        self.right_context_samples = int(round(right_context_ms * 16))
        self.prefix_training = prefix_training

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            value = json.loads(handle.readline())
        if not isinstance(value, dict):
            raise TypeError(f"expected object at record {index}")
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._read(index)
        waveform, sample_rate = torchaudio.load(str(item["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.squeeze(0)
        full_samples = len(waveform)
        capped_samples = min(full_samples, self.max_samples)
        if self.prefix_training and capped_samples > self.min_prefix_samples:
            min_ticks = max(1, math.ceil(self.min_prefix_samples / self.chunk_samples))
            max_ticks = max(min_ticks, capped_samples // self.chunk_samples)
            ticks = int(torch.randint(min_ticks, max_ticks + 1, (1,)).item())
            utterance_samples = min(capped_samples, ticks * self.chunk_samples)
        else:
            utterance_samples = capped_samples
        utterance_samples = max(400, utterance_samples)
        required_samples = utterance_samples + self.right_context_samples
        visible = waveform[:required_samples]
        if len(visible) < required_samples:
            visible = torch.nn.functional.pad(visible, (0, required_samples - len(visible)))

        utterance_ms = round(utterance_samples / 16)
        teacher_glm = [int(value) + 1 for value in item["source_glm"]]
        teacher_ends = [int(value) for value in item["source_glm_end_ms"]]
        teacher_count = bisect.bisect_right(teacher_ends, utterance_ms)
        max_output_frames = max(1, math.ceil(max(1, utterance_samples - 240) / 640))
        teacher_glm = teacher_glm[: min(teacher_count, max_output_frames)]
        if not teacher_glm:
            teacher_glm = [int(item["source_glm"][0]) + 1]

        visible_fraction = min(1.0, utterance_samples / max(1, full_samples))
        transcription = str(item["transcription"])
        source_chars = max(1, math.ceil(len(transcription) * visible_fraction))
        source_policy = self.policy_tokenizer.encode_ctc(transcription[:source_chars])
        source_policy = source_policy[:max_output_frames]

        return {
            "waveform": visible,
            "utterance_samples": torch.tensor(utterance_samples, dtype=torch.long),
            "teacher_glm": torch.tensor(teacher_glm, dtype=torch.long),
            "source_policy": torch.tensor(source_policy, dtype=torch.long),
            "target_capacity": torch.tensor(visible_fraction, dtype=torch.float32),
        }


def collate_stage_b(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    waveform_lengths = torch.tensor([len(item["waveform"]) for item in batch], dtype=torch.long)
    utterance_sample_lengths = torch.stack([item["utterance_samples"] for item in batch])
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()), dtype=torch.float32)
    for row, item in enumerate(batch):
        waveform[row, : len(item["waveform"])] = item["waveform"]
    result: dict[str, torch.Tensor] = {
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "utterance_sample_lengths": utterance_sample_lengths,
        "target_capacity": torch.stack([item["target_capacity"] for item in batch]),
    }
    for key in ("teacher_glm", "source_policy"):
        lengths = torch.tensor([len(item[key]) for item in batch], dtype=torch.long)
        result[f"{key}_lengths"] = lengths
        result[key] = torch.cat([item[key] for item in batch])
    return result
