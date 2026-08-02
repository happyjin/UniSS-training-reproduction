"""Formal Stage-B dataset for fixed-rate WhisperVQ latent distillation."""

from __future__ import annotations

from typing import Mapping

import torch

from training.simul_uniss.subsecond_v1.data import StageBAudioDataset


class LatentStageBAudioDataset(StageBAudioDataset):
    """Reuse audited Stage-A loading while retaining uncollapsed GLM IDs."""

    def __init__(self, *args: object, stability_future_ms: int = 320, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.stability_future_ms = int(stability_future_ms)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = super().__getitem__(index)
        # Historical CTC data reserves zero for blank and therefore adds one.
        # Latent regression uses the original frozen codebook IDs directly.
        teacher_ids = value.pop("teacher_glm") - 1
        if bool(((teacher_ids < 0) | (teacher_ids >= 16_384)).any()):
            raise ValueError("teacher GLM ID is outside the frozen codebook")
        value["teacher_glm_ids"] = teacher_ids
        token_count = len(teacher_ids)
        stable_count = max(0, token_count - max(1, self.stability_future_ms // 80))
        stability = torch.zeros(token_count, dtype=torch.float32)
        stability[:stable_count] = 1.0
        value["stability_target"] = stability
        return value


def collate_stage_b_latent(
    batch: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    waveform_lengths = torch.tensor([len(item["waveform"]) for item in batch], dtype=torch.long)
    utterance_sample_lengths = torch.stack([item["utterance_samples"] for item in batch])
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()), dtype=torch.float32)
    for row, item in enumerate(batch):
        waveform[row, : len(item["waveform"])] = item["waveform"]
    teacher_lengths = torch.tensor([len(item["teacher_glm_ids"]) for item in batch], dtype=torch.long)
    max_teacher = int(teacher_lengths.max())
    teacher = torch.zeros(len(batch), max_teacher, dtype=torch.long)
    stability = torch.zeros(len(batch), max_teacher, dtype=torch.float32)
    for row, item in enumerate(batch):
        count = len(item["teacher_glm_ids"])
        teacher[row, :count] = item["teacher_glm_ids"]
        stability[row, :count] = item["stability_target"]
    source_lengths = torch.tensor([len(item["source_policy"]) for item in batch], dtype=torch.long)
    return {
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "utterance_sample_lengths": utterance_sample_lengths,
        "teacher_glm_ids": teacher,
        "teacher_glm_lengths": teacher_lengths,
        "source_policy": torch.cat([item["source_policy"] for item in batch]),
        "source_policy_lengths": source_lengths,
        "target_capacity": torch.stack([item["target_capacity"] for item in batch]),
        "stability_target": stability,
    }
