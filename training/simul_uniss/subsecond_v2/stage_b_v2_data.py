"""Indexed audio plus Stage-A-v3 sidecar dataset for causal Stage-B-v2."""

from __future__ import annotations

import bisect
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.policy_tokenizer import PolicyTokenizer


class StageBV2SidecarDataset(Dataset):
    def __init__(
        self,
        sidecar_manifest: str | Path,
        source_manifest: str | Path,
        policy_tokenizer: PolicyTokenizer,
        *,
        max_audio_seconds: float = 8.0,
        min_prefix_ms: int = 640,
        chunk_ms: int = 160,
        right_context_ms: int = 80,
        prefix_training: bool = True,
    ) -> None:
        self.sidecar_path = Path(sidecar_manifest)
        self.source_path = Path(source_manifest)
        offsets = load_index(self.sidecar_path)
        if offsets is None:
            raise ValueError(f"missing sidecar index for {self.sidecar_path}")
        self.offsets = offsets
        self.policy_tokenizer = policy_tokenizer
        self.max_samples = round(max_audio_seconds * 16_000)
        self.min_prefix_samples = round(min_prefix_ms * 16)
        self.chunk_samples = round(chunk_ms * 16)
        self.right_context_samples = round(right_context_ms * 16)
        self.prefix_training = prefix_training

    def __len__(self) -> int:
        return len(self.offsets)

    def _sidecar_row(self, index: int) -> dict[str, object]:
        with self.sidecar_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def _source_row(self, offset: int) -> dict[str, object]:
        with self.source_path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    @lru_cache(maxsize=16)
    def _shard(self, path: str) -> dict[str, object]:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sidecar = self._sidecar_row(index)
        source = self._source_row(int(sidecar["source_manifest_offset"]))
        shard = self._shard(str(sidecar["shard_path"]))
        target_start, target_end = int(sidecar["target_start"]), int(sidecar["target_end"])
        reference_start = int(sidecar["reference_start"])
        reference_end = int(sidecar["reference_end"])
        target = shard["target_tokens"][target_start:target_end].long()  # type: ignore[index]
        reference = shard["full_reference_tokens"][reference_start:reference_end].long()  # type: ignore[index]
        stability = shard["stability"][target_start:target_end].float()  # type: ignore[index]

        waveform, sample_rate = torchaudio.load(str(source["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        waveform = waveform.squeeze(0)
        capped_samples = min(len(waveform), self.max_samples)
        if self.prefix_training and capped_samples > self.min_prefix_samples:
            minimum_ticks = max(1, math.ceil(self.min_prefix_samples / self.chunk_samples))
            maximum_ticks = max(minimum_ticks, capped_samples // self.chunk_samples)
            ticks = int(torch.randint(minimum_ticks, maximum_ticks + 1, (1,)).item())
            utterance_samples = min(capped_samples, ticks * self.chunk_samples)
        else:
            utterance_samples = capped_samples
        utterance_samples = max(400, utterance_samples)
        visible_samples = utterance_samples + self.right_context_samples
        visible = waveform[:visible_samples]
        if len(visible) < visible_samples:
            visible = F.pad(visible, (0, visible_samples - len(visible)))
        target_count = min(len(target), max(1, math.ceil(utterance_samples / 1_280)))
        target = target[:target_count]
        stability = stability[:target_count]
        reference = reference[: min(len(reference), target_count)]

        hidden = None
        topk_ids = None
        topk_distances = None
        if "pre_vq_hidden" in shard:
            hidden = shard["pre_vq_hidden"][target_start : target_start + target_count].float()  # type: ignore[index]
            topk_ids = shard["topk_ids"][target_start : target_start + target_count].long()  # type: ignore[index]
            topk_distances = shard["topk_distances"][target_start : target_start + target_count].float()  # type: ignore[index]

        utterance_ms = round(utterance_samples / 16)
        source_words = source.get("source_words")
        if isinstance(source_words, list) and source_words:
            visible_words = [
                str(value["text"])
                for value in source_words
                if isinstance(value, Mapping) and int(value.get("end_ms", 0)) <= utterance_ms
            ]
            if not visible_words:
                visible_words = [str(source_words[0]["text"])]
            separator = "" if str(source.get("src_lang", "")).lower() in {"cmn", "zh"} else " "
            source_prefix = separator.join(visible_words)
        else:
            fraction = min(1.0, utterance_samples / max(1, capped_samples))
            transcription = str(source["transcription"])
            source_prefix = transcription[: max(1, math.ceil(len(transcription) * fraction))]
        source_policy = self.policy_tokenizer.encode_ctc(source_prefix)
        source_policy = source_policy[: max(1, target_count * 2)]

        target_support = source.get("target_support")
        if isinstance(target_support, list) and target_support:
            supported = sum(
                int(value.get("support_end_ms", capped_samples / 16 + 1)) <= utterance_ms
                for value in target_support
                if isinstance(value, Mapping)
            )
            target_capacity = supported / len(target_support)
        else:
            target_capacity = min(1.0, utterance_samples / max(1, capped_samples))

        return {
            "waveform": visible,
            "utterance_samples": torch.tensor(utterance_samples, dtype=torch.long),
            "target_ids": target,
            "full_reference_ids": reference,
            "stability_target": stability,
            "teacher_hidden": hidden
            if hidden is not None
            else torch.zeros(target_count, 1, dtype=torch.float32),
            "has_teacher_hidden": torch.tensor(hidden is not None, dtype=torch.bool),
            "topk_ids": topk_ids
            if topk_ids is not None
            else torch.zeros(target_count, 1, dtype=torch.long),
            "topk_distances": topk_distances
            if topk_distances is not None
            else torch.zeros(target_count, 1, dtype=torch.float32),
            "source_policy": torch.tensor(source_policy, dtype=torch.long),
            "target_capacity": torch.tensor(target_capacity, dtype=torch.float32),
        }


def collate_stage_b_v2(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    waveform_lengths = torch.tensor([len(value["waveform"]) for value in batch], dtype=torch.long)
    utterance_lengths = torch.stack([value["utterance_samples"] for value in batch])
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()))
    for row, value in enumerate(batch):
        waveform[row, : len(value["waveform"])] = value["waveform"]
    target_lengths = torch.tensor([len(value["target_ids"]) for value in batch], dtype=torch.long)
    reference_lengths = torch.tensor(
        [len(value["full_reference_ids"]) for value in batch], dtype=torch.long
    )
    maximum = int(target_lengths.max())
    target = torch.zeros(len(batch), maximum, dtype=torch.long)
    reference = torch.zeros(len(batch), maximum, dtype=torch.long)
    stability = torch.zeros(len(batch), maximum)
    hidden_dim = max(int(value["teacher_hidden"].shape[-1]) for value in batch)
    topk = max(int(value["topk_ids"].shape[-1]) for value in batch)
    hidden = torch.zeros(len(batch), maximum, hidden_dim)
    topk_ids = torch.zeros(len(batch), maximum, topk, dtype=torch.long)
    topk_distances = torch.zeros(len(batch), maximum, topk)
    for row, value in enumerate(batch):
        count = len(value["target_ids"])
        target[row, :count] = value["target_ids"]
        ref_count = len(value["full_reference_ids"])
        reference[row, :ref_count] = value["full_reference_ids"]
        stability[row, :count] = value["stability_target"]
        source_hidden = value["teacher_hidden"]
        hidden[row, :count, : source_hidden.shape[-1]] = source_hidden
        source_topk = value["topk_ids"]
        topk_ids[row, :count, : source_topk.shape[-1]] = source_topk
        source_distances = value["topk_distances"]
        topk_distances[row, :count, : source_distances.shape[-1]] = source_distances
    source_lengths = torch.tensor([len(value["source_policy"]) for value in batch], dtype=torch.long)
    return {
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "utterance_sample_lengths": utterance_lengths,
        "target_ids": target,
        "target_lengths": target_lengths,
        "full_reference_ids": reference,
        "full_reference_lengths": reference_lengths,
        "stability_target": stability,
        "teacher_hidden": hidden,
        "has_teacher_hidden": torch.stack([value["has_teacher_hidden"] for value in batch]),
        "topk_ids": topk_ids,
        "topk_distances": topk_distances,
        "source_policy": torch.cat([value["source_policy"] for value in batch]),
        "source_policy_lengths": source_lengths,
        "target_capacity": torch.stack([value["target_capacity"] for value in batch]),
    }
