"""Bootstrap causal student distilled from UniST BiCodec/GLM token pairs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from training import constants_uniss as c
from training.simul_uniss.policy_tokenizer import PolicyTokenizer


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(inputs, (self.left_padding, 0)))


class StreamingTokenStudent(nn.Module):
    """Small causal bootstrap model operating on 50 Hz source BiCodec tokens.

    It validates distillation and policy-head training before the same heads are
    attached to the full audio Streaming GLM student.
    """

    def __init__(
        self,
        policy_vocab_size: int,
        *,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_embedding = nn.Embedding(c.BICODEC_SEMANTIC_SIZE, hidden_size)
        self.downsample1 = CausalConv1d(hidden_size, hidden_size, kernel_size=5, stride=2)
        self.downsample2 = CausalConv1d(hidden_size, hidden_size, kernel_size=5, stride=2)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.teacher_glm_head = nn.Linear(hidden_size, c.GLM_SEMANTIC_SIZE + 1)
        self.source_ctc_head = nn.Linear(hidden_size, policy_vocab_size)
        self.target_ctc_head = nn.Linear(hidden_size, policy_vocab_size)

    @staticmethod
    def output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, source_bicodec: torch.Tensor, input_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.input_embedding(source_bicodec)
        hidden = self.downsample1(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = F.gelu(hidden)
        hidden = self.downsample2(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = F.gelu(hidden)
        output_lengths = self.output_lengths(input_lengths)
        time = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=hidden.device), diagonal=1
        )
        positions = torch.arange(time, device=hidden.device).unsqueeze(0)
        padding_mask = positions >= output_lengths.unsqueeze(1)
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        return {
            "hidden": hidden,
            "output_lengths": output_lengths,
            "teacher_glm_logits": self.teacher_glm_head(hidden),
            "source_ctc_logits": self.source_ctc_head(hidden),
            "target_ctc_logits": self.target_ctc_head(hidden),
        }


def _prefix_text(text: str, fraction: float) -> str:
    if not text:
        return text
    length = max(1, min(len(text), int(math.ceil(len(text) * fraction))))
    return text[:length]


class StreamingStudentDataset(Dataset):
    def __init__(
        self,
        schedule_path: str | Path,
        policy_tokenizer: PolicyTokenizer,
        *,
        max_source_tokens: int = 1024,
        prefix_training: bool = True,
    ) -> None:
        self.path = Path(schedule_path)
        self.policy_tokenizer = policy_tokenizer
        self.max_source_tokens = max_source_tokens
        self.prefix_training = prefix_training
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"{self.path} contains no schedules")

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._read(index)
        events = item["events"]
        event_count = len(events)
        if self.prefix_training and event_count > 1 and index % 5:
            prefix_count = 1 + ((index * 104729) % event_count)
        else:
            prefix_count = event_count
        selected = events[:prefix_count]
        source_bicodec = [token for event in selected for token in event["source_bicodec"]]
        teacher_glm = [token + 1 for event in selected for token in event["source_glm"]]
        if not source_bicodec or not teacher_glm:
            raise ValueError(f"empty bootstrap token prefix for {item['id']}")

        full_source_length = max(1, int(item["source_bicodec_length"]))
        if len(source_bicodec) > self.max_source_tokens:
            fraction = self.max_source_tokens / len(source_bicodec)
            source_bicodec = source_bicodec[: self.max_source_tokens]
            teacher_glm = teacher_glm[: max(1, int(math.floor(len(teacher_glm) * fraction)))]
        visible_fraction = min(1.0, len(source_bicodec) / full_source_length)
        source_text = _prefix_text(str(item["transcription"]), visible_fraction)
        target_proxy = int(selected[-1]["target_ctc_count_proxy"])
        target_total = max(1, int(item["target_text_length"]))
        target_fraction = min(1.0, target_proxy / target_total)
        target_text = _prefix_text(str(item["translation"]), target_fraction)

        source_policy = self.policy_tokenizer.encode_ctc(source_text)
        target_policy = self.policy_tokenizer.encode_ctc(target_text)
        output_length = max(1, int(math.ceil(math.ceil(len(source_bicodec) / 2) / 2)))
        teacher_glm = teacher_glm[:output_length]
        source_policy = source_policy[:output_length]
        target_policy = target_policy[:output_length]
        return {
            "source_bicodec": torch.tensor(source_bicodec, dtype=torch.long),
            "teacher_glm": torch.tensor(teacher_glm, dtype=torch.long),
            "source_policy": torch.tensor(source_policy, dtype=torch.long),
            "target_policy": torch.tensor(target_policy, dtype=torch.long),
        }


def collate_student_batch(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")
    input_lengths = torch.tensor([len(item["source_bicodec"]) for item in batch], dtype=torch.long)
    max_length = int(input_lengths.max())
    inputs = torch.zeros(len(batch), max_length, dtype=torch.long)
    for row, item in enumerate(batch):
        values = item["source_bicodec"]
        inputs[row, : len(values)] = values

    result: dict[str, torch.Tensor] = {
        "source_bicodec": inputs,
        "input_lengths": input_lengths,
    }
    for key in ("teacher_glm", "source_policy", "target_policy"):
        lengths = torch.tensor([len(item[key]) for item in batch], dtype=torch.long)
        result[f"{key}_lengths"] = lengths
        result[key] = torch.cat([item[key] for item in batch])
    return result


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    return F.ctc_loss(
        logits.log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
