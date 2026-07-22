"""Causal log-Mel audio student for formal Streaming GLM distillation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from training import constants_uniss as c
from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.streaming_student import CausalConv1d, ctc_loss


class AudioStreamingStudent(nn.Module):
    def __init__(
        self,
        policy_vocab_size: int,
        *,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        sample_rate: int = 16000,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )
        self.input_projection = CausalConv1d(n_mels, hidden_size, kernel_size=5, stride=2)
        self.downsample1 = CausalConv1d(hidden_size, hidden_size, kernel_size=5, stride=2)
        self.downsample2 = CausalConv1d(hidden_size, hidden_size, kernel_size=5, stride=2)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.teacher_glm_head = nn.Linear(hidden_size, c.GLM_SEMANTIC_SIZE + 1)
        self.source_ctc_head = nn.Linear(hidden_size, policy_vocab_size)
        self.target_ctc_head = nn.Linear(hidden_size, policy_vocab_size)

    @staticmethod
    def mel_lengths(waveform_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div((waveform_lengths - 400).clamp_min(0), 160, rounding_mode="floor") + 1

    @staticmethod
    def output_lengths(waveform_lengths: torch.Tensor) -> torch.Tensor:
        lengths = AudioStreamingStudent.mel_lengths(waveform_lengths)
        for _ in range(3):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp_min(1)

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        mel = torch.log(self.mel(waveform).clamp_min(1e-5))
        hidden = F.gelu(self.input_projection(mel)).transpose(1, 2)
        hidden = F.gelu(self.downsample1(hidden.transpose(1, 2))).transpose(1, 2)
        hidden = F.gelu(self.downsample2(hidden.transpose(1, 2))).transpose(1, 2)
        output_lengths = self.output_lengths(waveform_lengths)
        time = hidden.shape[1]
        causal_mask = torch.triu(torch.ones(time, time, dtype=torch.bool, device=hidden.device), diagonal=1)
        positions = torch.arange(time, device=hidden.device).unsqueeze(0)
        padding_mask = positions >= output_lengths.unsqueeze(1)
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        return {
            "hidden": hidden,
            "output_lengths": output_lengths,
            "teacher_glm_logits": self.teacher_glm_head(hidden),
            "source_ctc_logits": self.source_ctc_head(hidden),
            "target_ctc_logits": self.target_ctc_head(hidden),
        }


class AudioStudentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        policy_tokenizer: PolicyTokenizer,
        *,
        max_audio_seconds: float = 12.0,
        prefix_training: bool = True,
    ) -> None:
        self.path = Path(manifest_path)
        self.policy_tokenizer = policy_tokenizer
        self.max_samples = int(round(max_audio_seconds * 16000))
        self.prefix_training = prefix_training
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"{self.path} contains no records")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            item = json.loads(handle.readline().decode("utf-8"))
        waveform, sample_rate = torchaudio.load(item["source_audio"])
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform[:1]
        full_samples = waveform.shape[-1]
        if self.prefix_training and index % 5 and full_samples > 3200:
            fractions = (0.25, 0.5, 0.75, 1.0)
            fraction = fractions[index % len(fractions)]
        else:
            fraction = 1.0
        visible_samples = min(self.max_samples, max(400, int(full_samples * fraction)))
        waveform = waveform[..., :visible_samples].squeeze(0)
        visible_fraction = min(1.0, visible_samples / max(1, full_samples))
        teacher = [int(token) + 1 for token in item["source_glm"]]
        teacher = teacher[: max(1, int(math.ceil(len(teacher) * visible_fraction)))]
        source_chars = max(1, int(math.ceil(len(item["transcription"]) * visible_fraction)))
        target_chars = max(1, int(math.ceil(len(item["translation"]) * visible_fraction)))
        source_policy = self.policy_tokenizer.encode_ctc(item["transcription"][:source_chars])
        target_policy = self.policy_tokenizer.encode_ctc(item["translation"][:target_chars])
        output_length = int(AudioStreamingStudent.output_lengths(torch.tensor([visible_samples]))[0])
        return {
            "waveform": waveform,
            "teacher_glm": torch.tensor(teacher[:output_length], dtype=torch.long),
            "source_policy": torch.tensor(source_policy[:output_length], dtype=torch.long),
            "target_policy": torch.tensor(target_policy[:output_length], dtype=torch.long),
        }


def collate_audio_student(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    waveform_lengths = torch.tensor([len(item["waveform"]) for item in batch], dtype=torch.long)
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()), dtype=torch.float32)
    for index, item in enumerate(batch):
        waveform[index, : len(item["waveform"])] = item["waveform"]
    result: dict[str, torch.Tensor] = {"waveform": waveform, "waveform_lengths": waveform_lengths}
    for key in ("teacher_glm", "source_policy", "target_policy"):
        result[key] = torch.cat([item[key] for item in batch])
        result[f"{key}_lengths"] = torch.tensor([len(item[key]) for item in batch], dtype=torch.long)
    return result


def audio_student_losses(
    model: AudioStreamingStudent, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    outputs = model(batch["waveform"], batch["waveform_lengths"])
    lengths = outputs["output_lengths"]
    teacher = ctc_loss(
        outputs["teacher_glm_logits"], batch["teacher_glm"], lengths, batch["teacher_glm_lengths"]
    )
    source = ctc_loss(
        outputs["source_ctc_logits"], batch["source_policy"], lengths, batch["source_policy_lengths"]
    )
    target = ctc_loss(
        outputs["target_ctc_logits"], batch["target_policy"], lengths, batch["target_policy_lengths"]
    )
    return {"total": teacher + 0.3 * source + 0.3 * target, "teacher": teacher, "source": source, "target": target}
