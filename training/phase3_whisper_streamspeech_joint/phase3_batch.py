"""Build Phase3-compatible policy-conditioned Qwen batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from training.sample_builders import build_performance_sample

from .policy_mask import phase3_prediction_attention_allowed
from .whisper_multichunk import additive_attention_mask


PERFORMANCE_SUFFIX_TOKENS = 5


@dataclass
class Phase3ARBatch:
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    source_starts: torch.Tensor
    target_starts: torch.Tensor
    target_positions: torch.Tensor


def build_policy_conditioned_phase3_batch(
    *,
    embedding_layer: torch.nn.Module,
    text_encoder: Callable[[str], list[int]],
    records: Sequence[Mapping[str, object]],
    source_embeddings: torch.Tensor,
    source_lengths: torch.Tensor,
    g: torch.Tensor,
) -> Phase3ARBatch:
    """Replace Phase3 source GLM IDs with STE embeddings and apply ``g(i)``."""

    if source_embeddings.ndim != 3 or source_lengths.shape != source_embeddings.shape[:1]:
        raise ValueError("invalid source embedding geometry")
    if len(records) != len(source_embeddings):
        raise ValueError("record/source batch sizes differ")
    device = source_embeddings.device
    sequences = []
    label_rows = []
    source_starts = []
    target_starts = []
    target_lengths = []
    for row, record in enumerate(records):
        speech_length = int(source_lengths[row].item())
        sample = build_performance_sample(
            source_glm=[0] * speech_length,
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            tgt_lang=str(record["tgt_lang"]),
            translation=str(record["translation"]),
            target_bicodec=record["target_bicodec"],  # type: ignore[arg-type]
            text_encoder=text_encoder,
            source_id=str(record.get("id", "")),
        )
        translation_start, translation_end = sample.segment_spans["performance_translation_text"]
        translation_ids = sample.target_ids[translation_start:translation_end]
        input_ids = torch.tensor(
            [*sample.prompt_ids, *translation_ids], dtype=torch.long, device=device
        )
        embeddings = embedding_layer(input_ids)
        source_start = sample.prompt_length - PERFORMANCE_SUFFIX_TOKENS - speech_length
        target_start = sample.prompt_length
        if source_start < 0 or source_start + speech_length > target_start:
            raise ValueError("invalid Phase3 performance prompt geometry")
        embeddings = embeddings.clone()
        embeddings[source_start : source_start + speech_length] = source_embeddings[
            row, :speech_length
        ]
        labels = torch.full_like(input_ids, -100)
        labels[target_start:] = input_ids[target_start:]
        sequences.append(embeddings)
        label_rows.append(labels)
        source_starts.append(source_start)
        target_starts.append(target_start)
        target_lengths.append(len(translation_ids))

    maximum = max(len(value) for value in sequences)
    hidden = sequences[0].shape[-1]
    inputs = source_embeddings.new_zeros(len(sequences), maximum, hidden)
    labels = torch.full((len(sequences), maximum), -100, dtype=torch.long, device=device)
    position_ids = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    target_positions = torch.full(
        (len(sequences), max(target_lengths)), -1, dtype=torch.long, device=device
    )
    sequence_lengths = torch.tensor([len(value) for value in sequences], device=device)
    source_starts_tensor = torch.tensor(source_starts, device=device)
    target_starts_tensor = torch.tensor(target_starts, device=device)
    target_lengths_tensor = torch.tensor(target_lengths, device=device)
    for row, (embeddings, row_labels) in enumerate(zip(sequences, label_rows)):
        length = len(embeddings)
        inputs[row, :length] = embeddings
        labels[row, :length] = row_labels
        position_ids[row, :length] = torch.arange(length, device=device)
        target_positions[row, : target_lengths[row]] = torch.arange(
            target_starts[row], target_starts[row] + target_lengths[row], device=device
        )
    allowed = phase3_prediction_attention_allowed(
        sequence_lengths=sequence_lengths,
        source_starts=source_starts_tensor,
        source_lengths=source_lengths,
        target_starts=target_starts_tensor,
        target_lengths=target_lengths_tensor,
        g=g[:, : max(target_lengths)],
    )
    return Phase3ARBatch(
        inputs_embeds=inputs,
        labels=labels,
        attention_mask=additive_attention_mask(allowed, inputs.dtype),
        position_ids=position_ids,
        source_starts=source_starts_tensor,
        target_starts=target_starts_tensor,
        target_positions=target_positions,
    )


def gather_target_hidden(hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or positions.ndim != 2 or hidden.shape[0] != positions.shape[0]:
        raise ValueError("hidden/position geometry differs")
    valid = positions >= 0
    safe = positions.clamp_min(0)
    gathered = torch.gather(hidden, 1, safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    return gathered * valid.unsqueeze(-1), valid.sum(dim=1)
