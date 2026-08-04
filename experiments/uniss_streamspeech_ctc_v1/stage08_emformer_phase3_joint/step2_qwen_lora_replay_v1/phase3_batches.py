"""Streaming and teacher-source Phase3 batches for Step2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from bridge import replace_embedding_span
from training.sample_builders import build_performance_sample


PERFORMANCE_SUFFIX_TOKENS = 5


def _pad(
    sequences: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    maximum = max(len(value) for value in sequences)
    hidden = sequences[0].shape[-1]
    inputs = sequences[0].new_zeros(len(sequences), maximum, hidden)
    label_batch = torch.full(
        (len(sequences), maximum), -100, dtype=torch.long, device=device
    )
    attention = torch.zeros(len(sequences), maximum, dtype=torch.long, device=device)
    target_tokens = 0
    for row, (embeddings, target_labels) in enumerate(zip(sequences, labels)):
        inputs[row, : len(embeddings)] = embeddings
        label_batch[row, : len(target_labels)] = target_labels
        attention[row, : len(embeddings)] = 1
        target_tokens += int((target_labels != -100).sum())
    return inputs, attention, label_batch, target_tokens


def _performance_sample(record: Mapping[str, object], text_encoder, source_glm):
    return build_performance_sample(
        source_glm=source_glm,
        bicodec_global=record["bicodec_global"],
        tgt_lang=str(record["tgt_lang"]),
        translation=str(record["translation"]),
        target_bicodec=record["target_bicodec"],
        text_encoder=text_encoder,
        source_id=str(record["id"]),
    )


def streaming_lm_batch(qwen, text_encoder, records, bridge_output, device):
    embedding_layer = qwen.get_input_embeddings()
    sequences, labels = [], []
    for row, record in enumerate(records):
        speech_length = int(bridge_output.token_lengths[row])
        sample = _performance_sample(record, text_encoder, [0] * speech_length)
        ids = torch.tensor(sample.input_ids, dtype=torch.long, device=device)
        embeddings = embedding_layer(ids)
        span_start = sample.prompt_length - PERFORMANCE_SUFFIX_TOKENS - speech_length
        embeddings = replace_embedding_span(
            embeddings,
            bridge_output.qwen_speech_embeddings[row],
            span_start=span_start,
            speech_length=speech_length,
        )
        target_labels = ids.clone()
        target_labels[: sample.prompt_length] = -100
        sequences.append(embeddings)
        labels.append(target_labels)
    return _pad(sequences, labels, device)


def offline_replay_lm_batch(qwen, text_encoder, records, device):
    embedding_layer = qwen.get_input_embeddings()
    sequences, labels = [], []
    for record in records:
        source_glm = record.get("source_glm")
        if not isinstance(source_glm, list) or not source_glm:
            raise ValueError("offline replay requires a non-empty source_glm list")
        sample = _performance_sample(record, text_encoder, source_glm)
        ids = torch.tensor(sample.input_ids, dtype=torch.long, device=device)
        target_labels = ids.clone()
        target_labels[: sample.prompt_length] = -100
        sequences.append(embedding_layer(ids))
        labels.append(target_labels)
    return _pad(sequences, labels, device)
