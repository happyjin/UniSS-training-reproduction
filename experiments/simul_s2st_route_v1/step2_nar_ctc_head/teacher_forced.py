"""Teacher-forced Phase3 hidden states for the NAR CTC head."""

from __future__ import annotations

import json
from typing import Callable, Mapping, Sequence

import torch

from training.phase3_whisper_streamspeech_joint.phase3_batch import gather_target_hidden
from training.sample_builders import build_performance_sample


@torch.no_grad()
def target_text_hidden(
    qwen,
    text_encoder: Callable[[str], list[int]],
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    tgt_lang: str,
    translation: str,
    target_bicodec: Sequence[int],
    source_id: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(text_hidden [1,T,H], text_lengths [1], prompt_length)``."""

    text_hidden, text_lengths, prompt_lengths = batched_target_text_hidden(
        qwen,
        text_encoder,
        source_glm=[list(source_glm)],
        bicodec_global=[list(bicodec_global)],
        tgt_lang=[tgt_lang],
        translation=[translation],
        target_bicodec=[list(target_bicodec)],
        source_id=[source_id],
        device=device,
    )
    return text_hidden, text_lengths, int(prompt_lengths[0].item())


@torch.no_grad()
def batched_target_text_hidden(
    qwen,
    text_encoder: Callable[[str], list[int]],
    *,
    source_glm: Sequence[Sequence[int]],
    bicodec_global: Sequence[Sequence[int]],
    tgt_lang: Sequence[str],
    translation: Sequence[str],
    target_bicodec: Sequence[Sequence[int]],
    source_id: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched teacher-forced text hiddens for a micro-batch.

    Returns ``(text_hidden [B,T,H], text_lengths [B], prompt_lengths [B])``.
    """

    batch = len(source_id)
    if batch == 0:
        raise ValueError("empty micro-batch")
    if not (
        len(source_glm)
        == len(bicodec_global)
        == len(tgt_lang)
        == len(translation)
        == len(target_bicodec)
        == batch
    ):
        raise ValueError("teacher-forced field lengths disagree")

    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    text_token_lengths: list[int] = []
    for row in range(batch):
        sample = build_performance_sample(
            source_glm=list(source_glm[row]),
            bicodec_global=list(bicodec_global[row]),
            tgt_lang=tgt_lang[row],
            translation=translation[row],
            target_bicodec=list(target_bicodec[row]),
            text_encoder=text_encoder,
            source_id=source_id[row],
        )
        start, end = sample.segment_spans["performance_translation_text"]
        translation_ids = sample.target_ids[start:end]
        sequences.append([*sample.prompt_ids, *translation_ids])
        prompt_lengths.append(int(sample.prompt_length))
        text_token_lengths.append(len(translation_ids))

    max_len = max(len(sequence) for sequence in sequences)
    pad_id = int(getattr(getattr(qwen, "config", None), "pad_token_id", None) or 0)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    positions = torch.full((batch, max(text_token_lengths)), -1, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        text = text_token_lengths[row]
        if text:
            positions[row, :text] = torch.arange(
                prompt_lengths[row],
                prompt_lengths[row] + text,
                device=device,
            )

    hidden = qwen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    ).hidden_states[-1]
    text_hidden, text_lengths = gather_target_hidden(hidden, positions)
    return (
        text_hidden,
        text_lengths,
        torch.tensor(prompt_lengths, dtype=torch.long, device=device),
    )


def batch_fields(batch: Mapping[str, object]) -> dict[str, object]:
    """Normalize a collated micro-batch into python lists + padded tensors."""

    record_json = batch["record_json"]
    if isinstance(record_json, (list, tuple)):
        metas = [json.loads(value) for value in record_json]
    else:
        metas = [json.loads(str(record_json))]

    source_glm = batch["source_glm"]
    target_bicodec = batch["target_bicodec"]
    bicodec_global = batch["bicodec_global"]
    duration = batch["source_duration_ms"]
    unit_repeats = batch["unit_repeats"]
    unit_lengths = batch["target_bicodec_length"]

    if isinstance(source_glm, torch.Tensor) and source_glm.ndim == 1:
        # Legacy micro-batch size 1 without padding dim.
        source_glm = source_glm.unsqueeze(0)
        target_bicodec = target_bicodec.unsqueeze(0)
        bicodec_global = bicodec_global.unsqueeze(0)
        duration = duration.reshape(1)
        unit_repeats = unit_repeats.reshape(1)
        unit_lengths = unit_lengths.reshape(1)

    batch_size = len(metas)
    glm_lengths = batch.get("source_glm_lengths")
    if glm_lengths is None:
        glm_lengths = torch.tensor(
            [int((source_glm[row] >= 0).sum()) for row in range(batch_size)],
            dtype=torch.long,
        )
    global_lengths = batch.get("bicodec_global_lengths")
    if global_lengths is None:
        global_lengths = torch.full((batch_size,), bicodec_global.shape[-1], dtype=torch.long)

    source_glm_lists = [
        [int(value) for value in source_glm[row, : int(glm_lengths[row])].tolist()]
        for row in range(batch_size)
    ]
    target_lists = [
        [int(value) for value in target_bicodec[row, : int(unit_lengths[row])].tolist()]
        for row in range(batch_size)
    ]
    global_lists = [
        [int(value) for value in bicodec_global[row, : int(global_lengths[row])].tolist()]
        for row in range(batch_size)
    ]
    return {
        "ids": [str(meta["id"]) for meta in metas],
        "tgt_lang": [str(meta["tgt_lang"]) for meta in metas],
        "translation": [str(meta["translation"]) for meta in metas],
        "source_duration_ms": duration.long().reshape(batch_size),
        "source_glm": source_glm_lists,
        "target_bicodec": target_lists,
        "target_bicodec_tensor": target_bicodec.long(),
        "target_bicodec_lengths": unit_lengths.long().reshape(batch_size),
        "unit_repeats": unit_repeats.long().reshape(batch_size),
        "bicodec_global": global_lists,
    }
