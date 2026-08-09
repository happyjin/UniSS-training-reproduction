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

    sample = build_performance_sample(
        source_glm=list(source_glm),
        bicodec_global=list(bicodec_global),
        tgt_lang=tgt_lang,
        translation=translation,
        target_bicodec=list(target_bicodec),
        text_encoder=text_encoder,
        source_id=source_id,
    )
    start, end = sample.segment_spans["performance_translation_text"]
    translation_ids = sample.target_ids[start:end]
    input_ids = torch.tensor(
        [[*sample.prompt_ids, *translation_ids]], dtype=torch.long, device=device
    )
    hidden = qwen(input_ids=input_ids, use_cache=False, output_hidden_states=True).hidden_states[-1]
    positions = torch.arange(
        sample.prompt_length,
        sample.prompt_length + len(translation_ids),
        device=device,
    ).unsqueeze(0)
    text_hidden, text_lengths = gather_target_hidden(hidden, positions)
    return text_hidden, text_lengths, sample.prompt_length


def batch_fields(batch: Mapping[str, object]) -> dict[str, object]:
    """Normalize Megatron's collated micro-batch (size 1) into python scalars/lists."""

    def first(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            return value[0]
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    meta = json.loads(first(batch["record_json"]))
    source_glm = first(batch["source_glm"])
    target_bicodec = first(batch["target_bicodec"])
    bicodec_global = first(batch["bicodec_global"])
    return {
        "id": str(meta["id"]),
        "tgt_lang": str(meta["tgt_lang"]),
        "translation": str(meta["translation"]),
        "source_duration_ms": first(batch["source_duration_ms"]).reshape(()).long(),
        "source_glm": [int(value) for value in source_glm.tolist()],
        "target_bicodec": [int(value) for value in target_bicodec.tolist()],
        "target_bicodec_tensor": target_bicodec.long(),
        "unit_repeats": first(batch["unit_repeats"]).reshape(()).long(),
        "bicodec_global": [int(value) for value in bicodec_global.tolist()],
    }
