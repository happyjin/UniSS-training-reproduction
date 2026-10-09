"""Data for the Stage 1 Talker warmup: target text in, BiCodec codes out.

The warmup never touches source audio, so it reads the UniST parquets
directly rather than going through the manifests. ``translation`` and
``target_bicodec`` there are byte-identical to the manifest's
``target_text`` and the codes it points at (checked on 200 rows), and
dropping the manifest means the stage is not limited to the shards whose
audio tars happen to be local -- all 1M rows are usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

COLUMNS = ["id", "translation", "tgt_lang", "target_bicodec", "bicodec_global"]

LANGUAGE_NAME = {"cmn": "Chinese", "eng": "English", "zh": "Chinese", "en": "English"}


def read_rows(
    parquets: list[str | Path],
    *,
    limit_per_file: int = 0,
    min_codes: int = 4,
    max_codes: int = 1024,
) -> list[dict]:
    """Load (text, codes) pairs, dropping ones no stage could learn from."""
    rows: list[dict] = []
    for path in parquets:
        table = pq.read_table(path, columns=COLUMNS)
        taken = 0
        for record in table.to_pylist():
            text = (record["translation"] or "").strip()
            codes = record["target_bicodec"]
            globals_ = record.get("bicodec_global")
            if not text or codes is None or not globals_:
                continue
            if not min_codes <= len(codes) <= max_codes:
                continue
            rows.append(
                {
                    "id": record["id"],
                    "text": text,
                    "lang": record["tgt_lang"],
                    "codes": np.asarray(codes, dtype=np.int16),
                    "globals": np.asarray(globals_, dtype=np.int16),
                }
            )
            taken += 1
            if limit_per_file and taken >= limit_per_file:
                break
    return rows


def tts_prompt(text: str, lang: str) -> list[dict]:
    """A conversation whose assistant turn is the text to be spoken.

    The Talker is conditioned on the Thinker's *reply* hidden states, so for
    a TTS warmup the reply has to be the target text itself. The system
    prompt is Omni's own: the model warns that speech output is only
    reliable under it.
    """
    language = LANGUAGE_NAME.get(lang, lang)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team,"
                        " Alibaba Group, capable of perceiving auditory and visual"
                        " inputs, as well as generating text and speech."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Read this {language} text aloud: {text}"}
            ],
        },
    ]


@dataclass
class ThinkerBatch:
    """Prompt left-padded, reply right-padded, so replies start together."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    reply_start: int
    reply_ids: torch.Tensor
    reply_mask: torch.Tensor


def build_thinker_batch(
    prompt_ids: list[list[int]],
    reply_ids: list[list[int]],
    *,
    pad_id: int,
    device: torch.device | str = "cpu",
) -> ThinkerBatch:
    """Lay out prompts and replies so one slice recovers every reply.

    Prompts are left-padded and replies right-padded. Without that, each
    sample's reply would begin at its own offset and the hidden states would
    have to be gathered row by row -- correct but easy to get quietly wrong,
    and this costs nothing.
    """
    if len(prompt_ids) != len(reply_ids):
        raise ValueError("prompts and replies must have the same batch size")
    batch = len(prompt_ids)
    if batch == 0:
        raise ValueError("empty batch")
    prompt_width = max(len(p) for p in prompt_ids)
    reply_width = max(len(r) for r in reply_ids)
    total = prompt_width + reply_width

    input_ids = torch.full((batch, total), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, total), dtype=torch.long)
    replies = torch.full((batch, reply_width), pad_id, dtype=torch.long)
    reply_mask = torch.zeros((batch, reply_width), dtype=torch.long)

    for row, (prompt, reply) in enumerate(zip(prompt_ids, reply_ids)):
        start = prompt_width - len(prompt)
        input_ids[row, start:prompt_width] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, start:prompt_width] = 1
        input_ids[row, prompt_width : prompt_width + len(reply)] = torch.tensor(
            reply, dtype=torch.long
        )
        attention_mask[row, prompt_width : prompt_width + len(reply)] = 1
        replies[row, : len(reply)] = torch.tensor(reply, dtype=torch.long)
        reply_mask[row, : len(reply)] = 1

    return ThinkerBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        reply_start=prompt_width,
        reply_ids=replies.to(device),
        reply_mask=reply_mask.to(device),
    )
