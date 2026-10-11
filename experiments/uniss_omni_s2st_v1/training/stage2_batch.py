"""Turn a collated Stage-2 batch into the tensors the model forward wants.

The prompt comes from the chat template, the target text is appended to
it, and the text labels are -100 over everything but the target. The
Talker needs the target's own token ids and its hidden states, so the
reply is also returned separately -- and because prompts differ in length
across a batch, they are left-padded so every reply starts at one index.

Audio tasks additionally carry mel features; the processor expands
``<|AUDIO|>`` into as many placeholder tokens as the encoder will emit, so
the prompt length is not knowable before this runs.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import build_codec_sequence
from experiments.uniss_omni_s2st_v1.training import stage2_tasks


def build_batch(batch: dict, processor, talker, layout, *, device) -> dict:
    """One collated batch -> model keyword arguments."""
    task = batch["task"]
    if stage2_tasks.uses_tts_cache(task):
        return _build_cached(batch, processor, talker, layout, device=device)
    conversations = [
        stage2_tasks.conversation(
            stage2_tasks.TaskSample(
                task,
                "x.wav" if stage2_tasks.needs_audio(task) else None,
                user_text, None, None, None,
            )
        )
        for user_text in batch["user_text"]
    ]
    templated = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    call = {"text": list(templated), "return_tensors": "pt", "padding": True}
    if "waveforms" in batch:
        call["audio"] = batch["waveforms"]
        call["sampling_rate"] = 16_000
    encoded = processor(**call)

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or 0
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids)

    supervise_text = stage2_tasks.supervises_text(task)
    reply = tokenizer(list(batch["text_target"]), add_special_tokens=False)["input_ids"]

    rows = prompt_ids.shape[0]
    prompt_width = prompt_ids.shape[1]
    reply_width = max(len(r) for r in reply)
    total = prompt_width + reply_width

    input_ids = torch.full((rows, total), pad_id, dtype=torch.long)
    attention = torch.zeros((rows, total), dtype=torch.long)
    text_labels = torch.full((rows, total), -100, dtype=torch.long)
    reply_ids = torch.full((rows, reply_width), pad_id, dtype=torch.long)
    reply_mask = torch.zeros((rows, reply_width), dtype=torch.long)

    input_ids[:, :prompt_width] = prompt_ids
    attention[:, :prompt_width] = prompt_mask
    for row, ids in enumerate(reply):
        length = len(ids)
        tensor = torch.as_tensor(ids, dtype=torch.long)
        input_ids[row, prompt_width : prompt_width + length] = tensor
        attention[row, prompt_width : prompt_width + length] = 1
        reply_ids[row, :length] = tensor
        reply_mask[row, :length] = 1
        if supervise_text:
            text_labels[row, prompt_width : prompt_width + length] = tensor

    out = {
        "thinker_input_ids": input_ids.to(device),
        "thinker_attention_mask": attention.to(device),
        "text_labels": text_labels.to(device),
        "reply_ids": reply_ids.to(device),
        "reply_mask": reply_mask.to(device),
        "reply_start": torch.tensor(prompt_width, dtype=torch.long, device=device),
        "supervise_text": torch.tensor(
            bool(supervise_text), dtype=torch.bool, device=device
        ),
    }
    for key in ("input_features", "feature_attention_mask"):
        if key in encoded:
            out[key] = encoded[key].to(device)

    if stage2_tasks.supervises_codes(task):
        codec_ids, codec_labels, codec_mask = build_codec_sequence(
            [c.to(device) for c in batch["codes"]],
            bos_id=talker.codec_bos_token,
            pad_id=talker.codec_pad_token,
            eos_id=layout.special_ids["eos"],
        )
        out.update(
            codec_input_ids=codec_ids,
            codec_labels=codec_labels,
            codec_mask=codec_mask,
            bicodec_global=batch["globals"].to(device),
        )
    return out


def _build_cached(batch: dict, processor, talker, layout, *, device) -> dict:
    """The TTS path, whose prompt and reply are already token ids.

    The cache exists so this task never re-tokenises 16M prompts; going
    back through the chat template would also need raw text the cache
    does not store.
    """
    pad_id = processor.tokenizer.pad_token_id or 0
    prompts = [np.asarray(p) for p in batch["prompt_ids"]]
    replies = [np.asarray(r) for r in batch["reply_ids"]]
    rows = len(prompts)
    prompt_width = max(len(p) for p in prompts)
    reply_width = max(len(r) for r in replies)
    total = prompt_width + reply_width

    input_ids = torch.full((rows, total), pad_id, dtype=torch.long)
    attention = torch.zeros((rows, total), dtype=torch.long)
    reply_ids = torch.full((rows, reply_width), pad_id, dtype=torch.long)
    reply_mask = torch.zeros((rows, reply_width), dtype=torch.long)

    for row, (prompt, reply) in enumerate(zip(prompts, replies)):
        # Prompts are left-padded so every reply begins at prompt_width.
        start = prompt_width - len(prompt)
        input_ids[row, start:prompt_width] = torch.as_tensor(prompt, dtype=torch.long)
        attention[row, start:prompt_width] = 1
        tensor = torch.as_tensor(reply, dtype=torch.long)
        input_ids[row, prompt_width : prompt_width + len(reply)] = tensor
        attention[row, prompt_width : prompt_width + len(reply)] = 1
        reply_ids[row, : len(reply)] = tensor
        reply_mask[row, : len(reply)] = 1

    codec_ids, codec_labels, codec_mask = build_codec_sequence(
        [c.to(device) for c in batch["codes"]],
        bos_id=talker.codec_bos_token,
        pad_id=talker.codec_pad_token,
        eos_id=layout.special_ids["eos"],
    )
    return {
        "thinker_input_ids": input_ids.to(device),
        "thinker_attention_mask": attention.to(device),
        "text_labels": torch.full((rows, total), -100, dtype=torch.long, device=device),
        "reply_ids": reply_ids.to(device),
        "reply_mask": reply_mask.to(device),
        "reply_start": torch.tensor(prompt_width, dtype=torch.long, device=device),
        "supervise_text": torch.tensor(False, dtype=torch.bool, device=device),
        "codec_input_ids": codec_ids,
        "codec_labels": codec_labels,
        "codec_mask": codec_mask,
        "bicodec_global": batch["globals"].to(device),
    }
