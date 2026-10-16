"""One sequence, text then speaker then codes, for the decoder-only route.

The two-stream route gives the Talker a text stream and a code stream and
adds them in the Thinker's width. Here there is no Talker: the codes are
positions in the Thinker's own sequence, embedded by a table of their own
and read back by a head of their own.

    [audio pads][instruction][reply text][speaker x32][codes x T]
                             ^ lm_head                ^ audio_head

Why separate tables rather than UniSS's single vocabulary: Qwen3's 152,064
rows carry the zero-shot translation this project is built on -- 38.34 and
39.24 Text-BLEU -- and growing that table puts 8,193 randomly initialised
code rows into the same softmax the text competes in. With two heads the
modality is decided by position, not by the model, and the code softmax is
over 8,193 rather than 160,257.

Speaker tokens sit between the text and the codes because they are given,
not predicted: BiCodec's first 32 tokens carry the voice and reach the
vocoder directly, so the model needs to see them before it starts emitting
semantics but is never scored on them.

Placeholder positions are filled by index, not by a reserved token id. The
thinker counts <|audio_pad|> occurrences to scatter audio, so any id used
for speaker or code slots has to be one it will leave alone; picking the
pad id and overwriting by index keeps that question from arising.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.uniss_omni_s2st_v1.training import stage2_tasks

BICODEC_GLOBAL_COUNT = 32
IGNORE = -100

# The codec geometry lives here rather than with the model because the
# inference script needs it and importing the model would pull in
# megatron, which the evaluation environment does not have.
BICODEC_CODES = 8192          # the semantic codebook
BICODEC_GLOBAL_SIZE = 4096    # the speaker codebook
CODE_EOS = BICODEC_CODES      # one past the codes, so 8,193 classes in all
CODE_VOCAB = BICODEC_CODES + 1
THINKER_HIDDEN = 2048


def build_deconly_batch(batch: dict, processor, *, device,
                        code_eos: int | None = None) -> dict:
    """One collated batch -> keyword arguments for DecOnlyModel.forward."""
    task = batch["task"]
    if stage2_tasks.uses_tts_cache(task):
        raise SystemExit(
            f"{task} draws from the TTS cache, which holds Qwen2.5 token ids;"
            " the decoder-only route templates its own prompts")

    # "x.wav" is a placeholder: the waveform reaches the processor through
    # the audio= argument, and the template only needs to know an audio
    # turn exists. Passing the real path would send the processor to disk
    # for a file the collate already read.
    conversations = [
        stage2_tasks.conversation(
            stage2_tasks.TaskSample(
                task,
                "x.wav" if stage2_tasks.needs_audio(task) else None,
                user_text, None, None, None))
        for user_text in batch["user_text"]
    ]
    templated = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False)
    call = {"text": list(templated), "return_tensors": "pt", "padding": True}
    if "waveforms" in batch:
        call["audio"] = batch["waveforms"]
        call["sampling_rate"] = 16_000
    encoded = processor(**call)

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or 0
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eot_id is None or eot_id < 0:
        eot_id = tokenizer.eos_token_id

    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids)
    rows, prompt_width = prompt_ids.shape

    reply = tokenizer(list(batch["text_target"]), add_special_tokens=False)["input_ids"]
    # The end-of-turn token is supervised: without it the model is never
    # told a reply ends, and a trainable backbone drifts into repeating --
    # measured once already, text-BLEU 38.34 to 5.25.
    reply = [list(r) + [eot_id] for r in reply]
    text_width = max(len(r) for r in reply)

    supervises_codes = stage2_tasks.supervises_codes(task)
    if supervises_codes:
        codes = [np.asarray(c, dtype=np.int64) for c in batch["codes"]]
        if code_eos is not None:
            codes = [np.concatenate([c, [code_eos]]) for c in codes]
        code_width = max(len(c) for c in codes)
        globals_ = [np.asarray(g, dtype=np.int64) for g in batch["globals"]]
    else:
        codes, globals_, code_width = [], [], 0

    speaker_width = BICODEC_GLOBAL_COUNT if supervises_codes else 0
    total = prompt_width + text_width + speaker_width + code_width

    input_ids = torch.full((rows, total), pad_id, dtype=torch.long)
    attention = torch.zeros((rows, total), dtype=torch.long)
    text_labels = torch.full((rows, total), IGNORE, dtype=torch.long)
    code_labels = torch.full((rows, max(code_width, 1)), IGNORE, dtype=torch.long)
    speaker_ids = torch.zeros((rows, max(speaker_width, 1)), dtype=torch.long)
    code_ids = torch.zeros((rows, max(code_width, 1)), dtype=torch.long)

    input_ids[:, :prompt_width] = prompt_ids
    attention[:, :prompt_width] = prompt_mask
    supervise_text = stage2_tasks.supervises_text(task)
    for row, ids in enumerate(reply):
        n = len(ids)
        at = prompt_width
        input_ids[row, at : at + n] = torch.as_tensor(ids, dtype=torch.long)
        attention[row, at : at + n] = 1
        if supervise_text:
            text_labels[row, at : at + n] = torch.as_tensor(ids, dtype=torch.long)
    if supervises_codes:
        for row in range(rows):
            at = prompt_width + text_width
            attention[row, at : at + speaker_width] = 1
            speaker_ids[row] = torch.as_tensor(globals_[row][:speaker_width])
            c = codes[row]
            at += speaker_width
            attention[row, at : at + len(c)] = 1
            code_ids[row, : len(c)] = torch.as_tensor(c)
            code_labels[row, : len(c)] = torch.as_tensor(c)

    out = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "text_labels": text_labels.to(device),
        "speaker_ids": speaker_ids.to(device),
        "code_ids": code_ids.to(device),
        "code_labels": code_labels.to(device),
        "code_start": torch.tensor(prompt_width + text_width + speaker_width,
                                   dtype=torch.long, device=device),
        "supervise_text": torch.tensor(bool(supervise_text), dtype=torch.bool,
                                       device=device),
        "supervise_codes": torch.tensor(bool(supervises_codes), dtype=torch.bool,
                                        device=device),
    }
    for key in ("input_features", "feature_attention_mask"):
        if key in encoded:
            out[key] = encoded[key].to(device)
    return out
