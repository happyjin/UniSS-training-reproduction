"""Qwen3-Omni's generate with the thinker at batch one and the talker batched.

Batching the whole of `generate` left-pads the prompts, and in bf16 the
padded thinker no longer decodes greedily to the same text: on 96 CVSS-T
sentences a fifth to two fifths of the translations changed. The thinker
is the small part of the time -- a sentence is a few dozen tokens, while
its speech is 12.5 talker steps a second, each followed by 15 residual-code
steps -- so the thinker stays at batch one, where its text is the text
phase_h and the cascade were scored on, and only the talker is batched.

Between the two is transformers 5.19's
Qwen3OmniMoeForConditionalGeneration.generate: its talker-input step run
per utterance at batch one, which is that function's own single-sample
path, then left-padded into one talker call, and its batched decode after
that. A different transformers version may build the talker input
differently, so this checks the version it was written against.

With a batch of one the result is generate's, sample for sample: the
thinker is greedy and the talker draws from the same seed in the same
order. That equality is the test this file is held to.
"""

from __future__ import annotations

import torch
import transformers

THINKER_EOS = 151645  # generate's thinker_eos_token_id default
if not transformers.__version__.startswith("5.19."):
    raise ImportError(f"written against transformers 5.19, found {transformers.__version__}")


def talker_kwargs(model, max_new_tokens: int = 4096) -> dict:
    """generate's talker settings, unchanged."""
    c = model.config.talker_config
    vocab = c.text_config.vocab_size
    return {
        "max_new_tokens": max_new_tokens, "do_sample": True, "top_k": 50, "top_p": 1.0,
        "temperature": 0.9, "repetition_penalty": 1.05, "eos_token_id": c.codec_eos_token_id,
        "suppress_tokens": [i for i in range(vocab - 1024, vocab) if i != c.codec_eos_token_id],
        "output_hidden_states": True, "return_dict_in_generate": True,
    }


@torch.no_grad()
def think(model, inputs: dict, max_new_tokens: int):
    """One utterance through the thinker, called as generate calls it."""
    rest = {k: v for k, v in inputs.items() if k != "input_ids"}
    return model.thinker.generate(
        input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens, eos_token_id=THINKER_EOS,
        do_sample=False, use_audio_in_video=False,
        output_hidden_states=True, return_dict_in_generate=True, **rest)


@torch.no_grad()
def talker_prompt(model, input_ids, result, speaker_id: int):
    """generate's step 2 for one unpadded utterance: (embeds, ids, trailing text, pad embed)."""
    cfg = model.config
    device = input_ids.device
    thinker_embed = torch.cat([h[0] for h in result.hidden_states], dim=1).to(device)
    thinker_hidden = torch.cat(
        [h[cfg.talker_config.accept_hidden_layer] for h in result.hidden_states], dim=1).to(device)
    tc = cfg.thinker_config
    multimodal = ((result.sequences == tc.audio_token_id) | (result.sequences == tc.image_token_id)
                  | (result.sequences == tc.video_token_id)).to(device)

    # The user turn: positions whose latest <|im_start|> opens a user message.
    n = input_ids.shape[1]
    positions = torch.arange(n, device=device).unsqueeze(0)
    starts = torch.where(input_ids == cfg.im_start_token_id, positions, positions.new_full((), -1))
    role = input_ids.gather(1, (torch.cummax(starts, dim=-1).values.clamp_min(0) + 1).clamp_max(n - 1))
    user = (role == cfg.user_token_id)[0]
    user_embeds = model._get_talker_user_parts(0, n, multimodal, thinker_hidden, thinker_embed)[:, user]
    user_ids = result.sequences[:, :n][:, user]

    opens = (input_ids[:, :-1] == cfg.im_start_token_id) & (input_ids[:, 1:] == cfg.assistant_token_id)
    start = torch.nonzero(opens[0], as_tuple=False).flatten()[-1].item()
    special = torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                           device=model.thinker.device, dtype=input_ids.dtype)
    bos, eos, pad = model.talker.text_projection(
        model.thinker.get_input_embeddings()(special)).to(device).chunk(3, dim=1)
    embeds, ids, trailing = model._get_talker_assistant_parts(
        start, result.sequences.shape[-1], speaker_id, thinker_embed, pad, bos, eos)
    # generate then moves each row's tts_eos to just after its own text; a
    # single sequence ends on its EOS, where the marker already is.
    return torch.cat((user_embeds, embeds), dim=1), torch.cat((user_ids, ids), dim=1), trailing, pad


@torch.no_grad()
def speak(model, prompts: list, audio_lengths, kwargs: dict) -> list:
    """One talker call over several talker_prompt()s; a 24 kHz waveform each."""
    b = len(prompts)
    pad = prompts[0][3]
    longest = max(p[0].shape[1] for p in prompts)
    text = max(p[2].shape[1] for p in prompts)
    first = prompts[0][0]
    embeds = first.new_zeros(b, longest, first.shape[-1])
    ids = prompts[0][1].new_zeros(b, longest)
    mask = prompts[0][1].new_zeros(b, longest)
    trailing = pad.expand(b, text, -1).clone()  # a shorter text is followed by pads, as generate does
    for i, (e, t, tr, _) in enumerate(prompts):
        embeds[i, longest - e.shape[1]:] = e[0]
        ids[i, longest - t.shape[1]:] = t[0]
        mask[i, longest - t.shape[1]:] = 1
        trailing[i, : tr.shape[1]] = tr[0]

    result = model.talker.generate(
        inputs_embeds=embeds, trailing_text_hidden=trailing, tts_pad_embed=pad.expand(b, -1, -1),
        talker_input_ids=ids, attention_mask=mask, audio_feature_lengths=audio_lengths, **kwargs)
    codes = (torch.stack([h[-1] for h in result.hidden_states if h[-1] is not None], dim=1)
             .transpose(1, 2).to(result.hidden_states[-1][-1].device))
    if b == 1:
        return [model.code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25).float().reshape(-1)]

    # generate's batched decode: codes past a row's own EOS are zeroed, the
    # batch decoded once, and each waveform cut at its own length.
    n = codes.shape[-1]
    sequences = result.sequences[:, :n]
    eos_ids = torch.as_tensor(kwargs["eos_token_id"], device=sequences.device).flatten()
    at = torch.arange(n, device=sequences.device)
    lengths = torch.where((sequences.unsqueeze(-1) == eos_ids).any(dim=-1), at, n).min(dim=-1).values
    codes = codes.masked_fill((at >= lengths.unsqueeze(-1)).unsqueeze(1), 0)
    chunk = 300
    wavs = model.code2wav.chunked_decode(codes, chunk_size=chunk, left_context_size=25)
    per_code = int(model.code2wav.total_upsample)
    lost = (n * per_code - wavs.shape[-1]) // ((n + chunk - 1) // chunk)
    cut = lengths * per_code - lost * ((lengths + chunk - 1) // chunk)
    return [w[..., :k].reshape(-1).float() for w, k in zip(wavs, cut.tolist())]
