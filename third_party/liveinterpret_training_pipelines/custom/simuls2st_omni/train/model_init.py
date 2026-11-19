"""Shared model construction for the three training stages.

`_resize_talker_vocab` lives in the inference module, where it runs *before*
loading trained weights. Training needs the same operation as a genuine
re-initialisation, so it is lifted here rather than imported from inference:
Stage 1 depends on the fresh init, and an accidental change on the inference
side must not silently alter training behaviour.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DualCodec semantic codebook plus the four control tokens the talker needs.
N_SEMANTIC_CODES = 16_384
N_CONTROL_TOKENS = 4


def resize_talker_vocab(talker, n_audio_codes: int = N_SEMANTIC_CODES) -> dict:
    """Re-initialise the talker codec embedding and head for `n_audio_codes`.

    Control token ids are assigned immediately above the codebook:
    `pad = n`, `bos = n+1`, `eos = n+2`, `mask = n+3`.

    Returns the control token id map.
    """
    new_vocab_size = n_audio_codes + N_CONTROL_TOKENS
    codec_pad_token_id = n_audio_codes
    codec_bos_token_id = n_audio_codes + 1
    codec_eos_token_id = n_audio_codes + 2
    codec_mask_token_id = n_audio_codes + 3

    old_embed = talker.model.embed_tokens
    old_head = talker.codec_head
    device = old_embed.weight.device
    dtype = old_embed.weight.dtype

    needs_resize = (
        old_embed.num_embeddings != new_vocab_size
        or old_head.out_features != new_vocab_size
    )
    if needs_resize:
        talker.model.embed_tokens = nn.Embedding(
            new_vocab_size,
            old_embed.embedding_dim,
            padding_idx=codec_pad_token_id,
            device=device,
            dtype=dtype,
        )
        talker.codec_head = nn.Linear(
            old_head.in_features,
            new_vocab_size,
            bias=False,
            device=device,
            dtype=old_head.weight.dtype,
        )
        nn.init.normal_(talker.model.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(talker.codec_head.weight, mean=0.0, std=0.02)
        logger.info(
            "Re-initialised talker codec embedding/head: %s -> %s",
            old_embed.num_embeddings,
            new_vocab_size,
        )
    else:
        logger.info("Talker codec vocab already at %s", new_vocab_size)

    talker.config.vocab_size = new_vocab_size
    talker.vocab_size = new_vocab_size
    talker.codebook_size = new_vocab_size
    talker.config.tts_codec_pad_token_id = codec_pad_token_id
    talker.config.tts_codec_start_token_id = codec_bos_token_id
    talker.config.tts_codec_end_token_id = codec_eos_token_id
    talker.config.tts_codec_mask_token_id = codec_mask_token_id
    talker.codec_pad_token = codec_pad_token_id
    talker.codec_bos_token = codec_bos_token_id
    talker.codec_eos_token = codec_eos_token_id
    talker.codec_mask_token = codec_mask_token_id

    return {
        "codec_pad": codec_pad_token_id,
        "codec_bos": codec_bos_token_id,
        "codec_eos": codec_eos_token_id,
        "codec_mask": codec_mask_token_id,
        "vocab_size": new_vocab_size,
    }


def load_base_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    n_audio_codes: int = N_SEMANTIC_CODES,
):
    """Load Qwen2.5-Omni with the talker enabled and its codec vocab reset.

    Mirrors the inference-side preparation: the talker is enabled explicitly,
    and `token2wav` / `visual` are dropped because this stack never uses them
    and they cost several GB of VRAM per process.
    """
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map=device_map,
        trust_remote_code=True,
    )
    if not hasattr(model, "talker"):
        model.enable_talker()
    if hasattr(model, "token2wav"):
        del model.token2wav
    if getattr(model.thinker, "visual", None) is not None:
        del model.thinker.visual
        model.thinker.visual = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The talker conditions on thinker embeddings of its text control tokens, so
    # the thinker vocab must cover those ids.
    min_vocab = max(
        len(processor.tokenizer),
        model.talker.config.tts_text_pad_token_id + 1,
        model.talker.config.tts_text_start_token_id + 1,
        model.talker.config.tts_text_end_token_id + 1,
        model.config.thinker_config.audio_token_index + 1,
    )
    if model.thinker.get_input_embeddings().weight.shape[0] < min_vocab:
        model.thinker.resize_token_embeddings(min_vocab)

    codec_tokens = resize_talker_vocab(model.talker, n_audio_codes)
    return model, processor, codec_tokens
