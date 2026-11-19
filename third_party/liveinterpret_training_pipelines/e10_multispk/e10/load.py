"""Model loading outside swift (tests / inference), mirroring StreamingSILoader + the S5 inference loader."""
from __future__ import annotations

import json
import os

import torch


def load_base(model_dir, dtype="bfloat16", device="cuda:0", text_w=1.0, speech_w=0.0, decision_w=1.0):
    from transformers import AutoTokenizer
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens, s4_stream_token_ids
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration
    model = StreamingSIForConditionalGeneration.from_pretrained(model_dir, dtype=getattr(torch, dtype))
    tok = AutoTokenizer.from_pretrained(model_dir)
    add_s4_stream_tokens(tok, capacity=int(model.config.vocab_size))
    ids = s4_stream_token_ids(tok)
    model.config.s4_stream_token_ids = ids
    model.config.text_loss_weight, model.config.speech_loss_weight = float(text_w), float(speech_w)
    model.config.s4_decision_weight = decision_w
    return model.to(device), tok, ids


def load_s5(model_dir, adapter, merge=True, **kw):
    """Base + the stock S5 adapter (E10-0 / reference), exactly as infer_s5_streaming_dynchunk.load_model."""
    import peft
    model, tok, ids = load_base(model_dir, **kw)
    meta = json.load(open(os.path.join(adapter, "s4_metadata.json")))
    if [int(x) for x in meta["stream_token_ids"]] != ids:
        raise SystemExit(f"stream ids {ids} != adapter {meta['stream_token_ids']}")
    pm = peft.PeftModel.from_pretrained(model, adapter)
    return (pm.merge_and_unload().eval() if merge else pm), tok, ids


def load_e10_model(model_dir, adapter, merge=True, **kw):
    from .tuner import load_e10
    model, tok, ids = load_base(model_dir, **kw)
    pm = load_e10(model, adapter, is_trainable=False)
    return (pm.merge_and_unload().eval() if merge else pm), tok, ids
