"""sst_hsmm_v1 Thinker side (reports/sst_hsmm_v1/prereg.zh-CN.md sections 0, 2): the UE Thinker (wave17 merged_enc_wise05)
with LoRA on the 48 text layers (q/k/v/o, r16, a32; wave5's injection and checkpointing), and the prefix forward: one
teacher-forced pass per chunk c (audio <= min(c s, X) + the reference + <|im_end|>, wave5's prompt), batched; returns
lp_tok (C, n) = log p(y_i | audio <= c, y_<i) and lp_wait (C, n) = log P(<|fim_pad|>) at the same positions. lm_head is
applied only at those positions (the forward runs with an identity head).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
THINKER = REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"
WAIT = 151662          # <|fim_pad|>
IM_START, IM_END = 151644, 151645
TGT = {"en2zh": "cmn", "zh2en": "eng"}
RANK_LORA, ALPHA, N_LAST = 16, 32.0, 4


def processor_and_heads():
    from transformers import Qwen3OmniMoeProcessor
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    proc = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3))
    proc.tokenizer.padding_side = "right"
    ins = inference_instruction()
    head = {}
    for d in ("zh2en", "en2zh"):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]]}]}]
        head[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return proc, head


def build_thinker(dev, lora: bool):
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M
    from experiments.uniss_streaming_p2st_traj_v1.training import lora as L
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(THINKER), torch_dtype=torch.bfloat16,
                                                                device_map={"": dev}, attn_implementation="sdpa")
    thinker = omni.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni, attr):
            delattr(omni, attr)
    del omni
    thinker.requires_grad_(False)
    names = {}
    if lora:
        text = thinker.model
        ad = L.inject(text, rank=RANK_LORA, alpha=ALPHA, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
        n_layers = len(text.layers)
        assert len(ad) == 4 * n_layers, (len(ad), n_layers)
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        eager = M.Qwen3OmniMoeThinkerTextExperts.forward.__wrapped__
        for li in range(n_layers - N_LAST, n_layers):
            text.layers[li].gradient_checkpointing = False
            ex = text.layers[li].mlp.experts
            ex.forward = types.MethodType(eager, ex)
        names = {nm: p for nm, p in text.named_parameters() if p.requires_grad}
        thinker.train()
    else:
        thinker.eval()
    thinker.audio_tower.eval()
    return thinker, names


def prefix_batch(proc, head, row: dict, wav):
    """row: data row (dir, ref, tok, X/C); wav: float32 16 kHz source (clean or mixed). -> processor inputs, positions"""
    C, X = row["C"], row["X"]
    tg = row["tok"] + [IM_END]          # prereg change 5: the end-of-turn token is the last target token
    n = len(tg)
    pre = [wav[: min(c * 16000, X * 16)] for c in range(1, C + 1)]
    text = head[row["dir"]] + row["ref"] + "<|im_end|>"
    inp = proc(text=[text] * C, audio=pre, return_tensors="pt", padding=True)
    ids = inp["input_ids"]
    pos = []
    for b in range(C):
        r = ids[b].tolist()
        st = max(i for i, t in enumerate(r) if t == IM_START) + 3
        if r[st: st + n] != tg:
            return None, None
        pos.append(list(range(st - 1, st - 1 + n)))
    return inp, torch.tensor(pos)


def emissions(thinker, inp, pos, dev):
    """-> lp_tok (C, n) fp32, lp_wait (C, n) fp32 (graph kept if grads are enabled)"""
    inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
    head = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        out = thinker(**inp, output_router_logits=False, use_cache=False)
    finally:
        thinker.lm_head = head
    hid = out.logits                                            # (C, L, H): hidden states
    C, n = pos.shape
    b = torch.arange(C, device=dev)[:, None].expand(C, n)
    h = hid[b, pos.to(dev)]                                     # (C, n, H)
    lsm = torch.log_softmax(head(h).float(), dim=-1)
    ids = inp["input_ids"][b, pos.to(dev) + 1]
    return lsm.gather(-1, ids[..., None])[..., 0], lsm[..., WAIT]


def prefix_batch_multi(proc, head, rows: list, wavs: list):
    """several samples' prefix batches in ONE processor call (one forward; speed only, same per-sample math as
    prefix_batch). -> (inp, [(row index, offset, pos (C_j, n_j))]) for the samples whose tokens match; (None, []) if none."""
    keep = list(range(len(rows)))
    while keep:
        texts, auds = [], []
        for j in keep:
            r, w = rows[j], wavs[j]
            C, X = r["C"], r["X"]
            auds += [w[: min(c * 16000, X * 16)] for c in range(1, C + 1)]
            texts += [head[r["dir"]] + r["ref"] + "<|im_end|>"] * C
        inp = proc(text=texts, audio=auds, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        meta, bad, off = [], [], 0
        for j in keep:
            r = rows[j]
            tg = r["tok"] + [IM_END]        # prereg change 5
            n, C = len(tg), r["C"]
            pos, ok = [], True
            for b in range(off, off + C):
                t = ids[b].tolist()
                st = max(i for i, x in enumerate(t) if x == IM_START) + 3
                if t[st: st + n] != tg:
                    ok = False
                    break
                pos.append(list(range(st - 1, st - 1 + n)))
            if ok:
                meta.append((j, off, torch.tensor(pos)))
            else:
                bad.append(j)
            off += C
        if not bad:
            return inp, meta
        keep = [j for j in keep if j not in bad]
    return None, []


def emissions_multi(thinker, inp, meta, dev):
    """one forward for prefix_batch_multi's batch -> [(row index, lp_tok (C_j, n_j), lp_wait (C_j, n_j))]"""
    inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
    head = thinker.lm_head
    thinker.lm_head = nn.Identity()
    try:
        out = thinker(**inp, output_router_logits=False, use_cache=False)
    finally:
        thinker.lm_head = head
    hid = out.logits
    res = []
    for j, off, pos in meta:
        C, n = pos.shape
        b = (off + torch.arange(C, device=dev))[:, None].expand(C, n)
        p = pos.to(dev)
        lsm = torch.log_softmax(head(hid[b, p]).float(), dim=-1)
        ids = inp["input_ids"][b, p + 1]
        res.append((j, lsm.gather(-1, ids[..., None])[..., 0], lsm[..., WAIT]))
    return res
