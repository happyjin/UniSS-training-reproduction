"""Plan 4A: the Thinker learns, at each read, to say only what its sampled futures agree on, then the wait marker.

Qwen3-Omni's Thinker with LoRA on the attention projections of its top layers (deconly's injection, its frozen
backbone, its adapter-only saves). Rows (data_clean/build_thinker_selfdistill.py): the audio heard so far and the
translation already committed as the start of the reply; the target is the vote's commit at that read and
<|box_end|> (wait), or the rest of the translation and <|im_end|> at the end. Loss: cross-entropy over the target
and its end token only.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_streaming_p2st_traj_v1.training import lora

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
METRIC_NAMES = ("loss", "end_acc", "wait_recall", "target_tokens")
IGNORE = -100
WAIT_ID, EOS_ID = 151649, 151645   # <|box_end|>, <|im_end|>


class ThinkerSDModel(HuggingFaceModule):
    def __init__(self, config, *, qwen3_path: str, lora_rank: int, lora_alpha: float, lora_from: int,
                 recompute: bool = True, pg_collection=None):
        super().__init__(config)
        from transformers import Qwen3OmniMoeForConditionalGeneration

        self.pg_collection = pg_collection
        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(qwen3_path, torch_dtype=torch.bfloat16)
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        self.thinker = omni3.thinker
        del omni3

        def keep(parent: str) -> bool:   # the text model's blocks from lora_from up; the audio tower never
            m = re.search(r"(?:^|\.)model\.layers\.(\d+)\.", parent + ".")
            return m is not None and int(m.group(1)) >= lora_from

        self.adapters = lora.inject(self.thinker, rank=lora_rank, alpha=lora_alpha, targets=ATTN, layer_filter=keep)
        if not self.adapters:
            raise SystemExit("no adapter was injected; the layer filter matched nothing")
        if recompute:
            self.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The adapters only; the 31.7B frozen weights come back from from_pretrained on every start."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters() if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def trainable_report(self) -> dict:
        return {"adapters": len(self.adapters),
                "adapter_m": round(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 2)}

    def forward(self, input_ids, attention_mask, labels, input_features=None, feature_attention_mask=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if input_features is not None:
            kwargs["input_features"] = input_features
            kwargs["feature_attention_mask"] = feature_attention_mask
        out = self.thinker(**kwargs)
        lab = labels[:, 1:]
        sel = lab != IGNORE
        logits = out.logits[:, :-1, :][sel].float()   # gather before the cast: the reply is a few dozen positions
        tgt = lab[sel]
        loss = F.cross_entropy(logits, tgt)
        with torch.no_grad():
            pred = logits.argmax(-1)
            end = (tgt == WAIT_ID) | (tgt == EOS_ID)
            end_acc = (pred[end] == tgt[end]).float().mean() if end.any() else torch.zeros((), device=tgt.device)
            wait = tgt == WAIT_ID
            wait_recall = (pred[wait] == WAIT_ID).float().mean() if wait.any() else torch.zeros((), device=tgt.device)
        return torch.stack([loss, end_acc.float(), wait_recall.float(),
                            torch.tensor(float(tgt.numel()), device=tgt.device)])
