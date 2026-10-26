"""G-STAR-SI (docs/multi_speaker_si_plan_v7.zh-CN.md): the gstar-nemotron branch's G-STAR model with the speech
LLM swapped for the Qwen3-Omni Thinker and the transcript for a Chinese translation.

  audio -> AuT encoder (frozen) -> a_i in R^2048, ~13 / s
  Nemotron-3 posteriors [F80, 8] -> interpolated to the audio-token grid -> SD projector (the branch's SDProjector:
  stride-5 Conv1d + MLP) -> s_j in R^2048, one per five audio tokens
  K:1 interleave inside <|audio_start|> ... <|audio_end|>: a0..a4, s0, a5..a9, s1, ...
  Thinker LLM (frozen, LoRA on the attention of every text layer) -> <|spk k|><|t a|>translation<|t b|> ...

The 251 time tokens (80 ms, <= 20 s) and 8 speaker tokens take the unused vocabulary rows 151676..151934, so the
vocabulary keeps its size; their input and output rows are trained (new_emb, new_head) and written back on merge.
Positions are plain arange: with audio and text only Qwen3-Omni's M-RoPE uses one index on all sections. Loss:
cross-entropy on the reply, time tokens x1.5 and speaker tokens x2 (the branch's hierarchical weights). The logits
are computed only where a label is supervised. Rows without posteriors ("inst": "sd", plain translation) carry no
speaker tokens.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_streaming_p2st_traj_v1.training import lora

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
METRIC_NAMES = ("loss", "spk_acc", "time_acc", "text_acc", "eos_acc", "target_tokens")
IGNORE = -100
EOS_ID = 151645           # <|im_end|>
AUDIO_PAD = 151675        # <|audio_pad|>
N_TIME, N_SLOT = 251, 8
NEW0 = 151676             # first unused row: <|t0|>
SPK0 = NEW0 + N_TIME      # <|spk1|>
NEW_END = SPK0 + N_SLOT   # one past <|spk8|>
K = 5                     # audio tokens per speaker token


def special_tokens() -> list[str]:
    return [f"<|t{i}|>" for i in range(N_TIME)] + [f"<|spk{k}|>" for k in range(1, N_SLOT + 1)]


def n_speaker_tokens(n_audio: int) -> int:
    return math.ceil(n_audio / K)


def post_to_tokens(post: np.ndarray, n_tokens: int, dur_s: float) -> np.ndarray:
    """[F80, 8] posteriors at 80 ms frame centres -> [n_tokens, 8] at the audio-token centres (linear)."""
    if n_tokens <= 0:
        return np.zeros((0, post.shape[1]), dtype=np.float32)
    fc = (np.arange(len(post)) + 0.5) * 0.08
    tc = (np.arange(n_tokens) + 0.5) * dur_s / n_tokens
    return np.stack([np.interp(tc, fc, post[:, j].astype(np.float32)) for j in range(post.shape[1])], 1)


def interleave(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """a [N, D], s [ceil(N/K), D] -> [N + ceil(N/K), D]: K audio tokens, then their speaker token."""
    out = []
    for j in range(s.shape[0]):
        out.append(a[j * K: (j + 1) * K])
        out.append(s[j: j + 1])
    return torch.cat(out, 0)


class SDProjector(nn.Module):
    """The gstar-nemotron branch's speaker-stream projector (gstar/model.py)."""

    def __init__(self, c_in: int, d_llm: int, hidden: int = 1024, stride: int = K):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(c_in, hidden, kernel_size=stride, stride=stride)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        nn.init.zeros_(self.mlp[-1].weight)   # the model starts as the plain Thinker reading zeros where s_j sit
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, L, C] -> [B, ceil(L/stride), d]
        pad = (-x.shape[1]) % self.stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return self.mlp(self.conv(x.transpose(1, 2)).transpose(1, 2))


class ThinkerGStarModel(HuggingFaceModule):
    def __init__(self, config, *, qwen3_path: str, lora_rank: int, lora_alpha: float, lora_from: int,
                 recompute: bool = True, use_sd: bool = True, init_merged: bool = False, pg_collection=None):
        super().__init__(config)
        from transformers import Qwen3OmniMoeForConditionalGeneration

        self.pg_collection = pg_collection
        self.use_sd = use_sd
        self.init_merged = init_merged
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
        d = self.thinker.config.text_config.hidden_size
        emb = self.thinker.get_input_embeddings().weight
        head = self.thinker.lm_head.weight
        # created after inject (which freezes everything else), so these train
        self.sd_proj = SDProjector(N_SLOT, d).float()
        if init_merged:   # continue a merged G-STAR-SI model: its trained token rows and projector are the start
            from pathlib import Path

            self.new_emb = nn.Parameter(emb[NEW0:NEW_END].float().clone())
            self.new_head = nn.Parameter(head[NEW0:NEW_END].float().clone())
            self.sd_proj.load_state_dict(torch.load(Path(qwen3_path) / "sd_proj.pt", map_location="cpu"))
        else:
            self.new_emb = nn.Parameter(emb[:NEW0].float().mean(0, keepdim=True).repeat(N_TIME + N_SLOT, 1)
                                        + 0.02 * emb[:NEW0].float().std() * torch.randn(N_TIME + N_SLOT, d))
            self.new_head = nn.Parameter(head[:NEW0].float().mean(0, keepdim=True).repeat(N_TIME + N_SLOT, 1).clone())
        if recompute:
            self.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Trainable tensors only: the adapters, the projector, the new token rows."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters() if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def trainable_report(self) -> dict:
        return {"adapters": len(self.adapters),
                "trainable_m": round(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 2),
                "use_sd": self.use_sd, "init_merged": self.init_merged}

    def embed(self, input_ids, input_features, feature_attention_mask, sd_post, n_audio, has_sd):
        """Text embeddings with the new rows, then the audio (and speaker) tokens scattered over <|audio_pad|>."""
        is_new = (input_ids >= NEW0) & (input_ids < NEW_END)
        emb = self.thinker.get_input_embeddings()(input_ids.masked_fill(is_new, 0))
        if is_new.any():
            emb = emb.clone()
            emb[is_new] = self.new_emb[input_ids[is_new] - NEW0].to(emb.dtype)
        if input_features is None:
            return emb
        a = self.thinker.get_audio_features(input_features, feature_attention_mask).last_hidden_state.to(emb.dtype)
        assert a.shape[0] == int(n_audio.sum()), (a.shape, n_audio)
        parts, pos = [], 0
        for i in range(input_ids.shape[0]):
            n = int(n_audio[i])
            ai = a[pos: pos + n]
            pos += n
            if bool(has_sd[i]):
                s = self.sd_proj(sd_post[i: i + 1, :n].to(self.sd_proj.conv.weight.dtype))[0]
                if not self.use_sd:
                    s = torch.zeros_like(s)
                ai = interleave(ai, s.to(emb.dtype))
            parts.append(ai)
        feats = torch.cat(parts, 0)
        mask = (input_ids == AUDIO_PAD).unsqueeze(-1)
        assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
        return emb.masked_scatter(mask.to(emb.device), feats)

    def logits_at(self, h: torch.Tensor) -> torch.Tensor:
        """Output logits with the new tokens' rows replaced by new_head."""
        logits = self.thinker.lm_head(h).float()
        logits[:, NEW0:NEW_END] = h.float() @ self.new_head.float().t()
        return logits

    def forward(self, input_ids, attention_mask, labels, input_features=None, feature_attention_mask=None,
                sd_post=None, n_audio=None, has_sd=None):
        emb = self.embed(input_ids, input_features, feature_attention_mask, sd_post, n_audio, has_sd)
        h = self.thinker.model(inputs_embeds=emb, attention_mask=attention_mask, return_dict=True).last_hidden_state
        lab = labels[:, 1:]
        sel = lab != IGNORE
        logits = self.logits_at(h[:, :-1][sel])
        tgt = lab[sel]
        is_time = (tgt >= NEW0) & (tgt < SPK0)
        is_spk = (tgt >= SPK0) & (tgt < NEW_END)
        w = torch.ones_like(tgt, dtype=torch.float32) + 0.5 * is_time.float() + 1.0 * is_spk.float()
        loss = (F.cross_entropy(logits, tgt, reduction="none") * w).sum() / w.sum()
        with torch.no_grad():
            pred = logits.argmax(-1)
            hit = (pred == tgt).float()
            zero = torch.zeros((), device=tgt.device)
            acc = lambda m: hit[m].mean() if m.any() else zero   # noqa: E731
            is_eos = tgt == EOS_ID
        return torch.stack([loss, acc(is_spk), acc(is_time), acc(~is_spk & ~is_time & ~is_eos), acc(is_eos),
                            torch.tensor(float(tgt.numel()), device=tgt.device)])
