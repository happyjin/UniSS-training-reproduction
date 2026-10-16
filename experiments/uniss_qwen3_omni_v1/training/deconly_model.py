"""Decoder-only: Qwen3's Thinker emits the codes itself, no Talker.

Four training routes have now put a separately-trained Talker in front of
audio-conditioned states and all four stopped near 20 against a 31.08
cascade -- s2st at 75% and at 100%, a five-task mixture with a thinker
LoRA, and an audio-conditioned warmup. The talker reads TTS-prompt states
at 31.08 and audio-conditioned ones at 0.69; a least-squares map between
the two families moved held-out cosine 0.594 to 0.605, so there is no
fixed transform either. Every one of those failures lives in the boundary
between two modules.

This route removes the boundary. The model that just heard the audio is
the model that emits the codes, so there is no second module with its own
learned input distribution and nothing for the 54 degrees to be measured
across.

It is SimulS2ST-Omni's Dec-only baseline, which that paper measures at
27.12 offline against 31.12 for Thinker-Talker, and worse again under
streaming. That is a lower ceiling than the route this replaces -- but the
route this replaces has not reached its own ceiling in four attempts, and
27.12 reached beats 31.08 unreached.

Two tables rather than UniSS's single vocabulary. Qwen3's 152,064 rows
carry the zero-shot translation this project is built on, and growing that
table drops 8,193 randomly initialised code rows into the softmax the text
competes in. Here position decides the modality: lm_head reads the reply
positions, audio_head reads the code positions, and neither sees the
other's classes.

    L = w_text * text_ce + w_code * code_ce

Both terms are causal-LM shifted. A task that supervises one stream
contributes one term, exactly as in the two-stream route.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_omni_s2st_v1.training.stage2_tasks import THINKER_TARGETS
from experiments.uniss_streaming_p2st_traj_v1.training import lora

from experiments.uniss_qwen3_omni_v1.training.deconly_batch import (  # noqa: F401
    BICODEC_CODES,
    BICODEC_GLOBAL_SIZE,
    CODE_EOS,
    CODE_VOCAB,
    IGNORE,
    THINKER_HIDDEN,
)

# Only the top layers carry adapters: autograd keeps activations back to
# the earliest parameter needing a gradient, so this bounds memory by how
# many layers sit above the first one. "all" adapts the whole text LM.
LORA_LAYERS_FROM = os.environ.get("UNISS_DECONLY_LORA_FROM", "32")
# "none" leaves the thinker bit-identical to the release: the only trained
# tensors are the two embeddings and the head, all at the edges of the
# sequence. The translation is then provably unchanged, and the question
# the run answers narrows to whether a frozen backbone can emit codes
# given nothing but a new input table and a new output head.
#
# global_embed can start from a two-stream export rather than from noise.
# phase_d trained one for 14,000 steps at exactly this shape -- 4096 rows
# of 2048 -- so the speaker conditioning does not have to be relearned.
GLOBAL_INIT = os.environ.get("UNISS_DECONLY_GLOBAL_INIT", "")
# Mixture over stage2_tasks. tts is excluded: it draws from a cache of
# Qwen2.5 token ids, which this route cannot read.
MIXTURE = {"s2st": 1.0}
if os.environ.get("UNISS_DECONLY_MIXTURE") == "warmup":
    # The paper gives Dec-only a warmup "to prevent modality underfitting"
    # -- without it one head learns and the other is ignored. asr and s2tt
    # hold the text side while s2st teaches the code side.
    MIXTURE = {"asr": 0.2, "s2tt": 1.0, "s2st": 1.5}


def _layer_filter():
    if LORA_LAYERS_FROM == "all":
        return None
    first = int(LORA_LAYERS_FROM)

    def keep(parent_name: str) -> bool:
        # The text LM's blocks only, from `first` up. The audio tower is
        # excluded at the source rather than frozen afterwards.
        m = re.search(r"(?:^|\.)model\.layers\.(\d+)\.", parent_name + ".")
        return m is not None and int(m.group(1)) >= first

    return keep


class DecOnlyModel(HuggingFaceModule):
    """Qwen3's Thinker with a code embedding and a code head bolted on."""

    def __init__(self, config, *, qwen3_path: str, lora_rank: int = 32,
                 lora_alpha: float = 64.0, text_weight: float = 1.0,
                 code_weight: float = 1.0, recompute: bool = True):
        super().__init__(config)
        from transformers import Qwen3OmniMoeForConditionalGeneration

        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_path, torch_dtype=torch.bfloat16)
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        self.thinker = omni3.thinker
        del omni3

        self.text_weight = float(text_weight)
        self.code_weight = float(code_weight)

        # inject() freezes everything it walks, so the adapters go in first
        # and the new modules are created after.
        if LORA_LAYERS_FROM == "none":
            self.thinker_lora = []
            self.thinker.requires_grad_(False)
            # text_ce's only gradient outlet is the adapter: lm_head and
            # every layer under it are frozen. Left on it would cost a
            # 152,064-way softmax over the whole sequence to produce a
            # gradient that is identically zero.
            if self.text_weight:
                print("[deconly] LORA_FROM=none: text_weight -> 0"
                      " (no trainable parameter lies behind text_ce)",
                      flush=True)
                self.text_weight = 0.0
        else:
            self.thinker_lora = lora.inject(
                self.thinker, rank=lora_rank, alpha=lora_alpha,
                targets=THINKER_TARGETS, layer_filter=_layer_filter(),
            )
        self.lora_from = LORA_LAYERS_FROM

        dtype = self.thinker.get_input_embeddings().weight.dtype
        self.audio_embed = nn.Embedding(CODE_VOCAB, THINKER_HIDDEN, dtype=dtype)
        self.global_embed = nn.Embedding(BICODEC_GLOBAL_SIZE, THINKER_HIDDEN,
                                         dtype=dtype)
        self.audio_head = nn.Linear(THINKER_HIDDEN, CODE_VOCAB, dtype=dtype)
        # At the scale of the thinker's own embedding rows: drawn larger the
        # code positions shout over the text ones in the first steps, drawn
        # smaller they are ignored and the code head never gets a gradient
        # worth the name.
        scale = float(self.thinker.get_input_embeddings().weight.float().std())
        for mod in (self.audio_embed, self.global_embed):
            nn.init.normal_(mod.weight, std=scale)
        nn.init.normal_(self.audio_head.weight, std=scale)
        nn.init.zeros_(self.audio_head.bias)

        if GLOBAL_INIT:
            donor = torch.load(GLOBAL_INIT, map_location="cpu",
                               weights_only=False).get("global_embed")
            if donor is None:
                raise SystemExit(f"{GLOBAL_INIT} carries no global_embed")
            if tuple(donor.shape) != tuple(self.global_embed.weight.shape):
                raise SystemExit(
                    f"global_embed is {tuple(self.global_embed.weight.shape)}"
                    f" but {GLOBAL_INIT} holds {tuple(donor.shape)}")
            with torch.no_grad():
                self.global_embed.weight.copy_(donor)
            print(f"[deconly] global_embed warm-started from {GLOBAL_INIT}",
                  flush=True)
        self.global_init = GLOBAL_INIT

        for mod in (self.audio_embed, self.global_embed, self.audio_head):
            mod.requires_grad_(True)

        if recompute:
            # use_reentrant=False because the base weights are frozen: the
            # reentrant form needs an input that requires grad, and with
            # LoRA the first one is several layers in.
            self.thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.recompute = recompute

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Only what trains -- the thinker's 31.7B come back from
        from_pretrained on every start, and writing them made a checkpoint
        65 GB in the two-stream route."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters()
                  if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def trainable_report(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        adapter = sum(p.numel() for layer in self.thinker_lora
                      for p in (layer.a, layer.b))
        return {
            "model": "deconly",
            "thinker_lora_m": round(adapter / 1e6, 2),
            "thinker_lora_layers": len(self.thinker_lora),
            "lora_from": self.lora_from,
            "global_init": self.global_init or "random",
            "text_weight": self.text_weight,
            "thinker_frozen_m": round(
                sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1),
            "audio_embed_m": round(count(self.audio_embed) / 1e6, 2),
            "global_embed_m": round(count(self.global_embed) / 1e6, 2),
            "audio_head_m": round(count(self.audio_head) / 1e6, 2),
            "mixture": str(MIXTURE),
        }

    def forward(self, input_ids, attention_mask, text_labels, speaker_ids,
                code_ids, code_labels, code_start, supervise_text,
                supervise_codes, input_features=None,
                feature_attention_mask=None):
        device = input_ids.device
        has_codes = bool(supervise_codes.flatten()[0])
        start = int(code_start.flatten()[0])

        # Build the embeddings ourselves so the code and speaker slots can
        # be written. The thinker still scatters audio into whatever
        # inputs_embeds it is handed -- it locates the slots from input_ids,
        # so passing both keeps the audio path intact.
        embeds = self.thinker.get_input_embeddings()(input_ids)
        if has_codes:
            width = code_ids.shape[1]
            spk = self.global_embed(speaker_ids)
            embeds[:, start - spk.shape[1] : start, :] = spk.to(embeds.dtype)
            embeds[:, start : start + width, :] = (
                self.audio_embed(code_ids).to(embeds.dtype))

        kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if input_features is not None:
            kwargs["input_features"] = input_features
            kwargs["feature_attention_mask"] = feature_attention_mask
        out = self.thinker(**kwargs)

        zero = torch.zeros((), device=device, dtype=torch.float32)

        text_loss = zero
        if bool(supervise_text.flatten()[0]):
            labels = text_labels[:, 1:]
            # Gather before the float cast: the prompt is hundreds of
            # positions and the reply a dozen, and casting the whole
            # sequence over a 152,064 vocabulary spends gigabytes to score
            # the few that carry a label.
            sel = labels != IGNORE
            if sel.any():
                text_loss = F.cross_entropy(
                    out.logits[:, :-1, :][sel].float(), labels[sel])

        code_loss = zero
        if has_codes:
            width = code_ids.shape[1]
            # Position start-1 is the last speaker slot, and it is what
            # predicts the first code. Reading from `start` instead would
            # train every step on the code before it.
            hidden = out.hidden_states[-1][:, start - 1 : start + width - 1, :]
            logits = self.audio_head(hidden)
            code_loss = F.cross_entropy(
                logits.float().reshape(-1, CODE_VOCAB),
                code_labels.reshape(-1), ignore_index=IGNORE)

        total = self.text_weight * text_loss + self.code_weight * code_loss
        return torch.stack([total, text_loss.detach(), code_loss.detach()])
