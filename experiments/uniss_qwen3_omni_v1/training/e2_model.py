"""E2: Qwen3-Omni's Thinker and UniSS as one model, the Talker trained on the Thinker.

P2 step zero showed the chain works untrained: UniSS reading the Thinker's
translation tokens holds P1's ASR-BLEU and keeps the speaker's voice,
register and timing. E2 trains it as one system, on the Thinker's own
states, with the speech loss reaching across the boundary.

  Thinker   Qwen3-Omni 30B-A3B, frozen. Reads the source speech and the
            translation; its last-layer state at each translation token is
            the hidden input below.
  Talker    UniSS (Qwen2 1.5B), trained. Reads exactly P2 step zero's
            sequence, with three inputs added to its embeddings:
              hidden    each translation token gets gate * W(h), h the
                        Thinker's RMS-normalised state for that token;
              prosody   each source GLM-4 token gets the source's pitch
                        and energy at that moment;
              duration  <|start_semantic_token|> gets the length to speak
                        for, absolute and relative to the source.
            The gate and both projections start at zero, so step zero is
            P2 step zero exactly, and the ASR-BLEU it starts from is the one
            measured there.
  Codec     BiCodec, outside the model: the source's 32 global tokens
            decode the predicted semantic codes in the speaker's voice.

The objective is cross-entropy on the semantic codes and the end token.
Gradients reach the Talker, the gate and the three projections; the
Thinker is frozen (E3 would open it with an adapter).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_qwen3_omni_v1.training.e2_batch import DURATION_DIM
from experiments.uniss_qwen3_omni_v1.training.e2_data import PROSODY_DIM

THINKER_HIDDEN = 2048
# UniSS's embedding rows sit at RMS 0.02-0.04; an added input at about that
# size can move the sequence without swamping the token it is added to.
ADD_SCALE = 0.03
# The KL anchor compares distributions over what UniSS may write at a code
# position: the 8,192 BiCodec semantic codes and the end token.
SEMANTIC_0, SEMANTIC_LAST, END_SEMANTIC = 155761, 163952, 180392


class E2Model(HuggingFaceModule):
    """added_inputs=False: UniSS alone -- no Thinker loaded, no added inputs
    (step 6000's ablation: switching all three off changed nothing). The
    E2 runs to date trained them; the default keeps that.

    kl_anchor: a frozen copy of UniSS as it started; loss = CE + kl_weight *
    KL(anchor || model) over code positions. Step 4000 learned to follow the
    length it was asked for and lost code-switched pronunciation on the way;
    the anchor keeps what it was not asked to change."""

    def __init__(self, config, *, qwen3_path: str, uniss_path: str, recompute: bool = True,
                 added_inputs: bool = True, kl_anchor: str | None = None, kl_weight: float = 0.0):
        super().__init__(config)
        from transformers import AutoModelForCausalLM, Qwen3OmniMoeForConditionalGeneration

        self.added_inputs, self.kl_weight = added_inputs, float(kl_weight)
        talker = AutoModelForCausalLM.from_pretrained(uniss_path, torch_dtype=torch.bfloat16)
        self.talker = talker
        self.anchor = None
        if kl_anchor and self.kl_weight > 0:
            self.anchor = AutoModelForCausalLM.from_pretrained(kl_anchor, torch_dtype=torch.bfloat16)
            self.anchor.requires_grad_(False)
            self.anchor.eval()
        self.thinker = None
        if not added_inputs:
            self.talker.requires_grad_(True)
            if recompute:
                self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return
        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(qwen3_path, torch_dtype=torch.bfloat16)
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        self.thinker = omni3.thinker
        del omni3
        width = talker.config.hidden_size

        self.hid_proj = torch.nn.Linear(THINKER_HIDDEN, width, bias=False)
        torch.nn.init.normal_(self.hid_proj.weight, std=THINKER_HIDDEN ** -0.5)
        self.hid_gate = torch.nn.Parameter(torch.zeros(width))
        self.pros_proj = torch.nn.Linear(PROSODY_DIM, width)
        self.dur_proj = torch.nn.Linear(DURATION_DIM, width)
        for layer in (self.pros_proj, self.dur_proj):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        for module in (self.hid_proj, self.pros_proj, self.dur_proj):
            module.to(torch.bfloat16)
        self.hid_gate.data = self.hid_gate.data.to(torch.bfloat16)
        # Megatron's --decoupled-lr applies to parameters carrying this flag:
        # the added inputs start at zero and learn at their own rate, while
        # UniSS, already trained, moves at the small base rate.
        for p in (*self.hid_proj.parameters(), self.hid_gate, *self.pros_proj.parameters(),
                  *self.dur_proj.parameters()):
            p.is_embedding_or_output_parameter = True

        self.thinker.requires_grad_(False)
        self.thinker.eval()
        self.talker.requires_grad_(True)
        if recompute:
            self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Only what trains: the frozen Thinker is 63 GB and comes back from
        from_pretrained on every start, as in phase E."""
        full = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        frozen = {prefix + name for name, param in self.named_parameters() if not param.requires_grad}
        kept = {k: v for k, v in full.items() if k not in frozen}
        if not kept:
            raise SystemExit("every key was filtered out; the predicate is wrong")
        return kept

    def train(self, mode: bool = True):
        super().train(mode)
        for frozen in (self.thinker, self.anchor):
            if frozen is not None:
                frozen.eval()
        return self

    def trainable_report(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        report = {"talker_m": round(count(self.talker) / 1e6, 1), "added_inputs": self.added_inputs,
                  "kl_weight": self.kl_weight, "anchor": self.anchor is not None}
        if self.added_inputs:
            report["added_m"] = round((count(self.hid_proj) + self.hid_gate.numel() + count(self.pros_proj)
                                       + count(self.dur_proj)) / 1e6, 2)
            report["thinker_frozen_m"] = round(sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1)
        return report

    def _kl(self, uniss_ids, uniss_mask, label_b, label_p, logits):
        if self.anchor is None:
            return torch.zeros((), device=logits.device)
        with torch.no_grad():
            states = self.anchor.model(inputs_embeds=self.anchor.get_input_embeddings()(uniss_ids),
                                       attention_mask=uniss_mask, return_dict=True).last_hidden_state
            ref = self.anchor.lm_head(states[label_b, label_p]).float()
        keep = torch.cat([torch.arange(SEMANTIC_0, SEMANTIC_LAST + 1, device=logits.device),
                          torch.tensor([END_SEMANTIC], device=logits.device)])
        ref_lp = F.log_softmax(ref[:, keep], dim=-1)
        own_lp = F.log_softmax(logits[:, keep], dim=-1)
        return (ref_lp.exp() * (ref_lp - own_lp)).sum(-1).mean()

    def forward(self, uniss_ids, uniss_mask, label_b, label_p, labels, thinker_input_ids=None,
                thinker_attention_mask=None, reply_start=None, content_b=None, content_p=None, content_j=None,
                glm_b=None, glm_p=None, glm_feat=None, start_b=None, start_p=None, start_feat=None,
                input_features=None, feature_attention_mask=None):
        if not self.added_inputs:
            embeds = self.talker.get_input_embeddings()(uniss_ids)
            states = self.talker.model(inputs_embeds=embeds, attention_mask=uniss_mask,
                                       return_dict=True).last_hidden_state
            logits = self.talker.lm_head(states[label_b, label_p]).float()
            ce = F.cross_entropy(logits, labels)
            kl = self._kl(uniss_ids, uniss_mask, label_b, label_p, logits)
            return torch.stack([ce + self.kl_weight * kl, ce.detach(), kl.detach(),
                                torch.tensor(float(labels.numel()), device=ce.device)])
        with torch.no_grad():
            kwargs = {"input_ids": thinker_input_ids, "attention_mask": thinker_attention_mask,
                      "output_hidden_states": True, "return_dict": True}
            if input_features is not None:
                kwargs["input_features"] = input_features
                kwargs["feature_attention_mask"] = feature_attention_mask
            out = self.thinker(**kwargs)
            hidden = out.hidden_states[-1][content_b, int(reply_start) + content_j].float()
            hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
            del out
        embeds = self.talker.get_input_embeddings()(uniss_ids)
        dtype = embeds.dtype
        added = self.hid_gate.to(dtype) * (ADD_SCALE * self.hid_proj(hidden.to(dtype)))
        embeds = embeds.index_put((content_b, content_p), added, accumulate=True)
        embeds = embeds.index_put((glm_b, glm_p), ADD_SCALE * self.pros_proj(glm_feat.to(dtype)), accumulate=True)
        embeds = embeds.index_put((start_b, start_p), ADD_SCALE * self.dur_proj(start_feat.to(dtype)),
                                  accumulate=True)
        states = self.talker.model(inputs_embeds=embeds, attention_mask=uniss_mask,
                                   return_dict=True).last_hidden_state
        logits = self.talker.lm_head(states[label_b, label_p]).float()
        ce = F.cross_entropy(logits, labels)
        kl = self._kl(uniss_ids, uniss_mask, label_b, label_p, logits)
        return torch.stack([ce + self.kl_weight * kl, ce.detach(), kl.detach(),
                            torch.tensor(float(labels.numel()), device=ce.device)])
