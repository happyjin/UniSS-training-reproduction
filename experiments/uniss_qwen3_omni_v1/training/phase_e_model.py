"""Phase E: the Talker learns to read Qwen3 in the context it will be used.

Phase D taught it to read Qwen3's states for a TTS prompt -- text in,
nothing else -- and it does that well: 70.16 ASR-BLEU in TTS mode. Fed
the states of an S2TT reply instead, where the prompt carries the source
audio, it scored 0.15. The Talker has never seen that conditioning, and
the difference between 70.16 and 0.15 is the whole of this phase.

It is stage 1 -> stage 2 again, and stage 2's post-mortem sets the
configuration. There the Talker saw 549,000 joint-conditioned samples
against stage 1's 15.36M clean ones -- a twenty-eighth, on a harder
task -- and lost 6 ASR-BLEU for it. So s2st carries twice the weight it
did, and tts stays to hold the clean path open.

The Thinker is frozen and carries no adapter, which departs from the
paper. The evidence is ours: Qwen3 zero-shot translates at 38.40 and
30.83 Text-BLEU, the best this project has measured, and a LoRA on a
Thinker has never improved translation here -- stage 2's cost 3.05 on
Zh->En. With the Thinker frozen there is nothing for a text loss to
train, so the objective is the code cross-entropy alone.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import (
    build_text_stream_batched,
    prefix_labels,
    prefix_mask,
)
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    BICODEC_GLOBAL_COUNT,
    BICODEC_GLOBAL_SIZE,
    apply_output_mask,
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import (
    TEXT_EOS_ROW,
    TEXT_PAD_ROW,
    StreamMarkers,
)

THINKER_HIDDEN = 2048
# s2st at twice stage 2's weight; tts kept so the clean path is not
# forgotten while the joint one is learned. asr/s2tt/mt are dropped --
# they only supervise text, and no text is supervised here.
MIXTURE = {"s2st": 3.0, "tts": 1.0}
# The tts task trains on the state family the talker already knows: a TTS
# prompt, median norm 143.6. The s2st task trains on what inference sends,
# median norm 191.8 and 48 degrees away. Measured on 267 token positions,
# cosine 0.674 -- two different regions, not a small correction. Spending
# a quarter of the budget pulling the projection back toward the region
# inference never visits is worse than spending none, which is what
# UNISS_PHASE_E_MIXTURE=s2st does. Unset, the mixture is as v1 and v2 ran.
if os.environ.get("UNISS_PHASE_E_MIXTURE") == "s2st":
    MIXTURE = {"s2st": 1.0}


class PhaseEModel(HuggingFaceModule):
    """Qwen3's Thinker frozen; the Phase D Talker trained on S2ST states."""

    def __init__(self, config, *, qwen3_path: str, qwen25_path: str,
                 phase_d_checkpoint: str, head_scale: float = 0.1,
                 seed: int = 20261013, recompute: bool = True):
        super().__init__(config)
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen3OmniMoeForConditionalGeneration,
        )

        donor = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            qwen25_path, torch_dtype=torch.bfloat16)
        talker = donor.talker
        self.layout = read_codec_layout(talker.config)
        retarget_talker_codebook(talker, seed=seed, head_scale=head_scale)
        global_embed = torch.nn.Embedding(BICODEC_GLOBAL_SIZE, THINKER_HIDDEN)
        markers = StreamMarkers(THINKER_HIDDEN)

        state = torch.load(phase_d_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = talker.load_state_dict(state["talker"], strict=False)
        with torch.no_grad():
            global_embed.weight.copy_(state["global_embed"])
            markers.eos.copy_(state["markers"]["eos"])
            markers.pad.copy_(state["markers"]["pad"])
        norm = talker.state_dict().get("model.norm.weight")
        if norm is not None and float(norm.float().abs().mean()) < 0.01:
            raise SystemExit("model.norm.weight is near zero; an optimiser moment")
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected),
                            "phase_d_step": int(state.get("step", -1))}

        del donor.thinker
        if hasattr(donor, "token2wav"):
            del donor.token2wav
        del donor

        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_path, torch_dtype=torch.bfloat16)
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        thinker = omni3.thinker
        del omni3

        self.thinker = thinker
        self.talker = talker
        self.global_embed = global_embed.to(torch.bfloat16)
        self.markers = markers

        self.thinker.requires_grad_(False)
        self.thinker.eval()
        self.talker.requires_grad_(True)
        self.global_embed.requires_grad_(True)
        self.markers.requires_grad_(True)

        if recompute:
            # Source audio now runs through the 48-layer MoE as well, and a
            # full shuffle puts a 25-second clip in an ordinary batch. Stage
            # 2's OOM came from sizing on a subset that never held one.
            self.talker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.register_buffer("output_mask", valid_output_mask(self.layout),
                             persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.thinker.eval()
        return self

    def trainable_report(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"talker_m": round(count(self.talker) / 1e6, 1),
                "global_embed_m": round(count(self.global_embed) / 1e6, 2),
                "markers": count(self.markers),
                "thinker_frozen_m": round(
                    sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1),
                **self.load_report}

    def forward(self, thinker_input_ids, thinker_attention_mask, reply_ids,
                reply_mask, reply_start, codec_input_ids, codec_labels,
                codec_mask, bicodec_global, input_features=None,
                feature_attention_mask=None):
        with torch.no_grad():
            kwargs = {"input_ids": thinker_input_ids,
                      "attention_mask": thinker_attention_mask,
                      "output_hidden_states": True, "return_dict": True}
            if input_features is not None:
                kwargs["input_features"] = input_features
                kwargs["feature_attention_mask"] = feature_attention_mask
            out = self.thinker(**kwargs)
            start = int(reply_start.flatten()[0])
            hidden = out.hidden_states[-1][:, start : start + reply_ids.shape[1], :]
            embeds = self.thinker.get_input_embeddings()(reply_ids)
        device = hidden.device

        steps = codec_input_ids.shape[1] + BICODEC_GLOBAL_COUNT
        eos_embed, pad_embed = self.markers.embeds(torch.float32, device)
        text_stream = build_text_stream_batched(
            hidden.float(), embeds.float(), reply_mask.sum(dim=1),
            steps=steps, eos_embed=eos_embed, pad_embed=pad_embed,
        ).to(dtype=hidden.dtype)
        codec_embeds = torch.cat(
            [self.global_embed(bicodec_global),
             self.talker.get_input_embeddings()(codec_input_ids)], dim=1)
        labels = prefix_labels(codec_labels, BICODEC_GLOBAL_COUNT)
        mask = prefix_mask(codec_mask, BICODEC_GLOBAL_COUNT)
        lm_input = self.talker.thinker_to_talker_proj(codec_embeds + text_stream)
        positions = (torch.arange(steps, device=device).view(1, 1, -1)
                     .expand(3, codec_input_ids.shape[0], -1))
        talker_out = self.talker.model(
            inputs_embeds=lm_input, attention_mask=mask,
            position_ids=positions, return_dict=True)
        logits = apply_output_mask(
            self.talker.codec_head(talker_out.last_hidden_state), self.output_mask)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               labels.view(-1), ignore_index=-100)
        supervised = (labels != -100).sum()
        return torch.stack([loss, supervised.to(loss.dtype)])
