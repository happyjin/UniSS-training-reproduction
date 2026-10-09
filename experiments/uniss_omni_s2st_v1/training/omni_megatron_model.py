"""Qwen2.5-Omni's Thinker and Talker, as a MegatronModule.

Megatron ships a HuggingFace bridge, but its Qwen wrapper is hardcoded to
``Qwen2ForCausalLM`` -- a plain decoder-only LM -- so it cannot load Omni,
which is an audio encoder plus a 36-layer Thinker plus a 24-layer Talker
with a cross-conditioning projection. The *base* class it derives from,
``HuggingFaceModule``, is generic, so this subclasses that instead and the
run gets Megatron's distributed optimiser, gradient accumulation,
checkpointing and logging around the real model.

The objective is Stage 1's: target text in, BiCodec codes out. The Thinker
is frozen and runs under no_grad; only the Talker trains, which includes
the re-drawn code embedding, the re-drawn output head and
``thinker_to_talker_proj``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import build_text_stream
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    apply_output_mask,
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)


class OmniTalkerWarmupModel(HuggingFaceModule):
    """Teacher-forced code cross-entropy, with the Thinker held frozen."""

    def __init__(self, config, *, model_path: str, head_scale: float,
                 embed_scale: float, seed: int):
        super().__init__(config)
        from transformers import Qwen2_5OmniForConditionalGeneration

        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        # The vocoder has no role in a code-space objective and would
        # otherwise occupy memory on every rank.
        if hasattr(omni, "token2wav"):
            del omni.token2wav

        self.layout = read_codec_layout(omni.talker.config)
        retarget_talker_codebook(
            omni.talker, seed=seed, head_scale=head_scale, embed_scale=embed_scale
        )

        # Assigned through __setattr__ so HuggingFaceModule tags every
        # parameter for the cross-TP gradient all-reduce it needs.
        self.thinker = omni.thinker
        self.talker = omni.talker

        self.thinker.requires_grad_(False)
        self.thinker.eval()
        self.talker.requires_grad_(True)

        self.register_buffer(
            "output_mask", valid_output_mask(self.layout), persistent=False
        )

    def train(self, mode: bool = True):
        """Keep the Thinker in eval whatever Megatron does to the wrapper."""
        super().train(mode)
        self.thinker.eval()
        return self

    def forward(
        self,
        thinker_input_ids,
        thinker_attention_mask,
        reply_ids,
        reply_mask,
        reply_start,
        codec_input_ids,
        codec_labels,
        codec_mask,
    ):
        reply_start = int(reply_start.flatten()[0])
        reply_width = reply_ids.shape[1]

        with torch.no_grad():
            out = self.thinker(
                input_ids=thinker_input_ids,
                attention_mask=thinker_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = out.hidden_states[-1][
                :, reply_start : reply_start + reply_width, :
            ]
            embeds = self.thinker.get_input_embeddings()(reply_ids)
            device = hidden.device
            eos_embed = self.thinker.get_input_embeddings()(
                torch.tensor([[self.talker.text_eos_token]], device=device)
            )
            pad_embed = self.thinker.get_input_embeddings()(
                torch.tensor([[self.talker.text_pad_token]], device=device)
            )

        steps = codec_input_ids.shape[1]
        # Each row's stream must run out at its own reply length. Padded to
        # the batch width, a short reply would be conditioned on a longer
        # neighbour's pad embeddings.
        streams = []
        for row in range(hidden.shape[0]):
            length = int(reply_mask[row].sum())
            streams.append(
                build_text_stream(
                    hidden[row : row + 1, :length].float(),
                    embeds[row : row + 1, :length].float(),
                    steps=steps,
                    eos_embed=eos_embed.float(),
                    pad_embed=pad_embed.float(),
                )
            )
        text_stream = torch.cat(streams, dim=0).to(dtype=hidden.dtype)

        codec_embeds = self.talker.get_input_embeddings()(codec_input_ids)
        lm_input = self.talker.thinker_to_talker_proj(codec_embeds + text_stream)
        position_ids = (
            torch.arange(steps, device=device)
            .view(1, 1, -1)
            .expand(3, codec_input_ids.shape[0], -1)
        )
        talker_out = self.talker.model(
            inputs_embeds=lm_input,
            attention_mask=codec_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = apply_output_mask(
            self.talker.codec_head(talker_out.last_hidden_state), self.output_mask
        )
        loss = F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]),
            codec_labels.view(-1),
            ignore_index=-100,
        )
        supervised = (codec_labels != -100).sum()
        return torch.stack([loss, supervised.to(loss.dtype)])
