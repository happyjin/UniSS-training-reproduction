"""Phase D: the Talker learns to read Qwen3's hidden states.

Stage 1 taught this Talker to turn Qwen2.5-Omni's reply states into
BiCodec codes, and it does that well -- 24.82 and 28.70 ASR-BLEU through
the cascade. Phase C replaced the Thinker, and the two models' 2048
dimensions line up exactly while meaning nothing to each other. So the
Talker keeps everything it knows about producing speech and relearns only
where the conditioning comes from.

Which is why the Thinker is frozen here and carries no adapter. Training
it would risk the one thing measured to be worth having: Qwen3 zero-shot
translates at 38.40 and 30.83 Text-BLEU, and in this project a LoRA on the
Thinker has never improved translation and once cost 3.05 BLEU.

The objective is stage 1's, unchanged: teacher-forced cross-entropy on the
codes, masked to the 8,192 BiCodec slots plus eos. No text loss -- the TTS
prompt already contains the text, so supervising it would train a frozen
model to copy its own input.

Two things differ from stage 1 besides the Thinker. The Talker starts from
Stage 1's trained weights rather than from the UniSS transfer, because it
already emits codes and only the input mapping is new. And the vectors
that end the text stream are parameters rather than embedding rows: in
Qwen2.5-Omni the reserved pad and eos rows hold the *same* vector, so the
distinction never carried a signal, and Qwen3's rows at those indices were
never trained for the job at all.
"""

from __future__ import annotations

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


class PhaseDModel(HuggingFaceModule):
    """Qwen3's Thinker frozen, the Stage 1 Talker trained to read it."""

    def __init__(self, config, *, qwen3_path: str, qwen25_path: str,
                 talker_checkpoint: str, head_scale: float = 0.1,
                 seed: int = 20261012, recompute: bool = True):
        super().__init__(config)
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen3OmniMoeForConditionalGeneration,
        )

        donor = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            qwen25_path, torch_dtype=torch.bfloat16
        )
        talker = donor.talker
        self.layout = read_codec_layout(talker.config)
        # The re-draw is what stage 1 trained under; the checkpoint then
        # overwrites every row it covers. Skipping it would leave Omni's
        # own codec rows in the slots the checkpoint does not reach.
        retarget_talker_codebook(talker, seed=seed, head_scale=head_scale)

        hidden = talker.get_input_embeddings().weight.shape[1]
        if hidden != THINKER_HIDDEN:
            raise SystemExit(f"talker input embedding is {hidden}, not {THINKER_HIDDEN}")
        global_embed = torch.nn.Embedding(BICODEC_GLOBAL_SIZE, hidden)

        state = torch.load(talker_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = talker.load_state_dict(state["talker"], strict=False)
        if state.get("global_embed") is None:
            raise SystemExit(
                f"{talker_checkpoint} has no global_embed; the speaker prefix"
                " would be random and nothing downstream would say so"
            )
        with torch.no_grad():
            global_embed.weight.copy_(state["global_embed"])
        norm = talker.state_dict().get("model.norm.weight")
        if norm is not None and float(norm.float().abs().mean()) < 0.01:
            raise SystemExit("model.norm.weight is near zero; that is an optimiser moment")
        self.load_report = {
            "missing": len(missing), "unexpected": len(unexpected),
            "stage1_step": int(state.get("step", -1)),
        }

        del donor.thinker
        if hasattr(donor, "token2wav"):
            del donor.token2wav
        del donor

        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_path, torch_dtype=torch.bfloat16
        )
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        thinker = omni3.thinker
        del omni3

        embed = thinker.get_input_embeddings().weight
        markers = StreamMarkers(
            THINKER_HIDDEN,
            eos_init=embed[TEXT_EOS_ROW].detach().cpu(),
            pad_init=embed[TEXT_PAD_ROW].detach().cpu(),
        )

        # Assigned through __setattr__ so HuggingFaceModule tags every
        # parameter for the gradient all-reduce it needs.
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
            # A 30B frozen Thinker leaves little room for activations, and
            # stage 2 showed the ceiling has to stop depending on what the
            # shuffle drew. The Thinker runs under no_grad so only the
            # Talker's recompute matters for the backward.
            self.talker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.register_buffer(
            "output_mask", valid_output_mask(self.layout), persistent=False
        )

    def train(self, mode: bool = True):
        """Keep the Thinker in eval whatever Megatron does to the wrapper."""
        super().train(mode)
        self.thinker.eval()
        return self

    def trainable_report(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "talker_m": round(count(self.talker) / 1e6, 1),
            "global_embed_m": round(count(self.global_embed) / 1e6, 2),
            "markers": count(self.markers),
            "thinker_frozen_m": round(
                sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1),
            **self.load_report,
        }

    def forward(self, thinker_input_ids, thinker_attention_mask, reply_ids,
                reply_mask, reply_start, codec_input_ids, codec_labels,
                codec_mask, bicodec_global):
        with torch.no_grad():
            out = self.thinker(
                input_ids=thinker_input_ids, attention_mask=thinker_attention_mask,
                output_hidden_states=True, return_dict=True)
            # The collate left-pads the prompt and right-pads the reply, so
            # every row's reply begins at the same index.
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
        mask_with_prefix = prefix_mask(codec_mask, BICODEC_GLOBAL_COUNT)
        lm_input = self.talker.thinker_to_talker_proj(codec_embeds + text_stream)
        positions = (torch.arange(steps, device=device).view(1, 1, -1)
                     .expand(3, codec_input_ids.shape[0], -1))
        talker_out = self.talker.model(
            inputs_embeds=lm_input, attention_mask=mask_with_prefix,
            position_ids=positions, return_dict=True)
        logits = apply_output_mask(
            self.talker.codec_head(talker_out.last_hidden_state), self.output_mask)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), labels.view(-1),
            ignore_index=-100)
        supervised = (labels != -100).sum()
        return torch.stack([loss, supervised.to(loss.dtype)])
