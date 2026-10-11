"""Stage 2: Thinker on LoRA r32, Talker full, audio encoder frozen.

The scope is the paper's (arXiv 2607.19810, Table 13): Stage 1 trains
"-- / Full", Stage 2 "LoRA r32 / Full", Stage 3 "LoRA r8 / LoRA r8". So
the Talker keeps training the way it did in Stage 1 and the Thinker moves
only through a rank-32 adapter, which is what lets the translation
quality the zero-shot evaluation measured survive the mixture.

The objective is the plan's:

    L = L_text + lambda * L_code

with each term already averaged over its own supervised positions, so
``lambda`` is a plain weight between two per-token means rather than a
correction for sequence length. Tasks that supervise only one of the two
contribute only that term -- TTS has no text loss by design, and ASR,
S2TT and MT have no codes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core.models.huggingface import HuggingFaceModule

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import (
    build_codec_sequence,
    build_text_stream_batched,
    prefix_labels,
    prefix_mask,
)
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    BICODEC_GLOBAL_COUNT,
    BICODEC_GLOBAL_SIZE,
    apply_output_mask,
    read_codec_layout,
    valid_output_mask,
)
from experiments.uniss_streaming_p2st_traj_v1.training import lora

# Attention only. The CUHKSZ IWSLT 2026 system adapts q/k/v/o at r=16 and
# leaves the MLP alone; widening the adapter to the MLP is a change to
# make deliberately and measure, not to inherit by default.
THINKER_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class Stage2Model(HuggingFaceModule):
    """Joint text and code supervision over the five-task mixture."""

    def __init__(self, config, *, model_path: str, talker_state: str | None,
                 lora_rank: int = 32, lora_alpha: float = 64.0,
                 code_weight: float = 1.0, seed: int = 20261011,
                 recompute: bool = True):
        super().__init__(config)
        from transformers import Qwen2_5OmniForConditionalGeneration

        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        if hasattr(omni, "token2wav"):
            del omni.token2wav
        self.layout = read_codec_layout(omni.talker.config)
        self.code_weight = float(code_weight)

        # Stage 1's Talker, including its re-drawn codebook and the speaker
        # prefix. Loading before the LoRA injection keeps the two concerns
        # separate: the Talker is weights, the Thinker is an adapter.
        self.global_embed = torch.nn.Embedding(BICODEC_GLOBAL_SIZE, 2048)
        if talker_state:
            state = torch.load(talker_state, map_location="cpu", weights_only=False)
            missing, unexpected = omni.talker.load_state_dict(state["talker"], strict=False)
            if state.get("global_embed") is None:
                raise SystemExit(
                    f"{talker_state} carries no speaker prefix; Stage 1 trained with one"
                )
            with torch.no_grad():
                self.global_embed.weight.copy_(state["global_embed"])
            self.stage1_step = state.get("step")
            self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        else:
            self.stage1_step = None
            self.load_report = None
        self.global_embed = self.global_embed.to(
            omni.talker.get_input_embeddings().weight.dtype
        )

        # LoRA freezes everything it touches, so the Thinker goes first and
        # the Talker is re-enabled afterwards.
        self.thinker = omni.thinker
        self.thinker_lora = lora.inject(
            self.thinker, rank=lora_rank, alpha=lora_alpha, targets=THINKER_TARGETS
        )
        # The audio tower stays frozen even where it matched a target name.
        tower = getattr(self.thinker, "audio_tower", None)
        if tower is not None:
            for parameter in tower.parameters():
                parameter.requires_grad_(False)

        self.talker = omni.talker
        self.talker.requires_grad_(True)
        self.global_embed.requires_grad_(True)

        if recompute:
            # Activation memory, not compute, is what bounds the batch here:
            # source audio runs from 1 to 24.9 seconds, and a full shuffle
            # puts the longest utterances in an ordinary batch. At micro
            # batch 16 a five-second average fits in 65 GB and a
            # 25-second one would want roughly 240. Recomputing the
            # Thinker's activations trades about a third of the step for a
            # ceiling that does not depend on what the shuffle drew.
            #
            # use_reentrant=False because the base weights are frozen: the
            # reentrant form needs an input that requires grad, and with
            # LoRA the first thing that does is several layers in.
            self.thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.talker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.recompute = recompute

        self.register_buffer(
            "output_mask", valid_output_mask(self.layout), persistent=False
        )

    def trainable_summary(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        adapter = sum(p.numel() for layer in self.thinker_lora
                      for p in (layer.a, layer.b))
        return {
            "thinker_lora_m": round(adapter / 1e6, 2),
            "thinker_frozen_m": round(
                sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1
            ),
            "talker_m": round(count(self.talker) / 1e6, 1),
            "global_embed_m": round(count(self.global_embed) / 1e6, 2),
            "stage1_step": self.stage1_step,
            "stage1_load": self.load_report,
        }

    def forward(
        self,
        thinker_input_ids,
        thinker_attention_mask,
        text_labels,
        reply_ids,
        reply_mask,
        reply_start,
        supervise_text,
        codec_input_ids=None,
        codec_labels=None,
        codec_mask=None,
        bicodec_global=None,
        input_features=None,
        feature_attention_mask=None,
    ):
        kwargs = {
            "input_ids": thinker_input_ids,
            "attention_mask": thinker_attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if input_features is not None:
            kwargs["input_features"] = input_features
            kwargs["feature_attention_mask"] = feature_attention_mask
        out = self.thinker(**kwargs)

        device = out.logits.device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        text_loss = zero
        if bool(supervise_text.flatten()[0]):
            # Megatron's forward_step does no shifting, so it is done here:
            # position t predicts token t+1.
            logits = out.logits[:, :-1, :].float()
            labels = text_labels[:, 1:]
            text_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100,
            )

        code_loss = zero
        if codec_input_ids is not None:
            start = int(reply_start.flatten()[0])
            width = reply_ids.shape[1]
            hidden = out.hidden_states[-1][:, start : start + width, :]
            embeds = self.thinker.get_input_embeddings()(reply_ids)
            eos_embed = self.thinker.get_input_embeddings()(
                torch.tensor([[self.talker.text_eos_token]], device=device)
            )
            pad_embed = self.thinker.get_input_embeddings()(
                torch.tensor([[self.talker.text_pad_token]], device=device)
            )
            steps = codec_input_ids.shape[1] + BICODEC_GLOBAL_COUNT
            text_stream = build_text_stream_batched(
                hidden.float(), embeds.float(), reply_mask.sum(dim=1),
                steps=steps, eos_embed=eos_embed.float(), pad_embed=pad_embed.float(),
            ).to(dtype=hidden.dtype)
            codec_embeds = torch.cat(
                [self.global_embed(bicodec_global),
                 self.talker.get_input_embeddings()(codec_input_ids)],
                dim=1,
            )
            labels = prefix_labels(codec_labels, BICODEC_GLOBAL_COUNT)
            mask = prefix_mask(codec_mask, BICODEC_GLOBAL_COUNT)
            lm_input = self.talker.thinker_to_talker_proj(codec_embeds + text_stream)
            positions = (
                torch.arange(steps, device=device)
                .view(1, 1, -1)
                .expand(3, codec_input_ids.shape[0], -1)
            )
            talker_out = self.talker.model(
                inputs_embeds=lm_input, attention_mask=mask,
                position_ids=positions, return_dict=True,
            )
            code_logits = apply_output_mask(
                self.talker.codec_head(talker_out.last_hidden_state), self.output_mask
            )
            code_loss = F.cross_entropy(
                code_logits.float().view(-1, code_logits.shape[-1]),
                labels.view(-1), ignore_index=-100,
            )

        total = text_loss + self.code_weight * code_loss
        return torch.stack([total, text_loss.detach(), code_loss.detach()])
