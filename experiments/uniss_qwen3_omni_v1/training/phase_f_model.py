"""Phase F: the paper's stage 2, on Qwen3's Thinker.

Phase E asked the Talker alone to learn a state family 48 degrees from
the one stage 1 fit it on -- measured, cosine 0.674 over 267 token
positions, with the audio-conditioned states carrying 25% more norm. It
got to 20.00 end-to-end against a 34.21 cascade and stopped moving:
+1.08 over one 2,000-step span, -0.67 over the next, and no change from
dropping the competing TTS task.

SimulS2ST-Omni (arXiv 2607.19810) reaches 31.12 on offline CVSS-T En->Zh
through the same single ``thinker_to_talker_proj`` linear, the same
``hidden + embeds`` addition and the same codec-side sum. Its repository
confirms it: ``freeze_thinker`` defaults to False. The projection is not
what differs. What differs is that both sides move, so the 48 degrees
never opens up.

``stage2_model`` already implements that recipe -- its docstring cites
the paper's Table 13 -- but against Qwen2.5-Omni, and it was measured
after 549k samples where stage 1 had taken 15.36M. The recipe was never
disproved; it was never fed. This is that model with:

  * Qwen3's Thinker in place of Qwen2.5's,
  * the Talker, speaker prefix and stream markers initialised from phase
    D rather than stage 1, because phase D is the Qwen3-compatible
    equivalent of stage 1 (63.55 / 73.91 in TTS mode), and
  * trained markers rather than embedding rows: Qwen2.5 holds the same
    vector at rows 151859 and 151861, Qwen3 holds two unrelated vectors
    that were never trained, which is why phase D learned them instead.

Objective and mixture are unchanged from stage 2:

    L = L_text + lambda * L_code

over asr/s2tt/mt/tts/s2st at 0.2:1:0.5:1:1.5 -- the paper's ratio, which
stage2_tasks already carried. Tasks that supervise one stream contribute
one term.
"""

from __future__ import annotations

import os
import re

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
from experiments.uniss_omni_s2st_v1.training.stage2_tasks import THINKER_TARGETS
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import (
    StreamMarkers,
    THINKER_HIDDEN,
)
from experiments.uniss_streaming_p2st_traj_v1.training import lora

# The paper's ratio, which stage2_tasks already carried. Imported rather
# than restated so the two cannot drift.
from experiments.uniss_omni_s2st_v1.training.stage2_tasks import MIXTURE  # noqa: F401

# Which thinker layers carry adapters. Autograd keeps activations back to
# the earliest parameter that needs a gradient, so "40" means 8 layers of
# activations rather than 48. "all" is the stage 2 behaviour.
LORA_LAYERS_FROM = os.environ.get("UNISS_PHASE_F_LORA_FROM", "40")


def _layer_filter():
    if LORA_LAYERS_FROM == "all":
        return None
    first = int(LORA_LAYERS_FROM)

    def keep(parent_name: str) -> bool:
        # Only the text LM's blocks, and only from `first` upward. The
        # audio tower is excluded here rather than frozen afterwards:
        # stage 2 wrapped it and then froze it, leaving 7.86M dead
        # parameters in every forward and every checkpoint.
        m = re.search(r"(?:^|\.)model\.layers\.(\d+)\.", parent_name + ".")
        return m is not None and int(m.group(1)) >= first

    return keep


class PhaseFModel(HuggingFaceModule):
    """Qwen3's Thinker on LoRA, phase D's Talker full, joint text+code."""

    def __init__(self, config, *, qwen3_path: str, qwen25_path: str,
                 phase_d_checkpoint: str, lora_rank: int = 32,
                 lora_alpha: float = 64.0, code_weight: float = 1.0,
                 head_scale: float = 0.1, seed: int = 20261014,
                 recompute: bool = True):
        super().__init__(config)
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen3OmniMoeForConditionalGeneration,
        )

        self.code_weight = float(code_weight)

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
                            "init_step": int(state.get("step", -1))}

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

        # inject() freezes everything it walks, so the Thinker goes first
        # and the Talker is re-enabled after.
        self.thinker_lora = lora.inject(
            self.thinker, rank=lora_rank, alpha=lora_alpha,
            targets=THINKER_TARGETS, layer_filter=_layer_filter(),
        )
        self.lora_from = LORA_LAYERS_FROM
        self.talker.requires_grad_(True)
        self.global_embed.requires_grad_(True)
        self.markers.requires_grad_(True)

        if recompute:
            # The thinker now runs a backward, and a full shuffle puts a
            # 25-second clip in an ordinary batch. Recompute rather than
            # size the batch for the worst draw -- stage 2's OOM came from
            # sizing on a subset that never held one.
            #
            # use_reentrant=False because the base weights are frozen: the
            # reentrant form needs an input that requires grad, and with
            # LoRA the first one is several layers in.
            self.thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.talker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.recompute = recompute

        self.register_buffer("output_mask", valid_output_mask(self.layout),
                             persistent=False)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Only what trains. The thinker is 31.7B of frozen weights that
        come back from from_pretrained on every start, and writing them
        made a checkpoint 65 GB -- ten of those is 650 GB against 679 GB
        free, so a 10,000-step run would have filled the disk around
        iteration 9,000.

        The adapter and the talker are what this run produces: 393M plus
        4.46M, about 5 GB once the optimiser's fp32 copies are counted.
        Save and load filter on the same predicate, so the key sets
        match and a resume finds exactly what it wrote.
        """
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
            "thinker_lora_m": round(adapter / 1e6, 2),
            "thinker_lora_layers": len(self.thinker_lora),
            "lora_from": self.lora_from,
            "thinker_frozen_m": round(
                sum(p.numel() for p in self.thinker.parameters()) / 1e6, 1),
            "talker_m": round(count(self.talker) / 1e6, 1),
            "global_embed_m": round(count(self.global_embed) / 1e6, 2),
            "markers": count(self.markers),
            "code_weight": self.code_weight,
            **self.load_report,
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
            labels = text_labels[:, 1:]
            # Gather before the float cast. An audio prompt runs 1000+
            # positions and the reply is about 20 of them, so casting the
            # whole sequence to float32 over a 152,064 vocabulary spends
            # 9.7 GB at micro batch 16 to score positions that are all
            # -100. Selecting first makes it 195 MB.
            sel = labels != -100
            if sel.any():
                text_loss = F.cross_entropy(
                    out.logits[:, :-1, :][sel].float(), labels[sel])

        code_loss = zero
        if codec_input_ids is not None:
            start = int(reply_start.flatten()[0])
            width = reply_ids.shape[1]
            hidden = out.hidden_states[-1][:, start : start + width, :]
            embeds = self.thinker.get_input_embeddings()(reply_ids)
            steps = codec_input_ids.shape[1] + BICODEC_GLOBAL_COUNT
            # Phase D's learned markers, not embedding rows: Qwen3 never
            # trained rows 151859 and 151861, and they are two unrelated
            # vectors there rather than Qwen2.5's single shared one.
            eos_embed, pad_embed = self.markers.embeds(torch.float32, device)
            text_stream = build_text_stream_batched(
                hidden.float(), embeds.float(), reply_mask.sum(dim=1),
                steps=steps, eos_embed=eos_embed, pad_embed=pad_embed,
            ).to(dtype=hidden.dtype)
            codec_embeds = torch.cat(
                [self.global_embed(bicodec_global),
                 self.talker.get_input_embeddings()(codec_input_ids)],
                dim=1,
            )
            labels = prefix_labels(codec_labels, BICODEC_GLOBAL_COUNT)
            mask = prefix_mask(codec_mask, BICODEC_GLOBAL_COUNT)
            lm_input = self.talker.thinker_to_talker_proj(codec_embeds + text_stream)
            positions = (torch.arange(steps, device=device).view(1, 1, -1)
                         .expand(3, codec_input_ids.shape[0], -1))
            talker_out = self.talker.model(
                inputs_embeds=lm_input, attention_mask=mask,
                position_ids=positions, return_dict=True,
            )
            code_logits = apply_output_mask(
                self.talker.codec_head(talker_out.last_hidden_state), self.output_mask)
            code_loss = F.cross_entropy(
                code_logits.float().view(-1, code_logits.shape[-1]),
                labels.view(-1), ignore_index=-100,
            )

        total = text_loss + self.code_weight * code_loss
        return torch.stack([total, text_loss.detach(), code_loss.detach()])
