"""Compound Phase3 WhisperVQ + StreamSpeech single-stage model."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder

from .config import JointLossWeights, MultiChunkConfig
from .ctc_heads import TaskCTCHeads
from .losses import NormalizedLoss, ctc_normalized_loss, masked_ce_normalized
from .nar_bicodec_ctc import NARBiCodecCTC
from .phase3_batch import build_policy_conditioned_phase3_batch, gather_target_hidden
from .phase3_ste_bridge import Phase3STEBridge
from .policy_mask import build_g_from_ctc_logits, packed_causal_attention_allowed
from .tokenizer_maps import CompactCTCMap
from .whisper_frontend import TrainableMultiChunkWhisperVQ
from .whisper_multichunk import additive_attention_mask


DIRECTIONS = {
    0: ("eng", "cmn", "asr_eng", "nar_s2tt_cmn"),
    1: ("cmn", "eng", "asr_cmn", "nar_s2tt_eng"),
}
COMPONENTS = (
    "bicodec_ctc",
    "ar_s2tt",
    "asr_ctc",
    "nar_s2tt_ctc",
    "phase3_replay",
)


def _flatten_rows(padded: torch.Tensor, lengths: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    values = [padded[int(row), : int(lengths[int(row)].item())] for row in rows.tolist()]
    return torch.cat(values) if values else padded.new_empty((0,), dtype=torch.long)


def _parameter_anchor(parameters: Iterable[nn.Parameter], reference: torch.Tensor) -> torch.Tensor:
    anchor = reference.sum() * 0.0
    for parameter in parameters:
        if parameter.requires_grad and parameter.numel():
            anchor = anchor + parameter.reshape(-1)[0] * 0.0
    return anchor


def _zero_loss(anchor: torch.Tensor) -> NormalizedLoss:
    return NormalizedLoss(anchor, anchor.detach().new_zeros(()))


def distributed_component_losses(
    losses: OrderedDict[str, NormalizedLoss], weights: JointLossWeights
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-normalize globally while preserving correct DDP gradient scale."""

    if tuple(losses) != COMPONENTS:
        raise ValueError(f"component order must be {COMPONENTS}")
    numerators = torch.stack([losses[name].numerator for name in COMPONENTS])
    denominators = torch.stack(
        [losses[name].denominator.to(numerators.dtype) for name in COMPONENTS]
    )
    global_denominators = denominators.detach().clone()
    global_numerators = numerators.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominators)
        dist.all_reduce(global_numerators)
        world_size = dist.get_world_size()
    active = global_denominators > 0
    scaled_local = torch.where(
        active,
        world_size * numerators / global_denominators.clamp_min(1),
        numerators * 0.0,
    )
    weight_vector = numerators.new_tensor(
        [
            weights.bicodec_ctc,
            weights.ar_s2tt,
            weights.asr_ctc,
            weights.nar_s2tt_ctc,
            weights.phase3_replay,
        ]
    )
    total = (scaled_local * weight_vector).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    return total, {
        name: global_means[index] for index, name in enumerate(COMPONENTS)
    }


class Phase3WhisperStreamSpeechJointModel(nn.Module):
    """One optimizer, one checkpoint stream, joint or exact-replay microbatch."""

    def __init__(
        self,
        *,
        whisper: TrainableMultiChunkWhisperVQ,
        qwen: nn.Module,
        tokenizer,
        ctc_maps: dict[str, CompactCTCMap],
        loss_weights: JointLossWeights | None = None,
        upsample_ratio: int = 48,
        bridge_surrogate: str = "projection",
        bridge_topk: int = 8,
        bridge_temperature: float = 0.1,
        bridge_gradient_scale: float = 1.0,
        teacher_temperature: float = 0.1,
        max_bridge_commitment: float | None = None,
        max_bridge_commitment_ratio: float | None = None,
        bridge_guard_baseline_microbatches: int = 0,
    ) -> None:
        super().__init__()
        self.whisper = whisper
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.text_encoder = load_hf_text_encoder(tokenizer)
        self.ctc_maps = ctc_maps
        self.loss_weights = loss_weights or JointLossWeights()
        self.max_bridge_commitment = max_bridge_commitment
        self.max_bridge_commitment_ratio = max_bridge_commitment_ratio
        self.bridge_guard_baseline_microbatches = int(bridge_guard_baseline_microbatches)
        self.register_buffer("bridge_guard_baseline_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("bridge_guard_baseline_count", torch.zeros((), dtype=torch.long))
        qwen_embeddings = qwen.get_input_embeddings().weight
        qwen_glm_embeddings = qwen_embeddings[
            c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
        ]
        self.bridge = Phase3STEBridge(
            whisper.hidden_size,
            qwen_embeddings.shape[-1],
            whisper.codebook,
            qwen_glm_embeddings,
            surrogate=bridge_surrogate,
            topk=bridge_topk,
            temperature=bridge_temperature,
            gradient_scale=bridge_gradient_scale,
            teacher_temperature=teacher_temperature,
        )
        output_sizes = {
            "asr_eng": ctc_maps["eng"].output_size,
            "asr_cmn": ctc_maps["cmn"].output_size,
            "nar_s2tt_eng": ctc_maps["eng"].output_size,
            "nar_s2tt_cmn": ctc_maps["cmn"].output_size,
        }
        self.ctc_heads = TaskCTCHeads(whisper.hidden_size, output_sizes)
        self.unit_ctc = NARBiCodecCTC(
            qwen_hidden_size=qwen_embeddings.shape[-1],
            semantic_vocab_size=c.BICODEC_SEMANTIC_SIZE,
            upsample_ratio=upsample_ratio,
        )
        self.tag_learning_rate_groups()

    @classmethod
    def from_pretrained(
        cls,
        *,
        whisper_path: str | Path,
        phase3_model: str | Path,
        tokenizer_map_dir: str | Path,
        chunk_config: MultiChunkConfig | None = None,
        loss_weights: JointLossWeights | None = None,
        upsample_ratio: int = 48,
        gradient_checkpointing: bool = True,
        bridge_surrogate: str = "projection",
        bridge_topk: int = 8,
        bridge_temperature: float = 0.1,
        bridge_gradient_scale: float = 1.0,
        teacher_temperature: float = 0.1,
        freeze_whisper_codebook: bool = False,
        freeze_whisper: bool = False,
        trainable_whisper_pre_vq_layers: int | None = None,
        freeze_whisper_post_vq: bool = False,
        freeze_qwen: bool = False,
        max_bridge_commitment: float | None = None,
        max_bridge_commitment_ratio: float | None = None,
        bridge_guard_baseline_microbatches: int = 0,
    ) -> "Phase3WhisperStreamSpeechJointModel":
        tokenizer = AutoTokenizer.from_pretrained(
            str(phase3_model), local_files_only=True
        )
        qwen = AutoModelForCausalLM.from_pretrained(
            str(phase3_model),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda()
        qwen.config.use_cache = False
        if gradient_checkpointing:
            qwen.gradient_checkpointing_enable()
        whisper = TrainableMultiChunkWhisperVQ(
            whisper_path,
            chunk_config=chunk_config,
            freeze_codebook_updates=freeze_whisper_codebook,
            freeze_encoder=freeze_whisper,
            trainable_pre_vq_layers=trainable_whisper_pre_vq_layers,
            freeze_post_vq=freeze_whisper_post_vq,
        )
        whisper.configure_gradient_checkpointing(gradient_checkpointing)
        if freeze_qwen:
            for parameter in qwen.parameters():
                parameter.requires_grad_(False)
        maps = {
            language: CompactCTCMap.load(
                Path(tokenizer_map_dir) / f"ctc_qwen_{language}.json"
            )
            for language in ("eng", "cmn")
        }
        return cls(
            whisper=whisper,
            qwen=qwen,
            tokenizer=tokenizer,
            ctc_maps=maps,
            loss_weights=loss_weights,
            upsample_ratio=upsample_ratio,
            bridge_surrogate=bridge_surrogate,
            bridge_topk=bridge_topk,
            bridge_temperature=bridge_temperature,
            bridge_gradient_scale=bridge_gradient_scale,
            teacher_temperature=teacher_temperature,
            max_bridge_commitment=max_bridge_commitment,
            max_bridge_commitment_ratio=max_bridge_commitment_ratio,
            bridge_guard_baseline_microbatches=bridge_guard_baseline_microbatches,
        ).cuda()

    def tag_learning_rate_groups(self) -> None:
        self.whisper.tag_learning_rate_groups()
        for parameter in self.bridge.parameters():
            if parameter.requires_grad:
                parameter.uniss_lr_bridge = True
        for module in (self.ctc_heads, self.unit_ctc):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.uniss_lr_new = True
        for parameter in self.qwen.parameters():
            if parameter.requires_grad:
                parameter.uniss_lr_qwen = True
        io_parameters = list(self.qwen.get_input_embeddings().parameters())
        output = self.qwen.get_output_embeddings()
        if output is not None:
            io_parameters.extend(output.parameters())
        for parameter in io_parameters:
            if parameter.requires_grad:
                parameter.uniss_lr_qwen = False
                parameter.uniss_lr_qwen_io = True

    def _joint_losses(
        self, batch: dict[str, object], chunk_ms: int | None
    ) -> tuple[OrderedDict[str, NormalizedLoss], dict[str, torch.Tensor]]:
        waveform = batch["waveform"]
        waveform_lengths = batch["waveform_lengths"]
        direction_ids = batch["direction_ids"]
        source_ctc = batch["source_ctc_ids"]
        source_ctc_lengths = batch["source_ctc_ids_lengths"]
        target_ctc = batch["target_ctc_ids"]
        target_ctc_lengths = batch["target_ctc_ids_lengths"]
        target_bicodec = batch["target_bicodec"]
        target_bicodec_lengths = batch["target_bicodec_lengths"]
        source_glm = batch["source_glm"]
        source_glm_lengths = batch["source_glm_lengths"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                waveform,
                waveform_lengths,
                direction_ids,
                source_ctc,
                source_ctc_lengths,
                target_ctc,
                target_ctc_lengths,
                target_bicodec,
                target_bicodec_lengths,
                source_glm,
                source_glm_lengths,
            )
        ):
            raise TypeError("joint batch tensor fields are malformed")
        whisper_output = self.whisper(
            waveform, waveform_lengths, chunk_ms=chunk_ms
        )
        ctc_logits = self.ctc_heads(whisper_output.pre_vq_hidden)
        head_anchor = sum(value.sum() * 0.0 for value in ctc_logits.values())
        asr_numerator = head_anchor
        asr_denominator = head_anchor.detach().new_zeros(())
        nar_numerator = head_anchor
        nar_denominator = head_anchor.detach().new_zeros(())
        asr_infeasible = head_anchor.detach().new_zeros(())
        nar_infeasible = head_anchor.detach().new_zeros(())
        g = torch.zeros(
            len(direction_ids),
            int(target_ctc_lengths.max().item()),
            dtype=torch.long,
            device=waveform.device,
        )
        for direction_id, (source_language, target_language, source_head, target_head) in DIRECTIONS.items():
            rows = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
            if not len(rows):
                continue
            source_targets = _flatten_rows(source_ctc, source_ctc_lengths, rows)
            target_targets = _flatten_rows(target_ctc, target_ctc_lengths, rows)
            source_loss, source_invalid = ctc_normalized_loss(
                ctc_logits[source_head][rows],
                source_targets,
                whisper_output.token_lengths[rows],
                source_ctc_lengths[rows],
                blank_id=self.ctc_maps[source_language].blank_id,
            )
            target_loss, target_invalid = ctc_normalized_loss(
                ctc_logits[target_head][rows],
                target_targets,
                whisper_output.token_lengths[rows],
                target_ctc_lengths[rows],
                blank_id=self.ctc_maps[target_language].blank_id,
            )
            asr_numerator = asr_numerator + source_loss.numerator
            asr_denominator = asr_denominator + source_loss.denominator
            nar_numerator = nar_numerator + target_loss.numerator
            nar_denominator = nar_denominator + target_loss.denominator
            asr_infeasible = asr_infeasible + source_invalid
            nar_infeasible = nar_infeasible + target_invalid
            direction_g = build_g_from_ctc_logits(
                ctc_logits[source_head][rows],
                ctc_logits[target_head][rows],
                asr_blank_id=self.ctc_maps[source_language].blank_id,
                target_blank_id=self.ctc_maps[target_language].blank_id,
                target_lengths=target_ctc_lengths[rows],
                encoder_lengths=whisper_output.token_lengths[rows],
            )
            g[rows, : direction_g.shape[1]] = direction_g

        bridge = self.bridge(
            whisper_output.pre_vq_hidden,
            whisper_output.token_lengths,
            teacher_code_ids=source_glm,
            teacher_lengths=source_glm_lengths,
        )
        record_values = batch["phase3_record_json"]
        if isinstance(record_values, str):
            record_values = [record_values]
        records = [json.loads(value) for value in record_values]  # type: ignore[arg-type]
        phase3_batch = build_policy_conditioned_phase3_batch(
            embedding_layer=self.qwen.get_input_embeddings(),
            text_encoder=self.text_encoder,
            records=records,
            source_embeddings=bridge.embeddings,
            source_lengths=whisper_output.token_lengths,
            g=g,
        )
        qwen_output = self.qwen(
            inputs_embeds=phase3_batch.inputs_embeds,
            attention_mask={"full_attention": phase3_batch.attention_mask},
            position_ids=phase3_batch.position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        ar_loss = masked_ce_normalized(
            qwen_output.logits[:, :-1], phase3_batch.labels[:, 1:]
        )
        target_hidden, text_lengths = gather_target_hidden(
            qwen_output.hidden_states[-1], phase3_batch.target_positions
        )
        unit_logits, unit_lengths = self.unit_ctc(target_hidden, text_lengths)
        unit_targets = _flatten_rows(
            target_bicodec,
            target_bicodec_lengths,
            torch.arange(len(target_bicodec), device=target_bicodec.device),
        )
        unit_loss, unit_infeasible = ctc_normalized_loss(
            unit_logits,
            unit_targets,
            unit_lengths,
            target_bicodec_lengths,
            blank_id=self.unit_ctc.blank_id,
        )
        losses = OrderedDict(
            (
                ("bicodec_ctc", unit_loss),
                ("ar_s2tt", ar_loss),
                ("asr_ctc", NormalizedLoss(asr_numerator, asr_denominator)),
                ("nar_s2tt_ctc", NormalizedLoss(nar_numerator, nar_denominator)),
                ("phase3_replay", _zero_loss(qwen_output.logits.sum() * 0.0)),
            )
        )
        diagnostics = {
            "asr_infeasible": asr_infeasible,
            "nar_infeasible": nar_infeasible,
            "unit_infeasible": unit_infeasible,
            "bridge_commitment": bridge.commitment_loss,
            "whisper_quantize": whisper_output.quantize_loss,
            "teacher_glm_ce": bridge.teacher_ce_loss,
            "teacher_glm_commitment": bridge.teacher_commitment_loss,
            "teacher_glm_agreement": bridge.teacher_agreement,
            "teacher_glm_coverage": bridge.teacher_coverage,
            "code_perplexity": bridge.code_perplexity,
            "active_code_fraction": bridge.active_code_fraction,
            "hidden_rms": bridge.hidden_rms,
        }
        return losses, diagnostics

    def _replay_losses(
        self, batch: dict[str, object]
    ) -> tuple[OrderedDict[str, NormalizedLoss], dict[str, torch.Tensor]]:
        tokens = batch["tokens"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]
        position_ids = batch["position_ids"]
        cu_seqlens = batch["cu_seqlens"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (tokens, labels, loss_mask, position_ids, cu_seqlens)
        ):
            raise TypeError("replay batch tensor fields are malformed")
        allowed = packed_causal_attention_allowed(cu_seqlens, tokens.shape[1])
        attention = additive_attention_mask(
            allowed, self.qwen.get_input_embeddings().weight.dtype
        )
        output = self.qwen(
            input_ids=tokens,
            attention_mask={"full_attention": attention},
            position_ids=position_ids,
            use_cache=False,
        )
        replay_labels = labels.masked_fill(loss_mask <= 0, -100)
        replay = masked_ce_normalized(output.logits, replay_labels)
        joint_parameters = list(self.whisper.parameters()) + list(self.bridge.parameters())
        joint_parameters += list(self.ctc_heads.parameters()) + list(self.unit_ctc.parameters())
        anchor = _parameter_anchor(joint_parameters, output.logits)
        losses = OrderedDict(
            (
                ("bicodec_ctc", _zero_loss(anchor)),
                ("ar_s2tt", _zero_loss(anchor)),
                ("asr_ctc", _zero_loss(anchor)),
                ("nar_s2tt_ctc", _zero_loss(anchor)),
                ("phase3_replay", replay),
            )
        )
        zero = replay.numerator.detach().new_zeros(())
        return losses, {
            "asr_infeasible": zero,
            "nar_infeasible": zero,
            "unit_infeasible": zero,
            "bridge_commitment": zero,
            "whisper_quantize": zero,
            "teacher_glm_ce": zero,
            "teacher_glm_commitment": zero,
            "teacher_glm_agreement": zero,
            "teacher_glm_coverage": zero,
            "code_perplexity": zero,
            "active_code_fraction": zero,
            "hidden_rms": zero,
        }

    def _guard_bridge_commitment(self, commitment: torch.Tensor, chunk_ms: int | None) -> None:
        guard_value = commitment.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(guard_value, op=dist.ReduceOp.MAX)
        if self.training and int(self.bridge_guard_baseline_count) < self.bridge_guard_baseline_microbatches:
            with torch.no_grad():
                self.bridge_guard_baseline_sum.add_(guard_value)
                self.bridge_guard_baseline_count.add_(1)
            return
        limits: list[tuple[str, float]] = []
        if self.max_bridge_commitment is not None:
            limits.append(("absolute", float(self.max_bridge_commitment)))
        if (
            self.max_bridge_commitment_ratio is not None
            and int(self.bridge_guard_baseline_count) > 0
        ):
            baseline = float(
                self.bridge_guard_baseline_sum / self.bridge_guard_baseline_count
            )
            limits.append(("relative", baseline * self.max_bridge_commitment_ratio))
        exceeded = [(name, limit) for name, limit in limits if float(guard_value) > limit]
        if exceeded:
            raise FloatingPointError(
                "bridge commitment exceeded safety gate: "
                f"value={float(guard_value):.6f}, limits={exceeded}, "
                f"baseline_count={int(self.bridge_guard_baseline_count)}, chunk_ms={chunk_ms}"
            )

    def forward(
        self,
        batch: dict[str, object],
        *,
        chunk_ms: int | None = None,
    ) -> torch.Tensor:
        sample_kind = batch["sample_kind"]
        if isinstance(sample_kind, (list, tuple)):
            if len(set(sample_kind)) != 1:
                raise ValueError("one microbatch cannot mix joint and replay samples")
            sample_kind = sample_kind[0]
        if sample_kind == "joint":
            losses, diagnostics = self._joint_losses(batch, chunk_ms)
            active_joint = 1.0
        elif sample_kind == "replay":
            losses, diagnostics = self._replay_losses(batch)
            active_joint = 0.0
        else:
            raise ValueError(f"unsupported sample kind: {sample_kind}")
        total, means = distributed_component_losses(losses, self.loss_weights)
        if sample_kind == "joint":
            commitment = diagnostics["bridge_commitment"]
            self._guard_bridge_commitment(commitment, chunk_ms)
            total = total + self.loss_weights.bridge_commitment * commitment
            total = total + self.loss_weights.whisper_quantize * diagnostics["whisper_quantize"]
            total = total + self.loss_weights.teacher_glm_ce * diagnostics["teacher_glm_ce"]
            total = total + self.loss_weights.teacher_glm_commitment * diagnostics["teacher_glm_commitment"]
        extras = torch.stack(
            [
                diagnostics["asr_infeasible"].detach().float(),
                diagnostics["nar_infeasible"].detach().float(),
                diagnostics["unit_infeasible"].detach().float(),
                diagnostics["bridge_commitment"].detach().float(),
                diagnostics["whisper_quantize"].detach().float(),
                diagnostics["teacher_glm_ce"].detach().float(),
                diagnostics["teacher_glm_commitment"].detach().float(),
                diagnostics["teacher_glm_agreement"].detach().float(),
                diagnostics["teacher_glm_coverage"].detach().float(),
                diagnostics["code_perplexity"].detach().float(),
                diagnostics["active_code_fraction"].detach().float(),
                diagnostics["hidden_rms"].detach().float(),
                total.detach().new_tensor(active_joint),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(extras)
            extras /= dist.get_world_size()
        values = [total]
        values.extend(means[name].detach() for name in COMPONENTS)
        values.extend(extras)
        finite_names = ["total", *COMPONENTS, *diagnostics, "sampler_joint_fraction"]
        non_finite = [
            name
            for name, value in zip(finite_names, values, strict=True)
            if not bool(torch.isfinite(value).all())
        ]
        if non_finite:
            raise FloatingPointError(
                "non-finite joint model loss or metric: "
                f"fields={non_finite}, sample_kind={sample_kind}, chunk_ms={chunk_ms}"
            )
        return torch.stack(values)
