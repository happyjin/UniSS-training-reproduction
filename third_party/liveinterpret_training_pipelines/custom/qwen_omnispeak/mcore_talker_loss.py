# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron-native talker codec loss for Qwen-OmniSpeak (non-invasive).

This module wires the *talker codec loss* (main-code CE + residual code-predictor
CE) into the Megatron ``megatron sft`` training step, so that the talker can be
trained under Megatron model parallelism (TP/EP/DP + ZeRO-1) instead of only the
DeepSpeed path.

Design (no edits to ms-swift / mcore-bridge source):

* The thinker is the Megatron ``GPTModel`` (TP/EP sharded). Its text CE is the
  usual Megatron per-token loss.
* The talker + bridge are loaded as replicated submodules of the mcore ``visual``
  module (see ``mcore_adaptation.QwenOmniSpeakVit.module_mapping``). They are
  DP-synchronised and checkpointed by the standard visual load/save path.
* A forward hook captures the thinker hidden states at ``accept_hidden_layer``
  (the SAME intermediate layer the DeepSpeed model uses). After the model forward
  returns the text loss, we run bridge -> talker -> codec heads on the captured
  hidden states and compute the codec loss, reusing the exact helpers from
  ``modeling_qwen_omnispeak`` for numerical parity with the DeepSpeed path.
* ``MegatronTrainer.forward_step`` / ``loss_func`` are monkey-patched. The patch
  is a strict no-op unless (a) the global enable flag is set, (b) the batch
  carries codec fields, and (c) the model actually owns a talker. Plain
  ``swift sft`` (DeepSpeed, variants 1/2) and all other Megatron models are
  therefore completely unaffected.

Supported parallelism for the codec path: ``TP>=1`` + ``EP>=1`` + ``DP>=1`` with
``PP==1``, ``CP==1``, ``sequence_parallel=False`` and no packing/padding-free.
The thinker still gets the full benefit of TP/EP/DP/ZeRO-1; the talker is
replicated. Unsupported configurations raise a clear error (fail fast) rather
than silently computing a wrong loss.
"""
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

try:
    from swift.utils import get_logger
    logger = get_logger()
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

# Reuse the EXACT codec helpers from the modeling file for parity.
try:
    from .modeling_qwen_omnispeak import (
        apply_speaker_prompt_to_codec_emb,
        build_block_causal_mask,
        build_icl_prefix,
        _codec_eos_loss_weight,
        _codec_history_corruption_prob,
        _corrupt_codec_history,
        _cross_entropy,
        _residual_loss_settings,
        _use_speaker_prompt,
        _weighted_codec_cross_entropy,
    )
except ImportError:  # pragma: no cover - top-level plugin import.
    from modeling_qwen_omnispeak import (  # type: ignore
        apply_speaker_prompt_to_codec_emb,
        build_block_causal_mask,
        build_icl_prefix,
        _codec_eos_loss_weight,
        _codec_history_corruption_prob,
        _corrupt_codec_history,
        _cross_entropy,
        _residual_loss_settings,
        _use_speaker_prompt,
        _weighted_codec_cross_entropy,
    )


# Codec batch keys produced by the bridge-alignment / continue-pretrain templates.
CODEC_KEYS = (
    'codec_input_ids',
    'codec_labels',
    'codec_attention_mask',
    'codec_residual_labels',
)

# Speaker-conditioning fields produced by the template/collator. Popped before
# the thinker GPTModel forward and consumed only by ``compute_codec_loss``.
SPEAKER_KEYS = (
    'speaker_embedding',
    'reference_audio_wav',
    'reference_audio_sr',
    'speaker_slot_index',
)

# Streaming read-write schedule produced by the simultaneous-policy template.
# When present it gates the ``codec -> sem`` attention block so the talker only
# attends to the source-audio prefix that has "arrived" by each codec frame
# (the READ half of the read-write policy). Absent for the offline bridge /
# continue-pretrain templates, in which case the block stays fully visible.
STREAMING_KEY = 'codec_to_sem_visible'

_PATCHED = False


# ---------------------------------------------------------------------------
# Enable flag
# ---------------------------------------------------------------------------
def codec_loss_enabled() -> bool:
    """Whether to compute the talker codec loss inside ``megatron sft``.

    Controlled by ``QWEN_OMNISPEAK_MCORE_CODEC_LOSS`` (default: enabled).
    The legacy ``QWEN3_OMNISPEAK_MCORE_CODEC_LOSS`` is still accepted.
    """
    val = os.environ.get('QWEN_OMNISPEAK_MCORE_CODEC_LOSS')
    if val is None:
        val = os.environ.get('QWEN3_OMNISPEAK_MCORE_CODEC_LOSS')
    if val is None:
        return True
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unwrap_mm(model):
    """Strip DDP / Float16Module wrappers to reach the MultimodalGPTModel."""
    m = model
    for _ in range(6):
        if hasattr(m, 'visual') or hasattr(m, 'language_model'):
            return m
        if hasattr(m, 'module'):
            m = m.module
        else:
            break
    return m


def _has_codec(data: Dict[str, Any]) -> bool:
    return data.get('codec_input_ids') is not None and data.get('codec_labels') is not None


def _has_talker(model) -> bool:
    mm = _unwrap_mm(model)
    visual = getattr(mm, 'visual', None)
    return visual is not None and getattr(visual, 'talker', None) is not None


def _resolve_hidden_layer_index(hf_config, num_local_layers: int) -> Tuple[int, bool]:
    """Map hidden-state selection to ``(decoder layer index, apply final norm)``.

    Integer selections follow HF ``hidden_states[k]`` semantics (the output of
    decoder layer ``k - 1``) and hook that layer directly. The string
    selections ``last``/``final`` correspond to HF ``hidden_states[-1]``,
    which is the tensor AFTER the thinker's final RMSNorm (HF appends the
    normed output as the last tuple element). Megatron-Core's
    ``TransformerBlock`` runs ``final_layernorm`` only after ALL layers, so a
    bare hook on ``decoder.layers[-1]`` captures the PRE-norm tensor — a
    different distribution into the adapter (P0-2, fixed 2026-08: the hook now
    applies the decoder's final_layernorm for ``last``/``final`` selections).
    """
    override = os.environ.get('QWEN_OMNISPEAK_MCORE_HIDDEN_LAYER')
    if override is None:
        override = os.environ.get('QWEN3_OMNISPEAK_MCORE_HIDDEN_LAYER')
    selection = override
    if selection is None:
        selection = getattr(hf_config, 'hidden_state_selection', getattr(hf_config, 'accept_hidden_layer', 24))
    if isinstance(selection, str) and selection.strip().lower() in ('last', 'final', '-1'):
        idx = num_local_layers - 1
        apply_final_norm = True
    else:
        accept = int(selection)
        idx = accept - 1
        apply_final_norm = False
    if idx < 0:
        idx = 0
    if idx > num_local_layers - 1:
        logger.warning(
            f'[qwen_omnispeak] accept_hidden_layer maps to decoder layer {idx} but only '
            f'{num_local_layers} layers are present on this stage; clamping to the last layer.')
        idx = num_local_layers - 1
    return idx, apply_final_norm


def _install_hidden_hook(mm) -> None:
    """Register a forward hook capturing the thinker hidden states at the
    bridge's ``accept_hidden_layer``. Installed once per model instance.
    """
    if getattr(mm, '_omnispeak_hook_installed', False):
        return
    decoder = mm.language_model.decoder
    layers = decoder.layers
    hf_config = mm.visual.hf_config
    idx, apply_final_norm = _resolve_hidden_layer_index(hf_config, len(layers))
    final_norm = decoder.final_layernorm if apply_final_norm else None

    def _hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Megatron-Core decoder layers emit [S, B, H]. Normalise to [B, S, H]
        # here, at the single capture point, so downstream consumers never
        # need to guess the layout from shape equality (which silently
        # mis-transposes when S == B; P2-1).
        hs = hs.transpose(0, 1).contiguous()
        if final_norm is not None:
            hs = final_norm(hs)
        mm._omnispeak_hidden = hs

    layers[idx].register_forward_hook(_hook)
    mm._omnispeak_hook_installed = True
    mm._omnispeak_hidden = None
    logger.info(
        f'[qwen_omnispeak] Installed thinker hidden-state capture hook on decoder layer {idx}'
        f' (final_layernorm={"applied" if final_norm is not None else "not applied"}).')


def _check_parallel_guards(args) -> None:
    """Fail fast for parallel configs the codec path cannot handle correctly."""
    pp = getattr(args, 'pipeline_model_parallel_size', 1) or 1
    cp = getattr(args, 'context_parallel_size', 1) or 1
    sp = bool(getattr(args, 'sequence_parallel', False))
    packing = bool(getattr(args, 'packing', False))
    padding_free = bool(getattr(args, 'padding_free', False))
    problems = []
    if pp != 1:
        problems.append(f'pipeline_model_parallel_size={pp} (require 1)')
    if cp != 1:
        problems.append(f'context_parallel_size={cp} (require 1)')
    if sp:
        problems.append('sequence_parallel=True (require False)')
    if packing:
        problems.append('packing=True (require False)')
    if padding_free:
        problems.append('padding_free=True (require False)')
    if problems:
        raise ValueError(
            'Qwen-OmniSpeak Megatron talker codec loss does not support: '
            + ', '.join(problems)
            + '. The thinker still uses TP/EP/DP/ZeRO-1; only the talker codec path is '
            'constrained. Either adjust these flags, or disable the codec loss with '
            'QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false (thinker text training only).')


def _pad_token_id(trainer) -> Optional[int]:
    for attr in ('processor', 'tokenizer'):
        obj = getattr(trainer.template, attr, None)
        pad = getattr(obj, 'pad_token_id', None)
        if pad is not None:
            return int(pad)
    return None


def _compute_text_embeds(mm, input_ids: torch.Tensor) -> torch.Tensor:
    """Pure thinker token embeddings ``[B, T, H]`` (no audio/vision injection),
    matching the DeepSpeed ``thinker.get_input_embeddings()(input_ids)`` route.
    """
    word_embeddings = mm.language_model.embedding.word_embeddings
    prev = getattr(word_embeddings, 'reduce_scatter_embeddings', False)
    try:
        if prev:
            word_embeddings.reduce_scatter_embeddings = False
        ids = input_ids.clamp(min=0)
        embeds = word_embeddings(ids)
    finally:
        if prev:
            word_embeddings.reduce_scatter_embeddings = prev
    # VocabParallelEmbedding returns [B, T, H] for a [B, T] input.
    return embeds


def _build_sem_keep_mask(thinker_attention_mask, sem_block):
    """Mirror modeling_qwen_omnispeak.forward sem_keep_mask alignment logic."""
    sem_len = sem_block.size(1)
    sem_keep_mask = thinker_attention_mask
    if sem_keep_mask is None and sem_len > 0:
        return torch.ones(sem_block.size(0), sem_len, dtype=torch.bool, device=sem_block.device)
    if sem_keep_mask is not None:
        sem_keep_mask = sem_keep_mask.to(device=sem_block.device, dtype=torch.bool)
        if sem_keep_mask.dim() == 1:
            sem_keep_mask = sem_keep_mask.unsqueeze(0)
        if sem_keep_mask.size(1) != sem_len:
            if sem_len % sem_keep_mask.size(1) == 0:
                sem_keep_mask = sem_keep_mask.repeat(1, sem_len // sem_keep_mask.size(1))
            elif sem_keep_mask.size(1) > sem_len:
                sem_keep_mask = sem_keep_mask[:, :sem_len]
            else:
                pad = sem_keep_mask.new_ones(sem_keep_mask.size(0), sem_len - sem_keep_mask.size(1))
                sem_keep_mask = torch.cat([sem_keep_mask, pad], dim=1)
    return sem_keep_mask


def _build_upstream_text_channel(
    thinker_adapter,
    talker,
    cfg,
    *,
    thinker_hidden_states: torch.Tensor,       # [B, T_thinker, H_thinker]
    thinker_text_embeds: Optional[torch.Tensor],  # [B, T_thinker, H_thinker] or None
    text_labels: Optional[torch.LongTensor],     # [B, T_thinker] reply positions = (labels != -100)
    target_text_ids: Optional[torch.LongTensor],  # [B, T_text] or None (unused now)
    codec_len: int,
    codec_bos_index: Optional[torch.Tensor],   # [B] or None
    thinker_attention_mask: Optional[torch.Tensor],
    codec_dtype: torch.dtype,
    codec_device: torch.device,
    full_add_prompt_len: Optional[torch.Tensor] = None,  # [B] or None
) -> torch.Tensor:
    """Megatron mirror of ``modeling_qwen_omnispeak.QwenOmniSpeakForConditional
    Generation._build_upstream_text_channel`` (V1 hidden+embed upstream protocol).

    The two implementations must stay numerically bit-identical — this is the
    *single* place that assembles the length-``codec_len`` text-side conditioning
    tensor ``text_channel`` for the upstream talker input protocol. Any change
    here must be mirrored to the DeepSpeed method (and vice versa).

    Returns ``[B, codec_len, H_talker]`` where each frame is either
    ``text_projection(adapter(reply_signal[k]))`` mapped to codec position
    ``codec_bos_index + k``, or ``text_projection(text_embedding(tts_pad_id))``
    elsewhere.
    """
    bsz = thinker_hidden_states.size(0)
    H = talker.model.config.hidden_size

    # ── 1. Extract reply-only positions via text_labels != -100 ───────
    if text_labels is None:
        logger.warning(
            '[upstream-mcore] text_labels is None; using entire thinker '
            'sequence (no reply-position filtering).')
        reply_hidden = thinker_hidden_states
        reply_embed = thinker_text_embeds
    else:
        reply_mask = (text_labels != -100)              # [B, T_thinker]
        reply_lens = reply_mask.sum(dim=-1).tolist()
        max_reply = max(reply_lens) if reply_lens else 0
        if max_reply == 0:
            logger.warning(
                '[upstream-mcore] text_labels has no reply positions (all -100); '
                'text_channel will be all pad_embed.')
            reply_hidden = thinker_hidden_states.new_zeros(
                bsz, 0, thinker_hidden_states.size(-1))
            reply_embed = (
                thinker_text_embeds.new_zeros(bsz, 0, thinker_text_embeds.size(-1))
                if thinker_text_embeds is not None else None
            )
        else:
            reply_hidden_list = []
            reply_embed_list = []
            for b in range(bsz):
                idx = reply_mask[b].nonzero(as_tuple=False).squeeze(-1)
                reply_hidden_list.append(thinker_hidden_states[b].index_select(0, idx))
                if thinker_text_embeds is not None:
                    reply_embed_list.append(thinker_text_embeds[b].index_select(0, idx))
            reply_hidden = torch.zeros(
                bsz, max_reply, thinker_hidden_states.size(-1),
                dtype=thinker_hidden_states.dtype, device=thinker_hidden_states.device)
            for b in range(bsz):
                lb = reply_lens[b]
                if lb > 0:
                    reply_hidden[b, :lb] = reply_hidden_list[b]
            if thinker_text_embeds is not None:
                reply_embed = torch.zeros(
                    bsz, max_reply, thinker_text_embeds.size(-1),
                    dtype=thinker_text_embeds.dtype, device=thinker_text_embeds.device)
                for b in range(bsz):
                    lb = reply_lens[b]
                    if lb > 0:
                        reply_embed[b, :lb] = reply_embed_list[b]
            else:
                reply_embed = None
            reply_lens_tensor = torch.tensor(
                reply_lens, dtype=torch.long, device=thinker_hidden_states.device)

    # ── 2. thinker_signal = reply_hidden + reply_embed ────────────────
    signal_type = getattr(cfg, 'thinker_signal_type', 'hidden_plus_embed')
    if signal_type == 'hidden_plus_embed':
        if reply_embed is None:
            logger.warning(
                '[upstream-mcore] thinker_text_embeds is None; using '
                'reply_hidden only (degraded V1 signal).')
            thinker_signal = reply_hidden
        elif reply_embed.shape != reply_hidden.shape:
            raise ValueError(
                f'reply_hidden {tuple(reply_hidden.shape)} and reply_embed '
                f'{tuple(reply_embed.shape)} must have identical shape.')
        else:
            thinker_signal = reply_hidden + reply_embed
    elif signal_type == 'logits_soft_embed':
        raise NotImplementedError(
            'thinker_signal_type=logits_soft_embed is not implemented.')
    else:
        raise ValueError(f'Unknown thinker_signal_type: {signal_type!r}')

    # ── 3. adapter -> text_projection ──────────────────────────────────
    adp_dtype = next(thinker_adapter.parameters()).dtype
    adapted = thinker_adapter(thinker_signal.to(adp_dtype))
    projected = talker.text_projection(
        adapted.to(next(talker.text_projection.parameters()).dtype))
    t_reply = projected.size(1)

    # ── 4. Per-sample reply length ─────────────────────────────────────
    if text_labels is not None and reply_lens and max_reply > 0:
        text_valid_lens = reply_lens_tensor
    elif target_text_ids is not None:
        text_valid_lens = (
            target_text_ids != cfg.talker_config.tts_pad_token_id).long().sum(dim=-1)
    elif thinker_attention_mask is not None:
        text_valid_lens = thinker_attention_mask.to(torch.long).sum(dim=-1)
    else:
        text_valid_lens = torch.full(
            (bsz,), t_reply, dtype=torch.long, device=projected.device)

    # ── 5. Build ``tts_pad_embed`` ─────────────────────────────────────
    pad_id = int(cfg.talker_config.tts_pad_token_id)
    pad_emb_1d = talker.text_projection(
        talker.model.text_embedding(
            torch.tensor([pad_id], device=projected.device, dtype=torch.long)
        ).to(next(talker.text_projection.parameters()).dtype)
    ).to(codec_dtype).squeeze(0)                                     # [H_talker]

    # ── 6. Scatter reply text into length-``codec_len`` buffer ─────────
    text_channel = pad_emb_1d.expand(bsz, codec_len, H).clone().to(codec_dtype)
    if codec_bos_index is None:
        codec_bos_index = torch.zeros(bsz, dtype=torch.long, device=projected.device)
    codec_bos_index = codec_bos_index.to(projected.device, dtype=torch.long)
    for b in range(bsz):
        start = int(codec_bos_index[b].item())
        span = int(text_valid_lens[b].item())
        span = min(span, t_reply, max(0, codec_len - start))
        if span > 0:
            text_channel[b, start:start + span] = projected[b, :span].to(codec_dtype)

    # ── 7. Full-add layout: thinker prompt signal + text_bos ───────────
    if full_add_prompt_len is not None:
        adp_dtype = next(thinker_adapter.parameters()).dtype
        tp_dtype = next(talker.text_projection.parameters()).dtype
        signal_type = getattr(cfg, 'thinker_signal_type', 'hidden_plus_embed')
        bos_id = int(getattr(cfg.talker_config, 'tts_bos_token_id', 151_672))
        bos_emb_1d = talker.text_projection(
            talker.model.text_embedding(
                torch.tensor([bos_id], device=projected.device, dtype=torch.long)
            ).to(tp_dtype)
        ).to(codec_dtype).squeeze(0)                                  # [H_talker]
        fap = full_add_prompt_len.to(projected.device, dtype=torch.long)
        for b in range(bsz):
            P = max(0, min(int(fap[b].item()), codec_len))
            if P > 0:
                ph = thinker_hidden_states[b, :P]
                if thinker_text_embeds is not None and signal_type == 'hidden_plus_embed':
                    ph = ph + thinker_text_embeds[b, :P]
                adapted_p = thinker_adapter(ph.to(adp_dtype))
                proj_p = talker.text_projection(adapted_p.to(tp_dtype)).to(codec_dtype)
                text_channel[b, :P] = proj_p
            bos_pos = int(codec_bos_index[b].item()) - 1
            if 0 <= bos_pos < codec_len:
                text_channel[b, bos_pos] = bos_emb_1d

    return text_channel


# ---------------------------------------------------------------------------
# Core: codec loss from captured thinker hidden states
# ---------------------------------------------------------------------------
def compute_codec_loss(
    mm,
    *,
    hf_config,
    input_ids: torch.Tensor,
    captured_hidden: torch.Tensor,
    codec: Dict[str, torch.Tensor],
    thinker_attention_mask: Optional[torch.Tensor],
    has_codec_supervision: Optional[torch.Tensor],
    codec_to_sem_visible: Optional[torch.Tensor] = None,
    use_upstream_input: bool = False,
    text_labels: Optional[torch.Tensor] = None,
    codec_bos_index: Optional[torch.Tensor] = None,
    target_text_ids: Optional[torch.Tensor] = None,
    full_add_prompt_len: Optional[torch.Tensor] = None,
    speaker_embedding: Optional[Any] = None,
    reference_audio_wav: Optional[list] = None,
    reference_audio_sr: Optional[list] = None,
    speaker_slot_index: Optional[torch.Tensor] = None,
    icl_source_codec: Optional[torch.Tensor] = None,
    icl_source_text_ids: Optional[torch.Tensor] = None,
    icl_source_codec_mask: Optional[torch.Tensor] = None,
    icl_source_text_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    """Run talker -> codec heads and return ``(codec_loss, num_tokens, metrics)``.

    Numerically mirrors ``QwenOmniSpeakForConditionalGeneration.forward`` codec
    branch (modeling_qwen_omnispeak.py). Two input protocols, selected by
    ``use_upstream_input`` (mirroring ``config.use_upstream_talker_input``):

    * ``False`` (default): the historical bridge path — ``bridge(...)`` then
      ``full_emb = cat([sem_block, codec_emb])``, block-causal 4D mask, and
      ``codec_hidden = last_hidden[:, sem_len:]``.
    * ``True``: the upstream Qwen3-TTS input protocol — ``text_channel`` from
      ``_build_upstream_text_channel`` then ``full_emb = codec_emb + text_channel``,
      2-D padding mask (talker adds causal), and ``codec_hidden = last_hidden``
      (every position is a codec frame). ``thinker_adapter`` must exist on
      ``mm.visual`` (built by ``mcore_adaptation.QwenOmniSpeakVit.prepare_model``).

    ``codec_loss`` is a mean (already including the residual weight);
    ``num_tokens`` counts supervised main-code positions; ``metrics`` carries
    ``[sum, count]`` vectors for logging.

    ``codec_to_sem_visible`` (simultaneous-policy template only) is the monotonic
    read-write schedule: a ``[B, codec_len]`` count of leading semantic tokens
    each codec frame may attend to. When ``None`` the ``codec -> sem`` block is
    fully visible (offline), exactly matching the bridge / continue-pretrain
    behaviour.

    Speaker prompt fields mirror DeepSpeed: after ``embed_codec``, the reserved
    speaker-slot embedding is replaced by an offline/online ECAPA x-vector via
    the shared ``apply_speaker_prompt_to_codec_emb`` helper.
    """
    visual = mm.visual
    bridge = visual.bridge
    talker = visual.talker
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype
    # Only present when ``use_upstream_input`` is True (see mcore_adaptation).
    thinker_adapter = getattr(visual, 'thinker_adapter', None)

    # ``captured_hidden`` already arrives as [B, S, H]: the capture hook
    # normalises the Megatron [S, B, H] layout once, at the source.
    hidden = captured_hidden
    thinker_hidden_states = hidden.to(device=device, dtype=dtype)
    thinker_text_embeds = _compute_text_embeds(mm, input_ids).to(device=device, dtype=dtype)

    codec_input_ids = codec['codec_input_ids'].to(device)
    codec_labels = codec['codec_labels'].to(device)
    if talker.training:
        codec_input_ids = _corrupt_codec_history(
            codec_input_ids,
            codec_labels,
            probability=_codec_history_corruption_prob(hf_config),
            acoustic_vocab_size=int(getattr(
                hf_config.talker_config, 'code_predictor_vocab_size', 2048)),
        )
    codec_attention_mask = codec.get('codec_attention_mask')
    if codec_attention_mask is not None:
        codec_attention_mask = codec_attention_mask.to(device)
    codec_residual_labels = codec.get('codec_residual_labels')
    if codec_residual_labels is not None:
        codec_residual_labels = codec_residual_labels.to(device)
    if thinker_attention_mask is not None:
        thinker_attention_mask = thinker_attention_mask.to(device)
    if codec_to_sem_visible is not None:
        codec_to_sem_visible = codec_to_sem_visible.to(device)

    metrics: Dict[str, torch.Tensor] = {}

    codec_emb = talker.model.embed_codec(codec_input_ids)
    # Speaker prompt (voice-clone x-vector injection): the SAME helper the
    # DeepSpeed forward uses, so both backends substitute an identical ECAPA
    # x-vector at the same control-prefix slot. Prefers offline
    # speaker_embedding (Lance / source_xvector); falls back to wav.
    # Missing fields under an enabled config mean the data pipeline dropped
    # them — fail fast instead of training on the placeholder.
    if _use_speaker_prompt(hf_config):
        if speaker_slot_index is None or (
                speaker_embedding is None and reference_audio_wav is None):
            raise RuntimeError(
                '[qwen_omnispeak] use_speaker_prompt is enabled but the batch '
                'carries no speaker_slot_index plus '
                'speaker_embedding/reference_audio_wav; the Megatron data '
                'pipeline dropped the speaker fields (check the '
                'pre-tokenisation schema and template._ensure_codec_fields).')
        codec_emb = apply_speaker_prompt_to_codec_emb(
            codec_emb,
            config=hf_config,
            speaker_slot_index=speaker_slot_index,
            speaker_embedding=speaker_embedding,
            reference_audio_wav=reference_audio_wav,
            reference_audio_sr=reference_audio_sr,
        )
    codec_len = codec_emb.size(1)
    codec_keep_mask = codec_attention_mask
    if codec_keep_mask is None and codec_len > 0:
        codec_keep_mask = torch.ones(codec_emb.size(0), codec_len, dtype=torch.bool, device=device)

    # Uns supervised ICL prefix length (upstream protocol only); the bridge
    # path never prepends one.
    _icl_len = 0
    if use_upstream_input:
        # ═══════════════════════════════════════════════════════════════
        #  Upstream Qwen3-TTS input protocol (V1: hidden+embed).
        #  inputs_embeds[t] = text_projection(adapter(thinker_signal))[t]
        #                    + codec_embedding(codec_input_ids[t])
        #  text side pads with tts_pad_embed outside the assistant span.
        #  Mirrors modeling_qwen_omnispeak upstream branch (full_emb =
        #  codec_emb + text_channel, 2-D padding mask, no sem slice).
        # ═══════════════════════════════════════════════════════════════
        if thinker_adapter is None:
            raise RuntimeError(
                '[qwen_omnispeak] use_upstream_input=True but mm.visual has no '
                'thinker_adapter; ensure mcore_adaptation builds it when '
                'use_upstream_talker_input is enabled.')
        # Robustness: mirror the DeepSpeed forward — derive codec_bos_index /
        # full_add_prompt_len from surviving tensors if not passed in.
        if codec_bos_index is None and codec_input_ids is not None:
            _bos = int(getattr(hf_config, 'codec_bos_id', 2149))
            _hits = (codec_input_ids == _bos)
            _bidx = []
            for _b in range(codec_input_ids.size(0)):
                _nz = _hits[_b].nonzero(as_tuple=False)
                _bidx.append(int(_nz[0].item()) if _nz.numel() > 0 else 0)
            codec_bos_index = torch.tensor(
                _bidx, dtype=torch.long, device=codec_input_ids.device)
        text_channel = _build_upstream_text_channel(
            thinker_adapter,
            talker,
            hf_config,
            thinker_hidden_states=thinker_hidden_states,
            thinker_text_embeds=thinker_text_embeds,
            text_labels=text_labels,
            target_text_ids=target_text_ids,
            codec_len=codec_len,
            codec_bos_index=codec_bos_index,
            thinker_attention_mask=thinker_attention_mask,
            codec_dtype=codec_emb.dtype,
            codec_device=codec_emb.device,
            full_add_prompt_len=full_add_prompt_len,
        )                                                 # [B, codec_len, H]
        full_emb = codec_emb + text_channel
        sem_len = 0
        attn = codec_keep_mask
        # ── ICL prefix: prepend the reference (source) codec+text segment ──
        # Mirrors the DeepSpeed forward's ICL branch (the prefix is NOT
        # supervised; it conditions the talker on the source speaker). Built
        # by the shared ``build_icl_prefix`` helper so both backends produce
        # an identical prefix (P1-7). Enabled via QWEN_OMNISPEAK_USE_ICL —
        # when on, missing source fields mean the pipeline dropped them, so
        # fail fast instead of silently degrading to non-ICL training.
        _icl_enabled = os.environ.get(
            'QWEN_OMNISPEAK_USE_ICL', 'false').strip().lower() in ('1', 'true', 'yes')
        if _icl_enabled and icl_source_codec is None:
            raise RuntimeError(
                '[qwen_omnispeak] QWEN_OMNISPEAK_USE_ICL is enabled but the '
                'batch carries no source_codec_codes/source_text_ids; the '
                'Megatron data pipeline dropped the ICL fields (check the '
                'pre-tokenisation schema and template._encode ICL pass-through).')
        # Gate on _icl_enabled as well — parity with the DeepSpeed forward
        # (modeling_qwen_omnispeak.py: ``_icl_enabled and icl_source_codec
        # is not None``) and with the parity-review spec's common config
        # (ICL off). The batch may carry source_codec_codes (kept alive by
        # the register whitelist) even when ICL is off; building the prefix
        # then would crash on the collator's list form and diverge the
        # forward structure between backends.
        if _icl_enabled and icl_source_codec is not None:
            icl_embed, icl_mask = build_icl_prefix(
                hf_config, talker,
                source_codec_codes=icl_source_codec,
                source_text_ids=icl_source_text_ids,
                source_codec_mask=icl_source_codec_mask,
                source_text_mask=icl_source_text_mask,
                codec_dtype=codec_emb.dtype, codec_device=codec_emb.device,
            )
            _icl_len = icl_embed.size(1)
            full_emb = torch.cat([icl_embed, full_emb], dim=1)
            attn = torch.cat([icl_mask, attn], dim=1)
    else:
        # ═══════════════════════════════════════════════════════════════
        #  Legacy DualRoute bridge + block-causal path (unchanged).
        # ═══════════════════════════════════════════════════════════════
        sem_block = bridge(
            thinker_text_embeds=thinker_text_embeds,
            thinker_hidden_states=thinker_hidden_states,
            semantic_route_mask=None,
            thinker_hidden_attention_mask=thinker_attention_mask,
        )
        full_emb = torch.cat([sem_block, codec_emb], dim=1)
        sem_len = sem_block.size(1)

        sem_keep_mask = _build_sem_keep_mask(thinker_attention_mask, sem_block)
        attn = build_block_causal_mask(
            sem_len=sem_len, codec_len=codec_len,
            sem_attention_mask=sem_keep_mask, codec_attention_mask=codec_keep_mask,
            dtype=full_emb.dtype, device=full_emb.device,
            codec_to_sem_visible=codec_to_sem_visible,
        )
    # Gradient checkpointing (recompute in backward): the talker backbone is a
    # *replicated* (unsharded) submodule, so its activations are NOT divided by TP.
    # Measured on H200: with codec_len up to ~9529 (full_emb up to ~9721) the
    # un-recomputed 28-layer dense backbone pushes peak activation VRAM to ~89GB on
    # the 3B run and >140GB on the larger model, OOMing at the code-predictor
    # down_proj. Recompute bounds peak activation memory independent of sequence
    # length (math identical: same grads).
    def _talker_backbone_fwd(emb, attn):
        return talker.model(inputs_embeds=emb, attention_mask=attn).last_hidden_state
    last_hidden = _grad_checkpoint(_talker_backbone_fwd, full_emb, attn, use_reentrant=False)
    # Upstream: ``sem_len`` is 0 (every position is a codec frame) and the
    # unsupervised ICL prefix (if any) is sliced off to realign with the codec
    # labels; bridge: drops the leading ``sem`` block.
    codec_hidden = last_hidden[:, _icl_len + sem_len:, :]
    main_codec_logits = talker.codec_head(codec_hidden)

    residual_loss_enabled, residual_loss_weight = _residual_loss_settings(hf_config)
    residual_codec_logits = None
    # 2026-09-17 fix: this ``codec_residual_labels is not None`` check is
    # DATA-dependent (per-microbatch batch content), not a config/model-level
    # flag — some microbatches' samples simply lack precomputed residual
    # codec labels. When that happens, ``talker.code_predictor`` (a distinct
    # trainable submodule — the residual code predictor head) is never
    # invoked, so none of its parameters get an autograd hook fire in this
    # microbatch. Under Megatron 0.17+'s ``overlap_grad_reduce`` grad-ready
    # invariant (every trainable param must register grad-ready in every
    # microbatch, see ``megatron/core/distributed/param_and_grad_buffer.py``
    # ``_ParamAndGradBucketGroup.reset()``'s
    # ``assert len(per_param_grad_ready_counts) == len(params)``), this
    # asymmetry crashes the *next* step's ``zero_grad_buffer()`` call the
    # first time step 1's last microbatch happens to skip this branch
    # (repro'd 2026-09-17 with ``OVERLAP_GRAD_REDUCE=true``: AssertionError on
    # all ranks right at the start of step 2). Same root-cause pattern as the
    # ``codec_loss``/``num_tokens>0`` fix below — anchor a ZERO gradient into
    # every ``code_predictor`` param when the branch is skipped (see the
    # ``elif residual_loss_enabled:`` block further down) so its
    # accumulate-grad hook still fires every microbatch, without touching the
    # loss value or the true gradient (PERF_TUNING.md already anticipated
    # this exact fix under "数据相关→加梯度锚点 loss").
    _residual_branch_ran = False
    if residual_loss_enabled and codec_labels is not None and codec_residual_labels is not None:
        residual_codec_logits = talker.code_predictor.forward_teacher_forced(
            talker_hidden=codec_hidden,
            main_codes=codec_labels,
            residual_codes=codec_residual_labels,
            main_codec_embedding=talker.model.codec_embedding,
        )
        _residual_branch_ran = True

    effective_labels = codec_labels
    effective_residual = codec_residual_labels
    if has_codec_supervision is not None:
        keep = has_codec_supervision.to(device=effective_labels.device, dtype=torch.bool)
        if keep.numel() == effective_labels.size(0):
            mask_main = keep[:, None].expand_as(effective_labels)
            effective_labels = effective_labels.masked_fill(~mask_main, -100)
            if effective_residual is not None:
                mask_res = keep[:, None, None].expand_as(effective_residual)
                effective_residual = effective_residual.masked_fill(~mask_res, -100)

    main_valid = (effective_labels != -100)
    num_tokens = main_valid.sum()

    # Always compute codec_loss (do NOT gate on num_tokens > 0).
    #
    # Megatron 0.17+ ``overlap_grad_reduce`` requires every trainable param to
    # register grad-ready in every micro-batch (``param_and_grad_buffer.py``
    # ``assert len(per_param_grad_ready_counts) == len(params)``).  The old
    # ``if bool(num_tokens > 0)`` guard skipped ``_weighted_codec_cross_entropy``
    # when all labels were -100, which broke the gradient chain from loss to
    # ``main_codec_logits`` (a local variable not returned to the trainer),
    # leaving talker/bridge/codec_head parameters without any gradient.
    #
    # ``_weighted_codec_cross_entropy`` already handles the zero-label case:
    # when ``valid_count == 0`` it returns ``flat_logits.sum() * 0.0`` — a zero
    # loss that still triggers autograd through ``main_codec_logits``, so every
    # talker parameter gets a zero gradient registered.  This is mathematically
    # identical to skipping the loss (no gradient contribution) while keeping
    # the backward graph intact for Megatron's grad-ready invariant.
    eos_id = int(getattr(hf_config, 'codec_eos_id', 2150))
    codec_loss = _weighted_codec_cross_entropy(
        main_codec_logits,
        effective_labels,
        eos_id=eos_id,
        eos_weight=_codec_eos_loss_weight(hf_config),
    )

    # main-code accuracy (logging) — always computed so the downstream
    # all_reduce key schema stays rank-identical (fixed ``_CODEC_METRIC_KEYS``
    # + zero-placeholder deadlock fix in ``_combined_loss_func``).
    with torch.no_grad():
        preds = main_codec_logits.argmax(dim=-1)
        correct = ((preds == effective_labels) & main_valid).sum().float()
        eos_boundary = effective_labels.eq(eos_id)
        boundary_correct = preds[eos_boundary].eq(eos_id).sum().float()
        metrics['eos_boundary_top1_acc'] = torch.stack([
            boundary_correct, eos_boundary.sum().float().clamp(min=1)])
        non_boundary = main_valid & ~eos_boundary
        premature = preds[non_boundary].eq(eos_id).sum().float()
        metrics['premature_eos_top1_rate'] = torch.stack([
            premature, non_boundary.sum().float().clamp(min=1)])
    metrics['codec_token_acc'] = torch.stack([correct, num_tokens.float().clamp(min=1)])

    # residual loss reporting. NOTE (2026-09-17, corrected): the comment here
    # used to claim this whole branch is "gated ONLY on config/model-level
    # flags, never data-dependent" — that was WRONG about the code_predictor
    # *forward call* above (``codec_labels/codec_residual_labels is not
    # None`` IS a per-microbatch data check, see the long comment above
    # ``_residual_branch_ran``). What IS still true and load-bearing here:
    # this ``if`` only reads ``.is not None`` (no ``.any()``/data-dependent
    # boolean reduction), so it doesn't force a cudaStreamSynchronize or make
    # the DP-wide all_reduce collective schedule asymmetric across ranks —
    # every rank enters this branch identically. The grad-ready asymmetry
    # (different problem: some *params* untouched, not an asymmetric
    # collective count) is what the ``elif`` below fixes.
    if residual_loss_enabled and residual_codec_logits is not None and effective_residual is not None:
        V = residual_codec_logits.shape[-1]
        resid_valid = (effective_residual != -100)
        resid_num = resid_valid.sum().clamp(min=1).float()
        resid_loss = _cross_entropy(
            residual_codec_logits.reshape(-1, V),
            effective_residual.reshape(-1),
            ignore_index=-100,
            reduction='sum',
        ) / resid_num
        codec_loss = codec_loss + residual_loss_weight * resid_loss
        # Report the residual mean over its OWN valid element count — the main
        # token count is a different denominator and made the metric drift on
        # batches whose residual/main valid ratio differs from 1 (P1-10).
        metrics['codec_residual_loss'] = torch.stack(
            [resid_loss.detach() * resid_num, resid_num])
    elif residual_loss_enabled and not _residual_branch_ran:
        # See the long comment above ``_residual_branch_ran`` for why this is
        # needed: ``code_predictor`` was skipped this microbatch purely
        # because of missing/None data fields (residual_loss_enabled — the
        # config flag — is still True). Anchor a mathematically-zero gradient
        # into every one of its parameters so Megatron's per-microbatch
        # grad-ready bookkeeping sees it touched: d/dp (sum(p) * 0.0) == 0
        # exactly, so this changes neither the loss value nor any real
        # gradient, it only makes autograd's accumulate-grad hook fire.
        _anchor = codec_loss.new_zeros(())
        for _p in talker.code_predictor.parameters():
            if _p.requires_grad:
                _anchor = _anchor + _p.sum()
        codec_loss = codec_loss + _anchor * 0.0
    metrics['codec_loss'] = torch.stack([codec_loss.detach() * num_tokens.float(), num_tokens.float()])

    return codec_loss, num_tokens.to(torch.int), metrics


# ---------------------------------------------------------------------------
# Trainer patches
# ---------------------------------------------------------------------------
def _run_default(trainer, data, model):
    """Faithful replica of MegatronTrainer.forward_step body (post get_batch).

    Used for every non-codec batch so behaviour is identical to upstream.
    """
    loss_scale = data.pop('loss_scale', None)
    channels = data.pop('channel', None)
    labels = data.get('labels')
    if trainer.args.task_type == 'seq_cls':
        data.pop('labels', None)
    output_tensor = model(**data)
    packed_seq_params = data.get('packed_seq_params')
    if trainer.args.task_type == 'seq_cls':
        loss_func = partial(
            trainer.seq_cls_loss_func,
            labels=labels,
            packed_seq_params=packed_seq_params,
            attention_mask=data.get('attention_mask'))
    else:
        loss_func = partial(
            trainer.loss_func,
            labels=labels,
            loss_scale=loss_scale,
            channels=channels,
            packed_seq_params=packed_seq_params)
    return output_tensor, loss_func


# Canonical, rank-identical ordering of codec reporting metrics. Every DP rank
# must all_reduce exactly these keys in this order (missing ones use a zero
# placeholder) to keep the collective schedule symmetric across ranks. The
# branch's EOS-boundary observation metrics are appended here (they are always
# written in ``compute_codec_loss`` without a data-dependent gate) so they are
# included in the fixed schema without breaking the symmetric collective.
_CODEC_METRIC_KEYS = (
    'codec_loss',
    'codec_token_acc',
    'codec_residual_loss',
    'eos_boundary_top1_acc',
    'premature_eos_top1_rate',
)

# Dedicated process group for metrics all-reduce (avoids NCCL communicator
# conflict with Megatron's async gradient all-reduce when overlap_grad_reduce=True).
# Pre-created in install_megatron_talker_codec_loss() to ensure synchronous
# new_group() across all ranks (lazy creation inside _all_reduce_vec would
# deadlock if one rank enters loss_func while another is still in backward).
_METRICS_PG = None


def _init_metrics_pg():
    """Create the dedicated metrics process group (idempotent).

    ``torch.distributed.new_group`` is a WORLD-WIDE collective: every process
    must issue the same sequence of group creations. With TP/EP>1 the world
    contains several data-parallel groups while each rank only knows its own,
    so each rank first all-gathers every rank's DP membership and then creates
    ALL groups in one canonical (sorted) order, keeping the handle of its own
    group (P1-6, fixed 2026-08 — creating only one's own group passed
    different rank lists at the same collective index and deadlocked under
    model parallelism).
    """
    global _METRICS_PG
    if _METRICS_PG is not None:
        return
    import torch.distributed as dist
    from megatron.core import mpu
    if not (dist.is_available() and dist.is_initialized()):
        return
    my_ranks = tuple(sorted(dist.get_process_group_ranks(
        mpu.get_data_parallel_group(with_context_parallel=True))))
    if dist.get_world_size() == len(my_ranks):
        # Single DP group covering the whole world (TP=PP=CP=1): no exchange
        # needed, identical to the historical behaviour.
        _METRICS_PG = dist.new_group(ranks=list(my_ranks), backend='nccl')
        return
    memberships = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, list(my_ranks))
    _METRICS_PG = None
    for ranks in sorted({tuple(r) for r in memberships}):
        handle = dist.new_group(ranks=list(ranks), backend='nccl')
        if ranks == my_ranks:
            _METRICS_PG = handle
    if _METRICS_PG is None:  # pragma: no cover - defensive
        raise RuntimeError(
            '[qwen_omnispeak] metrics process group creation did not cover '
            'this rank; DP group membership is inconsistent across ranks.')


def _all_reduce_vec(vec: torch.Tensor) -> torch.Tensor:
    """Cross-rank all-reduce for logging metrics.

    When ``overlap_grad_reduce=True``, Megatron dispatches asynchronous gradient
    all-reduce on the DP group communicator during backward.  A *synchronous*
    ``all_reduce`` on the same communicator from ``loss_func`` (called in the
    forward phase of every micro-batch) can interleave with Megatron's async
    gradient all-reduce and cause a NCCL deadlock.

    Fix: use a **dedicated process group** for metrics all-reduce so it never
    shares a NCCL communicator with Megatron's gradient all-reduce.  The group
    is created lazily on first use and cached for the process lifetime.
    """
    out = vec.detach().clone()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return out
    if _METRICS_PG is None:
        _init_metrics_pg()
    torch.distributed.all_reduce(out, group=_METRICS_PG)
    return out


def _combined_loss_func(trainer, output_tensor, *, labels, text_ce_labels,
                        loss_scale, channels, packed_seq_params,
                        codec_loss, codec_tokens, codec_metrics, bridge_mode):
    args = trainer.args

    # Loss-combination contract (P1-10, fixed 2026-08): return a per-micro-batch
    # MEAN with a unit token count so the trainer averages micro-batches
    # EQUALLY — the weighting the HF/swift trainer applies. The previous
    # ``codec_loss * codec_tokens`` / ``codec_loss * text_tokens`` forms made
    # Megatron weight each micro-batch by its token count, a different global
    # gradient weighting than DeepSpeed whenever sequence lengths varied
    # across micro-batches. NOTE: this intentionally changes the reported
    # loss/grad scale; curves before/after the fix are not directly comparable.
    # NOTE: MFU token accumulation moved to _patched_forward_step (runs once per
    # micro-batch); do NOT re-accumulate here — _combined_loss_func is only the
    # loss-reporting callback and would undercount.
    if bridge_mode:
        if text_ce_labels is not None and bool((text_ce_labels != -100).any()):
            # Thinker unfrozen (UNFREEZE_THINKER=1): match the HF objective
            # text_mean + codec_mean instead of silently dropping the thinker
            # text CE (P1-5). ``text_ce_labels`` are real thinker text labels
            # fed to the GPTModel forward; ``output_tensor`` is its per-token
            # loss.
            losses = output_tensor.float()
            mask = (text_ce_labels != -100)
            tok = mask.sum().clamp(min=1.0)
            lm_loss = torch.sum(losses * mask) / tok
            if codec_loss is not None:
                lm_loss = lm_loss + codec_loss
        elif codec_loss is not None:
            lm_loss = codec_loss
        else:
            lm_loss = output_tensor.float().sum() * 0.0
        # 0-dim scalar: mcore's schedule does ``total_num_tokens (0-dim) +=
        # num_tokens`` in place, which cannot broadcast a [1] tensor into a
        # 0-dim accumulator (RuntimeError). Keep the P1-10 unit-token-count
        # semantics, just as a scalar.
        num = torch.tensor(1, device=output_tensor.device, dtype=torch.int)
    else:
        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        text_sum = torch.sum(losses * loss_mask)
        text_tokens = loss_mask.sum().clamp(min=1.0)
        lm_loss = text_sum / text_tokens
        if codec_loss is not None:
            lm_loss = lm_loss + codec_loss
        # 0-dim scalar, same as the bridge_mode branch above (mcore
        # ``total_num_tokens += num_tokens`` in-place broadcast constraint).
        num = torch.tensor(1, device=output_tensor.device, dtype=torch.int)

    loss_vec = torch.cat([lm_loss.detach().view(1), num.detach().float().view(1)])
    reporting_loss = _all_reduce_vec(loss_vec)
    metrics = {'loss': reporting_loss}
    # IMPORTANT: iterate over a FIXED, rank-identical key schema instead of the
    # data-dependent ``codec_metrics.keys()``. Whether a given metric (e.g.
    # ``codec_residual_loss`` / ``codec_loss`` / ``codec_token_acc``) is present
    # depends on the batch content (num_tokens>0, residual labels valid, ...),
    # which differs across DP ranks. Driving ``all_reduce`` off those keys makes
    # the collective count asymmetric across ranks -> one rank issues an extra
    # all_reduce with no peer -> NCCL/CUDA stream hangs (silent deadlock).
    # Fixed schema + zero placeholders guarantees every rank runs the exact same
    # number/order of all_reduce calls. Zero [0, 0] placeholders keep the
    # sum/count reporting semantics intact.
    cm = codec_metrics or {}
    _placeholder = torch.zeros(2, device=loss_vec.device, dtype=loss_vec.dtype)
    for k in _CODEC_METRIC_KEYS:
        v = cm.get(k)
        metrics[k] = _all_reduce_vec(v if v is not None else _placeholder)
    return (lm_loss, num.detach().clone().to(torch.int), metrics)


def _patched_forward_step(trainer, data_iterator, model):
    vp_stage = model.module.module.vp_stage
    data = trainer.get_batch(data_iterator, vp_stage)

    if not (codec_loss_enabled() and _has_codec(data) and _has_talker(model)):
        return _run_default(trainer, data, model)

    _check_parallel_guards(trainer.args)
    mm = _unwrap_mm(model)
    _install_hidden_hook(mm)

    codec = {k: data.pop(k) for k in CODEC_KEYS if k in data}
    has_codec_supervision = data.pop('has_codec_supervision', None)
    # Pop the streaming read-write schedule (simultaneous-policy template only)
    # BEFORE ``model(**data)`` so it never reaches the thinker GPTModel forward
    # as an unexpected kwarg; pass it to the codec attention mask builder below.
    codec_to_sem_visible = data.pop(STREAMING_KEY, None)
    data.pop('run_thinker', None)
    # Upstream Qwen3-TTS input protocol (only present when the model config has
    # ``use_upstream_talker_input=True``). These fields are produced by the
    # template (custom/template.py) and are consumed ONLY by the upstream talker
    # input path in ``compute_codec_loss`` — pop them BEFORE ``model(**data)`` so
    # they never reach the mcore GPTModel forward as unexpected kwargs.
    use_upstream_input = bool(getattr(
        mm.visual.hf_config, 'use_upstream_talker_input', False))
    upstream_text_labels = data.pop('thinker_text_labels', None)
    upstream_codec_bos_index = data.pop('codec_bos_index', None)
    upstream_target_text_ids = data.pop('target_text_ids', None)
    upstream_full_add_prompt_len = data.pop('full_add_prompt_len', None)
    # Speaker-prompt / ICL fields are consumed ONLY by ``compute_codec_loss``;
    # pop them before ``model(**data)`` so they never reach the mcore GPTModel
    # forward as unexpected kwargs (same treatment as the codec fields above).
    speaker_fields = {k: data.pop(k, None) for k in SPEAKER_KEYS}
    icl_source_codec = data.pop('source_codec_codes', None)
    icl_source_text_ids = data.pop('source_text_ids', None)
    icl_source_codec_mask = data.pop('source_codec_mask', None)
    icl_source_text_mask = data.pop('source_text_mask', None)
    bridge_mode = has_codec_supervision is None  # bridge-alignment template

    loss_scale = data.pop('loss_scale', None)
    channels = data.pop('channel', None)
    labels = data.get('labels')
    input_ids = data.get('input_ids')

    pad_id = _pad_token_id(trainer)
    thinker_attention_mask = None
    if input_ids is not None and pad_id is not None:
        thinker_attention_mask = (input_ids != pad_id)

    # MFU accounting: accumulate the FULL per-micro-batch token count here
    # (forward_step runs once per micro-batch), NOT in _combined_loss_func which
    # is only invoked on the loss-reporting micro-batch and undercounts badly.
    # token components = non-pad thinker input_ids + codec_input_ids (main codec)
    # + codec_residual_labels (residual codec, [B,T,groups] supervised positions).
    # This matches the DeepSpeed token definition (text + main + residual codec),
    # so the two backends' MFU are comparable.
    try:
        from mfu_logging import accumulate_tokens
        _mfu_tok = 0
        _tok_input = 0
        _tok_ci = 0
        _tok_res = 0
        if input_ids is not None:
            _tok_input = int((input_ids != (pad_id if pad_id is not None else -1)).sum())
            _mfu_tok += _tok_input
        _ci = (codec or {}).get('codec_input_ids')
        if _ci is not None:
            # Use non-pad codec positions (codec_attention_mask) to match the
            # text/input_ids non-pad accounting; .numel() would over-count pads.
            _ci_mask = (codec or {}).get('codec_attention_mask')
            if _ci_mask is not None:
                _tok_ci = int((_ci_mask != 0).sum())
            else:
                _tok_ci = int(_ci.numel())
            _mfu_tok += _tok_ci
        _res = (codec or {}).get('codec_residual_labels')
        if _res is not None:
            _tok_res = int((_res != -100).sum())
            _mfu_tok += _tok_res
        # DIAG: print per-component token counts (input/main-codec/residual) once
        # per microbatch to validate the MFU token accounting.
        if os.environ.get('MFU_DIAG_SHAPES') == '1':
            _s = f'[mfu-diag] B={_ci.shape[0] if _ci is not None else 0}'
            _s += f' text={_tok_input}'
            _s += f' main_codec={_tok_ci}'
            _s += f' residual={_tok_res}'
            _s += f' shape_ci={tuple(_ci.shape) if _ci is not None else None}'
            _s += f' shape_res={tuple(_res.shape) if _res is not None else None}'
            _s += f' total={_mfu_tok}'
            print(_s, flush=True)
        if _mfu_tok > 0:
            accumulate_tokens(_mfu_tok)
    except Exception:
        pass

    # The bridge template overwrites `labels` with codec-accuracy labels whose
    # length equals `codec_len` (for token-acc logging). In Megatron the thinker
    # (text GPTModel) and the talker are SEPARATE submodules: the thinker is fed
    # the *text* `input_ids` (length `text_len`), while the codec labels belong
    # to the talker. Passing the mismatched codec `labels` into the thinker
    # forward makes mcore's cross-entropy do `logits[arange(text_len), labels]`
    # (shapes [text_len], [codec_len]) -> "indexing tensors could not be
    # broadcast together". With the thinker FROZEN we therefore feed it a
    # length-correct, fully masked labels tensor (no text supervision) and
    # optimise only the codec loss. With UNFREEZE_THINKER=1 (P1-5, fixed
    # 2026-08) we instead feed the REAL thinker text labels so the GPTModel's
    # per-token CE is meaningful and `_combined_loss_func` can add the text
    # CE the DeepSpeed objective includes. (In the DeepSpeed path the whole
    # model is fed `codec_input_ids` as `input_ids`, so the lengths already
    # match and the frozen-case override is a harmless no-op there.)
    text_ce_labels = None
    if input_ids is not None and 'labels' in data:
        if (upstream_text_labels is not None
                and tuple(upstream_text_labels.shape) == tuple(input_ids.shape)
                and any(p.requires_grad for p in mm.language_model.parameters())):
            text_ce_labels = upstream_text_labels.to(input_ids.device)
            data['labels'] = text_ce_labels
        else:
            data['labels'] = torch.full_like(input_ids, -100)

    mm._omnispeak_hidden = None
    output_tensor = model(**data)
    captured = getattr(mm, '_omnispeak_hidden', None)
    mm._omnispeak_hidden = None
    if captured is None:
        raise RuntimeError(
            '[qwen_omnispeak] Failed to capture thinker hidden states for codec loss. '
            'Check accept_hidden_layer / pipeline configuration.')

    codec_loss, codec_tokens, codec_metrics = compute_codec_loss(
        mm,
        hf_config=mm.visual.hf_config,
        input_ids=input_ids,
        captured_hidden=captured,
        codec=codec,
        thinker_attention_mask=thinker_attention_mask,
        has_codec_supervision=has_codec_supervision,
        codec_to_sem_visible=codec_to_sem_visible,
        use_upstream_input=use_upstream_input,
        text_labels=upstream_text_labels,
        codec_bos_index=upstream_codec_bos_index,
        target_text_ids=upstream_target_text_ids,
        full_add_prompt_len=upstream_full_add_prompt_len,
        speaker_embedding=speaker_fields['speaker_embedding'],
        reference_audio_wav=speaker_fields['reference_audio_wav'],
        reference_audio_sr=speaker_fields['reference_audio_sr'],
        speaker_slot_index=speaker_fields['speaker_slot_index'],
        icl_source_codec=icl_source_codec,
        icl_source_text_ids=icl_source_text_ids,
        icl_source_codec_mask=icl_source_codec_mask,
        icl_source_text_mask=icl_source_text_mask,
    )

    packed_seq_params = data.get('packed_seq_params')
    loss_func = partial(
        _combined_loss_func,
        trainer,
        labels=labels,
        text_ce_labels=text_ce_labels,
        loss_scale=loss_scale,
        channels=channels,
        packed_seq_params=packed_seq_params,
        codec_loss=codec_loss,
        codec_tokens=codec_tokens,
        codec_metrics=codec_metrics,
        bridge_mode=bridge_mode,
    )
    return output_tensor, loss_func


def install_megatron_talker_codec_loss() -> None:
    """Monkey-patch MegatronTrainer.forward_step to add talker codec loss.

    Safe to call multiple times. Raises ImportError if the Megatron trainer is
    unavailable (caught by the caller), in which case nothing is patched.
    """
    global _PATCHED
    if _PATCHED:
        return
    from swift.megatron.trainers.trainer import MegatronTrainer

    if getattr(MegatronTrainer, '_omnispeak_orig_forward_step', None) is None:
        MegatronTrainer._omnispeak_orig_forward_step = MegatronTrainer.forward_step

    def forward_step(self, data_iterator, model):
        return _patched_forward_step(self, data_iterator, model)

    MegatronTrainer.forward_step = forward_step
    _PATCHED = True

    # Pre-create the metrics process group so that ``torch.distributed.new_group``
    # is called synchronously on all ranks at plugin-install time, NOT lazily
    # inside ``_all_reduce_vec`` (which would deadlock if one rank enters the
    # loss function while another is still in backward).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _init_metrics_pg()

    logger.info('[qwen_omnispeak] Installed Megatron talker codec-loss forward_step patch.')


__all__ = [
    'CODEC_KEYS',
    'SPEAKER_KEYS',
    'codec_loss_enabled',
    'compute_codec_loss',
    'install_megatron_talker_codec_loss',
]
