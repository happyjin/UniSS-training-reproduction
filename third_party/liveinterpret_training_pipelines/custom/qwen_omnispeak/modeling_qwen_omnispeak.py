# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen-OmniSpeak modeling (single-file, ``trust_remote_code``-friendly).

This file is intentionally self-contained so that ``trust_remote_code=True``
exports can reference its classes via the strict two-segment
``module.ClassName`` shorthand expected by ``transformers``.

The model wires together:

    Qwen Omni Thinker  ──>  DualRoute Bridge  ──>  Qwen3-TTS Talker (sliced)
    (Qwen3 or Qwen2.5)                            └─> code predictor

Officially named variants (A/B/C/D/E/F/G/H); see
``QwenOmniSpeakConfig.SUPPORTED_VARIANTS``.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
except Exception:  # pragma: no cover - older transformers without Qwen2.5-Omni.
    Qwen2_5OmniThinkerForConditionalGeneration = None

# Configuration is in a sibling file. We support BOTH:
#   * direct execution / absolute import (when this file is loaded as a
#     top-level module), and
#   * relative import (when transformers' trust_remote_code loader turns the
#     export directory into a dynamic package).
# transformers' ``check_imports`` only reports *absolute* imports as missing
# external packages, so we keep the absolute-import path inside a ``try``
# block and use ``from .configuration_qwen_omnispeak import ...`` as the
# preferred relative path. The ``ast``-based importer used by
# ``get_imports`` *skips* ``Try`` blocks (see transformers source), which
# means the absolute import below is intentionally invisible to its
# dependency check.
try:
    from .configuration_qwen_omnispeak import (  # noqa: F401
        DualRouteBridgeConfig,
        QWEN25_THINKER_FAMILY,
        QwenOmniSpeakConfig,
        Qwen3TtsTalkerConfig,
    )
except ImportError:  # pragma: no cover
    from configuration_qwen_omnispeak import (  # noqa: F401
        DualRouteBridgeConfig,
        QWEN25_THINKER_FAMILY,
        QwenOmniSpeakConfig,
        Qwen3TtsTalkerConfig,
    )

# Frozen Qwen3-TTS ECAPA speaker embedder used for voice-clone x-vector
# injection. It lives outside the composite (module-level singleton) so it is
# invisible to save_pretrained / DeepSpeed / the optimizer. Import is best-effort
# and kept inside a ``try`` (skipped by transformers' ``get_imports`` check); the
# multiple fallbacks cover both the ``custom`` package layout and the flat
# trust_remote_code export directory. When exporting, copy ``speaker_encoder.py``
# next to this file so the injection path stays importable.
try:
    from speaker_encoder import get_speaker_embedder  # export dir (flat) / sys.path
except Exception:  # pragma: no cover
    try:
        from .speaker_encoder import get_speaker_embedder  # same package
    except Exception:
        # NOTE: previously we had ``from ..speaker_encoder import ...`` here as a
        # last-resort parent-package fallback. transformers'
        # ``dynamic_module_utils.get_relative_imports`` scans this file with the
        # regex ``r"^\s*from\s+\.(\S+)\s+import"``, which greedily captures the
        # second dot and turns ``..speaker_encoder`` into ``.speaker_encoder`` —
        # so save_pretrained / trust_remote_code loading then tries to open
        # ``<pkg_dir>/.speaker_encoder.py`` and crashes with FileNotFoundError.
        # Use importlib to avoid emitting a dot-prefixed relative import in
        # source that the regex can misparse.
        try:
            import importlib as _importlib
            _mod = _importlib.import_module('custom.speaker_encoder')
            get_speaker_embedder = _mod.get_speaker_embedder
        except Exception:
            get_speaker_embedder = None  # type: ignore


# ---------------------------------------------------------------------------
# Optional Triton-fused ops (liger-kernel) for project-local RMSNorm/SwiGLU/CE.
#
# swift's ``--use_liger_kernel`` dispatches on a single global ``model_type``
# string via ``apply_liger_kernel_to_qwen3()`` etc., which monkey-patches the
# *official* ``transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`` class.
# Our talker/adapter (``_RMSNorm``, ``_MLPProj`` below) are project-local
# ``nn.Module`` subclasses — never imported from that HF module — so that
# monkey-patch never reaches them; only the (frozen) thinker benefits.
#
# ``QWEN_OMNISPEAK_LIGER_OPS=1`` bypasses the model-type dispatch entirely and
# calls liger-kernel's underlying ``torch.autograd.Function`` ops directly
# inside these classes' ``forward``, so the *trainable* talker/adapter get the
# fused Triton kernels too. Lazy + cached + best-effort: any ImportError
# (liger-kernel not installed in this image) silently falls back to the pure
# PyTorch implementation.
#
# 2026-09-12 additions: this switch now ALSO covers the codec/text loss
# CrossEntropy (previously the only piece of the "RMSNorm/RoPE/SwiGLU/
# CrossEntropy" liger integration that was actually never wired up — see
# ``_cross_entropy`` below, dispatched via ``liger_kernel.transformers.
# functional.liger_cross_entropy``, a drop-in replacement for
# ``torch.nn.functional.cross_entropy`` with the same signature/semantics).
# ---------------------------------------------------------------------------
_LIGER_OPS_ENV = 'QWEN_OMNISPEAK_LIGER_OPS'
_liger_rmsnorm_fn_cache: Any = None
_liger_swiglu_fn_cache: Any = None
_liger_ops_checked = False
_liger_cross_entropy_fn_cache: Any = None
_liger_ce_checked = False


def _get_liger_rmsnorm_fn():
    global _liger_rmsnorm_fn_cache, _liger_ops_checked
    if os.environ.get(_LIGER_OPS_ENV) != '1':
        return None
    if not _liger_ops_checked:
        try:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
            global _liger_swiglu_fn_cache
            _liger_rmsnorm_fn_cache = LigerRMSNormFunction
            _liger_swiglu_fn_cache = LigerSiLUMulFunction
        except Exception:
            _liger_rmsnorm_fn_cache = None
            _liger_swiglu_fn_cache = None
        _liger_ops_checked = True
    return _liger_rmsnorm_fn_cache


def _get_liger_swiglu_fn():
    _get_liger_rmsnorm_fn()  # ensures the one-time import attempt has run
    return _liger_swiglu_fn_cache


def _get_liger_cross_entropy_fn():
    """Lazy-import ``liger_cross_entropy`` (gated by the same
    ``QWEN_OMNISPEAK_LIGER_OPS`` switch as RMSNorm/SwiGLU above).

    Separate cache/check flag from the RMSNorm/SwiGLU ops above: liger-kernel
    versions predating PR #435 (upstream, merged) had a bug where
    ``reduction="none"`` silently collapsed to a scalar instead of returning
    per-token losses (see ``_weighted_codec_cross_entropy`` below, which
    needs true per-token values for its EOS-weighting logic) — keeping this
    import attempt independent means a future liger-kernel downgrade/issue
    with just the CE op doesn't also disable the (already validated)
    RMSNorm/SwiGLU fusion, and vice versa.
    """
    global _liger_cross_entropy_fn_cache, _liger_ce_checked
    if os.environ.get(_LIGER_OPS_ENV) != '1':
        return None
    if not _liger_ce_checked:
        try:
            from liger_kernel.transformers.functional import liger_cross_entropy
            _liger_cross_entropy_fn_cache = liger_cross_entropy
        except Exception:
            _liger_cross_entropy_fn_cache = None
        _liger_ce_checked = True
    return _liger_cross_entropy_fn_cache


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *,
                    ignore_index: int = -100, reduction: str = 'mean') -> torch.Tensor:
    """Drop-in replacement for ``F.cross_entropy`` used at every codec/text
    loss call site. Dispatches to liger-kernel's fused Triton CE when
    ``QWEN_OMNISPEAK_LIGER_OPS=1`` and the package is importable; falls back
    to the pure PyTorch op otherwise (identical numerics, just no fusion) —
    same lazy/cached/best-effort pattern as ``_get_liger_rmsnorm_fn`` etc.

    Defensive shape check for ``reduction='none'``: liger-kernel versions
    predating upstream PR #435 had a bug where ``reduction='none'`` silently
    collapsed the result to a scalar instead of returning one loss value per
    (non-ignored) target position — a silent wrong-answer, not an exception.
    ``_weighted_codec_cross_entropy`` below relies on true per-token values
    for its EOS-weighting logic, so if the returned shape doesn't match the
    expected per-token count we discard it and fall back to ``F.cross_entropy``
    rather than risk silently corrupting the loss.
    """
    liger_ce = _get_liger_cross_entropy_fn()
    if liger_ce is not None:
        try:
            out = liger_ce(logits, labels, ignore_index=ignore_index, reduction=reduction)
            if reduction == 'none' and out.shape != labels.reshape(-1).shape:
                raise RuntimeError(
                    f'liger_cross_entropy(reduction="none") returned shape '
                    f'{tuple(out.shape)}, expected {tuple(labels.reshape(-1).shape)} '
                    f'(known upstream bug pre PR #435) — discarding, falling back')
            return out
        except Exception as e:  # pragma: no cover - best-effort, never break training
            try:
                from swift.utils import get_logger
                get_logger().warning(
                    f'liger_cross_entropy call failed, falling back to '
                    f'F.cross_entropy for this call: {e}')
            except Exception:
                pass
    return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)


def _apply_liger_to_talker_decoder(talker_inner_model) -> None:
    """Retroactively patch an ALREADY-INSTANTIATED talker decoder stack with
    liger-kernel's fused Triton RMSNorm/SwiGLU/RoPE.

    ``Qwen3TtsTalkerModel.__init__`` builds a temporary ``transformers``
    ``Qwen3Model`` and steals its ``layers``/``norm``/``rotary_emb`` (see that
    class's docstring) — so talker's ~28-layer decoder stack (the actual bulk
    of talker's forward/backward compute) is made of *genuine*
    ``transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer`` instances.

    swift's own ``--use_liger_kernel`` calls ``apply_liger_kernel_to_qwen3()``
    WITHOUT ``model=``, which only rebinds the ``modeling_qwen3`` MODULE-level
    class name (``modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm``) — this never
    reaches an already-built model's existing layer *instances* (verified: a
    running job with ``USE_LIGER_KERNEL=true`` still had the pristine,
    unpatched ``Qwen3RMSNorm.forward`` at runtime). Passing ``model=`` here
    makes liger-kernel's ``_patch_rms_norm_module``/``_patch_swiglu_module``
    directly rebind ``forward`` on each real ``decoder_layer.input_layernorm``
    / ``post_attention_layernorm`` / ``mlp`` in place — the fix that actually
    reaches the heavy compute (as opposed to ``_RMSNorm``/``_MLPProj`` above,
    which only cover the small bridge/adapter, not talker's main stack).
    """
    if os.environ.get(_LIGER_OPS_ENV) != '1':
        return
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(model=talker_inner_model)
    except Exception as e:  # pragma: no cover - best-effort, never break training
        try:
            from swift.utils import get_logger
            get_logger().warning(f'Failed to apply liger kernel to talker decoder: {e}')
        except Exception:
            pass


def _use_speaker_prompt(config) -> bool:
    """Whether to inject a reference-audio speaker x-vector (voice clone).

    Priority: env ``QWEN_OMNISPEAK_USE_SPEAKER_PROMPT`` > config flag. Off by
    default so existing runs are unaffected.
    """
    env = os.environ.get('QWEN_OMNISPEAK_USE_SPEAKER_PROMPT')
    if env is not None:
        return env.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(getattr(config, 'use_speaker_prompt', False))


def apply_speaker_prompt_to_codec_emb(
    codec_emb: torch.Tensor,
    *,
    config,
    speaker_slot_index: Optional[torch.Tensor] = None,
    speaker_embedding: Optional[List[Any]] = None,
    reference_audio_wav: Optional[List[Any]] = None,
    reference_audio_sr: Optional[List[int]] = None,
) -> torch.Tensor:
    """Replace the reserved speaker-slot embedding with an ECAPA x-vector.

    Shared by DeepSpeed ``forward`` and Megatron ``mcore_talker_loss`` so both
    backends inject timbre at the same place: immediately after
    ``embed_codec(codec_input_ids)``. Prefers offline ``speaker_embedding``;
    otherwise extracts online from ``reference_audio_wav``.
    """
    if not (_use_speaker_prompt(config) and speaker_slot_index is not None
            and (speaker_embedding is not None or reference_audio_wav is not None)):
        return codec_emb
    if speaker_embedding is None and get_speaker_embedder is None:
        raise ImportError(
            'use_speaker_prompt is on but speaker_encoder.py is not '
            'importable; copy it next to modeling_qwen_omnispeak.py in '
            'the export directory.')
    codec_emb = codec_emb.clone()
    n = codec_emb.size(0)
    if speaker_embedding is not None:
        n = min(n, len(speaker_embedding))
    if reference_audio_wav is not None:
        n = min(n, len(reference_audio_wav))
    embedder = None
    for b in range(n):
        slot_b = speaker_slot_index[b]
        slot = int(slot_b.item() if torch.is_tensor(slot_b) else slot_b)
        xvec = None
        if speaker_embedding is not None and b < len(speaker_embedding):
            emb_b = speaker_embedding[b]
            if emb_b is not None:
                xvec = torch.as_tensor(
                    emb_b, device=codec_emb.device, dtype=codec_emb.dtype
                ).reshape(-1)
        if xvec is None and reference_audio_wav is not None:
            wav_b = reference_audio_wav[b]
            if wav_b is None:
                continue
            if embedder is None:
                embedder = get_speaker_embedder(
                    enc_dim=int(getattr(config, 'speaker_embedding_dim', 2048)))
            sr_b = (int(reference_audio_sr[b])
                    if reference_audio_sr is not None else 24000)
            xvec = embedder.extract(
                wav_b, sr_b, device=codec_emb.device, dtype=codec_emb.dtype)
        if xvec is None:
            continue
        codec_emb[b, slot] = xvec
    return codec_emb


def _full_add_layout(config) -> bool:
    """Whether to use the Qwen2.5-Omni "full-add" talker input layout.

    Priority: env ``QWEN_OMNISPEAK_FULL_ADD_LAYOUT`` > config flag. Only
    meaningful together with ``use_upstream_talker_input``. When on, the text
    channel carries the full ``[prompt , text_bos , reply , pad]`` layout (added
    position-wise onto the ``[codec_mask × T_prompt , codec_pad , codec_bos ,
    frames]`` codec channel), exactly like ``Qwen2_5OmniForConditionalGeneration``.
    """
    env = os.environ.get('QWEN_OMNISPEAK_FULL_ADD_LAYOUT')
    if env is not None:
        return env.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(getattr(config, 'full_add_layout', False))


def apply_speaker_prompt(codec_emb, config, reference_audio_wav, reference_audio_sr,
                         speaker_slot_index):
    """Replace the reserved speaker-slot embedding with the frozen Qwen3-TTS
    ECAPA x-vector of the reference audio (voice clone / voice preserving).

    Direct substitution (no projection MLP) — mirrors upstream Qwen3-TTS,
    which concatenates ``speaker_embed`` verbatim at this control-prefix
    position. resample / mel / encoder all run here (the dataloader only
    loaded the raw waveform). The x-vector is detached (frozen encoder), so
    gradients flow to the other codec positions as usual.

    Shared by the DeepSpeed forward and the Megatron codec-loss path so both
    backends inject bit-identical speaker conditioning (P0-1, extracted
    2026-08; previously only the DeepSpeed forward performed the injection).
    """
    if get_speaker_embedder is None:
        raise ImportError(
            'use_speaker_prompt is on but speaker_encoder.py is not '
            'importable; copy it next to modeling_qwen_omnispeak.py in '
            'the export directory.')
    embedder = get_speaker_embedder(
        enc_dim=int(getattr(config, 'speaker_embedding_dim', 2048)))
    codec_emb = codec_emb.clone()
    n = min(codec_emb.size(0), len(reference_audio_wav))
    for b in range(n):
        wav_b = reference_audio_wav[b]
        if wav_b is None:
            continue
        sr_b = (int(reference_audio_sr[b])
                if reference_audio_sr is not None else 24000)
        xvec = embedder.extract(
            wav_b, sr_b, device=codec_emb.device, dtype=codec_emb.dtype)
        slot_b = speaker_slot_index[b]
        slot = int(slot_b.item() if torch.is_tensor(slot_b) else slot_b)
        codec_emb[b, slot] = xvec
    return codec_emb


def build_icl_prefix(config, talker,
                     source_codec_codes, source_text_ids,
                     source_codec_mask, source_text_mask,
                     *, codec_dtype, codec_device):
    """Build the in-context-learning (ICL) prefix, mirroring Qwen3-TTS
    ``generate_icl_prompt``: a leading segment of
    ``ref_text_embed + ref_codec_embed`` (element-wise, tts_pad-aligned)
    that lets the talker imitate the reference speaker's timbre/prosody.

    Returns ``(icl_embed [B, T_icl, H_talker], icl_mask [B, T_icl] bool)``
    prepended to the talker input. Uses the same code-group embedding sum +
    ``text_projection`` pipeline as the upstream path so the prefix lives in
    the same embedding space as ``codec_emb + text_channel``.

    Module-level (extracted from the model method, 2026-08) so the Megatron
    codec-loss path reuses the identical construction instead of a diverging
    copy (P1-7).
    """
    bsz = source_codec_codes.size(0)
    tp_dtype = next(talker.text_projection.parameters()).dtype

    # ── reference codec embedding: sum over 16 code groups (main + resid) ──
    ref_codes = source_codec_codes.to(codec_device, dtype=torch.long)   # [B, T_ref, 16]
    t_ref = ref_codes.size(1)
    code_sum = talker.get_input_embeddings()(ref_codes[..., 0])          # [B, T_ref, H]
    for g in range(1, talker.config.num_code_groups):
        code_sum = code_sum + talker.code_predictor.get_input_embeddings()[g - 1](
            ref_codes[..., g])
    ref_codec_embed = code_sum.to(codec_dtype)                          # [B, T_ref, H]

    # ── reference text embedding → text_projection (same as text channel) ──
    pad_id = int(config.talker_config.tts_pad_token_id)
    H = ref_codec_embed.size(-1)
    if source_text_ids is not None:
        txt_ids = source_text_ids.to(codec_device, dtype=torch.long)     # [B, T_txt]
        txt_emb = talker.text_projection(
            talker.model.text_embedding(txt_ids).to(tp_dtype)).to(codec_dtype)  # [B, T_txt, H]
        t_txt = txt_emb.size(1)
    else:
        txt_emb = None
        t_txt = 0

    # tts_pad embedding for aligning text/codec length
    pad_emb = talker.text_projection(
        talker.model.text_embedding(
            torch.tensor([pad_id], device=codec_device, dtype=torch.long)).to(tp_dtype)
    ).to(codec_dtype).squeeze(0)                                        # [H]

    # text channel over the ref-codec span: first t_txt positions carry ref
    # text, the rest are tts_pad (Qwen3-TTS non-streaming ICL alignment).
    icl_text = pad_emb.expand(bsz, t_ref, H).clone()
    if txt_emb is not None and t_txt > 0:
        n = min(t_txt, t_ref)
        icl_text[:, :n] = txt_emb[:, :n]
    icl_embed = ref_codec_embed + icl_text                              # [B, T_ref, H]

    if source_codec_mask is not None:
        icl_mask = source_codec_mask.to(codec_device, dtype=torch.bool)
    else:
        icl_mask = torch.ones(bsz, t_ref, dtype=torch.bool, device=codec_device)
    return icl_embed, icl_mask


# =============================================================================
#  Section 1: tiny shared modules (RMSNorm, SwiGLU MLP)
# =============================================================================


class _RMSNorm(nn.Module):
    """Plain RMSNorm (kept local so the export bundle has no extra deps).

    When ``QWEN_OMNISPEAK_LIGER_OPS=1`` and ``liger-kernel`` is importable,
    dispatches to ``LigerRMSNormFunction`` (fused Triton fwd+bwd) directly —
    bypassing swift's ``apply_liger_kernel_to_qwen3()`` model-type dispatch,
    which only monkey-patches ``transformers``' official Qwen3RMSNorm class
    and therefore never reaches this project-local class.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _fn = _get_liger_rmsnorm_fn()
        if _fn is not None:
            return _fn.apply(x, self.weight, self.variance_epsilon, 0.0, 'llama', False)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(in_dtype)


class _MLPProj(nn.Module):
    """A 2-layer SwiGLU-style projection block used by both bridge variants.

    When ``QWEN_OMNISPEAK_LIGER_OPS=1`` and ``liger-kernel`` is importable,
    fuses ``silu(gate) * up`` via ``LigerSiLUMulFunction`` (see ``_RMSNorm``
    docstring for why swift's model-type dispatch never reaches this class).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        _fn = _get_liger_swiglu_fn()
        if _fn is not None:
            return self.down_proj(_fn.apply(gate, up))
        return self.down_proj(F.silu(gate) * up)


# ---------------------------------------------------------------------------
#  Upstream Qwen3-TTS Talker components (feature/talker-upstream-omniinput)
#
#  Only the input-side plumbing is different from upstream; the attention /
#  decoder / RoPE are bit-perfect-equivalent to upstream's M-RoPE with our
#  1-D position_ids (see docs/ideas/talker_vs_upstream_diff.md §III.B). We therefore
#  only need the two lightweight modules below.
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerResizeMLP(nn.Module):
    """Faithful copy of upstream ``Qwen3TTSTalkerResizeMLP``.

    Two ``Linear`` layers with bias plus an activation, matching upstream's
    ``text_projection`` weight layout so ``linear_fc1.{weight,bias}`` and
    ``linear_fc2.{weight,bias}`` load key-by-key from Qwen3-TTS-12Hz-1.7B-Base
    (see M1b bit-perfect check in
    ``docs/ideas/talker_upstream_experiment_plan_v2.md`` §6.M1).
    """

    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int,
                 hidden_act: str = 'silu', bias: bool = True):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_dim, intermediate_dim, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_dim, output_dim, bias=bias)
        if hidden_act == 'silu':
            self.act_fn = nn.SiLU()
        elif hidden_act == 'gelu':
            self.act_fn = nn.GELU()
        else:
            # Fall back to torch's activation lookup for less common choices.
            self.act_fn = getattr(F, hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3TTSThinkerAdapter(nn.Module):
    """Adapter mapping ``thinker`` output signal to ``talker.text_embedding`` space.

    Structure: ``Linear -> GELU -> Dropout -> Linear -> LayerNorm``. Weights
    are initialised with the Qwen family default ``trunc_normal_(std=0.02)``,
    biases zero, LayerNorm gain 1 / bias 0.
    """

    def __init__(self, thinker_hidden_size: int, text_hidden_size: int,
                 intermediate_size: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        h_mid = int(intermediate_size) if intermediate_size else (2 * text_hidden_size)
        self.fc1 = nn.Linear(thinker_hidden_size, h_mid, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(h_mid, text_hidden_size, bias=True)
        self.norm = nn.LayerNorm(text_hidden_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.dropout(self.act(self.fc1(x)))))


# =============================================================================
#  Section 2: DualRoute Bridge implementations
# =============================================================================


class DualRouteMLPBridge(nn.Module):
    """V1 DualRoute Bridge: independent text/hidden MLP heads + entry adapter."""

    def __init__(self, config: DualRouteBridgeConfig):
        super().__init__()
        self.config = config
        h_in = config.thinker_hidden_size
        h_out = config.talker_hidden_size
        h_mid = config.intermediate_size
        self.text_norm = _RMSNorm(h_in, eps=config.rms_norm_eps)
        self.hidden_norm = _RMSNorm(h_in, eps=config.rms_norm_eps)
        self.text_projection = _MLPProj(h_in, h_mid, h_out)
        self.hidden_projection = _MLPProj(h_in, h_mid, h_out)
        self.entry_norm = _RMSNorm(h_out, eps=config.rms_norm_eps)
        self.entry_adapter = _MLPProj(h_out, h_mid, h_out)

    def forward(
        self,
        thinker_text_embeds: Optional[torch.Tensor] = None,
        thinker_hidden_states: Optional[torch.Tensor] = None,
        semantic_route_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Construct the *semantic block* of length ``T_sem`` with hidden=H_out.

        - If ``semantic_route_mask`` is provided (token-wise selection between
          the two streams, shape ``[B, T]``), the two streams must already be
          aligned to the same length and we mix them token-wise.
        - Otherwise we concatenate ``[text ; hidden]`` along the time dim.
        """
        if thinker_text_embeds is None and thinker_hidden_states is None:
            raise ValueError('At least one of text_embeds / hidden_states must be provided.')

        if semantic_route_mask is not None:
            assert thinker_text_embeds is not None and thinker_hidden_states is not None, (
                'semantic_route_mask requires both text_embeds and hidden_states.')
            assert thinker_text_embeds.shape == thinker_hidden_states.shape, (
                'text_embeds and hidden_states must have the same shape when route_mask is given.')
            text_proj = self.text_projection(self.text_norm(thinker_text_embeds))
            hidden_proj = self.hidden_projection(self.hidden_norm(thinker_hidden_states))
            mask = semantic_route_mask.to(text_proj.dtype).unsqueeze(-1)
            sem = text_proj * (1.0 - mask) + hidden_proj * mask
        else:
            parts = []
            if thinker_text_embeds is not None:
                parts.append(self.text_projection(self.text_norm(thinker_text_embeds)))
            if thinker_hidden_states is not None:
                parts.append(self.hidden_projection(self.hidden_norm(thinker_hidden_states)))
            sem = torch.cat(parts, dim=1)
        sem = sem + self.entry_adapter(self.entry_norm(sem))
        return sem


class _CrossAttnBlock(nn.Module):
    """A single decoder-style block (self-attn -> cross-attn -> MLP) used by Q-Former."""

    def __init__(self, dim: int, num_heads: int, intermediate: int, dropout: float, eps: float):
        super().__init__()
        self.norm_q = _RMSNorm(dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_kv = _RMSNorm(dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = _RMSNorm(dim, eps=eps)
        self.mlp = _MLPProj(dim, intermediate, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm_q(q)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + sa
        h = self.norm_kv(q)
        ca, _ = self.cross_attn(h, kv, kv, key_padding_mask=kv_pad_mask, need_weights=False)
        q = q + ca
        q = q + self.mlp(self.norm_ff(q))
        return q


class DualRouteQFormerBridge(nn.Module):
    """V2 DualRoute Bridge.

    Same projection heads as V1 plus a small Q-Former that compresses
    variable-length thinker hidden states into ``num_query_tokens`` learnable
    semantic queries.
    """

    def __init__(self, config: DualRouteBridgeConfig):
        super().__init__()
        self.config = config
        h_in = config.thinker_hidden_size
        h_out = config.talker_hidden_size
        h_mid = config.intermediate_size

        self.text_norm = _RMSNorm(h_in, eps=config.rms_norm_eps)
        self.hidden_norm = _RMSNorm(h_in, eps=config.rms_norm_eps)
        self.text_projection = _MLPProj(h_in, h_mid, h_out)
        self.hidden_projection = _MLPProj(h_in, h_mid, h_out)

        self.num_query_tokens = config.num_query_tokens
        self.queries = nn.Parameter(torch.randn(1, config.num_query_tokens, h_out) * 0.02)
        self.qformer_blocks = nn.ModuleList([
            _CrossAttnBlock(h_out, config.qformer_num_heads, h_mid, config.qformer_dropout,
                            config.rms_norm_eps) for _ in range(config.qformer_num_layers)
        ])
        self.entry_norm = _RMSNorm(h_out, eps=config.rms_norm_eps)
        self.entry_adapter = _MLPProj(h_out, h_mid, h_out)

    def forward(
        self,
        thinker_text_embeds: Optional[torch.Tensor] = None,
        thinker_hidden_states: Optional[torch.Tensor] = None,
        semantic_route_mask: Optional[torch.Tensor] = None,  # unused for QFormer
        thinker_hidden_attention_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if thinker_hidden_states is None and thinker_text_embeds is None:
            raise ValueError('At least one of text_embeds / hidden_states must be provided.')

        chunks = []
        if thinker_text_embeds is not None:
            chunks.append(self.text_projection(self.text_norm(thinker_text_embeds)))
        if thinker_hidden_states is not None:
            hidden = self.hidden_projection(self.hidden_norm(thinker_hidden_states))
            bsz = hidden.size(0)
            queries = self.queries.expand(bsz, -1, -1).to(hidden.dtype)
            kv_pad = None
            if thinker_hidden_attention_mask is not None:
                kv_pad = ~thinker_hidden_attention_mask.to(torch.bool)
            for blk in self.qformer_blocks:
                queries = blk(queries, hidden, kv_pad)
            chunks.append(queries)
        sem = torch.cat(chunks, dim=1)
        sem = sem + self.entry_adapter(self.entry_norm(sem))
        return sem


def build_bridge(config: DualRouteBridgeConfig) -> nn.Module:
    if config.bridge_type == 'mlp':
        return DualRouteMLPBridge(config)
    if config.bridge_type == 'qformer':
        return DualRouteQFormerBridge(config)
    raise ValueError(f'Unknown bridge_type: {config.bridge_type!r}')


# =============================================================================
#  Section 3: Qwen3-TTS Talker (Dense, Qwen3-style backbone)
#
#  Submodule layout matches upstream Qwen3-TTS-12Hz-1.7B-Base so loading is a
#  pure key-by-key copy (with optional layer slicing applied externally)::
#
#      Qwen3TtsTalkerForConditionalGeneration
#          model: Qwen3TtsTalkerModel
#              text_embedding              [V_text, H]            (= talker.model.text_embedding)
#              codec_embedding             [V_codec=3072, H]      (= talker.model.codec_embedding)
#              layers (Qwen3 dense, sliced or full 28)
#              norm
#          codec_head                      Linear(H, V_codec)     (= talker.codec_head)
#          code_predictor: Qwen3TtsCodePredictor
#              model: Qwen3CodePredictorTransformer
#                  layers (5 layers)
#                  norm
#                  codec_embedding         16 × Embedding(V_resid=2048, H=2048)
#              small_to_mtp_projection     Linear(H_talker=2048, H_cp=1024, bias=True)
#              lm_head                     16 × Linear(H_cp, V_resid, bias=False)
# =============================================================================


def _build_code_predictor_transformer_config(parent: Qwen3TtsTalkerConfig) -> Qwen3Config:
    """Mirror the upstream code-predictor inner transformer (a small Qwen3 dense)."""
    rope_params = getattr(parent, 'rope_parameters', None) or {
        'rope_theta': getattr(parent, 'rope_theta', 1_000_000.0), 'rope_type': 'default'
    }
    return Qwen3Config(
        vocab_size=1,  # we feed inputs_embeds, never input_ids
        hidden_size=parent.code_predictor_hidden_size,
        intermediate_size=parent.code_predictor_intermediate_size,
        num_hidden_layers=parent.code_predictor_num_layers,
        num_attention_heads=parent.code_predictor_num_attention_heads,
        num_key_value_heads=parent.code_predictor_num_key_value_heads,
        head_dim=parent.code_predictor_head_dim,
        max_position_embeddings=parent.max_position_embeddings,
        rms_norm_eps=parent.rms_norm_eps,
        rope_parameters=rope_params,
        tie_word_embeddings=False,
    )


def _cached_causal_bias(
    batch_size: int,
    sequence_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build the additive causal bias needed by a multi-token cache update."""
    if sequence_length == 1:
        return None
    total_length = past_length + sequence_length
    query_positions = torch.arange(
        past_length, total_length, device=device
    ).unsqueeze(1)
    key_positions = torch.arange(total_length, device=device).unsqueeze(0)
    future = key_positions > query_positions
    bias = torch.zeros(
        batch_size,
        1,
        sequence_length,
        total_length,
        dtype=dtype,
        device=device,
    )
    return bias.masked_fill(future[None, None, :, :], torch.finfo(dtype).min)


class Qwen3CodePredictorTransformer(nn.Module):
    """Transformer stack of the residual code predictor.

    Exposes the upstream layout ``code_predictor.model.{layers,norm,codec_embedding}``
    *as direct submodules* (not via ``@property``) so the state_dict naming
    matches Qwen3-TTS-Base exactly.
    """

    def __init__(self, parent_config: Qwen3TtsTalkerConfig):
        super().__init__()
        cfg = _build_code_predictor_transformer_config(parent_config)
        # Build a temporary ``Qwen3Model`` and *steal* its ``layers`` and
        # ``norm`` submodules into our own namespace so they appear as
        # ``code_predictor.model.layers.*`` / ``code_predictor.model.norm.*``
        # in state_dict.
        _tmp = Qwen3Model(cfg)
        self.layers = _tmp.layers
        self.norm = _tmp.norm
        self.rotary_emb = _tmp.rotary_emb
        self._cfg = cfg
        # Per-group residual codec embeddings (input side of the predictor).
        # NOTE: hidden dim is the *outer* talker hidden_size (2048), then a
        # projection to the predictor hidden_size (1024) is applied below.
        if parent_config.code_predictor_has_codec_embedding:
            self.codec_embedding = nn.ModuleList([
                nn.Embedding(parent_config.code_predictor_vocab_size, parent_config.hidden_size)
                for _ in range(parent_config.num_code_groups)
            ])
            # 2026-09-17: the LAST entry (index ``num_code_groups - 1``) is
            # structurally, permanently dead weight for the *input* role —
            # ``forward_teacher_forced`` builds the group sequence with
            # ``for i in range(groups - 1): seq.append(codec_embedding[i](...))``
            # (see below), i.e. only indices ``0..num_code_groups-2`` are ever
            # read as an embedding LOOKUP. The last group is only ever a
            # *prediction target* (via the separate ``lm_head[-1]``, a
            # different module), never fed back in as conditioning input —
            # by construction of the left-to-right group-autoregressive
            # roll-out there is no "next group" after the last one that would
            # need its embedding. Freeze it: not data-dependent (always true,
            # every microbatch, every sample), so it's the "结构性未用→
            # requires_grad=False" fix category, not a "加梯度锚点" one.
            # Repro'd 2026-09-17 on the Megatron side: leaving this trainable
            # crashes Megatron 0.17+'s ``overlap_grad_reduce`` grad-ready
            # invariant on literally every run (exact same missing param every
            # time, all ranks: ``code_predictor.model.codec_embedding.14.
            # weight``) since it can never register grad-ready in any
            # microbatch — freezing removes it from Megatron's tracked
            # trainable-param set entirely, which is the actual fix (an
            # anchor-loss workaround would be pointless here: there's no data
            # condition under which this param could ever legitimately
            # participate). Harmless on the DeepSpeed side too — it just
            # stops silently carrying dead (always-zero) gradient/optimizer
            # state for one embedding table.
            self.codec_embedding[-1].weight.requires_grad_(False)
        else:
            self.codec_embedding = None  # type: ignore[assignment]

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        # Manually run the layer stack: this is sufficient for our test/forward
        # path. We don't try to replicate every internal helper of Qwen3Model
        # (cache mgmt etc.); the loaded weights are the only artefact that
        # needs to be loadable for downstream training & save_pretrained.
        bsz, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(bsz, seq_len)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        if causal:
            # Autoregressive over the *codebook-group* dimension (matches the
            # upstream Qwen3-TTS code-predictor, which decodes residual groups
            # left-to-right). Group position ``i`` may only attend to ``0..i``.
            big_neg = torch.finfo(h.dtype).min
            causal_2d = torch.triu(
                torch.full((seq_len, seq_len), big_neg, dtype=h.dtype, device=h.device), diagonal=1)
            attention_mask = causal_2d[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
        # When ``attention_mask`` is a 2D padding mask, make it 4D additive bias.
        elif attention_mask is not None and attention_mask.dim() == 2:
            big_neg = torch.finfo(h.dtype).min
            mask4d = torch.zeros(bsz, 1, seq_len, seq_len, dtype=h.dtype, device=h.device)
            mask4d = mask4d.masked_fill(~attention_mask.to(torch.bool)[:, None, None, :], big_neg)
            attention_mask = mask4d
        for layer in self.layers:
            out = layer(h, attention_mask=attention_mask, position_ids=position_ids,
                        position_embeddings=position_embeddings)
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)
        return h

    def forward_cached(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """Run inference incrementally over residual codebook groups.

        This is deliberately separate from :meth:`forward`: training retains
        its existing full-sequence path, while inference can prefill the first
        two group positions and append one projected embedding at a time.  The
        caller owns the inference context: nesting ``torch.inference_mode`` in
        this method breaks AOTAutograd capture of the RoPE buffer in the
        current PyTorch runtime, while :class:`CodePredictorRunner` already
        supplies that context.
        """
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        batch_size, sequence_length, _ = inputs_embeds.shape
        if past_key_values is None:
            past_key_values = DynamicCache()
        cached_length = past_key_values.get_seq_length()
        past_length = int(cached_length) if cached_length is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + sequence_length,
                device=inputs_embeds.device,
            )
        position_ids = cache_position.unsqueeze(0).expand(
            batch_size, sequence_length
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        attention_bias = _cached_causal_bias(
            batch_size,
            sequence_length,
            past_length,
            inputs_embeds.dtype,
            inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_bias,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3TtsCodePredictor(nn.Module):
    """Residual code predictor: per-group input embeddings + 5-layer transformer + 16 grouped heads.

    Upstream key prefixes::

        small_to_mtp_projection.weight              [H_cp=1024, H_talker=2048]
        small_to_mtp_projection.bias                [H_cp]
        model.layers.{0..4}....
        model.norm.weight
        model.codec_embedding.{0..15}.weight        [V_resid, H_talker]
        lm_head.{0..15}.weight                      [V_resid, H_cp]
    """

    def __init__(self, parent_config: Qwen3TtsTalkerConfig):
        super().__init__()
        self.parent_config = parent_config
        # Talker -> CodePredictor hidden projection (with bias, matches upstream).
        self.small_to_mtp_projection = nn.Linear(
            parent_config.hidden_size, parent_config.code_predictor_hidden_size, bias=True)
        self.model = Qwen3CodePredictorTransformer(parent_config)
        self.lm_head = nn.ModuleList([
            nn.Linear(parent_config.code_predictor_hidden_size,
                      parent_config.code_predictor_vocab_size, bias=False)
            for _ in range(parent_config.num_code_groups)
        ])

    def forward(self, talker_hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Baseline (parallel) path — kept for the standalone talker forward / tests.

        Project ``talker_hidden`` to the predictor hidden size and predict all
        residual groups in parallel from the SAME position. This path does NOT
        perform the residual roll-out and should only be used when ground-truth
        codes are unavailable (e.g. a quick shape/test check). For real
        training use :meth:`forward_teacher_forced`.

        Returns logits of shape ``[B, T, num_code_groups, code_predictor_vocab_size]``.
        """
        x = self.small_to_mtp_projection(talker_hidden)
        h = self.model(inputs_embeds=x, attention_mask=attention_mask)
        per_group = [head(h) for head in self.lm_head]
        return torch.stack(per_group, dim=2)

    def forward_teacher_forced(
        self,
        talker_hidden: torch.Tensor,
        main_codes: torch.LongTensor,
        residual_codes: torch.LongTensor,
        main_codec_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Autoregressive residual roll-out along the codebook-group dimension.

        This mirrors the upstream Qwen3-TTS ``forward_sub_talker_finetune`` +
        ``code_predictor.forward_finetune`` exactly. For every codec frame we
        build a length ``G + 1`` *group sequence*::

            [ talker_hidden , E_main(main_code) , E_res[0](res_0) , ... , E_res[G-2](res_{G-2}) ]
              pos 0           pos 1               pos 2                     pos G

        run a CAUSAL transformer over the group dimension (so group position
        ``i`` only sees groups ``0..i-1`` plus the talker hidden), and let
        ``lm_head[i]`` at position ``i + 1`` predict residual group ``i``
        (``i = 0..G-1``). Teacher forcing feeds the ground-truth lower groups.

        Args:
            talker_hidden:   ``[B, T, H_talker]`` talker backbone hidden per frame.
            main_codes:      ``[B, T]`` ground-truth main code (group 0) per frame.
            residual_codes:  ``[B, T, G]`` ground-truth residual codes (groups 1..G).
            main_codec_embedding: the talker's main codec ``nn.Embedding`` (vocab
                ``codec_vocab_size``), used to embed the main code at group pos 1.

        Returns:
            logits ``[B, T, G, code_predictor_vocab_size]`` aligned with ``residual_codes``.
        """
        bsz, seq_t, hidden = talker_hidden.shape
        groups = self.parent_config.num_code_groups
        v_main = main_codec_embedding.num_embeddings
        v_res = self.parent_config.code_predictor_vocab_size
        num = bsz * seq_t

        flat_hidden = talker_hidden.reshape(num, hidden)
        # Clamp special / padded ids (EOS, -100, pad) to a valid embedding index.
        # Those frames carry -100 residual labels and are ignored by the loss.
        flat_main = main_codes.reshape(num).clamp(0, v_main - 1)
        flat_res = residual_codes.reshape(num, groups).clamp(0, v_res - 1)

        # Build the group sequence: [hidden, E_main(main), E_res[i](res_i) for i<G-1].
        seq = [flat_hidden.unsqueeze(1), main_codec_embedding(flat_main).unsqueeze(1)]
        for i in range(groups - 1):
            seq.append(self.model.codec_embedding[i](flat_res[:, i]).unsqueeze(1))
        seq = torch.cat(seq, dim=1)  # [num, G + 1, H_talker]

        x = self.small_to_mtp_projection(seq)         # [num, G + 1, H_cp]
        h = self.model(inputs_embeds=x, causal=True)  # [num, G + 1, H_cp]

        # lm_head[i] reads group position i + 1 to predict residual group i.
        logits = [self.lm_head[i](h[:, i + 1]) for i in range(groups)]
        logits = torch.stack(logits, dim=1)           # [num, G, V_res]
        return logits.reshape(bsz, seq_t, groups, v_res)


class Qwen3TtsTalkerModel(PreTrainedModel):
    """Dense Qwen3 backbone + codec embedding + main codec lm_head + code predictor.

    Submodule layout matches upstream Qwen3-TTS-12Hz-1.7B-Base for a
    parameter-perfect copy::

        text_embedding              [V_text, H]
        codec_embedding             [V_codec=3072, H]
        layers                      Qwen3 dense (sliced or full 28)
        norm
    """

    config_class = Qwen3TtsTalkerConfig
    base_model_prefix = 'qwen3_tts_talker'
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3TtsTalkerConfig):
        super().__init__(config)
        rope_params = getattr(config, 'rope_parameters', None) or {
            'rope_theta': getattr(config, 'rope_theta', 1_000_000.0), 'rope_type': 'default'
        }
        backbone_cfg = Qwen3Config(
            vocab_size=1,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_parameters=rope_params,
            tie_word_embeddings=False,
        )
        # Build a temporary ``Qwen3Model`` and *steal* ``layers`` / ``norm`` /
        # ``rotary_emb`` so that state_dict keys match upstream
        # ``talker.model.layers.*`` / ``talker.model.norm.*`` exactly.
        _tmp = Qwen3Model(backbone_cfg)
        self.layers = _tmp.layers
        self.norm = _tmp.norm
        self.rotary_emb = _tmp.rotary_emb
        self._backbone_cfg = backbone_cfg

        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.codec_embedding = nn.Embedding(config.codec_vocab_size, config.hidden_size)
        self.post_init()

    # HF embedding API ---------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.text_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_embedding = value

    def embed_codec(self, codec_input_ids: torch.LongTensor) -> torch.Tensor:
        return self.codec_embedding(codec_input_ids)

    def embed_text(self, text_input_ids: torch.LongTensor) -> torch.Tensor:
        return self.text_embedding(text_input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values=None,
        return_dict: bool = True,
    ):
        """Run the dense Qwen3 stack on ``inputs_embeds``.

        Returns a HF-style ``BaseModelOutputWithPast`` (only
        ``last_hidden_state`` is populated; we don't need cache/etc. here).
        """
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        # Build the attention bias. The talker is an AUTOREGRESSIVE decoder over
        # the codec time axis, so it MUST be causal — position ``t`` may only
        # attend to ``<= t``. This mirrors upstream Qwen3-TTS
        # ``Qwen3TTSTalkerModel.forward`` which uses ``create_causal_mask(...)``
        # (see project/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py).
        #
        # A 2-D ``attention_mask`` is only a *padding* mask; converting it to 4-D
        # without the causal lower-triangular term would let position ``t`` see
        # ``codec_input_ids[t+1] == codec_labels[t]`` and copy the answer, making
        # the main-codec ``token_acc`` collapse to ~1.0 (label leakage). We
        # therefore always superimpose the causal term. A pre-built 4-D bias
        # (e.g. from ``build_block_causal_mask`` on the legacy path) is assumed to
        # already encode causality and is passed through unchanged.
        big_neg = torch.finfo(inputs_embeds.dtype).min
        causal_block = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )  # True where key index > query index (future) -> must be masked
        if attention_mask is None:
            m4 = torch.zeros(bsz, 1, seq_len, seq_len, dtype=inputs_embeds.dtype, device=device)
            m4 = m4.masked_fill(causal_block[None, None, :, :], big_neg)
            attention_mask = m4
        elif attention_mask.dim() == 2:
            m4 = torch.zeros(bsz, 1, seq_len, seq_len, dtype=inputs_embeds.dtype, device=device)
            m4 = m4.masked_fill(~attention_mask.to(torch.bool)[:, None, None, :], big_neg)  # padding
            m4 = m4.masked_fill(causal_block[None, None, :, :], big_neg)                     # + causal
            attention_mask = m4
        # else: already a 4-D additive bias (legacy block-causal path) — use as-is.

        h = inputs_embeds
        for layer in self.layers:
            out = layer(h, attention_mask=attention_mask, position_ids=position_ids,
                        position_embeddings=position_embeddings)
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=h)

    @torch.inference_mode()
    def forward_cached(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """INFERENCE-ONLY incremental forward with KV-cache.

        This method is NOT used in training and NEVER touched by the existing
        ``forward``. It runs the same stolen Qwen3 decoder ``layers`` but relies
        on the HF ``Cache`` mechanism so that a single new codec frame only
        recomputes attention against the cached past (O(1) per step instead of
        the O(T^2) full-sequence recompute in ``forward``).

        Args:
            inputs_embeds: ``[B, S, H]``. ``S`` may be >1 (prefill) or 1 (step).
            past_key_values: a ``transformers.cache_utils.Cache`` (e.g.
                ``DynamicCache``). Updated in place by the attention modules.
            cache_position: ``[S]`` absolute positions of ``inputs_embeds`` in
                the full sequence. If ``None``, inferred from the current cache
                length so callers can omit it.

        Returns a HF ``BaseModelOutputWithPast`` with ``last_hidden_state`` and
        the (in-place updated) ``past_key_values``.

        Equivalence to ``forward``: the talker is a pure causal decoder, so
        position ``t`` attends to ``<= t`` regardless of whether the past frames
        are recomputed or cached. Padding is not a concern at inference (all
        codec positions are real). We therefore pass ``attention_mask=None`` and
        let the attention see the full cached history (causal by construction of
        left-to-right generation).
        """
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        if past_key_values is None:
            past_key_values = DynamicCache()
        past_len = int(past_key_values.get_seq_length()) if past_key_values.get_seq_length() is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bsz, seq_len)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Build a causal additive bias only when we process >1 query at once
        # (prefill). For a single-token step there is no future to mask, so we
        # can pass ``None`` (cheaper). The key length is the full cached history
        # (``past_len + seq_len``); queries occupy the last ``seq_len`` slots.
        attn_bias = None
        if seq_len > 1:
            total = past_len + seq_len
            big_neg = torch.finfo(inputs_embeds.dtype).min
            # query i (abs pos = past_len + i) may attend to key j iff j <= past_len + i.
            q_pos = torch.arange(past_len, total, device=device).unsqueeze(1)   # [seq_len,1]
            k_pos = torch.arange(0, total, device=device).unsqueeze(0)          # [1,total]
            future = k_pos > q_pos                                              # [seq_len,total]
            attn_bias = torch.zeros(bsz, 1, seq_len, total, dtype=inputs_embeds.dtype, device=device)
            attn_bias = attn_bias.masked_fill(future[None, None, :, :], big_neg)

        h = inputs_embeds
        for layer in self.layers:
            h = layer(
                h,
                attention_mask=attn_bias,     # None for single-step; causal bias for prefill
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if isinstance(h, tuple):
                h = h[0]
        h = self.norm(h)
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values)


class Qwen3TtsTalkerForConditionalGeneration(PreTrainedModel):
    """Top-level Qwen3-TTS Talker module.

    Submodule layout matches upstream::

        model: Qwen3TtsTalkerModel  (text_embedding, codec_embedding, layers, norm)
        codec_head: Linear(H, V_codec)
        code_predictor: Qwen3TtsCodePredictor
    """

    config_class = Qwen3TtsTalkerConfig
    base_model_prefix = 'qwen3_tts_talker'

    def __init__(self, config: Qwen3TtsTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TtsTalkerModel(config)
        self.codec_head = nn.Linear(config.hidden_size, config.codec_vocab_size, bias=False)
        self.code_predictor = Qwen3TtsCodePredictor(config)
        # ``text_projection`` mirrors upstream Qwen3-TTS: it maps the text-side
        # signal (either ``text_embedding(id)`` for the classic TTS path or the
        # ThinkerAdapter output on this branch) into the talker main hidden
        # space. Structure and weight layout match upstream so
        # ``talker.text_projection.linear_fc{1,2}.{weight,bias}`` load
        # bit-perfect from Qwen3-TTS-12Hz-1.7B-Base (verified in M1b).
        #
        # We build it unconditionally (regardless of the top-level
        # ``use_upstream_talker_input`` flag) so that existing checkpoints
        # containing these tensors continue to load cleanly; the bridge
        # forward path simply ignores this module.
        text_h = getattr(config, 'text_hidden_size', config.hidden_size)
        hidden_act = getattr(config, 'hidden_act', 'silu')
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            input_dim=text_h,
            intermediate_dim=text_h,
            output_dim=config.hidden_size,
            hidden_act=hidden_act,
            bias=True,
        )
        self.post_init()

    def forward(self, codec_input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None):
        emb = self.model.embed_codec(codec_input_ids)
        out = self.model(inputs_embeds=emb, attention_mask=attention_mask)
        h = out.last_hidden_state
        main_logits = self.codec_head(h)
        residual_logits = self.code_predictor(h, attention_mask)
        return {
            'main_codec_logits': main_logits,
            'residual_codec_logits': residual_logits,
            'last_hidden_state': h,
        }


# =============================================================================
#  Section 4: top-level Qwen-OmniSpeak model
# =============================================================================


@dataclass
class QwenOmniSpeakOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    main_codec_logits: Optional[torch.Tensor] = None
    residual_codec_logits: Optional[torch.Tensor] = None
    talker_hidden_states: Optional[torch.Tensor] = None
    semantic_block: Optional[torch.Tensor] = None
    text_logits: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, torch.Tensor]] = None


def _token_correctness(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Optional[torch.Tensor]:
    if logits is None or labels is None:
        return None
    with torch.no_grad():
        labels = labels.to(logits.device)
        mask = labels != ignore_index
        if not bool(mask.any()):
            return None
        preds = logits.argmax(dim=-1)
        correct = (preds == labels) & mask
        return correct[mask].float().detach()


def select_thinker_hidden(hidden_states: Tuple[torch.Tensor, ...], selection: Union[int, str, None]) -> torch.Tensor:
    """Select the hidden state passed to the bridge.

    Qwen3-Omni keeps the legacy layer index 24. Qwen2.5-Omni uses ``last``
    (``hidden_states[-1]``), matching the final thinker output.
    """
    if not hidden_states:
        raise RuntimeError('Thinker returned an empty hidden_states tuple.')
    if selection is None:
        selection = 24
    if isinstance(selection, str):
        value = selection.strip().lower()
        if value in ('last', 'final', '-1'):
            return hidden_states[-1]
        idx = int(value)
    else:
        idx = int(selection)
    return hidden_states[idx]


def _residual_loss_settings(config) -> Tuple[bool, float]:
    """Resolve residual-loss on/off + weight.

    Priority: runtime env vars override the model config, which in turn
    defaults to (enabled=True, weight=1.0). This lets a single training run
    flip the residual loss without re-exporting / editing ``config.json``::

        QWEN_OMNISPEAK_RESIDUAL_LOSS=false        # disable residual CE
        QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT=0.5   # scale residual CE
    """
    enabled = bool(getattr(config, 'residual_loss_enabled', True))
    weight = float(getattr(config, 'residual_loss_weight', 1.0))
    env_enabled = os.environ.get('QWEN_OMNISPEAK_RESIDUAL_LOSS')
    if env_enabled is None:
        env_enabled = os.environ.get('QWEN3_OMNISPEAK_RESIDUAL_LOSS')
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() in ('1', 'true', 'yes', 'on')
    env_weight = os.environ.get('QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT')
    if env_weight is None:
        env_weight = os.environ.get('QWEN3_OMNISPEAK_RESIDUAL_LOSS_WEIGHT')
    if env_weight is not None:
        try:
            weight = float(env_weight)
        except ValueError:
            pass
    if weight <= 0.0:
        enabled = False
    return enabled, weight


def _codec_eos_loss_weight(config) -> float:
    """Resolve the main-codec EOS CE weight (env > config > 1.0)."""
    weight = float(getattr(config, 'codec_eos_loss_weight', 1.0))
    env_weight = os.environ.get('QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT')
    if env_weight is not None:
        try:
            weight = float(env_weight)
        except ValueError as exc:
            raise ValueError(
                'QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT must be a number.') from exc
    if weight <= 0:
        raise ValueError('codec EOS loss weight must be > 0.')
    return weight


def _weighted_codec_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """MiniMind-style EOS numerator weighting with an unweighted denominator."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    valid_count = valid.sum()
    if not bool(valid_count > 0):
        return flat_logits.sum() * 0.0
    per_token = _cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction='none')
    weights = torch.ones_like(per_token)
    weights = torch.where(flat_labels.eq(eos_id), weights * eos_weight, weights)
    return (per_token * weights * valid).sum() / valid_count


def _codec_history_corruption_prob(config) -> float:
    """Resolve training-only acoustic-history corruption (env > config)."""
    probability = float(getattr(config, 'codec_history_corruption_prob', 0.0))
    env_probability = os.environ.get('QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB')
    if env_probability is not None:
        try:
            probability = float(env_probability)
        except ValueError as exc:
            raise ValueError(
                'QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB must be a number.') from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError('codec history corruption probability must be in [0, 1].')
    return probability


def _corrupt_codec_history(
    codec_input_ids: torch.Tensor,
    codec_labels: torch.Tensor,
    *,
    probability: float,
    acoustic_vocab_size: int,
) -> torch.Tensor:
    """Replace supervised acoustic-history inputs; never alter targets/control ids."""
    if probability == 0.0:
        return codec_input_ids
    eligible = (
        codec_input_ids.ge(0)
        & codec_input_ids.lt(acoustic_vocab_size)
        & codec_labels.ne(-100)
    )
    if not bool(eligible.any()):
        return codec_input_ids
    corruption_mask = eligible & torch.rand(
        codec_input_ids.shape, device=codec_input_ids.device).lt(probability)
    if not bool(corruption_mask.any()):
        return codec_input_ids
    replacements = torch.randint(
        0,
        acoustic_vocab_size,
        codec_input_ids.shape,
        dtype=codec_input_ids.dtype,
        device=codec_input_ids.device,
    )
    replacements = torch.where(
        replacements.eq(codec_input_ids),
        (replacements + 1) % acoustic_vocab_size,
        replacements,
    )
    corrupted = codec_input_ids.clone()
    corrupted[corruption_mask] = replacements[corruption_mask]
    return corrupted


def _text_loss_settings(config) -> Tuple[bool, float]:
    """Resolve thinker text-loss on/off + weight.

    Priority: runtime env vars override the model config, which in turn
    defaults to (enabled=True, weight=1.0). This lets a training run flip /
    scale the thinker-side CE without re-exporting ``config.json``::

        QWEN_OMNISPEAK_TEXT_LOSS=false          # disable thinker text CE
        QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT=0.5     # scale thinker text CE

    Note: the text loss only contributes *gradient* when the thinker is
    trainable (unfrozen). When frozen it is computed for metrics only and
    this weight has no effect on the backward pass.
    """
    enabled = bool(getattr(config, 'text_loss_enabled', True))
    weight = float(getattr(config, 'text_loss_weight', 1.0))
    env_enabled = os.environ.get('QWEN_OMNISPEAK_TEXT_LOSS')
    if env_enabled is None:
        env_enabled = os.environ.get('QWEN3_OMNISPEAK_TEXT_LOSS')
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() in ('1', 'true', 'yes', 'on')
    env_weight = os.environ.get('QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT')
    if env_weight is None:
        env_weight = os.environ.get('QWEN3_OMNISPEAK_TEXT_LOSS_WEIGHT')
    if env_weight is not None:
        try:
            weight = float(env_weight)
        except ValueError:
            pass
    if weight <= 0.0:
        enabled = False
    return enabled, weight


def build_block_causal_mask(
    sem_len: int,
    codec_len: int,
    sem_attention_mask: Optional[torch.Tensor],
    codec_attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    codec_to_sem_visible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Construct a 4D additive attention mask of shape ``[B, 1, T, T]``.

    Default rules (offline / full-context, used by Stage-1 bridge alignment):
      * codec tokens can attend to *all* semantic tokens (within the chunk)
      * codec tokens can only attend to *past* codec tokens (causal)
      * semantic tokens can attend bidirectionally within the semantic block
      * semantic tokens cannot attend to codec tokens (no future leakage)

    Streaming / simultaneous-interpretation extension point:
      * ``codec_to_sem_visible`` (shape ``[codec_len]`` or ``[B, codec_len]``)
        gives, per codec frame ``t``, how many *leading* semantic tokens that
        frame is allowed to attend to (``sem[:codec_to_sem_visible[t]]``).
        Passing a monotonically non-decreasing schedule turns the otherwise
        full-context ``codec -> sem`` block into a chunk-causal (read-write)
        mask, without changing any other behaviour. When ``None`` (default)
        the block stays fully visible, i.e. *exactly* the offline behaviour.
    """
    B = max(
        sem_attention_mask.size(0) if sem_attention_mask is not None else 1,
        codec_attention_mask.size(0) if codec_attention_mask is not None else 1,
        codec_to_sem_visible.size(0) if (codec_to_sem_visible is not None
                                         and codec_to_sem_visible.dim() == 2) else 1,
    )
    T = sem_len + codec_len
    big_neg = torch.finfo(dtype).min
    # Per-batch ``allowed`` so a batched ``codec_to_sem_visible`` is supported.
    allowed = torch.zeros(B, T, T, dtype=torch.bool, device=device)
    allowed[:, :sem_len, :sem_len] = True
    if codec_len > 0 and sem_len > 0:
        if codec_to_sem_visible is None:
            allowed[:, sem_len:, :sem_len] = True
        else:
            vis = torch.as_tensor(codec_to_sem_visible, device=device)
            if vis.dim() == 1:
                vis = vis.unsqueeze(0).expand(B, codec_len)
            sem_ar = torch.arange(sem_len, device=device)
            # [B, codec_len, sem_len]: frame t sees sem position j iff j < vis[t]
            allowed[:, sem_len:, :sem_len] = sem_ar[None, None, :] < vis[:, :, None]
    if codec_len > 0:
        causal = torch.tril(torch.ones(codec_len, codec_len, dtype=torch.bool, device=device))
        allowed[:, sem_len:, sem_len:] = causal[None]
    bias = torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), big_neg, dtype=dtype, device=device))
    bias = bias.unsqueeze(1)  # [B, 1, T, T]
    if sem_attention_mask is not None or codec_attention_mask is not None:
        key_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        if sem_attention_mask is not None:
            key_mask[:, :sem_len] = sem_attention_mask.to(torch.bool)
        if codec_attention_mask is not None:
            key_mask[:, sem_len:] = codec_attention_mask.to(torch.bool)
        bias = bias.masked_fill(~key_mask[:, None, None, :], big_neg)
    return bias


class QwenOmniSpeakModel(PreTrainedModel):
    """Bare composite that exposes thinker, bridge and talker as named children."""

    config_class = QwenOmniSpeakConfig
    base_model_prefix = 'qwen_omnispeak'
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn3 = True

    def __init__(self, config: QwenOmniSpeakConfig):
        super().__init__(config)
        self.config = config

        if getattr(config, '_thinker_is_full_omni', False):
            thinker_subcfg = config.thinker_config.thinker_config
        else:
            thinker_subcfg = config.thinker_config
        # We only ever need the Thinker subtree. Building it directly avoids
        # the upstream Talker/Code2Wav duplication that otherwise breaks
        # ``save_pretrained`` (HF flags shared tensors and refuses to save).
        if getattr(config, 'thinker_family', None) == QWEN25_THINKER_FAMILY:
            if Qwen2_5OmniThinkerForConditionalGeneration is None:
                raise ImportError('Qwen2.5-Omni thinker requires transformers with qwen2_5_omni support.')
            self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(thinker_subcfg)
        else:
            self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_subcfg)

        # ``bridge`` is kept even when the upstream talker input protocol is
        # enabled: it is skipped in ``forward``, but keeping the module means
        # legacy checkpoints (with ``bridge.*`` tensors) still load cleanly and
        # the same weights file can flip between the two forward paths via
        # ``config.use_upstream_talker_input``.
        self.bridge = build_bridge(config.bridge_config)
        # Use the full ``ForConditionalGeneration`` wrapper so the talker keeps
        # the upstream submodule layout (model / codec_head / code_predictor),
        # which makes loading raw Qwen3-TTS weights a key-by-key copy.
        self.talker = Qwen3TtsTalkerForConditionalGeneration(config.talker_config)

        # ── ThinkerAdapter (V1: hidden + token_embed) ──────────────────────
        # Only instantiated when the upstream talker input protocol is on. It
        # maps the thinker signal (``hidden + token_embed``) to
        # ``text_hidden_size`` so it can be fed into ``talker.text_projection``
        # exactly like an ``text_embedding(id)`` output would.
        self.thinker_adapter: Optional[Qwen3TTSThinkerAdapter] = None
        if getattr(config, 'use_upstream_talker_input', False):
            thinker_h = self._infer_thinker_hidden_size(thinker_subcfg)
            text_h = getattr(config.talker_config, 'text_hidden_size',
                             config.talker_config.hidden_size)
            self.thinker_adapter = Qwen3TTSThinkerAdapter(
                thinker_hidden_size=thinker_h,
                text_hidden_size=text_h,
                intermediate_size=getattr(config, 'adapter_hidden_size', None),
                dropout=float(getattr(config, 'adapter_dropout', 0.0) or 0.0),
            )

        if config.tie_codec_embeddings:
            self.talker.codec_head.weight = self.talker.model.codec_embedding.weight

        self.post_init()
        # Retroactively patch talker's already-built Qwen3 decoder stack with
        # liger-kernel's fused Triton ops (no-op unless
        # QWEN_OMNISPEAK_LIGER_OPS=1; see docstring above).
        _apply_liger_to_talker_decoder(self.talker.model)

    @staticmethod
    def _infer_thinker_hidden_size(thinker_subcfg) -> int:
        """Best-effort resolution of the thinker's text hidden size.

        Handles both ``Qwen2_5OmniThinkerConfig`` (``text_config.hidden_size``)
        and ``Qwen3OmniMoeThinkerConfig`` (also under ``text_config``). Falls
        back to a plain top-level ``hidden_size`` for defensive builds.
        """
        text_cfg = getattr(thinker_subcfg, 'text_config', None)
        if text_cfg is not None and getattr(text_cfg, 'hidden_size', None):
            return int(text_cfg.hidden_size)
        if getattr(thinker_subcfg, 'hidden_size', None):
            return int(thinker_subcfg.hidden_size)
        raise ValueError('Cannot infer thinker hidden_size for ThinkerAdapter.')

    @torch.no_grad()
    def _run_thinker_no_grad(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_text_embeds: bool = True,
        **thinker_kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self._run_thinker_inner(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, output_text_embeds=output_text_embeds,
            **thinker_kwargs)

    def _run_thinker_inner(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_text_embeds: bool = True,
        **thinker_kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            **thinker_kwargs,
        )
        selection = getattr(self.config, 'hidden_state_selection', getattr(self.config, 'accept_hidden_layer', 24))
        hidden_states_tuple = getattr(out, 'hidden_states', None)
        if hidden_states_tuple is None:
            raise RuntimeError('Thinker did not return hidden_states; check output_hidden_states=True.')
        thinker_hidden = select_thinker_hidden(hidden_states_tuple, selection)
        text_embeds = None
        if output_text_embeds and input_ids is not None:
            text_embeds = self.thinker.get_input_embeddings()(input_ids)
        return {
            'hidden_states': thinker_hidden,
            'text_embeds': text_embeds,
            'logits': getattr(out, 'logits', None),
        }

    def run_thinker(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_text_embeds: bool = True,
        **thinker_kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the thinker.

        During training (``self.training is True``) gradients flow through
        the thinker so end-to-end fine-tuning of thinker + bridge + talker
        is possible (used by continue-pretrain). During eval/inference we
        keep the original ``no_grad`` behaviour for memory efficiency.
        """
        thinker_trainable = any(p.requires_grad for p in self.thinker.parameters())
        if self.training and thinker_trainable:
            return self._run_thinker_inner(
                input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, output_text_embeds=output_text_embeds,
                **thinker_kwargs)
        return self._run_thinker_no_grad(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, output_text_embeds=output_text_embeds,
            **thinker_kwargs)


class QwenOmniSpeakForConditionalGeneration(PreTrainedModel):
    """Top-level Qwen-OmniSpeak model.

    Forward expects pre-computed thinker outputs (text_embeds, hidden_states),
    or alternatively raw ``input_ids`` from which the thinker side is run
    on-the-fly. The talker side consumes ``codec_input_ids`` (right-shifted
    codec labels with leading ``codec_bos`` / pad).
    """

    config_class = QwenOmniSpeakConfig
    base_model_prefix = 'qwen_omnispeak'
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn3 = True
    _no_split_modules: List[str] = [
        'Qwen3OmniMoeThinkerDecoderLayer', 'Qwen3DecoderLayer',
        'Qwen2_5OmniThinkerDecoderLayer', 'Qwen2_5OmniDecoderLayer',
    ]

    def __init__(self, config: QwenOmniSpeakConfig):
        super().__init__(config)
        self.config = config
        self.model = QwenOmniSpeakModel(config)
        self.post_init()
        # MFU accounting: per-step supervised-token counter accumulated across
        # micro-batches by forward() and reset by the trainer on_step_end hook.
        # Lives on the model (not in `metrics`) to bypass swift's MeanMetric.
        self._mfu_tokens = 0

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Backwards-compatible state_dict loader (torch API path).

        Historical exports of ``qwen2_5_3b_omnispeak_*`` checkpoints emit
        ``talker.text_projection.linear_fc{1,2}.{weight,bias}`` with a *naked*
        ``talker.`` prefix (no leading ``model.``) because the talker's
        ``text_projection`` module used to live outside ``QwenOmniSpeakModel``
        as an orphan tensor. Rewrite those keys to their current location
        under ``model.talker.*`` before delegating to the standard loader.

        The ``from_pretrained`` path uses ``_fix_state_dict_key_on_load``
        below to perform the equivalent rename.
        """
        needs_remap = any(k.startswith('talker.text_projection.') for k in state_dict)
        if needs_remap:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('talker.text_projection.'):
                    remapped['model.' + k] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _fix_state_dict_key_on_load(key: str):
        """Rename legacy naked ``talker.text_projection.*`` keys to their
        current location under ``model.talker.*`` during ``from_pretrained``.
        Delegates all other renames to the base implementation.
        """
        if key.startswith('talker.text_projection.'):
            return 'model.' + key, True
        return PreTrainedModel._fix_state_dict_key_on_load(key)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override to load ``talker.text_projection.*`` from the sidecar
        ``qwen3_tts_extras.safetensors`` file (which is NOT listed in
        ``model.safetensors.index.json`` and thus silently skipped by the
        standard ``from_pretrained`` path).
        """
        # ms-swift 用 `with torch.device('meta'):` 包裹本次调用，模型会被建在 meta 上，
        # transformers 随即走 _load_state_dict_into_meta_model 并对 meta 张量做
        # `param = param[...]` → NotImplementedError: Cannot copy out of meta tensor。
        # 修复：内层 `with torch.device(target)` 覆盖外层 meta 上下文，让模型以真实张量
        # 构建、走普通加载路径。约束（缺一不可）：
        #   - device_map 保持 ms-swift 原值 'cuda:N'；置 None 会触发 modeling_utils 的
        #     meta anti-pattern 守卫（该守卫仅在 device_map is None 时检查上下文设备）。
        #   - low_cpu_mem_usage=False，阻止 transformers 内部 init_empty_weights 回到 meta。
        import sys, torch
        kwargs['use_safetensors'] = True
        kwargs['low_cpu_mem_usage'] = False
        _dm = kwargs.get('device_map')
        _target = torch.device(_dm) if _dm is not None else torch.device('cpu')
        print(f"[omnispeak] from_pretrained device_map={_dm!r} build_device={_target!r} "
              f"outer_default_device={torch.get_default_device()!r}",
              file=sys.stderr, flush=True)
        with torch.device(_target):
            model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # Sidecar load for ``talker.text_projection.*`` (4 tensors in
        # ``qwen3_tts_extras.safetensors``, alongside the speaker_encoder
        # weights, using the *naked* ``talker.text_projection.*`` prefix).
        # text_projection is FROZEN but FORWARD-CRITICAL in adapter mode (the
        # adapter output and tts_pad embed both pass through it): a silent
        # failure leaves a random projection in the forward path and training
        # looks healthy while optimising against wrong semantics. Adapter mode
        # therefore fails fast on any missing file/key/shape error; bridge mode
        # keeps the historical best-effort behaviour (P1-1, fixed 2026-08).
        import os
        from pathlib import Path
        _upstream = bool(getattr(model.config, 'use_upstream_talker_input', False))
        model_dir = Path(pretrained_model_name_or_path)
        sidecar = model_dir / 'qwen3_tts_extras.safetensors'
        if not sidecar.is_file():
            if _upstream:
                raise FileNotFoundError(
                    f'[omnispeak] adapter mode requires the sidecar '
                    f'{sidecar.name} (talker.text_projection.*) but it is '
                    f'missing under {model_dir}.')
            return model
        try:
            from safetensors.torch import load_file as _load_sf
            extras = _load_sf(str(sidecar))
            tp_prefix = 'talker.text_projection.'
            tp_sd = {('model.' + k): v for k, v in extras.items()
                     if k.startswith(tp_prefix)}
            if not tp_sd:
                raise ValueError(f'no {tp_prefix}* tensors found in {sidecar.name}')
            # assign=True bypasses shape checks (it swaps references), so
            # validate key presence and shapes explicitly before loading.
            _md = next(model.parameters()).device
            tp_sd = {k: v.to(_md) for k, v in tp_sd.items()}
            for k, v in tp_sd.items():
                p = model.get_parameter(k)
                if tuple(p.shape) != tuple(v.shape):
                    raise ValueError(
                        f'sidecar tensor {k} shape {tuple(v.shape)} does not '
                        f'match model parameter {tuple(p.shape)}')
            missing, unexpected = model.load_state_dict(tp_sd, strict=False, assign=True)
            tp_missing = [k for k in missing
                          if k.startswith('model.talker.text_projection.')]
            if tp_missing:
                raise ValueError(f'sidecar keys not accepted by the model: {tp_missing}')
            if _upstream:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    f'[omnispeak] loaded talker.text_projection.* '
                    f'({len(tp_sd)} tensors) from sidecar {sidecar.name}')
        except Exception:
            if _upstream:
                raise
            # bridge mode: non-fatal; text_projection just stays randomly init
        return model

    @property
    def thinker(self):
        return self.model.thinker

    @property
    def bridge(self):
        return self.model.bridge

    @property
    def talker(self):
        return self.model.talker

    @property
    def thinker_adapter(self):
        return self.model.thinker_adapter

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    # -----------------------------------------------------------------------
    #  Upstream Qwen3-TTS talker input protocol helpers
    #  (only used when ``config.use_upstream_talker_input`` is True).
    # -----------------------------------------------------------------------

    def _build_icl_prefix(
        self,
        source_codec_codes: torch.Tensor,   # [B, T_ref, 16] reference (source) codec
        source_text_ids: Optional[torch.Tensor],  # [B, T_txt] reference (source) text token ids
        source_codec_mask: Optional[torch.Tensor],  # [B, T_ref] bool
        source_text_mask: Optional[torch.Tensor],   # [B, T_txt] bool
        codec_dtype: torch.dtype,
        codec_device: torch.device,
    ):
        """Build the ICL prefix (see the module-level ``build_icl_prefix``)."""
        return build_icl_prefix(
            self.config, self.talker,
            source_codec_codes, source_text_ids,
            source_codec_mask, source_text_mask,
            codec_dtype=codec_dtype, codec_device=codec_device)

    def _build_upstream_text_channel(
        self,
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
        """Assemble the length-``codec_len`` text-side conditioning tensor.

        Teacher-forced (training) path: extract the thinker's *reply* positions
        (where ``text_labels != -100``, i.e. the assistant text span that
        Qwen2.5-Omni would produce via ``generate()``) and use their hidden +
        token_embed as the text channel. This matches upstream Qwen2.5-Omni's
        ``thinker_reply_part`` (L166 of ``Qwen2_5OmniForConditionalGeneration.generate``)
        but computed in a single teacher-forced forward pass (differentiable
        through the thinker, no argmax/generate loop needed).

        Returns ``[B, codec_len, H_talker]`` where every frame is either:
          * ``text_projection(adapter(reply_signal[k]))`` mapped to codec position
            ``codec_bos_index + k``, or
          * ``text_projection(text_embedding(tts_pad_token_id))`` (a.k.a.
            ``tts_pad_embed``) elsewhere.

        Full-add layout (``full_add_prompt_len`` given, Qwen2.5-Omni): the
        positions *before* ``codec_bos_index`` are additionally filled with the
        thinker *prompt* signal — ``text_channel[b, 0:P_b]`` = projected
        ``(prompt_hidden + prompt_embed)`` and ``text_channel[b, P_b]`` =
        projected ``text_bos`` embedding (``P_b = full_add_prompt_len[b] =
        codec_bos_index[b] - 1`` when no speaker slot). This mirrors
        ``talker_inputs_embeds = cat([prompt_part, text_bos, reply_part])``.
        """
        assert self.thinker_adapter is not None, (
            'use_upstream_talker_input=True but ThinkerAdapter was not built; '
            'this usually means the flag was flipped after __init__.')
        cfg = self.config
        talker = self.talker
        bsz = thinker_hidden_states.size(0)
        H = talker.model.config.hidden_size

        # ── 1. Extract reply-only positions via text_labels != -100 ───────
        # ``text_labels`` is the thinker LM label tensor: -100 on prompt /
        # audio positions, actual token id on the assistant reply span.
        # We keep only the reply positions' hidden + embed (mirroring upstream
        # Qwen2.5-Omni ``thinker_reply_part`` which uses generated token hidden).
        if text_labels is None:
            # Fallback: if text_labels not provided (e.g. TTS task where labels
            # were reinterpreted as codec_labels), use the entire sequence.
            # This is the old behaviour; warn so the user knows.
            import logging as _lg
            _lg.getLogger(__name__).warning(
                '[upstream] text_labels is None; using entire thinker sequence '
                '(no reply-position filtering). This is incorrect for s2s; '
                'ensure text_labels reaches forward for s2s tasks.')
            reply_hidden = thinker_hidden_states
            reply_embed = thinker_text_embeds
        else:
            # text_labels: [B, T_thinker], reply_mask: [B, T_thinker] bool
            reply_mask = (text_labels != -100)   # [B, T_thinker]
            # Gather reply positions per sample. Reply lengths may differ per
            # sample, so we process per-sample and pad to the max reply length.
            reply_lens = reply_mask.sum(dim=-1).tolist()  # list[int]
            max_reply = max(reply_lens) if reply_lens else 0
            if max_reply == 0:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    '[upstream] text_labels has no reply positions (all -100); '
                    'text_channel will be all pad_embed.')
                reply_hidden = thinker_hidden_states.new_zeros(bsz, 0, thinker_hidden_states.size(-1))
                reply_embed = (
                    thinker_text_embeds.new_zeros(bsz, 0, thinker_text_embeds.size(-1))
                    if thinker_text_embeds is not None else None
                )
            else:
                reply_hidden_list = []
                reply_embed_list = []
                for b in range(bsz):
                    idx = reply_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [reply_len_b]
                    rh = thinker_hidden_states[b].index_select(0, idx)       # [reply_len_b, H]
                    reply_hidden_list.append(rh)
                    if thinker_text_embeds is not None:
                        re = thinker_text_embeds[b].index_select(0, idx)    # [reply_len_b, H]
                        reply_embed_list.append(re)
                # Pad to max_reply along time dim with zeros (will be masked by
                # text_valid_lens later, so the padding content doesn't matter).
                reply_hidden = torch.zeros(bsz, max_reply,
                                           thinker_hidden_states.size(-1),
                                           dtype=thinker_hidden_states.dtype,
                                           device=thinker_hidden_states.device)
                for b in range(bsz):
                    lb = reply_lens[b]
                    if lb > 0:
                        reply_hidden[b, :lb] = reply_hidden_list[b]
                if thinker_text_embeds is not None:
                    reply_embed = torch.zeros(bsz, max_reply,
                                              thinker_text_embeds.size(-1),
                                              dtype=thinker_text_embeds.dtype,
                                              device=thinker_text_embeds.device)
                    for b in range(bsz):
                        lb = reply_lens[b]
                        if lb > 0:
                            reply_embed[b, :lb] = reply_embed_list[b]
                else:
                    reply_embed = None
                # Store reply_lens as tensor for later use
                reply_lens_tensor = torch.tensor(reply_lens, dtype=torch.long,
                                                 device=thinker_hidden_states.device)

        # ── 2. thinker_signal = reply_hidden + reply_embed ────────────────
        signal_type = getattr(cfg, 'thinker_signal_type', 'hidden_plus_embed')
        if signal_type == 'hidden_plus_embed':
            if reply_embed is None:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    '[upstream] thinker_text_embeds is None; using '
                    'reply_hidden only (degraded V1 signal).')
                thinker_signal = reply_hidden
            elif reply_embed.shape != reply_hidden.shape:
                raise ValueError(
                    f'reply_hidden {tuple(reply_hidden.shape)} '
                    f'and reply_embed {tuple(reply_embed.shape)} '
                    'must have identical shape.')
            else:
                thinker_signal = reply_hidden + reply_embed
        elif signal_type == 'logits_soft_embed':
            raise NotImplementedError(
                'thinker_signal_type=logits_soft_embed is reserved for the V2 '
                'ablation and not implemented on this branch.')
        else:
            raise ValueError(f'Unknown thinker_signal_type: {signal_type!r}')

        # ── 3. adapter -> text_projection ──────────────────────────────────
        adp_dtype = next(self.thinker_adapter.parameters()).dtype
        adapted = self.thinker_adapter(thinker_signal.to(adp_dtype))
        projected = talker.text_projection(adapted.to(next(talker.text_projection.parameters()).dtype))
        # projected: [B, T_reply, H_talker]
        t_reply = projected.size(1)

        # ── 4. Per-sample reply length (how many text tokens to scatter) ──
        if text_labels is not None and max_reply > 0:
            text_valid_lens = reply_lens_tensor
        elif target_text_ids is not None:
            text_valid_lens = (target_text_ids != cfg.talker_config.tts_pad_token_id).long().sum(dim=-1)
        elif thinker_attention_mask is not None:
            text_valid_lens = thinker_attention_mask.to(torch.long).sum(dim=-1)
        else:
            text_valid_lens = torch.full((bsz,), t_reply, dtype=torch.long,
                                         device=projected.device)

        # ── 5. Build ``tts_pad_embed`` ─────────────────────────────────────
        pad_id = int(cfg.talker_config.tts_pad_token_id)
        pad_emb_1d = talker.text_projection(
            talker.model.text_embedding(
                torch.tensor([pad_id], device=projected.device, dtype=torch.long)
            ).to(next(talker.text_projection.parameters()).dtype)
        )
        pad_emb_1d = pad_emb_1d.to(codec_dtype).squeeze(0)                # [H_talker]

        # ── 6. Scatter reply text into a length-``codec_len`` buffer ───────
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

        # ── 7. Full-add layout: fill the thinker prompt signal + text_bos ──
        # (Qwen2.5-Omni ``talker_inputs_embeds = cat([prompt_part, text_bos,
        #  reply_part])``.) The prompt positions ``[0:P_b]`` carry the projected
        # thinker prompt ``(hidden + embed)`` and position ``P_b`` (= the
        # ``codec_pad`` frame right before ``codec_bos``) carries ``text_bos``.
        # Positions ``[codec_bos_index:]`` were already filled with the reply
        # signal above; everything else stays ``tts_pad_embed``.
        if full_add_prompt_len is not None:
            adp_dtype = next(self.thinker_adapter.parameters()).dtype
            tp_dtype = next(talker.text_projection.parameters()).dtype
            signal_type = getattr(cfg, 'thinker_signal_type', 'hidden_plus_embed')
            bos_id = int(getattr(cfg.talker_config, 'tts_bos_token_id', 151_672))
            bos_emb_1d = talker.text_projection(
                talker.model.text_embedding(
                    torch.tensor([bos_id], device=projected.device, dtype=torch.long)
                ).to(tp_dtype)
            ).to(codec_dtype).squeeze(0)                          # [H_talker]
            fap = full_add_prompt_len.to(projected.device, dtype=torch.long)
            for b in range(bsz):
                P = max(0, min(int(fap[b].item()), codec_len))
                if P > 0:
                    ph = thinker_hidden_states[b, :P]
                    if thinker_text_embeds is not None and signal_type == 'hidden_plus_embed':
                        ph = ph + thinker_text_embeds[b, :P]
                    adapted_p = self.thinker_adapter(ph.to(adp_dtype))
                    proj_p = talker.text_projection(adapted_p.to(tp_dtype)).to(codec_dtype)
                    text_channel[b, :P] = proj_p
                bos_pos = int(codec_bos_index[b].item()) - 1
                if 0 <= bos_pos < codec_len:
                    text_channel[b, bos_pos] = bos_emb_1d

        return text_channel

    def forward(
        self,
        # thinker inputs (any one path)
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # pre-computed thinker outputs (overrides on-the-fly run)
        thinker_text_embeds: Optional[torch.Tensor] = None,
        thinker_hidden_states: Optional[torch.Tensor] = None,
        thinker_attention_mask: Optional[torch.Tensor] = None,
        semantic_route_mask: Optional[torch.Tensor] = None,
        # talker / codec route
        codec_input_ids: Optional[torch.LongTensor] = None,
        codec_attention_mask: Optional[torch.Tensor] = None,
        codec_labels: Optional[torch.LongTensor] = None,
        codec_residual_labels: Optional[torch.LongTensor] = None,
        has_codec_supervision: Optional[torch.Tensor] = None,
        # Upstream-input protocol: per-sample target text token ids for the
        # talker text channel. Shape ``[B, T_text]`` with ``tts_pad_token_id``
        # padding. When present *and* ``config.use_upstream_talker_input=True``
        # the forward switches to the upstream Qwen3-TTS input protocol.
        target_text_ids: Optional[torch.LongTensor] = None,
        # Zero-indexed position within ``codec_input_ids`` from which the
        # talker starts consuming ``target_text_ids`` one token per frame
        # (i.e. the codec_bos position; positions strictly before it feed only
        # ``tts_pad_embed`` into the text channel). Shape ``[B]``.
        codec_bos_index: Optional[torch.Tensor] = None,
        # Full-add layout (Qwen2.5-Omni): per-sample thinker prompt length, i.e.
        # the number of leading ``codec_mask`` positions whose text channel is
        # filled with the thinker prompt signal (``text_bos`` sits at
        # ``codec_bos_index - 1``). ``None`` keeps the reply-add layout.
        full_add_prompt_len: Optional[torch.Tensor] = None,
        # Thinker text labels (upstream mode only): the original thinker text
        # labels (reply = target_text positions, -100 elsewhere) before the
        # collator overwrites ``labels`` with codec-aligned labels. Passed as a
        # separate field so the forward can compute ``token_text_acc`` and
        # identify reply positions for the upstream text channel, while the main
        # ``labels`` stream stays codec-aligned (identical to the original
        # bridge-concat behaviour for loss scaling and swift's _compute_acc).
        thinker_text_labels: Optional[torch.LongTensor] = None,
        # streaming / simultaneous-interpretation read-write schedule.
        # shape ``[codec_len]`` or ``[B, codec_len]``; per codec frame, how many
        # leading semantic tokens it may attend to. ``None`` (default) keeps the
        # offline full-context behaviour, so all existing stages are unaffected.
        codec_to_sem_visible: Optional[torch.Tensor] = None,
        # voice-clone speaker prompt (x-vector injection). ``reference_audio_wav``
        # is a per-sample list of mono waveforms (variable length), loaded by the
        # dataloader; resample / mel / ECAPA encoder all run here in forward. The
        # x-vector replaces the reserved speaker-slot embedding in the codec
        # control prefix (see ``speaker_slot_index``).
        # ``speaker_embedding`` is an optional offline-precomputed x-vector list
        # (same dim as ECAPA); when present it skips online extract.
        reference_audio_wav: Optional[List[Any]] = None,
        reference_audio_sr: Optional[List[int]] = None,
        speaker_embedding: Optional[List[Any]] = None,
        speaker_slot_index: Optional[torch.Tensor] = None,
        # Trainer compatibility
        labels: Optional[torch.LongTensor] = None,
        loss_scale: Optional[torch.Tensor] = None,
        # misc
        run_thinker: bool = False,
        return_dict: bool = True,
        **thinker_kwargs,
    ) -> QwenOmniSpeakOutput:
        # --- 0. Argument normalisation -----------------------------------
        # Bridge alignment template legacy path: when codec_labels is missing
        # but ``labels`` and ``codec_input_ids`` line up, treat ``labels`` as
        # codec labels (keeps Stage 1 behaviour).
        #
        # In upstream mode the collator always sets ``labels = acc_labels``
        # (codec-aligned) and passes the original thinker text labels via the
        # separate ``thinker_text_labels`` field. We prefer that field for all
        # text-related computations (text_loss, token_text_acc, reply-position
        # identification) so the main ``labels`` stream stays codec-aligned —
        # identical to the original bridge-concat behaviour for swift's loss
        # scaling and _compute_acc.
        text_labels = thinker_text_labels if thinker_text_labels is not None else labels
        if (codec_labels is not None and labels is not None and has_codec_supervision is None
                and labels.shape == codec_labels.shape):
            if thinker_text_labels is None:
                text_labels = None
        elif (codec_labels is None and labels is not None and codec_input_ids is not None
              and labels.shape == codec_input_ids.shape):
            codec_labels = labels
            if thinker_text_labels is None:
                text_labels = None
        # ── ICL mode: pop reference (source) codec + text before discarding ──
        # Enabled by env QWEN_OMNISPEAK_USE_ICL=true (default off → x-vector only).
        _icl_enabled = os.environ.get("QWEN_OMNISPEAK_USE_ICL", "false").lower() in ("1", "true", "yes")
        icl_source_codec = thinker_kwargs.pop("source_codec_codes", None)
        icl_source_text_ids = thinker_kwargs.pop("source_text_ids", None)
        icl_source_codec_mask = thinker_kwargs.pop("source_codec_mask", None)
        icl_source_text_mask = thinker_kwargs.pop("source_text_mask", None)
        for _ignored_key in (
                'loss_scale', 'num_items_in_batch', 'bridge_task', 'target_audio', 'target_text',
                'task_prompt', 'source_lang', 'target_lang', 'speaker_profile', 'reference_audios',
                'speaker_spans', 'latency_budget_ms', 'schema_version',
                'target_text_ids', 'codec_bos_index', 'source_codec_len',
                # Extra-field leftovers (collator already extracted the tensors).
                'source_speaker_embedding', 'source_xvector', 'source_input_features',
                'source_input_features_frames', 'source_feature_attention_mask'):
            thinker_kwargs.pop(_ignored_key, None)
        if thinker_attention_mask is None:
            thinker_attention_mask = attention_mask

        # --- 1. Thinker side ---------------------------------------------
        text_logits: Optional[torch.Tensor] = None
        if thinker_text_embeds is None and thinker_hidden_states is None:
            if not run_thinker:
                raise ValueError(
                    'Either provide thinker_text_embeds/thinker_hidden_states, '
                    'or set run_thinker=True with input_ids/inputs_embeds.')
            thinker_out = self.model.run_thinker(
                input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, **thinker_kwargs)
            # Extract thinker outputs from the returned dict

            thinker_hidden_states = thinker_out['hidden_states']
            thinker_text_embeds = thinker_out['text_embeds']
            text_logits = thinker_out['logits']
            # Fallback: if thinker_text_embeds was None (e.g. input_ids was
            # not passed to run_thinker — the multimodal/audio path feeds the
            # thinker via ``inputs_embeds`` so input_ids is None), reconstruct
            # the token/merged embeddings so the upstream input protocol can
            # compute hidden+embed. ``inputs_embeds`` here already contains the
            # audio-merged prompt embeddings (== Qwen2.5-Omni ``token_embeds[0]``).
            if thinker_text_embeds is None and input_ids is not None:
                thinker_text_embeds = self.thinker.get_input_embeddings()(input_ids)
            elif thinker_text_embeds is None and inputs_embeds is not None:
                thinker_text_embeds = inputs_embeds

        # --- 2. Text loss (thinker text CE) -------------------------------
        loss: Optional[torch.Tensor] = None
        acc_metrics: Dict[str, torch.Tensor] = {}
        if text_labels is not None and text_logits is not None and text_logits.shape[:2] == text_labels.shape:
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = text_labels[:, 1:].contiguous()
            text_loss = _cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            # Only add text_loss to the total loss when it carries gradient
            # (i.e. the thinker is trainable). When the thinker is frozen
            # (runs under @torch.no_grad), text_loss has no grad_fn and adding
            # it would inflate the reported loss without contributing gradient
            # — and it would also cause swift's loss scaling to use text token
            # counts instead of codec token counts, slightly perturbing the
            # gradient magnitude.  Accuracy is always computed for metrics.
            # ``text_loss_enabled`` / ``text_loss_weight`` (config + env) allow
            # disabling or scaling the thinker CE independently.
            text_loss_enabled, text_loss_weight = _text_loss_settings(self.config)
            if (text_loss_enabled and torch.isfinite(text_loss)
                    and text_loss.requires_grad):
                loss = text_loss_weight * text_loss
            text_correct = _token_correctness(shift_logits, shift_labels)
            if text_correct is not None:
                acc_metrics['token_text_acc'] = text_correct
            if os.environ.get('QWEN_DEBUG_TEXT_ACC'):
                _p = shift_logits[0].argmax(dim=-1)
                _m = shift_labels[0] != -100
                _nc = int((_p[_m] == shift_labels[0][_m]).sum())
                _nt = int(_m.sum())
                try:
                    from transformers import AutoTokenizer as _AT
                    _tp = os.environ.get('QWEN_THINKER_TOKENIZER', 'models/qwen2_5_3b_omnispeak_a')
                    _tk = _AT.from_pretrained(_tp, trust_remote_code=True)
                    print(f'[DBG-TXT] acc={_nc/max(1,_nt):.4f} ({_nc}/{_nt})', flush=True)
                    print(f'[DBG-TXT] TARGET: {_tk.decode(shift_labels[0][_m].cpu().tolist(), skip_special_tokens=True)[:250]}', flush=True)
                    print(f'[DBG-TXT] PRED:   {_tk.decode(_p[_m].cpu().tolist(), skip_special_tokens=True)[:250]}', flush=True)
                except Exception as _e:
                    print(f'[DBG-TXT] acc={_nc/max(1,_nt):.4f} ({_nc}/{_nt}) err={_e}', flush=True)

        # --- 3. Codec route (only when codec_input_ids is provided) ------
        main_codec_logits: Optional[torch.Tensor] = None
        residual_codec_logits: Optional[torch.Tensor] = None
        last_hidden: Optional[torch.Tensor] = None
        sem_block: Optional[torch.Tensor] = None

        if codec_input_ids is not None:
            if self.training and codec_labels is not None:
                codec_input_ids = _corrupt_codec_history(
                    codec_input_ids,
                    codec_labels.to(codec_input_ids.device),
                    probability=_codec_history_corruption_prob(self.config),
                    acoustic_vocab_size=int(getattr(
                        self.config.talker_config, 'code_predictor_vocab_size', 2048)),
                )
            _use_upstream_input = bool(getattr(self.config, 'use_upstream_talker_input', False))

            # ── Common: build codec_emb (+ speaker slot substitution) ──
            codec_emb = self.talker.model.embed_codec(codec_input_ids)
            # Voice-clone: replace the reserved speaker-slot embedding with the
            # frozen Qwen3-TTS ECAPA x-vector. Shared helper keeps DeepSpeed and
            # Megatron numerically aligned at this exact injection point.
            # Prefers offline speaker_embedding (Lance / source_xvector);
            # falls back to online extract from reference_audio_wav.
            codec_emb = apply_speaker_prompt_to_codec_emb(
                codec_emb,
                config=self.config,
                speaker_slot_index=speaker_slot_index,
                speaker_embedding=speaker_embedding,
                reference_audio_wav=reference_audio_wav,
                reference_audio_sr=reference_audio_sr,
            )

            codec_len = codec_emb.size(1)
            if codec_attention_mask is not None:
                codec_keep_mask = codec_attention_mask.to(device=codec_emb.device, dtype=torch.bool)
            else:
                codec_keep_mask = torch.ones(codec_emb.size(0), codec_len,
                                             dtype=torch.bool, device=codec_emb.device)

            if _use_upstream_input:
                # ═════════════════════════════════════════════════════════
                #  Upstream Qwen3-TTS input protocol (V1: hidden+embed).
                #  inputs_embeds[t] = text_projection(adapter(thinker_signal))[t]
                #                     + codec_embedding(codec_input_ids[t])
                #  text side pads with tts_pad_embed outside the assistant span;
                #  see ideas/talker_upstream_experiment_plan_v2.md §4.
                # ═════════════════════════════════════════════════════════
                # Robustness: the template-emitted upstream scalar fields
                # (codec_bos_index / full_add_prompt_len) are dropped by the
                # swift dataset pipeline before collation, so we derive them
                # here from surviving tensors instead of trusting the kwargs.
                #   * codec_bos_index = position of ``codec_bos_id`` in codec_input_ids
                #   * full_add_prompt_len (full-add) = first reply position in the
                #     thinker text labels (== the codec_mask prefix length)
                if codec_bos_index is None and codec_input_ids is not None:
                    _bos = int(getattr(self.config, 'codec_bos_id', 2149))
                    _hits = (codec_input_ids == _bos)
                    _bidx = []
                    for _b in range(codec_input_ids.size(0)):
                        _nz = _hits[_b].nonzero(as_tuple=False)
                        _bidx.append(int(_nz[0].item()) if _nz.numel() > 0 else 0)
                    codec_bos_index = torch.tensor(_bidx, dtype=torch.long,
                                                   device=codec_input_ids.device)
                if _full_add_layout(self.config) and full_add_prompt_len is None:
                    if text_labels is not None:
                        _fap = []
                        for _b in range(text_labels.size(0)):
                            _nz = (text_labels[_b] != -100).nonzero(as_tuple=False)
                            _fap.append(int(_nz[0].item()) if _nz.numel() > 0
                                        else int(text_labels.size(1)))
                        full_add_prompt_len = torch.tensor(_fap, dtype=torch.long,
                                                           device=text_labels.device)
                text_channel = self._build_upstream_text_channel(
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
                )                                                  # [B, T_codec, H]
                full_emb = codec_emb + text_channel
                # ── ICL prefix: prepend reference (source) codec+text segment ──
                # Mirrors Qwen3-TTS ``generate_icl_prompt``. The prefix conditions
                # the talker on the source speaker; it is NOT supervised, so we
                # slice it off the hidden states before the codec head/labels.
                _icl_len = 0
                if _icl_enabled and icl_source_codec is not None:
                    icl_embed, icl_mask = self._build_icl_prefix(
                        source_codec_codes=icl_source_codec,
                        source_text_ids=icl_source_text_ids,
                        source_codec_mask=icl_source_codec_mask,
                        source_text_mask=icl_source_text_mask,
                        codec_dtype=codec_emb.dtype, codec_device=codec_emb.device,
                    )
                    _icl_len = icl_embed.size(1)
                    full_emb = torch.cat([icl_embed, full_emb], dim=1)
                    codec_keep_mask = torch.cat([icl_mask, codec_keep_mask], dim=1)
                # Pass the 2-D padding mask; ``Qwen3TtsTalkerModel.forward``
                # superimposes the CAUSAL lower-triangular term (mirroring
                # upstream Qwen3-TTS ``create_causal_mask``) so position ``t``
                # cannot attend to future codec frames (no label leakage).
                talker_out = self.talker.model(inputs_embeds=full_emb,
                                               attention_mask=codec_keep_mask)
                last_hidden = talker_out.last_hidden_state
                assert last_hidden is not None
                # Drop the (unsupervised) ICL prefix positions → align with codec.
                if _icl_len > 0:
                    last_hidden = last_hidden[:, _icl_len:]
                    codec_keep_mask = codec_keep_mask[:, _icl_len:]
                codec_hidden = last_hidden                          # every position is a codec frame
                main_codec_logits = self.talker.codec_head(codec_hidden)
                # For diagnostics only; ``sem_block`` no longer exists in the
                # upstream input protocol so we surface ``text_channel``.
                sem_block = text_channel
            else:
                # ═════════════════════════════════════════════════════════
                #  Legacy DualRoute bridge + block-causal path (unchanged).
                # ═════════════════════════════════════════════════════════
                sem_block = self.bridge(
                    thinker_text_embeds=thinker_text_embeds,
                    thinker_hidden_states=thinker_hidden_states,
                    semantic_route_mask=semantic_route_mask,
                    thinker_hidden_attention_mask=thinker_attention_mask,
                )
                assert sem_block is not None
                full_emb = torch.cat([sem_block, codec_emb], dim=1)
                sem_len = sem_block.size(1)

                sem_keep_mask = thinker_attention_mask
                if sem_keep_mask is None and sem_len > 0:
                    sem_keep_mask = torch.ones(sem_block.size(0), sem_len, dtype=torch.bool,
                                               device=sem_block.device)
                elif sem_keep_mask is not None:
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

                attn_bias = build_block_causal_mask(
                    sem_len=sem_len, codec_len=codec_len,
                    sem_attention_mask=sem_keep_mask, codec_attention_mask=codec_keep_mask,
                    dtype=full_emb.dtype, device=full_emb.device,
                    codec_to_sem_visible=codec_to_sem_visible,
                )
                talker_out = self.talker.model(inputs_embeds=full_emb, attention_mask=attn_bias)
                last_hidden = talker_out.last_hidden_state
                assert last_hidden is not None
                codec_hidden = last_hidden[:, sem_len:, :]
                main_codec_logits = self.talker.codec_head(codec_hidden)
            # Residual code predictor: autoregressive roll-out along the
            # codebook-group dimension (teacher-forced with the ground-truth
            # lower groups), mirroring upstream Qwen3-TTS. Only run it when the
            # residual labels exist AND the residual loss is enabled (skipping
            # it when disabled saves the extra code-predictor forward).
            residual_loss_enabled, residual_loss_weight = _residual_loss_settings(self.config)
            residual_codec_logits = None
            if (residual_loss_enabled and codec_labels is not None
                    and codec_residual_labels is not None):
                # Guard against occasional collator/label shape mismatches that
                # would otherwise crash the whole DDP group with
                # ``reshape [B*T, G] invalid for input of size X`` inside
                # ``forward_teacher_forced``. When the residual labels' first
                # two dims don't match ``codec_hidden``, skip residual loss for
                # this micro-batch (main-code loss still counts) and log once.
                _ch_shape = tuple(codec_hidden.shape[:2])
                _cl_shape = tuple(codec_labels.shape[:2])
                _cr_shape = tuple(codec_residual_labels.shape[:2])
                _num_groups = getattr(self.config, 'num_code_groups', None)
                _cr_g = codec_residual_labels.shape[-1] if codec_residual_labels.ndim >= 3 else None
                if _ch_shape == _cl_shape == _cr_shape and (
                        _num_groups is None or _cr_g == _num_groups):
                    residual_codec_logits = self.talker.code_predictor.forward_teacher_forced(
                        talker_hidden=codec_hidden,
                        main_codes=codec_labels.to(codec_hidden.device),
                        residual_codes=codec_residual_labels.to(codec_hidden.device),
                        main_codec_embedding=self.talker.model.codec_embedding,
                    )
                else:
                    import warnings as _w
                    _w.warn(
                        f'[qwen_omnispeak] skipping residual loss due to shape '
                        f'mismatch: codec_hidden[:2]={_ch_shape} '
                        f'codec_labels[:2]={_cl_shape} '
                        f'codec_residual_labels[:2]={_cr_shape} '
                        f'residual_groups={_cr_g} expected={_num_groups}'
                    )

            # Codec loss: support per-row gating via has_codec_supervision so
            # mixed batches (continue-pretrain) only attribute codec loss to
            # rows that actually have target_audio.
            if codec_labels is not None:
                assert main_codec_logits is not None
                effective_labels = codec_labels.to(main_codec_logits.device)
                effective_residual = (codec_residual_labels.to(main_codec_logits.device)
                                      if codec_residual_labels is not None else None)
                if has_codec_supervision is not None:
                    keep = has_codec_supervision.to(device=effective_labels.device, dtype=torch.bool)
                    if keep.numel() == effective_labels.size(0):
                        mask_main = keep[:, None].expand_as(effective_labels)
                        effective_labels = effective_labels.masked_fill(~mask_main, -100)
                        if effective_residual is not None:
                            mask_res = keep[:, None, None].expand_as(effective_residual)
                            effective_residual = effective_residual.masked_fill(~mask_res, -100)
                main_correct = _token_correctness(main_codec_logits, effective_labels)
                if main_correct is not None:
                    acc_metrics['token_acc'] = main_correct
                with torch.no_grad():
                    eos_id = int(getattr(self.config, 'codec_eos_id', 2150))
                    main_predictions = main_codec_logits.argmax(dim=-1)
                    eos_boundary = effective_labels.eq(eos_id)
                    if bool(eos_boundary.any()):
                        acc_metrics['eos_boundary_top1_acc'] = (
                            main_predictions[eos_boundary].eq(eos_id).float().detach())
                    non_boundary = effective_labels.ne(-100) & ~eos_boundary
                    if bool(non_boundary.any()):
                        acc_metrics['premature_eos_top1_rate'] = (
                            main_predictions[non_boundary].eq(eos_id).float().detach())
                if residual_codec_logits is not None and effective_residual is not None:
                    resid_correct = _token_correctness(residual_codec_logits, effective_residual)
                    if resid_correct is not None:
                        acc_metrics['token_acoustic_token'] = resid_correct
                if (effective_labels != -100).any():
                    codec_loss = _weighted_codec_cross_entropy(
                        main_codec_logits,
                        effective_labels,
                        eos_id=int(getattr(self.config, 'codec_eos_id', 2150)),
                        eos_weight=_codec_eos_loss_weight(self.config),
                    )
                    if (residual_loss_enabled and residual_codec_logits is not None
                            and effective_residual is not None
                            and (effective_residual != -100).any()):
                        _, _, _, V = residual_codec_logits.shape
                        resid_loss = _cross_entropy(
                            residual_codec_logits.reshape(-1, V),
                            effective_residual.reshape(-1),
                            ignore_index=-100,
                        )
                        codec_loss = codec_loss + residual_loss_weight * resid_loss
                    loss = codec_loss if loss is None else (loss + codec_loss)

        if loss is None and (text_labels is not None or codec_labels is not None):
            # Zero-supervision fallback (e.g. a batch happens to have all -100
            # labels). Anchor the zero loss on the MAIN CODEC logits when the
            # codec route ran — the same tensor the Megatron codec-loss path
            # anchors on — so ``loss`` keeps a grad_fn through adapter/talker
            # and the ``grad is None`` semantics stay consistent across
            # backends (P1-4, fixed 2026-08; anchoring on text_logits first
            # produced a graph-less loss whenever the thinker ran frozen
            # under no_grad).
            ref = main_codec_logits if main_codec_logits is not None else text_logits
            if ref is not None:
                loss = ref.sum() * 0.0

        # Expose the right logits for the trainer's token-accuracy metric.
        # In codec-only mode (bridge alignment: no thinker-text supervision)
        # the meaningful accuracy is over the MAIN codec head, so surface
        # ``main_codec_logits`` as ``logits``. The bridge-alignment collator
        # provides ``labels`` aligned to the main codec stream (see template),
        # so swift's standard next-token shift yields the correct codec acc.
        # When thinker text supervision is active (continue-pretrain), keep
        # the text logits so text accuracy stays meaningful.
        returned_logits = text_logits
        if text_labels is None and main_codec_logits is not None:
            returned_logits = main_codec_logits

        # MFU accounting: accumulate this micro-batch's supervised-token count
        # onto the model-level counter (bypassing swift's MeanMetric, which
        # would average micro-batches instead of summing). Each
        # ``_token_correctness`` result is one 0/1 value per supervised token,
        # so ``.numel()`` == that route's supervised-token count.
        try:
            # MFU token accounting: only the 3 REAL supervised-token routes
            # (text / main-codec / residual-codec). eos_boundary_top1_acc and
            # premature_eos_top1_rate are drawn from the main-codec predictions,
            # i.e. they OVERLAP token_acc — summing all acc_metrics over-counts.
            _tok = 0
            _comp = {}
            for _k in ('token_text_acc', 'token_acc', 'token_acoustic_token'):
                _acc = acc_metrics.get(_k)
                if _acc is not None and _acc.dim() == 1 and _acc.numel() > 0:
                    _n = int(_acc.numel())
                    _tok += _n
                    _comp[_k] = _n
            if _tok > 0:
                self._mfu_tokens += _tok
            try:
                _mask = codec_attention_mask if codec_attention_mask is not None else None
                if _mask is not None:
                    from mfu_logging import accumulate_pad_waste
                    accumulate_pad_waste(_mask.sum(dim=-1).tolist())
            except Exception:
                pass
            # DIAG: print acc_metrics component token counts (text/main/residual)
            # once per micro-batch to validate MFU token accounting vs megatron.
            import os as _os
            if _os.environ.get('MFU_DIAG_SHAPES') == '1' and _comp:
                _s = f'[mfu-diag-ds] ' + ' '.join(f'{k}={v}' for k, v in _comp.items())
                _s += f' total={_tok}'
                print(_s, flush=True)
        except Exception:
            pass

        return QwenOmniSpeakOutput(
            loss=loss,
            logits=returned_logits,
            main_codec_logits=main_codec_logits,
            residual_codec_logits=residual_codec_logits,
            talker_hidden_states=last_hidden,
            semantic_block=sem_block,
            text_logits=text_logits,
            metrics=acc_metrics or None,
        )

    @torch.inference_mode()
    def make_cached_talker_session(
        self,
        *,
        thinker_text_embeds: torch.Tensor,
        thinker_hidden_states: torch.Tensor,
        thinker_attention_mask: torch.Tensor,
        thinker_text_labels: Optional[torch.Tensor],
        control_prefix_ids: torch.Tensor,
        speaker_slot_index: Optional[torch.Tensor] = None,
        source_xvector: Optional[torch.Tensor] = None,
        max_frames: int = 384,
    ):
        """INFERENCE-ONLY. Build a KV-cached decoding session for the upstream
        reply_add / full_add path and return a ``_CachedTalkerSession`` whose
        ``prefill()`` warms the cache over the control prefix (returning the
        first ``(main_logits, codec_hidden)``) and whose ``step(main_id)``
        advances one codec frame using the cache (O(1) attention vs the O(T^2)
        full recompute in ``forward``).

        Math mirrors ``forward`` (upstream branch): ``full_emb =
        codec_embedding(id) + text_channel``. It never touches ``forward`` or the
        training path. Assumes batch size 1 and no ICL prefix (source speaker via
        the x-vector slot inside ``control_prefix_ids``).
        """
        from transformers.cache_utils import DynamicCache

        cfg = self.config
        talker = self.talker
        device = control_prefix_ids.device
        assert control_prefix_ids.size(0) == 1, "cached talker session supports batch size 1"
        prefix_len = control_prefix_ids.size(1)
        codec_len = prefix_len + max_frames  # extra text_channel positions are pad_embed (harmless)

        codec_emb_prefix = talker.model.codec_embedding(control_prefix_ids)  # [1, prefix_len, H]
        codec_dtype = codec_emb_prefix.dtype
        if (_use_speaker_prompt(cfg) and speaker_slot_index is not None
                and source_xvector is not None):
            codec_emb_prefix = codec_emb_prefix.clone()
            xv = source_xvector.to(device=device, dtype=codec_dtype)
            slots = speaker_slot_index
            if not torch.is_tensor(slots):
                slots = torch.as_tensor(slots, dtype=torch.long)
            slots = slots.to(device)[:1]
            codec_emb_prefix[torch.arange(1, device=device), slots] = xv[:1]

        codec_bos_index = torch.tensor([prefix_len - 1], dtype=torch.long, device=device)
        full_add_prompt_len = None
        if _full_add_layout(cfg) and thinker_text_labels is not None:
            _nz = (thinker_text_labels[0] != -100).nonzero(as_tuple=False)
            _fap = int(_nz[0].item()) if _nz.numel() > 0 else int(thinker_text_labels.size(1))
            full_add_prompt_len = torch.tensor([_fap], dtype=torch.long, device=device)
        text_channel = self._build_upstream_text_channel(
            thinker_hidden_states=thinker_hidden_states,
            thinker_text_embeds=thinker_text_embeds,
            text_labels=thinker_text_labels,
            target_text_ids=None,
            codec_len=codec_len,
            codec_bos_index=codec_bos_index,
            thinker_attention_mask=thinker_attention_mask,
            codec_dtype=codec_dtype,
            codec_device=device,
            full_add_prompt_len=full_add_prompt_len,
        )  # [1, codec_len, H]

        return _CachedTalkerSession(
            talker=talker,
            cache=DynamicCache(),
            codec_emb_prefix=codec_emb_prefix,
            text_channel=text_channel,
            prefix_len=prefix_len,
            max_frames=max_frames,
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        thinker_config=None,
        **kwargs,
    ) -> 'QwenOmniSpeakForConditionalGeneration':
        cfg = QwenOmniSpeakConfig.from_variant(variant=variant, thinker_config=thinker_config,
                                                **kwargs)
        return cls(cfg)


class _CachedTalkerSession:
    """INFERENCE-ONLY helper holding the KV-cache + precomputed per-frame text
    channel for one utterance. Created by
    ``QwenOmniSpeakForConditionalGeneration.make_cached_talker_session``.

    Usage::

        session = model.make_cached_talker_session(...)
        main_logits, codec_hidden = session.prefill()
        # sample main_id from main_logits, predict residuals from codec_hidden
        while not done:
            main_logits, codec_hidden = session.step(main_id)
    """

    def __init__(self, *, talker, cache, codec_emb_prefix, text_channel,
                 prefix_len: int, max_frames: int):
        self.talker = talker
        self.cache = cache
        self.codec_emb_prefix = codec_emb_prefix        # [1, prefix_len, H]
        self.text_channel = text_channel                # [1, codec_len, H]
        self.prefix_len = int(prefix_len)
        self.max_frames = int(max_frames)
        self._frame_idx = self.prefix_len               # next codec position to write
        self._done = False

    @torch.inference_mode()
    def prefill(self):
        """Warm the cache over the control prefix; return first frame logits."""
        prefix_emb = self.codec_emb_prefix + self.text_channel[:, :self.prefix_len]
        out = self.talker.model.forward_cached(prefix_emb, past_key_values=self.cache)
        self.cache = out.past_key_values
        codec_hidden = out.last_hidden_state[:, -1:, :]
        main_logits = self.talker.codec_head(codec_hidden)
        return main_logits, codec_hidden

    @torch.inference_mode()
    def step(self, main_id: Union[int, torch.Tensor]):
        """Advance one codec frame given the sampled main id for the CURRENT
        position; return the next frame's ``(main_logits, codec_hidden)``."""
        if self._frame_idx >= self.prefix_len + self.max_frames:
            self._done = True
        device = self.text_channel.device
        if torch.is_tensor(main_id):
            ids = main_id.to(device=device, dtype=torch.long).reshape(1, 1)
        else:
            ids = torch.tensor([[int(main_id)]], dtype=torch.long, device=device)
        pos = min(self._frame_idx, self.text_channel.size(1) - 1)
        step_emb = self.talker.model.codec_embedding(ids) + self.text_channel[:, pos:pos + 1]
        out = self.talker.model.forward_cached(step_emb, past_key_values=self.cache)
        self.cache = out.past_key_values
        self._frame_idx += 1
        codec_hidden = out.last_hidden_state[:, -1:, :]
        main_logits = self.talker.codec_head(codec_hidden)
        return main_logits, codec_hidden


__all__ = [
    'DualRouteMLPBridge',
    'DualRouteQFormerBridge',
    'build_bridge',
    'Qwen3TtsCodePredictor',
    'Qwen3TtsTalkerModel',
    'Qwen3TtsTalkerForConditionalGeneration',
    'QwenOmniSpeakModel',
    'QwenOmniSpeakForConditionalGeneration',
    'QwenOmniSpeakOutput',
    'build_block_causal_mask',
]


# Backwards-compatible aliases for older exports/user code.
Qwen3OmniSpeakOutput = QwenOmniSpeakOutput
Qwen3OmniSpeakModel = QwenOmniSpeakModel
Qwen3OmniSpeakForConditionalGeneration = QwenOmniSpeakForConditionalGeneration
