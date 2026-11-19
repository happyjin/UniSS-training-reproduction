# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen-OmniSpeak configurations.

This file is intentionally self-contained (no relative imports) so that
``trust_remote_code=True`` exports can reference its classes via the strict
two-segment ``module.ClassName`` shorthand expected by ``transformers``.
"""
import os
from typing import Any, ClassVar, Dict, List, Optional, Union

from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
try:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniThinkerConfig,
    )
except Exception:  # pragma: no cover - older transformers without Qwen2.5-Omni.
    Qwen2_5OmniConfig = None
    Qwen2_5OmniThinkerConfig = None


QWEN3_THINKER_FAMILY = 'qwen3_omni_moe'
QWEN25_THINKER_FAMILY = 'qwen2_5_omni'
SUPPORTED_THINKER_FAMILIES = (QWEN3_THINKER_FAMILY, QWEN25_THINKER_FAMILY)


def infer_thinker_family(thinker_dict: Dict[str, Any], explicit: Optional[str] = None) -> str:
    if explicit in ('qwen3', 'qwen3_omni', 'qwen3_omni_moe'):
        return QWEN3_THINKER_FAMILY
    if explicit in ('qwen2_5', 'qwen2.5', 'qwen2_5_omni', 'qwen2.5_omni'):
        return QWEN25_THINKER_FAMILY
    if explicit:
        raise ValueError(f'Unsupported thinker_family: {explicit!r}. Expected one of {SUPPORTED_THINKER_FAMILIES}.')
    model_type = str(thinker_dict.get('model_type', '')).lower()
    if model_type in ('qwen2_5_omni', 'qwen2.5_omni'):
        return QWEN25_THINKER_FAMILY
    nested = thinker_dict.get('thinker_config')
    if isinstance(nested, dict):
        nested_model_type = str(nested.get('model_type', '')).lower()
        if nested_model_type in ('qwen2_5_omni_thinker', 'qwen2_5_omni_text') or 'audio_token_index' in nested:
            return QWEN25_THINKER_FAMILY
    if 'audio_token_index' in thinker_dict:
        return QWEN25_THINKER_FAMILY
    return QWEN3_THINKER_FAMILY


def default_hidden_state_selection(thinker_family: str) -> Union[int, str]:
    return 'last' if thinker_family == QWEN25_THINKER_FAMILY else 24


# ---------------------------------------------------------------------------
# DualRoute Bridge config
# ---------------------------------------------------------------------------


class DualRouteBridgeConfig(PretrainedConfig):
    """Config for the DualRoute Bridge module.

    Two implementation variants are supported:

    - ``mlp``      : DualRoute MLP Bridge (V1, default).
    - ``qformer``  : DualRoute Q-Former Bridge (V2).
    """

    model_type = 'qwen_omnispeak_bridge'

    def __init__(
        self,
        bridge_type: str = 'mlp',
        thinker_hidden_size: int = 2048,
        talker_hidden_size: int = 2048,
        intermediate_size: int = 8192,
        rms_norm_eps: float = 1e-6,
        # Q-Former specific
        num_query_tokens: int = 64,
        qformer_num_layers: int = 2,
        qformer_num_heads: int = 16,
        qformer_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bridge_type not in ('mlp', 'qformer'):
            raise ValueError(f'Unsupported bridge_type: {bridge_type!r}')
        self.bridge_type = bridge_type
        self.thinker_hidden_size = thinker_hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.num_query_tokens = num_query_tokens
        self.qformer_num_layers = qformer_num_layers
        self.qformer_num_heads = qformer_num_heads
        self.qformer_dropout = qformer_dropout


# ---------------------------------------------------------------------------
# Qwen3-TTS Talker config (Dense, 28 layers reference)
# ---------------------------------------------------------------------------


class Qwen3TtsTalkerConfig(Qwen3Config):
    """Config for the Qwen3-TTS Dense Talker backbone.

    The backbone shares the same dense Qwen3 transformer recipe, plus codec
    embedding/lm_head and a residual code predictor head.

    The ``num_hidden_layers`` field is shared with ``Qwen3Config``; combined
    with ``layer_selection`` it describes which slice of the original 28-layer
    Qwen3-TTS backbone is used.
    """

    model_type = 'qwen3_tts_talker'

    def __init__(
        self,
        # Qwen3-TTS-12Hz-1.7B-CustomVoice defaults
        vocab_size: int = 151_936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 32_768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        tie_word_embeddings: bool = False,
        # Codec / Talker-specific
        codec_vocab_size: int = 3072,
        # Residual code groups predicted by ``code_predictor`` *in addition to*
        # the main group produced by ``codec_head``. Qwen3-TTS-Base ships with
        # 15 residual groups; the design doc's ``16 = main + 15`` convention.
        num_code_groups: int = 15,
        # Code Predictor inner config — defaults match Qwen3-TTS-12Hz-1.7B-Base.
        code_predictor_hidden_size: int = 1024,
        code_predictor_intermediate_size: int = 3072,
        code_predictor_num_layers: int = 5,
        code_predictor_num_attention_heads: int = 16,
        code_predictor_num_key_value_heads: int = 8,
        code_predictor_head_dim: int = 128,
        code_predictor_vocab_size: int = 2048,
        # Whether the residual code predictor has its own per-group input
        # codec embedding tables (Qwen3-TTS-Base does; size 16 × [V, hidden]).
        code_predictor_has_codec_embedding: bool = True,
        # Layer selection: which slice of the original 28-layer backbone we keep.
        # ``mode`` ∈ {"full_28", "top_16", "top_12", "top_8", "top_4", "custom"}.
        # When ``mode == "custom"`` the explicit ``layer_indices`` is used.
        layer_selection_mode: str = 'full_28',
        layer_selection_indices: Optional[List[int]] = None,
        original_num_hidden_layers: int = 28,
        # ── Upstream Qwen3-TTS talker input protocol (feature/talker-upstream) ──
        # These fields describe the ORIGINAL Qwen3-TTS talker input format
        # (see ``docs/ideas/talker_upstream_experiment_plan_v2.md`` §4). They are
        # only consumed when ``QwenOmniSpeakConfig.use_upstream_talker_input``
        # is True; otherwise every field defaults to a value that mirrors the
        # historical bridge path and is safely ignored.
        text_hidden_size: int = 2048,   # dim of ``text_embedding`` (before text_projection)
        text_vocab_size: int = 151_936, # same as ``vocab_size`` (Qwen text vocab)
        # TTS control tokens (talker-side text vocab). ids taken from upstream
        # Qwen3-TTS-12Hz-1.7B-Base ``tokenizer_config.json``.
        tts_bos_token_id: int = 151_672,   # <tts_text_bos>
        tts_eos_token_id: int = 151_673,   # <tts_text_eod>
        tts_pad_token_id: int = 151_671,   # <tts_pad>
        # Wall-clock alignment used by upstream for the 3D M-RoPE. We stay
        # 1D-RoPE (see M-RoPE ≡ 1D-RoPE bit-perfect equivalence proof), so
        # this is stored for record only.
        position_id_per_seconds: int = 13,
        # Speaker prompt config (matches upstream Qwen3-TTS speaker slot layout).
        spk_id: int = 0,
        spk_is_dialect: bool = False,
        # Explicit rope_scaling (upstream sets ``interleaved`` M-RoPE); we
        # keep 1D-RoPE at runtime but store the field so a full HF from_dict
        # round-trip preserves the original config.
        rope_scaling: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # transformers>=5.x exposes RoPE as ``rope_parameters``; handle both
        # old (``rope_theta``) and new APIs.
        if 'rope_parameters' not in kwargs:
            kwargs['rope_parameters'] = {'rope_theta': rope_theta, 'rope_type': 'default'}
        # transformers>=5.x ``Qwen3Config`` validates that ``len(layer_types) ==
        # num_hidden_layers``. When the config is reloaded from a JSON dict the
        # ``layer_types`` key may still describe the *original* 28-layer backbone
        # while ``num_hidden_layers`` is the *sliced* count. Drop any inherited
        # ``layer_types`` so it gets regenerated from the new ``num_hidden_layers``.
        kwargs.pop('layer_types', None)
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.rope_theta = rope_theta
        self.codec_vocab_size = codec_vocab_size
        self.num_code_groups = num_code_groups
        self.code_predictor_hidden_size = code_predictor_hidden_size
        self.code_predictor_intermediate_size = code_predictor_intermediate_size
        self.code_predictor_num_layers = code_predictor_num_layers
        self.code_predictor_num_attention_heads = code_predictor_num_attention_heads
        self.code_predictor_num_key_value_heads = code_predictor_num_key_value_heads
        self.code_predictor_head_dim = code_predictor_head_dim
        self.code_predictor_vocab_size = code_predictor_vocab_size
        self.code_predictor_has_codec_embedding = code_predictor_has_codec_embedding
        self.layer_selection_mode = layer_selection_mode
        self.layer_selection_indices = layer_selection_indices
        self.original_num_hidden_layers = original_num_hidden_layers
        # Upstream Qwen3-TTS fields (only used when
        # ``QwenOmniSpeakConfig.use_upstream_talker_input`` is True).
        self.text_hidden_size = text_hidden_size
        self.text_vocab_size = text_vocab_size
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.spk_id = spk_id
        self.spk_is_dialect = spk_is_dialect
        if rope_scaling is not None:
            self.rope_scaling = rope_scaling

    # Convenience -----------------------------------------------------------------

    def resolved_layer_indices(self) -> List[int]:
        """Return the absolute indices into the original 28-layer Qwen3-TTS backbone
        that this talker keeps.

        - ``full_28`` -> [0, 1, ..., 27]
        - ``top_16``  -> [12, 13, ..., 27]
        - ``top_12``  -> [16, 17, ..., 27]
        - ``top_8``   -> [20, 21, ..., 27]
        - ``top_4``   -> [24, 25, ..., 27]
        - ``custom``  -> ``layer_selection_indices``
        """
        n = self.original_num_hidden_layers
        mode = self.layer_selection_mode
        if mode == 'full_28':
            return list(range(n))
        if mode == 'top_16':
            return list(range(n - 16, n))
        if mode == 'top_12':
            return list(range(n - 12, n))
        if mode == 'top_8':
            return list(range(n - 8, n))
        if mode == 'top_4':
            return list(range(n - 4, n))
        if mode == 'custom':
            if not self.layer_selection_indices:
                raise ValueError('layer_selection_mode=custom requires layer_selection_indices')
            return list(self.layer_selection_indices)
        raise ValueError(f'Unknown layer_selection_mode: {mode!r}')


# ---------------------------------------------------------------------------
# Top-level Qwen-OmniSpeak config
# ---------------------------------------------------------------------------


class QwenOmniSpeakConfig(PretrainedConfig):
    """Top-level config for the Qwen-OmniSpeak composite model."""

    model_type = 'qwen_omnispeak'
    is_composition = True

    # The official variants registered both on ms-swift side and via
    # ``ConfigDict.from_variant``. ``ClassVar`` prevents ``transformers>=5.x``
    # from treating this dict as a dataclass field.
    SUPPORTED_VARIANTS: ClassVar[Dict[str, Dict[str, Any]]] = {
        'A': {'bridge_type': 'mlp', 'layer_selection_mode': 'full_28'},
        'B': {'bridge_type': 'mlp', 'layer_selection_mode': 'top_16'},
        'C': {'bridge_type': 'mlp', 'layer_selection_mode': 'top_12'},
        'D': {'bridge_type': 'mlp', 'layer_selection_mode': 'top_8'},
        'E': {'bridge_type': 'mlp', 'layer_selection_mode': 'top_4'},
        'F': {'bridge_type': 'qformer', 'layer_selection_mode': 'full_28'},
        'G': {'bridge_type': 'qformer', 'layer_selection_mode': 'top_16'},
        'H': {'bridge_type': 'qformer', 'layer_selection_mode': 'top_12'},
        'I': {'bridge_type': 'qformer', 'layer_selection_mode': 'top_8'},
        'J': {'bridge_type': 'qformer', 'layer_selection_mode': 'top_4'},
    }

    def __init__(
        self,
        thinker_config: Optional[Union[Dict[str, Any], PretrainedConfig]] = None,
        bridge_config: Optional[Union[Dict[str, Any], DualRouteBridgeConfig]] = None,
        talker_config: Optional[Union[Dict[str, Any], Qwen3TtsTalkerConfig]] = None,
        accept_hidden_layer: Optional[Union[int, str]] = None,
        hidden_state_selection: Optional[Union[int, str]] = None,
        thinker_family: Optional[str] = None,
        thinker_model_id: Optional[str] = None,
        talker_model_id: Optional[str] = None,
        variant: str = 'B',
        codec_pad_id: int = 2148,
        codec_bos_id: int = 2149,
        codec_eos_id: int = 2150,
        # Talker control tokens (Qwen3-TTS-12Hz-1.7B-Base defaults). These are
        # the "mode-switch" tokens the talker saw at the start of every codec
        # sequence during its own pretraining: think / nothink markers plus a
        # per-language id. We prepend them before ``codec_bos`` so the talker
        # enters its familiar conditioning state. They are NOT supervised
        # (their label positions are set to -100).
        codec_think_id: int = 2154,
        codec_nothink_id: int = 2155,
        codec_think_bos_id: int = 2156,
        codec_think_eos_id: int = 2157,
        # Qwen2.5-Omni "full-add" layout: the codec control prefix that aligns
        # position-wise with the thinker prompt is a run of ``codec_mask`` ids
        # (one per thinker prompt token), followed by ``codec_pad`` + ``codec_bos``
        # — see ``full_add_layout`` below. Qwen3-TTS-12Hz has no dedicated mask
        # token, so we reuse ``codec_pad_id`` by default (these prefix positions
        # are never supervised; the real conditioning comes from the added
        # thinker prompt signal, mirroring Qwen2_5OmniTalker's ``codec_mask``).
        codec_mask_id: int = 2148,
        codec_language_id: Optional[Dict[str, int]] = None,
        # Residual (code-predictor) loss controls. ``residual_loss_enabled``
        # toggles the 15-group residual CE on/off; ``residual_loss_weight``
        # scales it relative to the main-code CE (total = main + weight*resid).
        # Both can be overridden at runtime via env vars
        # ``QWEN_OMNISPEAK_RESIDUAL_LOSS`` / ``QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT``.
        residual_loss_enabled: bool = True,
        residual_loss_weight: float = 0.3,
        # Main-codec EOS supervision and codec-history corruption. Defaults
        # preserve the historical training behavior. Runtime env overrides:
        # ``QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT`` and
        # ``QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB``.
        codec_eos_loss_weight: float = 1.0,
        codec_history_corruption_prob: float = 0.0,
        # Thinker text CE loss controls. ``text_loss_enabled`` toggles the
        # thinker-side next-token CE on/off; ``text_loss_weight`` scales it
        # relative to the codec loss (total = weight*text + codec). The text
        # loss only contributes gradient when the thinker is trainable
        # (unfrozen); when frozen it is reported as a metric only. Both can be
        # overridden at runtime via env vars ``QWEN_OMNISPEAK_TEXT_LOSS`` /
        # ``QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT``.
        text_loss_enabled: bool = True,
        text_loss_weight: float = 1.0,
        tie_codec_embeddings: bool = False,
        # Voice-clone speaker prompt (x-vector injection). When enabled, the
        # control prefix reserves one ``speaker slot`` (right after
        # ``codec_think_eos``) whose embedding is replaced in ``forward`` by the
        # frozen Qwen3-TTS ECAPA x-vector of the reference audio — mirroring the
        # upstream Qwen3-TTS speaker-conditioning position. The encoder itself is
        # loaded lazily outside the composite (see ``custom/speaker_encoder.py``);
        # nothing about the talker/bridge structure changes. Off by default so
        # existing runs are unaffected. Can be toggled at runtime via the env var
        # ``QWEN_OMNISPEAK_USE_SPEAKER_PROMPT``.
        use_speaker_prompt: bool = False,
        speaker_embedding_dim: int = 2048,
        speaker_encoder_sample_rate: int = 24000,
        # ── Upstream Qwen3-TTS talker input path (feature/talker-upstream) ──
        # When True, the composite forward switches to the *upstream* Qwen3-TTS
        # talker input protocol:
        #   inputs_embeds[t] = text_projection(text_signal[t]) + codec_embedding(codec_id[t])
        #   plus the upstream 9-position prefill assembly (role prefix + control
        #   tokens + trailing_text_hidden), skipping the DualRoute bridge and
        #   the block-causal ``[sem ; codec]`` concatenation entirely.
        # When False (default), the forward keeps the original bridge path so
        # all pre-existing checkpoints / training scripts on other branches
        # continue to work unchanged.
        use_upstream_talker_input: bool = False,
        # Qwen2.5-Omni "full-add" layout (feature/talker-full-add). Only takes
        # effect when ``use_upstream_talker_input=True``. When enabled the talker
        # text channel is assembled *exactly* like Qwen2_5OmniForConditionalGeneration:
        #   text  : [ prompt_hidden+embed (T_prompt) , text_bos , reply_hidden+embed , pad... ]
        #   codec : [ codec_mask × T_prompt          , codec_pad , codec_bos , frames...     ]
        # and the two channels are added position-wise
        # (``inputs_embeds = codec_embeds + thinker_reply_part``). This differs
        # from the reply-add layout, which fills the pre-``codec_bos`` positions
        # with ``tts_pad_embed`` (reply only). Can be toggled at runtime via the
        # env var ``QWEN_OMNISPEAK_FULL_ADD_LAYOUT``.
        full_add_layout: bool = False,
        # Thinker signal variant fed into the upstream text channel. Only
        # ``hidden_plus_embed`` is implemented in this branch (V1).
        # ``logits_soft_embed`` is reserved for a future V2 ablation.
        thinker_signal_type: str = 'hidden_plus_embed',
        # ThinkerAdapter (H_thinker -> H_text=2048) — a 2-layer MLP with GELU
        # + LayerNorm, initialised with the Qwen family default
        # ``trunc_normal_(std=0.02)``.
        adapter_hidden_size: Optional[int] = None,  # defaults to 2×talker.text_hidden_size
        adapter_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, PretrainedConfig):
            thinker_dict = thinker_config.to_dict()
        else:
            thinker_dict = dict(thinker_config)
        self.thinker_family = infer_thinker_family(thinker_dict, thinker_family)
        if self.thinker_family == QWEN25_THINKER_FAMILY and (Qwen2_5OmniConfig is None or Qwen2_5OmniThinkerConfig is None):
            raise ImportError('Qwen2.5-Omni thinker requires transformers with qwen2_5_omni support.')
        # Heuristic: a full Omni config has nested ``thinker_config`` + ``talker_config`` fields.
        if 'thinker_config' in thinker_dict and 'talker_config' in thinker_dict:
            if self.thinker_family == QWEN25_THINKER_FAMILY:
                self.thinker_config = Qwen2_5OmniConfig(**thinker_dict)  # type: ignore[misc]
            else:
                self.thinker_config = Qwen3OmniMoeConfig(**thinker_dict)
            self._thinker_is_full_omni = True
        else:
            if self.thinker_family == QWEN25_THINKER_FAMILY:
                self.thinker_config = (
                    Qwen2_5OmniThinkerConfig(**thinker_dict) if thinker_dict else Qwen2_5OmniThinkerConfig()  # type: ignore[misc]
                )
            else:
                self.thinker_config = Qwen3OmniMoeThinkerConfig(**thinker_dict) if thinker_dict else Qwen3OmniMoeThinkerConfig()
            self._thinker_is_full_omni = False

        if bridge_config is None:
            bridge_config = {}
        self.bridge_config = bridge_config if isinstance(bridge_config, PretrainedConfig) else DualRouteBridgeConfig(**bridge_config)

        if talker_config is None:
            talker_config = {}
        self.talker_config = talker_config if isinstance(talker_config, PretrainedConfig) else Qwen3TtsTalkerConfig(**talker_config)

        if hidden_state_selection is None:
            hidden_state_selection = accept_hidden_layer
        if hidden_state_selection is None:
            hidden_state_selection = default_hidden_state_selection(self.thinker_family)
        self.hidden_state_selection = hidden_state_selection
        self.accept_hidden_layer = hidden_state_selection
        self.thinker_model_id = thinker_model_id
        self.talker_model_id = talker_model_id
        self.variant = variant
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.codec_eos_id = codec_eos_id
        self.codec_think_id = codec_think_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_mask_id = codec_mask_id
        # Default language->codec-id map mirrors Qwen3-TTS-12Hz-1.7B-Base.
        self.codec_language_id = codec_language_id if codec_language_id is not None else {
            'chinese': 2055, 'english': 2050, 'german': 2053, 'italian': 2070,
            'portuguese': 2071, 'spanish': 2054, 'japanese': 2058, 'korean': 2064,
            'french': 2061, 'russian': 2069,
        }
        self.tie_codec_embeddings = tie_codec_embeddings
        self.residual_loss_enabled = residual_loss_enabled
        self.residual_loss_weight = residual_loss_weight
        if codec_eos_loss_weight <= 0:
            raise ValueError('codec_eos_loss_weight must be > 0.')
        if not 0.0 <= codec_history_corruption_prob <= 1.0:
            raise ValueError('codec_history_corruption_prob must be in [0, 1].')
        self.codec_eos_loss_weight = float(codec_eos_loss_weight)
        self.codec_history_corruption_prob = float(codec_history_corruption_prob)
        self.text_loss_enabled = text_loss_enabled
        self.text_loss_weight = text_loss_weight
        self.use_speaker_prompt = use_speaker_prompt
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_encoder_sample_rate = speaker_encoder_sample_rate
        # Upstream Qwen3-TTS talker input protocol switch + adapter config.
        # This runtime override lets all input-layout arms share one base
        # checkpoint without duplicating its multi-gigabyte weight files.
        upstream_env = os.environ.get('QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT')
        if upstream_env is not None:
            use_upstream_talker_input = upstream_env.lower() in {
                '1', 'true', 'yes', 'y', 'on'
            }
        self.use_upstream_talker_input = use_upstream_talker_input
        self.full_add_layout = full_add_layout
        if thinker_signal_type not in ('hidden_plus_embed', 'logits_soft_embed'):
            raise ValueError(f'Unsupported thinker_signal_type: {thinker_signal_type!r}')
        self.thinker_signal_type = thinker_signal_type
        self.adapter_hidden_size = adapter_hidden_size
        self.adapter_dropout = adapter_dropout

    @classmethod
    def from_variant(
        cls,
        variant: str,
        thinker_config: Union[Dict[str, Any], PretrainedConfig, None] = None,
        talker_overrides: Optional[Dict[str, Any]] = None,
        bridge_overrides: Optional[Dict[str, Any]] = None,
        thinker_family: Optional[str] = None,
        thinker_model_id: Optional[str] = None,
        talker_model_id: Optional[str] = None,
        **kwargs,
    ) -> 'QwenOmniSpeakConfig':
        if variant not in cls.SUPPORTED_VARIANTS:
            raise ValueError(
                f'Unknown variant {variant!r}. Expected one of {sorted(cls.SUPPORTED_VARIANTS)}.')
        spec = cls.SUPPORTED_VARIANTS[variant]
        bridge_kwargs = {'bridge_type': spec['bridge_type']}
        if bridge_overrides:
            bridge_kwargs.update(bridge_overrides)
        talker_kwargs = {'layer_selection_mode': spec['layer_selection_mode']}
        if talker_overrides:
            talker_kwargs.update(talker_overrides)
        talker_cfg = Qwen3TtsTalkerConfig(**talker_kwargs)
        talker_cfg.num_hidden_layers = len(talker_cfg.resolved_layer_indices())
        bridge_cfg = DualRouteBridgeConfig(**bridge_kwargs)
        return cls(
            thinker_config=thinker_config,  # pyright: ignore[reportArgumentType]
            bridge_config=bridge_cfg,
            talker_config=talker_cfg,
            variant=variant,
            thinker_family=thinker_family,
            thinker_model_id=thinker_model_id,
            talker_model_id=talker_model_id,
            **kwargs,
        )


# Backwards-compatible class alias for exported checkpoints or older user code.
Qwen3OmniSpeakConfig = QwenOmniSpeakConfig
