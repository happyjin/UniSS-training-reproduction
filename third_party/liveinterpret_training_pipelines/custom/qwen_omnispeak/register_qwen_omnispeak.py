# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-swift registration entry for Qwen-OmniSpeak (variants A/B/C/D/E/F/G/H/I/J).

Usage::

    swift sft \\
        --external_plugins training_pipelines/custom/qwen_omnispeak/register_qwen_omnispeak.py \\
        --model_type qwen3_omnispeak_b \\
        --model <path-to-exported-variant-B> \\
        ...

The registration file does *three* things:

1. Adds ``training_pipelines/custom/qwen_omnispeak`` to ``sys.path`` so the
   self-contained model package is importable both before and after export.
2. Registers a single ``MultiModelKeys`` model arch shared by all variants
   (they only differ in inner config, not parameter naming).
3. Registers one ``ModelMeta`` entry per variant:
   ``qwen_omnispeak_a / b / c / d / e / f / g / h / i / j``, plus a template alias ``qwen_omnispeak``
   that subclasses the existing Qwen3-Omni multimodal template.
"""
import os
import sys

# Make the local self-contained model package importable when this plugin is
# loaded directly via ``--external_plugins``.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from typing import Optional  # noqa: E402

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel  # noqa: E402


# Ensure upstream models support FA3 by dynamically injecting the attribute.
# transformers 4.57.3's Qwen3OmniMoe models define `_supports_flash_attn = True` but
# ms-swift may also check `_supports_flash_attn3`. Inject both to be safe.
def _ensure_fa3_support():
    """Inject FA3 support attributes into upstream model classes."""
    # Qwen3-Omni MoE models
    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
        for cls in [Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeThinkerForConditionalGeneration]:
            if cls is not None:
                if not getattr(cls, '_supports_flash_attn', False):
                    cls._supports_flash_attn = True
                if not getattr(cls, '_supports_flash_attn3', False):
                    cls._supports_flash_attn3 = True
    except Exception:
        pass

    # Qwen2.5-Omni models
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        for cls in [Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration]:
            if cls is not None:
                if not getattr(cls, '_supports_flash_attn', False):
                    cls._supports_flash_attn = True
                if not getattr(cls, '_supports_flash_attn3', False):
                    cls._supports_flash_attn3 = True
    except Exception:
        pass

_ensure_fa3_support()

from swift.model import (Model, ModelGroup, ModelLoader, ModelMeta,  # noqa: E402
                         MultiModelKeys, register_model, register_model_arch)
from swift.template import register_template  # noqa: E402
from swift.template.constant import MLLMTemplateType  # noqa: E402
from swift.template.template_meta import TemplateMeta  # noqa: E402
from swift.utils import Processor, get_logger  # noqa: E402

logger = get_logger()

# ---------------------------------------------------------------------------
# 1. Model architecture map (parts -> module name prefixes)
# ---------------------------------------------------------------------------
# The same prefix layout is shared across all variants because they only
# differ in *inner* config (bridge_type / talker layers).

QWEN_OMNISPEAK_ARCH = 'qwen_omnispeak'

register_model_arch(
    MultiModelKeys(
        QWEN_OMNISPEAK_ARCH,
        # The *language model* in ms-swift terminology = Thinker text backbone.
        # Reach the deepest possible language sub-module so freeze_llm/LoRA
        # behaviour matches the upstream qwen3_omni model arch.
        language_model=['model.thinker.model', 'model.thinker.lm_head'],
        # Vision/audio tower: also part of thinker side. Keep separated so
        # ``freeze_vit`` works.
        vision_tower=['model.thinker.audio_tower', 'model.thinker.visual'],
        # Aligners on the Thinker side (audio proj / vision merger).
        aligner=[
            # Qwen2.5-Omni audio projector.
            'model.thinker.audio_tower.proj',
            # Qwen3-Omni audio projectors.
            'model.thinker.audio_tower.proj1',
            'model.thinker.audio_tower.proj2',
            'model.thinker.visual.merger',
            'model.thinker.visual.merger_list',
            # And our DualRoute Bridge — by far the most important
            # trainable module for Stage 1.
            'model.bridge',
        ],
        # Generator-side modules: the talker backbone, codec embedding/head and
        # the residual code predictor. Putting them in ``generator`` keeps
        # freeze_llm/aligner behaviour focused on the thinker + bridge in
        # Stage 1.
        generator=[
            'model.talker.model',
            'model.talker.codec_head',
            'model.talker.code_predictor',
        ],
    ))


# ---------------------------------------------------------------------------
# 2. Custom loader used by all variants
# ---------------------------------------------------------------------------


# Token-id fields that ms-swift's Qwen3-Omni template reads off
# ``self.config.thinker_config`` directly. In transformers>=5.x the loaded
# top-level ``Qwen3OmniMoeConfig`` keeps these one level deeper (under its
# *own* ``thinker_config``), so ``self.config.thinker_config.audio_token_id``
# raises AttributeError. We mirror them up so the template encoder works.
_THINKER_FACING_TOKEN_KEYS = (
    # Qwen3-Omni token ids.
    'audio_token_id', 'audio_start_token_id', 'audio_end_token_id',
    'image_token_id', 'video_token_id',
    'vision_start_token_id', 'vision_end_token_id',
    # Qwen2.5-Omni token ids used by Qwen2_5OmniTemplate._post_encode.
    'audio_token_index', 'image_token_index', 'video_token_index',
)


def _hoist_token_ids_for_swift(config: PretrainedConfig) -> None:
    """Mirror nested token_id fields up one level so swift template lookups work.

    Affected lookups (qwen3_omni v3 template):
        ``self.config.thinker_config.audio_token_id`` and friends.

    Path resolution:
        config.thinker_config -> Qwen3OmniMoeConfig (the "full omni" wrapper)
                                  ^ this object lacks the token_id directly.
        config.thinker_config.thinker_config -> Qwen3OmniMoeThinkerConfig
                                                  ^ has audio_token_id etc.

    We copy each missing token_id up by one level. Same for talker_config.
    """
    outer = getattr(config, 'thinker_config', None)
    if outer is None:
        return
    sources = []
    for child_name in ('thinker_config', 'talker_config'):
        child = getattr(outer, child_name, None)
        if child is not None:
            sources.append(child)
    for key in _THINKER_FACING_TOKEN_KEYS:
        if hasattr(outer, key) and getattr(outer, key) is not None:
            continue
        for src in sources:
            value = getattr(src, key, None)
            if value is not None:
                try:
                    setattr(outer, key, value)
                except Exception:
                    pass
                break


class QwenOmniSpeakLoader(ModelLoader):
    """Loads an exported Qwen-OmniSpeak model directory.

    Config and model classes are imported from the local
    ``custom/qwen_omnispeak/`` package. The ``get_config`` method
    in this loader has a built-in fallback that directly imports
    ``QwenOmniSpeakConfig`` when ``configuration_qwen_omnispeak.py``
    is not present in the model directory.
    """

    def get_config(self, model_dir: str) -> PretrainedConfig:
        has_config_py = os.path.isfile(os.path.join(model_dir, 'configuration_qwen_omnispeak.py'))
        config = None
        if has_config_py:
            try:
                config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            except (KeyError, ValueError) as exc:
                # Some swift checkpoints copy configuration_qwen_omnispeak.py
                # but save an incomplete auto_map containing only
                # AutoModelForCausalLM. AutoConfig then cannot resolve the
                # custom ``qwen_omnispeak`` model_type. Fall through to the
                # local class just as we do when the copied source is absent.
                logger.warning(
                    'AutoConfig could not load %s (%s); using the locally '
                    'registered QwenOmniSpeakConfig.', model_dir, exc)
        if config is None:
            # Fallback: load config.json manually and instantiate the config
            # class from _CUR_DIR (already in sys.path).
            import json as _json
            from configuration_qwen_omnispeak import QwenOmniSpeakConfig

            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = _json.load(f)
            config = QwenOmniSpeakConfig.from_dict(config_dict)
            logger.info('Loaded config for %s via local class in %s.', model_dir, _CUR_DIR)
        _hoist_token_ids_for_swift(config)
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        # Reuse the processor copied from the selected upstream Thinker.
        # Qwen3-Omni uses Qwen3OmniMoeProcessor; Qwen2.5-Omni uses
        # Qwen2_5OmniProcessor. Fall back to tokenizer-only if processor loading
        # fails so text-only dry runs still work.
        from transformers import AutoTokenizer
        import json as _json

        thinker_family = getattr(config, 'thinker_family', 'qwen3_omni_moe')
        if thinker_family == 'qwen2_5_omni':
            try:
                from transformers import Qwen2_5OmniProcessor
                return Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
            except Exception as e:
                logger.warning(f'Failed to load Qwen2.5-Omni processor ({e}); falling back to tokenizer only.')
                return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        try:
            from transformers import Qwen3OmniMoeProcessor
            processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir, trust_remote_code=True)
        except AttributeError:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            tk_cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
            if os.path.isfile(tk_cfg_path):
                with open(tk_cfg_path, 'r', encoding='utf-8') as f:
                    tk_cfg = _json.load(f)
                for _k in ('image_token', 'audio_token', 'video_token',
                           'vision_bos_token', 'vision_eos_token',
                           'audio_bos_token', 'audio_eos_token'):
                    if _k in tk_cfg and not hasattr(tokenizer, _k):
                        setattr(tokenizer, _k, tk_cfg[_k])
            try:
                processor = Qwen3OmniMoeProcessor.from_pretrained(
                    model_dir, tokenizer=tokenizer, trust_remote_code=True)
            except Exception:
                # Last-resort fallback: tokenizer only (no audio/vision processing).
                processor = tokenizer
                logger.warning(
                    'Falling back to tokenizer-only processor; multimodal media '
                    'fields will not be processed by the template.')
        except Exception:
            from transformers import AutoTokenizer as _AT
            processor = _AT.from_pretrained(model_dir, trust_remote_code=True)
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        """Load the model, preferring .py files from the checkpoint directory.

        When the source files and a complete ``auto_map`` exist inside
        ``model_dir``, the normal HF ``AutoModelForCausalLM`` path (with
        ``trust_remote_code=True``) is used.

        When they are absent, we bypass HF's auto-resolution entirely: the model
        class is imported directly from ``custom/qwen_omnispeak/`` and the
        ``auto_map`` in config is temporarily cleared so ``from_pretrained`` does
        not attempt to fetch remote code (which can hang on machines without
        direct HF Hub access).
        """
        from transformers import AutoModelForCausalLM

        ### DEBUG: Log bridge config - commented out
        ### bridge_type = getattr(config, 'bridge_config', None)
        ### if bridge_type is not None:
        ###     bridge_type = getattr(bridge_type, 'bridge_type', None) if hasattr(bridge_type, 'bridge_type') else None
        ### logger.info(f'[DEBUG] Loading model from {model_dir}, bridge_type in config: {bridge_type}')

        has_modeling_py = os.path.isfile(os.path.join(model_dir, 'modeling_qwen_omnispeak.py'))
        has_config_py = os.path.isfile(os.path.join(model_dir, 'configuration_qwen_omnispeak.py'))
        auto_map = getattr(config, 'auto_map', None) or {}
        has_complete_auto_map = (
            'AutoConfig' in auto_map and 'AutoModelForCausalLM' in auto_map)

        if has_modeling_py and has_config_py and has_complete_auto_map:
            # Normal path: .py files present in the checkpoint — let HF
            # auto-load them via trust_remote_code.
            self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
            model = super().get_model(model_dir, config, processor, model_kwargs)
        else:
            # Fallback: source files are absent, or swift saved an incomplete
            # auto_map. Import directly from the local custom package to avoid
            # HF dynamic-module cache races across distributed ranks.
            from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration

            saved_auto_map = getattr(config, 'auto_map', None)
            saved_attn_impl = getattr(config, 'attn_implementation', None)

            # Temporarily switch to FA2 to bypass transformers' FA3 check,
            # then restore the original setting after model init.
            if saved_attn_impl == 'flash_attention_3':
                config.attn_implementation = 'flash_attention_2'

            if saved_auto_map:
                config.auto_map = {}
            try:
                model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
                    model_dir, config=config, torch_dtype=self.torch_dtype,
                    trust_remote_code=True, **model_kwargs)
            finally:
                if saved_auto_map is not None:
                    config.auto_map = saved_auto_map
                # Restore original attn_implementation so training uses FA3.
                if saved_attn_impl is not None:
                    config.attn_implementation = saved_attn_impl
            # Mark the auto class so deepspeed save/load works correctly.
            if getattr(model, '_auto_class', None) is None:
                model._auto_class = 'AutoModelForCausalLM'

        ### DEBUG: Log model's expected bridge keys - commented out
        ### if hasattr(model, 'model') and hasattr(model.model, 'bridge'):
        ###     bridge_module = model.model.bridge
        ###     expected_keys = list(bridge_module.state_dict().keys())
        ###     logger.info(f'[DEBUG] Model bridge module type: {type(bridge_module).__name__}')
        ###     logger.info(f'[DEBUG] Model expects {len(expected_keys)} bridge parameter keys:')
        ###     for k in sorted(expected_keys)[:30]:
        ###         logger.info(f'[DEBUG]   - {k}')
        ###     if len(expected_keys) > 30:
        ###         logger.info(f'[DEBUG]   ... and {len(expected_keys) - 30} more')


        return model


# ---------------------------------------------------------------------------
# 3. Template (reuse the Qwen3-Omni multimodal template logic)
# ---------------------------------------------------------------------------


QWEN_OMNISPEAK_TEMPLATE = 'qwen_omnispeak'

# We re-use the existing ``qwen3_omni`` template by registering an alias.
# Subclassing keeps the door open for tweaks (e.g. emitting a leading
# ``<|codec_bos|>`` token) without touching the parent template.

try:
    from swift.template.templates.qwen import Qwen2_5OmniTemplate, Qwen3OmniTemplate

    class QwenOmniSpeakTemplate(Qwen3OmniTemplate):
        """Template alias for Qwen-OmniSpeak.

        The same registered template supports Qwen3-Omni and Qwen2.5-Omni
        thinkers. For Qwen2.5 exports we switch the inherited omni template to
        ``omni_v2_5`` so placeholder tokens, processor kwargs, post-encode media
        embedding and RoPE position handling match the upstream Qwen2.5 path.
        """

        def _resolve_config(self, processor=None):
            # ``self.config`` is only set *inside* the base ``init_processor``.
            # During our own ``init_processor`` (before the super() call) it does
            # not exist yet, so fall back to the processor's model_info config.
            config = getattr(self, 'config', None)
            if config is None and processor is not None:
                model_info = getattr(processor, 'model_info', None)
                config = getattr(model_info, 'config', None)
            return config

        def _is_qwen25_thinker(self, processor=None) -> bool:
            config = self._resolve_config(processor)
            return getattr(config, 'thinker_family', None) == 'qwen2_5_omni'

        def init_processor(self, processor) -> None:
            if self._is_qwen25_thinker(processor):
                self.version = 'omni_v2_5'
                self.placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']
                return Qwen2_5OmniTemplate.init_processor(self, processor)
            self.version = 'omni_v3'
            self.placeholder_tokens = ['<|image_pad|>', '<|audio_pad|>', '<|video_pad|>']
            return Qwen3OmniTemplate.init_processor(self, processor)

        # Bridge/codec kwargs consumed directly by the QwenOmniSpeak model
        # forward. The upstream omni ``_post_encode`` returns only
        # ``{'inputs_embeds': ...}`` and ms-swift's ``pre_forward_hook`` only
        # re-adds a fixed whitelist (input_ids/attention_mask/labels/...), so
        # these would otherwise be dropped before reaching ``model.forward``.
        _BRIDGE_FORWARD_KEYS = (
            'run_thinker', 'codec_input_ids', 'codec_attention_mask',
            'codec_labels', 'codec_residual_labels', 'has_codec_supervision',
            'codec_to_sem_visible',
            # Upstream Qwen3-TTS / Qwen2.5-Omni full-add talker input protocol.
            # Without these in the whitelist the omni ``_post_encode`` (which
            # returns only ``inputs_embeds``) would drop them before forward.
            'target_text_ids', 'codec_bos_index', 'full_add_prompt_len',
            # Thinker text labels (upstream mode): original text labels before
            # the collator overwrites ``labels`` with codec-aligned labels.
            'thinker_text_labels',
            # Speaker x-vector (voice clone / voice preserving). Same reason —
            # these must survive ``_post_encode`` to reach the model forward,
            # otherwise the x-vector is silently never injected.
            'reference_audio_wav', 'reference_audio_sr', 'speaker_slot_index',
            'speaker_embedding',
            # ICL reference (source) segment. Same survival requirement: the
            # omni ``_post_encode`` drops unknown keys, which silently disabled
            # ICL even when QWEN_OMNISPEAK_USE_ICL=true (P1-7).
            'source_codec_codes', 'source_text_ids',
            'source_codec_mask', 'source_text_mask',
        )

        def _post_encode(self, model, inputs):
            # The upstream omni ``_post_encode`` reads token-id fields off
            # ``model.config.thinker_config`` (e.g. ``audio_token_index``). For
            # exported Qwen-OmniSpeak models those live one level deeper, so
            # mirror them up onto the model's own config before delegating.
            model_config = getattr(model, 'config', None)
            if model_config is not None:
                _hoist_token_ids_for_swift(model_config)
            if self._is_qwen25_thinker():
                encoded = Qwen2_5OmniTemplate._post_encode(self, model, inputs)
            else:
                encoded = Qwen3OmniTemplate._post_encode(self, model, inputs)
            if isinstance(encoded, dict) and isinstance(inputs, dict):
                for key in self._BRIDGE_FORWARD_KEYS:
                    if key not in encoded and key in inputs:
                        encoded[key] = inputs[key]
            return encoded

    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakTemplate,
        ))
except Exception as e:  # pragma: no cover
    logger.warning(
        f'Failed to subclass Qwen3OmniTemplate ({e}); falling back to a basic chat template.')
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
        ))


# ---------------------------------------------------------------------------
# 4. Register the variants as separate model_types.
# ---------------------------------------------------------------------------

VARIANT_DOCSTRING = {
    'A': 'Qwen Omni Thinker + DualRoute MLP Bridge + Qwen3-TTS Talker (full-28 layers)',
    'B': 'Qwen Omni Thinker + DualRoute MLP Bridge + Qwen3-TTS Talker (top-16 layers)',
    'C': 'Qwen Omni Thinker + DualRoute MLP Bridge + Qwen3-TTS Talker (top-12 layers)',
    'D': 'Qwen Omni Thinker + DualRoute MLP Bridge + Qwen3-TTS Talker (top-8 layers)',
    'E': 'Qwen Omni Thinker + DualRoute MLP Bridge + Qwen3-TTS Talker (top-4 layers)',
    'F': 'Qwen Omni Thinker + DualRoute Q-Former Bridge + Qwen3-TTS Talker (full-28 layers)',
    'G': 'Qwen Omni Thinker + DualRoute Q-Former Bridge + Qwen3-TTS Talker (top-16 layers)',
    'H': 'Qwen Omni Thinker + DualRoute Q-Former Bridge + Qwen3-TTS Talker (top-12 layers)',
    'I': 'Qwen Omni Thinker + DualRoute Q-Former Bridge + Qwen3-TTS Talker (top-8 layers)',
    'J': 'Qwen Omni Thinker + DualRoute Q-Former Bridge + Qwen3-TTS Talker (top-4 layers)',
}

# Each variant has its own ``model_type`` so users can reference them via
# ``--model_type qwen_omnispeak_a`` etc. Architectures string is shared
# (``QwenOmniSpeakForConditionalGeneration``), so additional automatic
# matching by config.architectures is also possible — but the default routing
# uses ``model_type`` for unambiguity across variants.
#
# Every Thinker family/size gets its own model_type prefix. They all share the
# same loader / arch / template and differ only in the weights baked in at
# export time (which Thinker family/size was used + hidden-state selection):
#   * ``qwen3_omnispeak_{a..j}``      -> Qwen3-Omni-30B-A3B Thinker (legacy 24)
#   * ``qwen2_5_7b_omnispeak_{a..j}`` -> Qwen2.5-Omni-7B Thinker (last hidden)
#   * ``qwen2_5_3b_omnispeak_{a..j}`` -> Qwen2.5-Omni-3B Thinker (last hidden)
#   * ``qwen_omnispeak_{a..j}``       -> backward-compatible default prefix
QWEN_OMNISPEAK_MODEL_PREFIXES = (
    'qwen_omnispeak',
    'qwen3_omnispeak',
    'qwen2_5_7b_omnispeak',
    'qwen2_5_3b_omnispeak',
)

for _prefix in QWEN_OMNISPEAK_MODEL_PREFIXES:
    for _variant in VARIANT_DOCSTRING:
        _suffix = _variant.lower()
        _model_type = f'{_prefix}_{_suffix}'
        register_model(
            ModelMeta(
                _model_type,
                [
                    # No public model_groups: the model is built locally via the
                    # export script. Listing dummy models is unnecessary because
                    # ``model_type`` is provided explicitly on the command line.
                    ModelGroup([Model(_model_type, _model_type)], QWEN_OMNISPEAK_TEMPLATE),
                ],
                QwenOmniSpeakLoader,
                template=QWEN_OMNISPEAK_TEMPLATE,
                is_multimodal=True,
                model_arch=QWEN_OMNISPEAK_ARCH,
                mcore_model_type=_model_type,
                architectures=['QwenOmniSpeakForConditionalGeneration'],
                requires=['transformers>=4.57.dev0', 'soundfile', 'decord', 'qwen_omni_utils>=0.0.9'],
                tags=['vision', 'video', 'audio', 'speech-generation'],
                additional_saved_files=[
                    'modeling_qwen_omnispeak.py',
                    'configuration_qwen_omnispeak.py',
                ],
            ))
        logger.info(f'Registered Qwen-OmniSpeak variant {_variant} as model_type={_model_type!r}: '
                    f'{VARIANT_DOCSTRING[_variant]}')


if 'QwenOmniSpeakTemplate' in globals():
    # Backwards-compatible alias for older helper modules.
    Qwen3OmniSpeakTemplate = QwenOmniSpeakTemplate

__all__ = [
    'QWEN_OMNISPEAK_ARCH', 'QWEN_OMNISPEAK_TEMPLATE', 'QWEN_OMNISPEAK_MODEL_PREFIXES',
    'QwenOmniSpeakLoader', 'QwenOmniSpeakTemplate', 'Qwen3OmniSpeakTemplate',
]
