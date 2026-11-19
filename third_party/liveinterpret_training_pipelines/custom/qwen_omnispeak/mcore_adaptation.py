# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-invasive mcore-bridge registration for Qwen-OmniSpeak.

This module is imported by the ms-swift external plugin. It only writes to
mcore-bridge's public registry and does not modify mcore-bridge source files.
"""
import importlib
import os
import sys
from functools import partial
from typing import Optional, Tuple

import torch
from transformers import PretrainedConfig

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.model.mm_gpts.qwen3_vl import Qwen3VLLoader, Qwen3VL_Vit
from mcore_bridge.model.register import ModelMeta, register_model
from mcore_bridge.model.mm_gpts.utils import HuggingFaceVit

try:
    from .modeling_qwen_omnispeak import build_bridge, Qwen3TTSThinkerAdapter
    from .configuration_qwen_omnispeak import QwenOmniSpeakConfig
except ImportError:  # pragma: no cover - imported as a top-level plugin by ms-swift.
    from modeling_qwen_omnispeak import build_bridge, Qwen3TTSThinkerAdapter
    from configuration_qwen_omnispeak import QwenOmniSpeakConfig


MCORE_QWEN_OMNISPEAK_BASE_TYPE = 'qwen_omnispeak'
# Keep these in sync with ``register_qwen_omnispeak.QWEN_OMNISPEAK_MODEL_PREFIXES``.
# Every Thinker family/size shares the same bridge/visual/loader; they differ
# only in the weights baked in at export time.
MCORE_QWEN_OMNISPEAK_PREFIXES: Tuple[str, ...] = (
    'qwen_omnispeak',
    'qwen3_omnispeak',
    'qwen2_5_7b_omnispeak',
    'qwen2_5_3b_omnispeak',
)
# Variant suffixes are derived from the single source of truth
# (``QwenOmniSpeakConfig.SUPPORTED_VARIANTS``) so new variants are picked up
# automatically without touching this file.
MCORE_QWEN_OMNISPEAK_SUFFIXES: Tuple[str, ...] = tuple(
    v.lower() for v in sorted(QwenOmniSpeakConfig.SUPPORTED_VARIANTS))
MCORE_QWEN_OMNISPEAK_VARIANTS: Tuple[str, ...] = tuple(
    f'{prefix}_{suffix}'
    for prefix in MCORE_QWEN_OMNISPEAK_PREFIXES
    for suffix in MCORE_QWEN_OMNISPEAK_SUFFIXES)


class QwenOmniSpeakBridge(GPTBridge):
    hf_layers_prefix = 'model.thinker.model.layers'
    hf_embed_key = 'model.thinker.model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.thinker.model.norm.weight'
    hf_lm_head_key = 'model.thinker.lm_head.weight'
    hf_score_key = 'model.thinker.score.weight'

    def load_weights(self, models, model_dir):
        """Store model_dir for sidecar lookup, then delegate."""
        self._current_model_dir = model_dir
        super().load_weights(models, model_dir)

    def _convert_pre_process(self, mg_model, hf_state_dict, prefix, to_mcore):
        """Inject sidecar ``talker.text_projection.*`` as ``LazyTensor``
        objects into ``hf_state_dict`` *before* delegating to the base
        ``_convert_pre_process``, so ``_set_module`` finds all keys and does
        not raise ``missing_keys``.

        The sidecar ``qwen3_tts_extras.safetensors`` contains 4 tensors
        (``talker.text_projection.linear_fc{1,2}.{weight,bias}``) that are
        NOT in ``model.safetensors.index.json``. Without injecting them here,
        ``_set_module`` asserts ``missing_keys == 0`` and fails.

        Likewise, ``model.thinker_adapter.*`` is absent from bridge-era
        checkpoints (the adapter was introduced on the upstream branch). When
        ``use_upstream_talker_input`` is on we inject freshly-initialised
        adapter weights (mirroring DeepSpeed's ``strict=False`` → random init
        behaviour) so that ``_set_module`` finds all keys; the adapter is then
        trained from this random init, exactly like the DeepSpeed path.
        """
        if to_mcore and getattr(self, '_current_model_dir', None):
            self._inject_sidecar_lazy(hf_state_dict)
        if to_mcore:
            self._inject_thinker_adapter_init(hf_state_dict)
        return super()._convert_pre_process(mg_model, hf_state_dict, prefix, to_mcore)

    def _inject_sidecar_lazy(self, hf_state_dict):
        """Merge sidecar ``talker.text_projection.*`` weights into
        ``hf_state_dict`` (in-place), wrapping each value in ``LazyTensor``
        so it is compatible with mcore-bridge's lazy safetensors loading.

        text_projection is frozen but forward-critical in adapter mode (the
        adapter output and tts_pad embed both pass through it): a silent
        failure leaves a random projection in the forward path and training
        looks healthy while optimising wrong semantics. Adapter mode
        therefore fails fast on a missing sidecar or any injection error;
        bridge mode keeps the historical best-effort skip (P1-1).
        """
        import os as _os
        hf_config = getattr(self.config, 'hf_config', None)
        upstream = bool(getattr(hf_config, 'use_upstream_talker_input', False))
        sidecar = _os.path.join(self._current_model_dir,
                                'qwen3_tts_extras.safetensors')
        if not _os.path.isfile(sidecar):
            if upstream:
                raise FileNotFoundError(
                    f'[omnispeak-mcore] adapter mode requires the sidecar '
                    f'{sidecar} (talker.text_projection.*) but it is missing.')
            return
        # Only inject if the text_projection keys are actually missing
        tp_keys = (
            'model.talker.text_projection.linear_fc1.weight',
            'model.talker.text_projection.linear_fc1.bias',
            'model.talker.text_projection.linear_fc2.weight',
            'model.talker.text_projection.linear_fc2.bias',
        )
        if all(k in hf_state_dict for k in tp_keys):
            return
        try:
            from mcore_bridge.utils.safetensors import LazyTensor
            from safetensors.torch import safe_open
            sidecar_handle = safe_open(sidecar, framework='pt')
            injected = 0
            for k in sidecar_handle.keys():
                if k.startswith('talker.text_projection.'):
                    hf_key = 'model.' + k  # module_mapping: model.talker -> talker
                    if hf_key not in hf_state_dict:
                        hf_state_dict[hf_key] = LazyTensor(
                            loader=partial(sidecar_handle.get_tensor, k))
                        injected += 1
            if injected == 0:
                raise ValueError(
                    f'no talker.text_projection.* tensors found in {sidecar}')
            if injected:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    f'[omnispeak-mcore] injected {injected} '
                    f'text_projection tensors from sidecar')
        except Exception:
            if upstream:
                raise
            # bridge mode: non-fatal; text_projection stays randomly init

    def _inject_thinker_adapter_init(self, hf_state_dict):
        """Inject freshly-initialised ``model.thinker_adapter.*`` weights into
        ``hf_state_dict`` when the checkpoint lacks them.

        Bridge-era checkpoints (e.g. ``qwen2_5_3b_omnispeak_a``) do not contain
        ``thinker_adapter`` tensors — the module was introduced on the upstream
        branch. mcore-bridge's ``_set_module`` asserts ``missing_keys == 0``, so
        without injection the load fails. We instantiate a CPU
        ``Qwen3TTSThinkerAdapter`` (which runs ``_init_weights`` → trunc_normal
        std=0.02, biases 0, LayerNorm 1/0) and inject its ``state_dict`` under
        the ``model.thinker_adapter.`` prefix. This mirrors DeepSpeed's
        ``load_state_dict(strict=False)`` → random-init behaviour for the first
        adapter training run; the adapter is subsequently trained from this
        init and saved back via the standard visual save path.

        A checkpoint that contains SOME but not all adapter tensors is corrupt
        or hand-merged: refuse to top it up with random init and name the
        missing keys explicitly (P2-2; the previous ``any(prefix)`` early-out
        left the failure to ``_set_module`` as an opaque missing_keys error).
        """
        hf_config = getattr(self.config, 'hf_config', None)
        if hf_config is None or not getattr(hf_config, 'use_upstream_talker_input', False):
            return
        adapter_prefix = 'model.thinker_adapter.'
        try:
            from mcore_bridge.utils.safetensors import LazyTensor
            thinker_config = QwenOmniSpeakVit._get_thinker_config(hf_config)
            thinker_h = QwenOmniSpeakVit._infer_thinker_hidden_size(thinker_config)
            text_h = getattr(
                hf_config.talker_config, 'text_hidden_size',
                hf_config.talker_config.hidden_size)
            tmp = Qwen3TTSThinkerAdapter(
                thinker_hidden_size=thinker_h,
                text_hidden_size=text_h,
                intermediate_size=getattr(hf_config, 'adapter_hidden_size', None),
                dropout=float(getattr(hf_config, 'adapter_dropout', 0.0) or 0.0),
            )
            canonical = [f'{adapter_prefix}{k}' for k in tmp.state_dict()]
            present = [k for k in hf_state_dict if k.startswith(adapter_prefix)]
            if present:
                missing = [k for k in canonical if k not in hf_state_dict]
                if missing:
                    raise ValueError(
                        f'[omnispeak-mcore] partial thinker_adapter checkpoint: '
                        f'{len(present)}/{len(canonical)} tensors present '
                        f'(missing: {missing}). Refusing to complete a '
                        f'partially saved adapter with random init.')
                return  # complete adapter already in the checkpoint
            for k, v in tmp.state_dict().items():
                hf_state_dict[f'{adapter_prefix}{k}'] = LazyTensor(tensor=v)
            import logging as _lg
            _lg.getLogger(__name__).info(
                f'[omnispeak-mcore] injected {len(tmp.state_dict())} '
                f'thinker_adapter tensors (random init, checkpoint lacked them)')
        except Exception as e:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                f'[omnispeak-mcore] thinker_adapter init injection failed: {e}; '
                f'load will likely fail with missing_keys')


class QwenOmniSpeakVit(HuggingFaceVit):
    # NOTE on ``model.talker``:
    # The talker is mapped here (rather than declared in ``_generator``) so that
    # mcore-bridge's standard visual load/save path
    # (``GPTBridge._convert_pre_process``) materialises it as a *real* replicated
    # submodule on every rank. This makes the talker trainable, DP-synchronised
    # and round-tripped back to ``model.talker.*`` on save -- which is what the
    # Megatron talker codec-loss path (mcore_talker_loss.py) needs. The bridge is
    # the post-thinker aligner; audio/visual are the pre-thinker encoders.
    module_mapping = {
        'model.thinker.audio_tower': 'audio_tower',
        'model.thinker.visual': 'visual',
        'model.bridge': 'bridge',
        'model.talker': 'talker',
        'model.thinker_adapter': 'thinker_adapter',
    }
    _vision_tower = ['audio_tower', 'visual']
    _aligner = ['audio_tower.proj', 'audio_tower.proj1', 'audio_tower.proj2', 'visual.merger', 'visual.merger_list', 'bridge']
    # Empty: the talker is now a loaded submodule (see module_mapping above), so
    # the generator weight-stashing path must NOT also handle it.
    _generator = []
    # Submodules that make up the talker codec route (used for freeze helpers).
    _talker = ['talker']
    test_mm_type = 'audio'

    @staticmethod
    def _get_thinker_config(hf_config: PretrainedConfig):
        thinker_config = getattr(hf_config, 'thinker_config', hf_config)
        nested_thinker = getattr(thinker_config, 'thinker_config', None)
        if nested_thinker is not None and hasattr(nested_thinker, 'audio_config'):
            return nested_thinker
        return thinker_config

    @staticmethod
    def _get_token_id(config, *names):
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
        raise AttributeError(f'None of token id fields exist: {names}')

    @staticmethod
    def _is_qwen25_thinker(hf_config: PretrainedConfig) -> bool:
        return getattr(hf_config, 'thinker_family', None) == 'qwen2_5_omni'

    @staticmethod
    def _infer_thinker_hidden_size(thinker_config) -> int:
        """Best-effort resolution of the thinker's text hidden size.

        Mirrors ``QwenOmniSpeakForConditionalGeneration._infer_thinker_hidden_size``
        (modeling_qwen_omnispeak.py): ``text_config.hidden_size`` for both
        ``Qwen2_5OmniThinkerConfig`` and ``Qwen3OmniMoeThinkerConfig``, falling
        back to a plain top-level ``hidden_size``.
        """
        text_cfg = getattr(thinker_config, 'text_config', None)
        if text_cfg is not None and getattr(text_cfg, 'hidden_size', None):
            return int(text_cfg.hidden_size)
        if getattr(thinker_config, 'hidden_size', None):
            return int(thinker_config.hidden_size)
        raise ValueError('Cannot infer thinker hidden_size for ThinkerAdapter.')

    def prepare_model(self, hf_config: PretrainedConfig):
        thinker_config = self._get_thinker_config(hf_config)
        if self._is_qwen25_thinker(hf_config):
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoder, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder)
            self.model_cls = Qwen2_5OmniThinkerForConditionalGeneration
            self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(thinker_config.audio_config)
            self.visual = Qwen2_5OmniVisionEncoder._from_config(thinker_config.vision_config)
        else:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeVisionEncoder)
            self.model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
            self.audio_tower = Qwen3OmniMoeAudioEncoder._from_config(thinker_config.audio_config)
            self.visual = Qwen3OmniMoeVisionEncoder._from_config(thinker_config.vision_config)
        self.bridge = build_bridge(hf_config.bridge_config)
        # Build the talker as a real submodule so it is loaded/saved via the
        # standard visual module_mapping path and trained under Megatron DP.
        # Imported lazily to keep plain (non-mcore) imports cheap.
        try:
            from .modeling_qwen_omnispeak import Qwen3TtsTalkerForConditionalGeneration
        except ImportError:  # pragma: no cover - top-level plugin import.
            from modeling_qwen_omnispeak import Qwen3TtsTalkerForConditionalGeneration
        self.talker = Qwen3TtsTalkerForConditionalGeneration(hf_config.talker_config)
        if getattr(hf_config, 'tie_codec_embeddings', False):
            self.talker.codec_head.weight = self.talker.model.codec_embedding.weight

        # ── ThinkerAdapter (V1: hidden + token_embed) ──────────────────────
        # Mirror ``modeling_qwen_omnispeak.QwenOmniSpeakForConditionalGeneration``:
        # only build it when the upstream talker input protocol is on. It maps the
        # thinker signal (``hidden + token_embed``) to ``text_hidden_size`` so it
        # can be fed into ``talker.text_projection`` exactly like an
        # ``text_embedding(id)`` output would. Registered in ``module_mapping`` so
        # its weights load/save via the standard visual path and train under DP.
        self.thinker_adapter: Optional['Qwen3TTSThinkerAdapter'] = None
        if getattr(hf_config, 'use_upstream_talker_input', False):
            thinker_h = self._infer_thinker_hidden_size(thinker_config)
            text_h = getattr(
                hf_config.talker_config, 'text_hidden_size',
                hf_config.talker_config.hidden_size)
            self.thinker_adapter = Qwen3TTSThinkerAdapter(
                thinker_hidden_size=thinker_h,
                text_hidden_size=text_h,
                intermediate_size=getattr(hf_config, 'adapter_hidden_size', None),
                dropout=float(getattr(hf_config, 'adapter_dropout', 0.0) or 0.0),
            )

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        thinker_config = self._get_thinker_config(self.hf_config)
        if self._is_qwen25_thinker(self.hf_config):
            inputs_embeds = self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, thinker_config)
            res = {'inputs_embeds': inputs_embeds}
        else:
            res = Qwen3VL_Vit._get_inputs_embeds(self, inputs_embeds, kwargs, self.visual, thinker_config)
            inputs_embeds = res['inputs_embeds']
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')
        audio_token_id = self._get_token_id(
            thinker_config, 'audio_token_id', 'audio_token_index')
        audio_token_count = int(input_ids.eq(audio_token_id).sum().item())

        if input_features is None:
            if audio_token_count:
                raise ValueError(
                    'Megatron batch contains audio tokens but no `input_features`. '
                    'For Lance mel rows, verify that `source_input_features` was '
                    'resolved by the template and preserved by the collator.')
            # Keep the audio tower connected to the graph for text-only batches;
            # this mirrors mcore-bridge's multimodal unused-parameter handling.
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_res = self.get_audio_features(input_features, feature_attention_mask)
            audio_embeds = audio_res.last_hidden_state if hasattr(audio_res, 'last_hidden_state') else audio_res
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            if not audio_token_count:
                raise ValueError(
                    'Megatron batch contains `input_features` but no audio tokens; '
                    'the Lance mel manifest and template output are inconsistent.')
            audio_res = self.get_audio_features(input_features, feature_attention_mask)
            audio_embeds = audio_res.last_hidden_state if hasattr(audio_res, 'last_hidden_state') else audio_res
            if audio_embeds.numel() != audio_token_count * inputs_embeds.shape[-1]:
                raise ValueError(
                    'Megatron audio embedding/token mismatch: '
                    f'{audio_embeds.numel() // audio_embeds.shape[-1]} audio embeddings '
                    f'for {audio_token_count} audio tokens. Check Lance mel `frames` '
                    'and `feature_attention_mask` metadata.')
            audio_mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        res['inputs_embeds'] = inputs_embeds
        return res

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


def _register_one(model_type: str, aliases):
    register_model(
        ModelMeta(
            model_type,
            aliases,
            bridge_cls=QwenOmniSpeakBridge,
            visual_cls=QwenOmniSpeakVit,
            loader=Qwen3VLLoader,
        ),
        exist_ok=True,
    )


def register_qwen_omnispeak_mcore() -> None:
    # Register each family prefix as a stand-alone base type as well, so
    # ``--model_type qwen2_5_7b_omnispeak`` (without an A-H suffix) also resolves.
    for base_type in MCORE_QWEN_OMNISPEAK_PREFIXES:
        _register_one(base_type, [base_type])
    for model_type in MCORE_QWEN_OMNISPEAK_VARIANTS:
        _register_one(model_type, [model_type])

    # get_mcore_model_type caches alias mapping. Reset it so late plugin imports
    # remain visible even if another path queried the mapping earlier.
    register_mod = importlib.import_module('mcore_bridge.model.register')
    register_mod.model_type_mapping = None


register_qwen_omnispeak_mcore()


# Install the Megatron talker codec-loss integration (non-invasive monkey-patch
# of the Megatron SFT trainer). This is a no-op for plain ``swift sft`` and for
# non-omnispeak models; it only activates inside ``megatron sft`` when codec
# fields are present in the batch.
#
# Failure policy (P1-2, fixed 2026-08): under the Megatron backend an
# unpatched trainer still accepts the job but dies at the FIRST codec batch
# with an unrelated TypeError — after hours of dataset preprocessing. So when
# the CLI selected the megatron backend (``swift megatron sft ...`` puts
# ``megatron`` in argv) and the codec loss was not explicitly disabled, any
# install failure is fatal at import time. The DeepSpeed backend keeps the
# historical best-effort skip (mcore deps are optional there).
def _megatron_backend_requested() -> bool:
    return 'megatron' in sys.argv[1:3]


try:
    try:
        from . import mcore_talker_loss
    except ImportError:
        import mcore_talker_loss  # type: ignore[no-redef]
    mcore_talker_loss.install_megatron_talker_codec_loss()
except Exception as e:  # pragma: no cover - optional, depends on Megatron stack.
    # Mirror ``mcore_talker_loss.codec_loss_enabled()`` via env (the module may
    # itself be the thing that failed to import).
    _flag = os.environ.get('QWEN_OMNISPEAK_MCORE_CODEC_LOSS')
    if _flag is None:
        _flag = os.environ.get('QWEN3_OMNISPEAK_MCORE_CODEC_LOSS')
    _codec_loss_on = _flag is None or _flag.strip().lower() in ('1', 'true', 'yes', 'on')
    if _megatron_backend_requested() and _codec_loss_on:
        raise
    try:
        from swift.utils import get_logger
        get_logger().info(f'Qwen-OmniSpeak Megatron talker codec-loss not installed: {e}')
    except Exception:
        pass


__all__ = [
    'MCORE_QWEN_OMNISPEAK_BASE_TYPE',
    'MCORE_QWEN_OMNISPEAK_PREFIXES',
    'MCORE_QWEN_OMNISPEAK_SUFFIXES',
    'MCORE_QWEN_OMNISPEAK_VARIANTS',
    'QwenOmniSpeakBridge',
    'QwenOmniSpeakVit',
    'register_qwen_omnispeak_mcore',
]
