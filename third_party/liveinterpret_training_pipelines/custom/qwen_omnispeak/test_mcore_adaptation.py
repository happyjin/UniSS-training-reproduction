#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Smoke checks for the non-invasive Qwen-OmniSpeak mcore registration."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Resolve the custom/ package THIS file lives in (parents[1]) instead of a
# sibling checkout named liveinterpret_training_pipelines: the hardcoded
# sibling path made any other workspace name (e.g. *_2) silently smoke-test
# the WRONG repository (P2-4). QWEN_OMNISPEAK_CUSTOM_DIR overrides for
# out-of-tree runs.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUSTOM_DIR = Path(os.environ.get(
    'QWEN_OMNISPEAK_CUSTOM_DIR',
    os.fspath(Path(__file__).resolve().parents[1])))
_MCORE_SRC = _REPO_ROOT / 'mcore-bridge' / 'src'
for _path in (_CUSTOM_DIR, _MCORE_SRC):
    _path_str = os.fspath(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

try:
    import qwen_omnispeak.mcore_adaptation as mcore_adaptation  # noqa: E402
    from mcore_bridge.model import get_mcore_model_type, get_model_meta  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover - depends on local Megatron stack.
    if e.name == 'transformer_engine':
        print('Skip mcore adaptation smoke test: transformer_engine is not installed.')
        sys.exit(0)
    raise


def main() -> None:
    expected = tuple(f'qwen_omnispeak_{suffix}' for suffix in 'abcdefgh')
    for model_type in expected:
        assert get_mcore_model_type(model_type) == model_type
        meta = get_model_meta(model_type)
        assert meta is not None
        assert meta.bridge_cls is mcore_adaptation.QwenOmniSpeakBridge
        assert meta.visual_cls is mcore_adaptation.QwenOmniSpeakVit

    bridge_cls = mcore_adaptation.QwenOmniSpeakBridge
    assert bridge_cls.hf_layers_prefix == 'model.thinker.model.layers'
    assert bridge_cls.hf_embed_key == 'model.thinker.model.embed_tokens.weight'
    assert bridge_cls.hf_final_layernorm_key == 'model.thinker.model.norm.weight'
    assert bridge_cls.hf_lm_head_key == 'model.thinker.lm_head.weight'

    module_mapping = mcore_adaptation.QwenOmniSpeakVit.module_mapping
    assert module_mapping['model.thinker.audio_tower'] == 'audio_tower'
    assert module_mapping['model.thinker.visual'] == 'visual'
    assert module_mapping['model.bridge'] == 'bridge'
    assert module_mapping['model.talker'] == 'talker'
    assert module_mapping['model.thinker_adapter'] == 'thinker_adapter'
    assert 'audio_tower.proj' in mcore_adaptation.QwenOmniSpeakVit._aligner
    assert 'audio_tower.proj1' in mcore_adaptation.QwenOmniSpeakVit._aligner
    assert mcore_adaptation.QwenOmniSpeakVit._generator == []
    assert mcore_adaptation.QwenOmniSpeakVit._talker == ['talker']

    class _FakeVit:
        get_inputs_embeds = mcore_adaptation.QwenOmniSpeakVit.get_inputs_embeds
        _get_thinker_config = staticmethod(lambda config: config.thinker_config)
        _is_qwen25_thinker = staticmethod(lambda config: True)
        _get_token_id = staticmethod(mcore_adaptation.QwenOmniSpeakVit._get_token_id)
        _hf_get_inputs_embeds = staticmethod(
            lambda inputs_embeds, kwargs, visual, thinker_config: inputs_embeds)

        def __init__(self):
            thinker = SimpleNamespace(audio_token_id=9)
            self.hf_config = SimpleNamespace(
                thinker_family='qwen2_5_omni',
                thinker_config=thinker,
            )
            self.visual = None

    fake = _FakeVit()
    embeds = torch.zeros(1, 3, 4)
    try:
        fake.get_inputs_embeds(
            embeds,
            input_ids=torch.tensor([[1, 9, 2]]),
            input_features=None,
        )
    except ValueError as exc:
        assert 'audio tokens but no `input_features`' in str(exc)
    else:
        raise AssertionError('Missing Lance mel must fail instead of using zero audio.')

    try:
        fake.get_inputs_embeds(
            embeds,
            input_ids=torch.tensor([[1, 2, 3]]),
            input_features=torch.zeros(1, 128, 2),
        )
    except ValueError as exc:
        assert '`input_features` but no audio tokens' in str(exc)
    else:
        raise AssertionError('Orphan Lance mel features must fail clearly.')

    print('Qwen-OmniSpeak mcore adaptation smoke test passed.')


if __name__ == '__main__':
    main()
