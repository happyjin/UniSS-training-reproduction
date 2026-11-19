#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Smoke tests for the Qwen-OmniSpeak custom model.

These tests *do not* require any real Qwen3-Omni or Qwen3-TTS weights. They
build a tiny synthetic config (so it fits in CPU memory), then exercise:

1. forward of all variants (A/B/C/D/E/F/G/H/I/J) on dummy thinker outputs + dummy
   codec input ids;
2. save_pretrained / from_pretrained round-trip for one variant, simulating
   the export -> reload flow used by ms-swift;
3. Thinker-family hidden-state selection defaults (Qwen3 index 24 vs Qwen2.5 last).

Run::

    python training_pipelines/custom/qwen_omnispeak/test_qwen_omnispeak.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_DIR = os.path.dirname(_CUR_DIR)
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)
if _CUSTOM_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_DIR)

from modeling_qwen_omnispeak import (  # noqa: E402
    QwenOmniSpeakForConditionalGeneration,
    select_thinker_hidden,
)
from configuration_qwen_omnispeak import (  # noqa: E402, F401
    QWEN25_THINKER_FAMILY,
    QWEN3_THINKER_FAMILY,
    QwenOmniSpeakConfig,
    Qwen3TtsTalkerConfig,
    DualRouteBridgeConfig,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeThinkerConfig,
)
from token_accuracy_metrics import (  # noqa: E402
    compute_causal_text_token_values,
    compute_direct_token_values,
    install_qwen_omnispeak_accuracy_patch,
)


def _make_tiny_thinker_config() -> Qwen3OmniMoeThinkerConfig:
    """A minimal Qwen3-Omni Thinker config that fits easily on CPU."""
    cfg = Qwen3OmniMoeThinkerConfig()
    # Shrink text backbone.
    cfg.text_config.hidden_size = 64
    cfg.text_config.intermediate_size = 128
    cfg.text_config.num_hidden_layers = 2
    cfg.text_config.num_attention_heads = 4
    cfg.text_config.num_key_value_heads = 2
    cfg.text_config.head_dim = 16
    cfg.text_config.vocab_size = 256
    cfg.text_config.num_local_experts = 2
    cfg.text_config.num_experts_per_tok = 1
    cfg.text_config.shared_expert_intermediate_size = 64
    cfg.text_config.moe_intermediate_size = 32
    cfg.text_config.rope_scaling = {'rope_type': 'default'}
    # Shrink encoders.
    cfg.audio_config.encoder_layers = 1
    cfg.audio_config.num_hidden_layers = 1
    cfg.audio_config.d_model = 64
    cfg.audio_config.encoder_attention_heads = 4
    cfg.audio_config.encoder_ffn_dim = 128
    cfg.audio_config.output_dim = 64
    cfg.vision_config.depth = 1
    cfg.vision_config.num_hidden_layers = 1
    cfg.vision_config.hidden_size = 64
    cfg.vision_config.intermediate_size = 128
    cfg.vision_config.out_hidden_size = 64
    cfg.vision_config.num_heads = 4
    return cfg


def _make_tiny_omni_speak_config(variant: str) -> QwenOmniSpeakConfig:
    thinker_cfg = _make_tiny_thinker_config()
    talker_overrides = {
        'hidden_size': 64,
        'intermediate_size': 128,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 16,
        'vocab_size': 256,
        'codec_vocab_size': 64,
        'code_predictor_hidden_size': 32,
        'code_predictor_num_layers': 1,
        'code_predictor_vocab_size': 32,
        'num_code_groups': 4,
        'original_num_hidden_layers': 28,  # keep canonical 28 so layer-selection logic is exercised
        'max_position_embeddings': 256,
    }
    bridge_overrides = {
        'thinker_hidden_size': thinker_cfg.text_config.hidden_size,
        'talker_hidden_size': 64,
        'intermediate_size': 128,
        'num_query_tokens': 8,
        'qformer_num_layers': 1,
        'qformer_num_heads': 4,
    }
    cfg = QwenOmniSpeakConfig.from_variant(
        variant=variant,
        thinker_config=thinker_cfg.to_dict(),
        talker_overrides=talker_overrides,
        bridge_overrides=bridge_overrides,
        accept_hidden_layer=1,
        codec_eos_id=63,
    )
    return cfg


def test_hidden_state_selection_defaults():
    print('=== test_hidden_state_selection_defaults ===')
    tensors = tuple(torch.full((1, 1, 2), float(i)) for i in range(4))
    assert torch.equal(select_thinker_hidden(tensors, 'last'), tensors[-1])
    assert torch.equal(select_thinker_hidden(tensors, 'final'), tensors[-1])
    assert torch.equal(select_thinker_hidden(tensors, 2), tensors[2])

    qwen3_cfg = QwenOmniSpeakConfig(thinker_config=_make_tiny_thinker_config().to_dict())
    assert qwen3_cfg.thinker_family == QWEN3_THINKER_FAMILY
    assert qwen3_cfg.hidden_state_selection == 24

    try:
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    except Exception:
        print('  Qwen2.5-Omni config unavailable; skip qwen2.5 config default check.')
        return
    qwen25 = Qwen2_5OmniThinkerConfig()
    qwen25_cfg = QwenOmniSpeakConfig(
        thinker_config=qwen25.to_dict(), thinker_family=QWEN25_THINKER_FAMILY)
    assert qwen25_cfg.thinker_family == QWEN25_THINKER_FAMILY
    assert qwen25_cfg.hidden_state_selection == 'last'
    print('  hidden-state selection defaults: OK')


def test_forward_all_variants():
    print('=== test_forward_all_variants ===')
    torch.manual_seed(0)
    B, T_text, T_codec = 1, 4, 6
    for variant in QwenOmniSpeakConfig.SUPPORTED_VARIANTS:
        cfg = _make_tiny_omni_speak_config(variant)
        model = QwenOmniSpeakForConditionalGeneration(cfg)
        model.eval()
        H = cfg.talker_config.hidden_size

        text_embeds = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
        hidden_states = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
        codec_ids = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
        codec_labels = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
        codec_labels[:, -1] = cfg.codec_eos_id
        residual_labels = torch.randint(
            0,
            cfg.talker_config.code_predictor_vocab_size,
            (B, T_codec, cfg.talker_config.num_code_groups))

        with torch.no_grad():
            out = model(
                thinker_text_embeds=text_embeds,
                thinker_hidden_states=hidden_states,
                codec_input_ids=codec_ids,
                codec_labels=codec_labels,
                codec_residual_labels=residual_labels,
            )

        # Validate output shapes.
        expected_main = (B, T_codec, cfg.talker_config.codec_vocab_size)
        expected_resid = (B, T_codec, cfg.talker_config.num_code_groups,
                          cfg.talker_config.code_predictor_vocab_size)
        assert out.main_codec_logits.shape == expected_main, (
            f'variant {variant}: main {out.main_codec_logits.shape} != {expected_main}')
        assert out.residual_codec_logits.shape == expected_resid, (
            f'variant {variant}: resid {out.residual_codec_logits.shape} != {expected_resid}')
        assert out.loss is not None and torch.isfinite(out.loss), (
            f'variant {variant}: invalid loss {out.loss}')
        assert out.metrics is not None, f'variant {variant}: missing accuracy metrics'
        assert set(out.metrics) == {
            'token_acc',
            'token_acoustic_token',
            'eos_boundary_top1_acc',
            'premature_eos_top1_rate',
        }, (
            f'variant {variant}: unexpected metric keys {set(out.metrics)}')
        assert out.metrics['token_acc'].ndim == 1 and out.metrics['token_acc'].numel() > 0
        assert (out.metrics['token_acoustic_token'].ndim == 1
                and out.metrics['token_acoustic_token'].numel() > 0)
        # Sem block length check.
        if cfg.bridge_config.bridge_type == 'qformer':
            # Q-Former: text route (T_text) + queries (num_query_tokens)
            expected_sem = T_text + cfg.bridge_config.num_query_tokens
        else:
            expected_sem = T_text + T_text
        assert out.semantic_block.shape == (B, expected_sem, H), (
            f'variant {variant}: semantic block {out.semantic_block.shape} != ({B},{expected_sem},{H})')
        # Talker hidden full length
        assert out.talker_hidden_states.shape == (B, expected_sem + T_codec, H), (
            f'variant {variant}: talker hidden {out.talker_hidden_states.shape}')
        print(f'  variant {variant}: OK '
              f'(main={tuple(out.main_codec_logits.shape)}, '
              f'resid={tuple(out.residual_codec_logits.shape)}, '
              f'loss={out.loss.item():.4f}, '
              f'talker_layers={cfg.talker_config.num_hidden_layers})')


def test_codec_training_controls():
    print('=== test_codec_training_controls ===')
    cfg = _make_tiny_omni_speak_config('A')
    cfg.residual_loss_enabled = False
    model = QwenOmniSpeakForConditionalGeneration(cfg)
    B, T_text, T_codec = 1, 4, 8
    text_embeds = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
    hidden_states = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
    codec_ids = torch.arange(T_codec).unsqueeze(0).long()
    codec_labels = torch.tensor([[-100, 1, 2, 3, 4, 5, 6, cfg.codec_eos_id]])

    env_keys = (
        'QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT',
        'QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB',
    )
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ[env_keys[0]] = '10'
        os.environ[env_keys[1]] = '0'
        model.train()
        clean = model(
            thinker_text_embeds=text_embeds,
            thinker_hidden_states=hidden_states,
            codec_input_ids=codec_ids,
            codec_labels=codec_labels,
        )
        from modeling_qwen_omnispeak import _weighted_codec_cross_entropy
        expected_loss = _weighted_codec_cross_entropy(
            clean.main_codec_logits,
            codec_labels,
            eos_id=cfg.codec_eos_id,
            eos_weight=10.0,
        )
        torch.testing.assert_close(clean.loss, expected_loss)

        os.environ[env_keys[1]] = '1'
        torch.manual_seed(9)
        corrupted = model(
            thinker_text_embeds=text_embeds,
            thinker_hidden_states=hidden_states,
            codec_input_ids=codec_ids,
            codec_labels=codec_labels,
        )
        assert not torch.equal(clean.main_codec_logits, corrupted.main_codec_logits)

        # Evaluation/inference must ignore history corruption even when the env
        # remains set, so two different RNG seeds produce identical logits.
        model.eval()
        with torch.no_grad():
            torch.manual_seed(1)
            eval_a = model(
                thinker_text_embeds=text_embeds,
                thinker_hidden_states=hidden_states,
                codec_input_ids=codec_ids,
                codec_labels=codec_labels,
            )
            torch.manual_seed(2)
            eval_b = model(
                thinker_text_embeds=text_embeds,
                thinker_hidden_states=hidden_states,
                codec_input_ids=codec_ids,
                codec_labels=codec_labels,
            )
        torch.testing.assert_close(eval_a.main_codec_logits, eval_b.main_codec_logits)
        print('  weighted loss + train-only history corruption: OK')
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_save_load_roundtrip():
    print('=== test_save_load_roundtrip ===')
    cfg = _make_tiny_omni_speak_config('B')
    cfg.codec_eos_loss_weight = 10.0
    cfg.codec_history_corruption_prob = 0.05
    model = QwenOmniSpeakForConditionalGeneration(cfg)
    model.eval()

    # Run once for a reference output.
    torch.manual_seed(1)
    B, T_text, T_codec = 1, 3, 5
    text_embeds = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
    hidden = torch.randn(B, T_text, cfg.bridge_config.thinker_hidden_size)
    codec_ids = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))

    with torch.no_grad():
        ref_out = model(
            thinker_text_embeds=text_embeds, thinker_hidden_states=hidden, codec_input_ids=codec_ids)

    tmp = tempfile.mkdtemp(prefix='qwen_omnispeak_test_')
    try:
        model.save_pretrained(tmp, safe_serialization=True)

        # Reload from the export dir, simulating a fresh load.
        # The exported directory does NOT contain modeling/configuration .py
        # files or auto_map entries — config and model classes are imported
        # from _CUR_DIR (the local custom/qwen_omnispeak/ package), which
        # mirrors how QwenOmniSpeakLoader handles real training/inference.
        for mod_name in list(sys.modules):
            if (mod_name.startswith('modeling_qwen_omnispeak')
                    or mod_name.startswith('configuration_qwen_omnispeak')):
                del sys.modules[mod_name]

        # Re-import from _CUR_DIR (already in sys.path from the module header).
        from configuration_qwen_omnispeak import QwenOmniSpeakConfig as _ReloadedConfig
        import json as _json
        cfg_path = os.path.join(tmp, 'config.json')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            jcfg = _json.load(f)
        cfg2 = _ReloadedConfig.from_dict(jcfg)
        assert cfg2.codec_eos_loss_weight == 10.0
        assert cfg2.codec_history_corruption_prob == 0.05
        print(f'  reloaded config type: {cfg2.__class__.__name__}, '
              f'variant={getattr(cfg2, "variant", None)}')

        from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration as _ReloadedModel
        model2 = _ReloadedModel.from_pretrained(tmp)
        model2.eval()
        with torch.no_grad():
            out2 = model2(
                thinker_text_embeds=text_embeds,
                thinker_hidden_states=hidden,
                codec_input_ids=codec_ids)
        err = (ref_out.main_codec_logits - out2.main_codec_logits).abs().max().item()
        print(f'  max |Δ main_codec_logits| after save/load = {err:.2e}')
        assert err < 1e-4, f'save/load round-trip diverged: {err}'
        print('  OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_token_accuracy_helpers():
    print('=== test_token_accuracy_helpers ===')
    logits = torch.tensor([[[9.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 9.0]]])
    labels = torch.tensor([[0, 1, -100]])
    values = compute_direct_token_values(logits, labels)
    assert values.tolist() == [1.0, 0.0], values

    text_logits = torch.zeros(1, 4, 4)
    text_logits[0, 0, 1] = 9.0
    text_logits[0, 1, 0] = 9.0
    text_labels = torch.tensor([[42, 1, 2, -100]])
    text_values = compute_causal_text_token_values(text_logits, text_labels)
    assert text_values.tolist() == [1.0, 0.0], text_values

    residual_logits = torch.zeros(1, 2, 2, 3)
    residual_logits[0, 0, 0, 0] = 9.0
    residual_logits[0, 0, 1, 2] = 9.0
    residual_logits[0, 1, 0, 2] = 9.0
    residual_labels = torch.tensor([[[0, 1], [2, -100]]])
    residual_values = compute_direct_token_values(residual_logits, residual_labels)
    assert residual_values.tolist() == [1.0, 0.0, 1.0], residual_values
    print('  OK')


def test_trainer_metric_patch():
    print('=== test_trainer_metric_patch ===')
    install_qwen_omnispeak_accuracy_patch()
    install_qwen_omnispeak_accuracy_patch()

    class _MetricBucket:
        def __init__(self):
            self.values = []

        def update(self, value):
            self.values.append(value.detach().cpu())

    class _MetricStore(dict):
        def __missing__(self, key):
            self[key] = _MetricBucket()
            return self[key]

    class _Model:
        training = True

    class _Trainer:
        model = _Model()
        custom_metrics = {'train': _MetricStore(), 'eval': _MetricStore()}

    class _Outputs:
        metrics = {'token_acc': torch.tensor([1.0, 0.0]), 'token_text_acc': torch.tensor([1.0])}

    from swift.trainers.mixin import SwiftMixin
    SwiftMixin._compute_acc(_Trainer(), _Outputs(), labels=torch.tensor([[0]]))
    assert _Trainer.custom_metrics['train']['token_acc'].values[0].tolist() == [1.0, 0.0]
    assert _Trainer.custom_metrics['train']['token_text_acc'].values[0].tolist() == [1.0]
    print('  OK')


if __name__ == '__main__':
    test_hidden_state_selection_defaults()
    test_token_accuracy_helpers()
    test_trainer_metric_patch()
    test_forward_all_variants()
    test_codec_training_controls()
    test_save_load_roundtrip()
    print('\nAll smoke tests passed.')
