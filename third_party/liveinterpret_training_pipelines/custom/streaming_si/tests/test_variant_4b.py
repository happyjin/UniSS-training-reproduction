"""streaming_si_4b 变体（Thinker Qwen3.5 / Talker 0.6B 结构）的模型级测试（tiny 尺寸，随机权重）。

    python -m pytest custom/streaming_si/tests/test_variant_4b.py -q
"""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.modeling_streaming_si import (StreamingSIConfig,  # noqa: E402
                                                StreamingSIForConditionalGeneration)
from streaming_si.model.qwen3_5_compat import Qwen3_5CompatForCausalLM  # noqa: E402
from streaming_si.tests.test_streaming_si import (AUDIO_ID, TINY_VOXTRAL_AUDIO, TINY_VOXTRAL_PROJ,  # noqa: E402
                                            reference_s3_loss, s3_batch)
from streaming_si.model.variants import TALKER_PRESETS, variant_of  # noqa: E402

THINKER_TEXT = dict(vocab_size=512, hidden_size=80, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
                    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                    linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=True,
                    layer_types=['linear_attention'] * 3 + ['full_attention'])


def tiny_4b_config(**over):
    talker = dict(TALKER_PRESETS['0.6b'], talker_num_hidden_layers=2, talker_hidden_size=64,
                  talker_intermediate_size=96, talker_head_dim=16)
    kw = dict({k: THINKER_TEXT[k] for k in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                                            'num_attention_heads', 'num_key_value_heads', 'head_dim',
                                            'rms_norm_eps', 'rope_theta', 'tie_word_embeddings')},
              thinker_arch='qwen3_5', thinker_text_config=THINKER_TEXT, talker_preset='0.6b', **talker,
              audio_ids=[AUDIO_ID], speech_token_size=100, voxtral_audio_config=TINY_VOXTRAL_AUDIO,
              voxtral_projector_config=TINY_VOXTRAL_PROJ)
    kw.update(over)
    return StreamingSIConfig(**kw)


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return StreamingSIForConditionalGeneration(tiny_4b_config()).eval()


def test_structure(model):
    assert variant_of(model.config) == 'streaming_si_4b'
    assert isinstance(model.thinker_model, Qwen3_5CompatForCausalLM)
    kinds = [('linear_attn' if hasattr(l, 'linear_attn') else 'self_attn') for l in model.thinker.layers]
    assert kinds == ['linear_attn'] * 3 + ['self_attn']
    assert model.lm_head.weight.data_ptr() == model.thinker.embed_tokens.weight.data_ptr()
    assert model.talker.decoder.config.num_key_value_heads == 8          # 0.6B 预设：与 Qwen3-0.6B 同形
    assert model.talker.thinker_proj.up.in_features == 80 and model.audio_proj.mlp[-1].out_features == 80
    assert tuple(model.talker_text_embed.weight.shape) == (512, 80)


def test_flat_fields_must_match_text_config():
    with pytest.raises(ValueError, match='不一致'):
        StreamingSIForConditionalGeneration(tiny_4b_config(hidden_size=64))


def test_s3_loss_matches_reference_formula(model):
    batch = s3_batch(model.config)
    with torch.no_grad():
        out = model(**batch)
        ref_text, ref_speech = reference_s3_loss(model, batch)
    assert torch.allclose(out.metrics['text_loss'], ref_text, atol=1e-6)
    assert torch.allclose(out.metrics['speech_loss'], ref_speech, atol=1e-6)
    assert torch.allclose(out.loss, ref_text + ref_speech, atol=1e-6)


def test_right_padding_does_not_change_valid_positions(model):
    batch = s3_batch(model.config, n_frames=(3, 2))
    with torch.no_grad():
        both = model(**batch)
        one = model(**{k: (v[:1] if torch.is_tensor(v) and v.size(0) == 2 else v) for k, v in batch.items()})
    n = int(batch['attention_mask'][0].sum()) - 1
    assert torch.allclose(both.logits[0, :n], one.logits[0, :n], atol=1e-5)


def test_backward_and_gradient_checkpointing():
    torch.manual_seed(1)
    m = StreamingSIForConditionalGeneration(tiny_4b_config()).train()
    m.audio_encoder.requires_grad_(False)
    m.talker_text_embed.weight.requires_grad = False
    batch = s3_batch(m.config)
    loss0 = m(**batch).loss
    loss0.backward()
    lin = m.thinker.layers[0].linear_attn
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in
               (lin.in_proj_qkv.weight, lin.in_proj_z.weight, lin.A_log, lin.dt_bias, lin.conv1d.weight))
    assert m.talker.decoder.layers[0].self_attn.q_proj.weight.grad is not None
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    m.gradient_checkpointing_enable({'use_reentrant': False})
    loss1 = m(**batch).loss
    loss1.backward()
    assert torch.allclose(loss0, loss1, atol=1e-6)
    worst = max((grads[n] - p.grad).abs().max().item() for n, p in m.named_parameters() if n in grads)
    assert worst < 1e-5, worst


def test_save_and_reload_roundtrip(model):
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        again = StreamingSIForConditionalGeneration.from_pretrained(d)
    assert again.config.thinker_arch == 'qwen3_5' and again.config.talker_preset == '0.6b'
    sa, sb = model.state_dict(), again.state_dict()
    assert set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert again.lm_head.weight.data_ptr() == again.thinker.embed_tokens.weight.data_ptr()


def test_s4_lora_covers_linear_attention_and_ties_tokens():
    pytest.importorskip('peft')
    from streaming_si.integration.s4_peft import apply_s4_train_scope, attach_s4_peft, expected_s4_lora_modules, unwrap_s4_model
    torch.manual_seed(2)
    pm = attach_s4_peft(StreamingSIForConditionalGeneration(tiny_4b_config()), [505, 506, 507, 508])
    raw = unwrap_s4_model(pm)
    lora = sorted(n for n, mod in raw.named_modules() if hasattr(mod, 'lora_A'))
    assert len(lora) == expected_s4_lora_modules(raw) == 3 * 6 + 1 * 7 + 2 * 7
    assert 'thinker_model.model.layers.0.linear_attn.in_proj_qkv' in lora
    assert 'thinker_model.model.layers.0.linear_attn.out_proj' in lora
    assert not any(n.endswith(('in_proj_b', 'in_proj_a')) for n in lora)
    assert raw.thinker_model.lm_head.token_adapter.tied_adapter is raw.thinker_model.model.embed_tokens.token_adapter
    names = apply_s4_train_scope(pm, 'thinker')
    assert any('linear_attn.in_proj_z.lora_A' in n for n in names)
    assert not any(n.startswith('talker.') for n in names)
