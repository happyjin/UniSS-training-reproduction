"""S2 / S4 的 fused text CE（`SI_FUSED_TEXT_CE`）等价性与守卫测试。

核心主张：`fused_linear_cross_entropy` 把 `lm_head` 投影与 CE 融合、不物化 151936 词表的
全序列 logits，其 text_loss 与原实现 `F.cross_entropy(lm_head(h).float(), ...)` 在
数值容差内一致（不保证逐位：liger 在 Triton 内 fp32 累加，结合律不同）。

liger 的 Triton kernel **必须在 CUDA 上**跑，所以等价性主测用 `@pytest.mark.skipif` 守 GPU；
不依赖 GPU 的守卫逻辑（开关/前置条件/回退）在 CPU 上也要全过。
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from streaming_si.model.base_model import OmniSIConfig  # noqa: E402
from streaming_si.model.model_si import SI, _liger_flce  # noqa: E402

CUDA = torch.cuda.is_available()
HAS_LIGER = _liger_flce() is not None
needs_gpu_liger = pytest.mark.skipif(
    not (CUDA and HAS_LIGER),
    reason=f'fused CE 的 Triton kernel 需要 CUDA + liger（cuda={CUDA} liger={HAS_LIGER}）')


def _cfg(**over):
    """tiny 配置，但 vocab 刻意放大到 2048 以体现 fused CE 的适用场景。"""
    c = dict(
        vocab_size=2048, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64,
        speech_token_size=27, talker_special_tokens=4,
        talker_hidden_size=16, talker_num_hidden_layers=2, talker_num_attention_heads=4,
        talker_num_key_value_heads=2, talker_head_dim=4, talker_intermediate_size=32,
        max_position_embeddings=128, audio_special_token='<|audio_pad|>', audio_ids=[2047],
        thinker_arch='qwen3', talker_preset='0.1b',
        text_loss_weight=1.0, speech_loss_weight=0.0, cond_mode='full',
    )
    c.update(over)
    return OmniSIConfig(**c)


def _build(fused, device='cpu', dtype=torch.float32):
    torch.manual_seed(0)
    m = SI(_cfg(fused_text_ce=fused))
    m.eval()
    return m.to(device=device, dtype=dtype)


def _batch(B=2, L=9, vocab=2048, device='cpu'):
    torch.manual_seed(7)
    ids = torch.randint(0, vocab - 1, (B, L), device=device)
    labels = ids.clone()
    labels[:, :2] = -100          # 前两位 ignore，覆盖 ignore_index 路径
    return ids, labels


# ---------------------------------------------------------------------------
# 1. 等价性（需要 GPU + liger）
# ---------------------------------------------------------------------------
@needs_gpu_liger
def test_fused_text_ce_matches_reference_loss():
    from streaming_si.model.modeling_streaming_si import compute_si_losses
    ref = _build(False, device='cuda', dtype=torch.float32)
    alt = _build(True, device='cuda', dtype=torch.float32)
    alt.load_state_dict(ref.state_dict())

    ids, labels = _batch(device='cuda')
    with torch.no_grad():
        o_ref = ref(ids, skip_text_logits=False)
        o_alt = alt(ids, skip_text_logits=False)

    assert o_ref.logits is not None and o_ref.text_hidden is None, 'ref 应物化 logits'
    assert o_alt.logits is None and o_alt.text_hidden is not None, 'alt 应走 fused 路径'

    t_ref, _, m_ref = compute_si_losses(o_ref, labels)
    t_alt, _, m_alt = compute_si_losses(o_alt, labels)
    assert torch.allclose(t_ref, t_alt, rtol=1e-4, atol=1e-5), \
        f'text_loss 不一致: ref={t_ref.item():.8f} alt={t_alt.item():.8f}'
    # 已知代价：fused 路径拿不到 text_acc
    assert 'text_acc' in m_ref
    assert 'text_acc' not in m_alt, 'fused 路径不应产出 text_acc（无 logits 可 argmax）'


@needs_gpu_liger
def test_fused_text_ce_backward_matches():
    """梯度也要一致 —— 否则训练轨迹会漂。"""
    from streaming_si.model.modeling_streaming_si import compute_si_losses
    ref = _build(False, device='cuda', dtype=torch.float32)
    alt = _build(True, device='cuda', dtype=torch.float32)
    alt.load_state_dict(ref.state_dict())
    ids, labels = _batch(device='cuda')

    g = {}
    for name, m in (('ref', ref), ('alt', alt)):
        m.zero_grad(set_to_none=True)
        t, _, _ = compute_si_losses(m(ids, skip_text_logits=False), labels)
        t.backward()
        g[name] = m.lm_head.weight.grad.detach().clone()
    assert torch.allclose(g['ref'], g['alt'], rtol=1e-3, atol=1e-4), \
        f'lm_head 梯度不一致，最大差 {(g["ref"]-g["alt"]).abs().max().item():.3e}'


# ---------------------------------------------------------------------------
# 2. 守卫（CPU 即可）
# ---------------------------------------------------------------------------
def test_flag_off_keeps_logits_path():
    m = _build(False)
    assert m.use_fused_text_ce() is False
    ids, _ = _batch()
    with torch.no_grad():
        o = m(ids, skip_text_logits=False)
    assert o.logits is not None and o.text_hidden is None


def test_use_cache_disables_fused():
    """推理要真的 logits 做采样，不能融合。"""
    m = _build(True)
    assert m.use_fused_text_ce(use_cache=True) is False


@pytest.mark.skipif(not HAS_LIGER, reason='无 liger 时该断言由 test_no_liger_falls_back 覆盖')
def test_flag_on_enables_fused():
    m = _build(True)
    assert m.use_fused_text_ce() is True


@pytest.mark.skipif(HAS_LIGER, reason='仅在无 liger 的镜像上验证回退')
def test_no_liger_falls_back():
    """基线镜像没有 liger：开关打开也必须回退到 logits 路径，不能崩。"""
    m = _build(True)
    assert m.use_fused_text_ce() is False
    ids, _ = _batch()
    with torch.no_grad():
        o = m(ids, skip_text_logits=False)
    assert o.logits is not None, '无 liger 时必须回退到原 logits 路径'


def test_skip_text_logits_takes_precedence():
    """S1（text_w=0）即使开了 fused 开关，也应整个跳过 text 路。"""
    m = _build(True)
    ids, _ = _batch()
    with torch.no_grad():
        o = m(ids, skip_text_logits=True)
    assert o.logits is None and o.text_hidden is None, 'text_w=0 时不该走 fused，应全跳过'
