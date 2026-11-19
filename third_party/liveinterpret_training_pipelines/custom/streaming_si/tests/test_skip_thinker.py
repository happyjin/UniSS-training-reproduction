"""S1 跳过 thinker 主干前向（`SI_SKIP_THINKER`）的等价性与守卫测试。

核心主张：`cond_mode='text_only'` 且 `text_loss_weight=0` 时，thinker 的输出会被
`_fuse_talker_condition` 的 `torch.zeros_like` 整段丢弃，因此把真实 thinker 前向换成
同形状零张量后，**talker 看到的 condition 逐位相同**，speech loss / logits 也逐位相同。

这里用 tiny 夹具（`make_smoke_fixture.py` 的同规格小模型）在 CPU 上做对拍，不需要 GPU。
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from streaming_si.model.base_model import OmniSIConfig  # noqa: E402
from streaming_si.model.model_si import SI  # noqa: E402


def _tiny_config(**over):
    """足够小的 SI 配置（CPU 秒级），talker 与 thinker 维度刻意不同以暴露形状错误。"""
    cfg = dict(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64,
        speech_token_size=27, talker_special_tokens=4,
        talker_hidden_size=16, talker_num_hidden_layers=2, talker_num_attention_heads=4,
        talker_num_key_value_heads=2, talker_head_dim=4, talker_intermediate_size=32,
        max_position_embeddings=128, audio_special_token='<|audio_pad|>', audio_ids=[63],
        thinker_arch='qwen3', talker_preset='0.1b',
    )
    cfg.update(over)
    return OmniSIConfig(**cfg)


def _build(cond_mode, skip):
    torch.manual_seed(0)
    cfg = _tiny_config(cond_mode=cond_mode, skip_thinker_when_text_only=skip,
                       text_loss_weight=0.0, speech_loss_weight=1.0)
    model = SI(cfg)
    model.eval()
    return model


def _batch(B=2, L=9, S=5, vocab=64, speech=27):
    torch.manual_seed(123)
    input_ids = torch.randint(0, vocab - 1, (B, L))
    answer_mask = torch.zeros(B, L, dtype=torch.bool)
    answer_mask[:, L // 2:] = True          # 后半段是 answer（talker 条件来源）
    speech_tokens = torch.randint(0, speech, (B, S))
    speech_lens = torch.full((B,), S, dtype=torch.long)
    return dict(input_ids=input_ids, answer_mask=answer_mask,
                speech_tokens=speech_tokens, speech_token_lens=speech_lens)


# ---------------------------------------------------------------------------
# 1. 核心等价性：text_only 下跳过 thinker，speech_logits 逐位相同
# ---------------------------------------------------------------------------
def test_skip_thinker_bitwise_equal_under_text_only():
    ref = _build('text_only', skip=False)
    alt = _build('text_only', skip=True)
    alt.load_state_dict(ref.state_dict())   # 同权重，唯一差异 = 是否跑 thinker

    b = _batch()
    with torch.no_grad():
        o_ref = ref(**b, skip_text_logits=True)
        o_alt = alt(**b, skip_text_logits=True)

    assert ref.can_skip_thinker(skip_text_logits=True) is False, 'ref 臂不应跳过'
    assert alt.can_skip_thinker(skip_text_logits=True) is True, 'alt 臂应当跳过'
    assert o_ref.speech_logits is not None
    # 逐位相等（不是 allclose）：cond 路两边最终都是 zeros，text 路与 thinker 无关
    assert torch.equal(o_ref.speech_logits, o_alt.speech_logits), \
        f'speech_logits 不一致，最大差 {(o_ref.speech_logits - o_alt.speech_logits).abs().max()}'
    assert torch.equal(o_ref.speech_target, o_alt.speech_target)


# ---------------------------------------------------------------------------
# 2. thinker_proj(0) == 0：等价性的数学依据（GatedBridge 全 bias=False + RMSNorm）
# ---------------------------------------------------------------------------
def test_gated_bridge_maps_zero_to_zero():
    m = _build('text_only', skip=True)
    z = torch.zeros(2, 3, m.config.hidden_size)
    out = m.talker.thinker_proj(z)
    assert torch.equal(out, torch.zeros_like(out)), 'GatedBridge(0) 必须为 0，否则等价性不成立'


# ---------------------------------------------------------------------------
# 3. 守卫：任一前置条件不满足就不能跳过
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cond_mode,skip_flag,kwargs,expected,why', [
    ('full',      True,  dict(skip_text_logits=True),                      False, 'full 模式真的要用 thinker hidden'),
    ('text_only', False, dict(skip_text_logits=True),                      False, '开关默认关 = 零回归'),
    ('text_only', True,  dict(skip_text_logits=False),                     False, 'text_w>0 时 lm_head 要用 last_hidden_state'),
    ('text_only', True,  dict(skip_text_logits=True, use_cache=True),      False, '推理要 past_key_values'),
    ('text_only', True,  dict(skip_text_logits=True, audio_inputs=object()), False, '有音频输入必须经 thinker 注入'),
    ('text_only', True,  dict(skip_text_logits=True),                      True,  'S1 的正常情形'),
])
def test_can_skip_thinker_guards(cond_mode, skip_flag, kwargs, expected, why):
    m = _build(cond_mode, skip=skip_flag)
    assert m.can_skip_thinker(**kwargs) is expected, why


def test_mid_layer_bridge_blocks_skip():
    """talker_bridge_layer>=0 要取中间层 hidden_states，零张量替代未实现，必须不跳。"""
    torch.manual_seed(0)
    cfg = _tiny_config(cond_mode='text_only', skip_thinker_when_text_only=True,
                       text_loss_weight=0.0, speech_loss_weight=1.0, talker_bridge_layer=1)
    m = SI(cfg)
    assert m.can_skip_thinker(skip_text_logits=True) is False


# ---------------------------------------------------------------------------
# 4. full 模式下行为必须逐字节不变（S3 / S3-w / S4 回归保护）
# ---------------------------------------------------------------------------
def test_full_mode_unaffected_by_flag():
    ref = _build('full', skip=False)
    alt = _build('full', skip=True)        # 开了开关，但 full 模式下不该生效
    alt.load_state_dict(ref.state_dict())

    b = _batch()
    with torch.no_grad():
        o_ref = ref(**b, skip_text_logits=True)
        o_alt = alt(**b, skip_text_logits=True)
    assert torch.equal(o_ref.speech_logits, o_alt.speech_logits), \
        'full 模式开启 SI_SKIP_THINKER 后结果变了 —— S3/S4 会被破坏'


# ---------------------------------------------------------------------------
# 5. 零张量的形状 / dtype / device 必须与真实 thinker 输出一致
# ---------------------------------------------------------------------------
def test_zero_thinker_out_matches_real_shape():
    m = _build('text_only', skip=True)
    b = _batch()
    with torch.no_grad():
        real = m.forward_thinker(b['input_ids'])
        fake = m.zero_thinker_out(b['input_ids'])
    assert fake.last_hidden_state.shape == real.last_hidden_state.shape
    assert fake.last_hidden_state.dtype == real.last_hidden_state.dtype
    assert fake.last_hidden_state.device == real.last_hidden_state.device
    assert torch.equal(fake.last_hidden_state, torch.zeros_like(fake.last_hidden_state))
