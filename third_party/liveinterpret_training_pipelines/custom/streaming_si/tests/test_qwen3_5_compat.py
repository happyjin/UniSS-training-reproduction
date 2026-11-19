"""Qwen3.5 文本主干兼容实现对拍：transformers>=5.2 官方 `Qwen3_5ForCausalLM` vs 本仓库 4.57 兼容实现。

两个版本的 transformers 不能共存，分两步（随机权重，不需要下载 checkpoint）：

  # 1) 装有 transformers>=5.2 的环境生成参考
  python custom/streaming_si/tests/test_qwen3_5_compat.py reference /tmp/qwen3_5_ref.pt
  # 2) 本仓库环境（transformers 4.57.3）加载同一份权重与输入，逐元素比较
  python custom/streaming_si/tests/test_qwen3_5_compat.py compare /tmp/qwen3_5_ref.pt
  # 3) 增量缓存 vs 整段前向（只需本仓库环境；pytest 也会跑）
  python custom/streaming_si/tests/test_qwen3_5_compat.py incremental [cpu|cuda]
  # 4) 定长缓存 + CUDA Graph（qwen3_5_static）vs 整段前向
  python custom/streaming_si/tests/test_qwen3_5_compat.py static [cpu|cuda]

覆盖：键名一一对应、tied lm_head、线性注意力（跨 64 的 chunk 边界、GQA 头扩展）、门控全注意力
（部分旋转 RoPE 对纯文本等价于 interleaved mRoPE）、右 padding 下有效位的 logits、全部层 hidden。
"""
import os
import sys

import torch

TINY = dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, max_position_embeddings=4096, rms_norm_eps=1e-6,
            linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, tie_word_embeddings=True,
            layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'])
ROPE = dict(rope_type='default', rope_theta=10000000.0, partial_rotary_factor=0.25,
            mrope_interleaved=True, mrope_section=[1, 1, 2])     # 旋转维 32*0.25=8 -> 4 个频率
SEQ, LENS = 150, (150, 97)                                      # 150 > 2x64：跨两个 chunk 边界


def _inputs():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, TINY['vocab_size'], (2, SEQ), generator=g)
    mask = torch.zeros(2, SEQ, dtype=torch.long)
    for i, n in enumerate(LENS):
        mask[i, :n] = 1
    return ids, mask


def reference(path):
    import transformers
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(**TINY, rope_parameters=ROPE)
    cfg._attn_implementation = 'eager'
    model = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():                      # 官方初始化把 (1+w) norm 置零、A_log 随机；再加扰动覆盖所有参数
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    ids, mask = _inputs()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        out_nomask = model(input_ids=ids[:1], use_cache=False)
    torch.save(dict(version=transformers.__version__, config=dict(TINY, **{
        'rope_theta': ROPE['rope_theta'], 'partial_rotary_factor': ROPE['partial_rotary_factor']}),
        state_dict={k: v.clone() for k, v in model.state_dict().items()},
        ids=ids, mask=mask, logits=out.logits, hidden=[h.clone() for h in out.hidden_states],
        logits_nomask=out_nomask.logits), path)
    print(f'reference saved: {path} (transformers {transformers.__version__}, {len(model.state_dict())} keys)')


def compare(path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/
    import transformers
    from streaming_si.model.qwen3_5_compat import Qwen3_5CompatForCausalLM, Qwen3_5TextCompatConfig
    ref = torch.load(path, weights_only=False)
    cfg = Qwen3_5TextCompatConfig(**ref['config'])
    cfg._attn_implementation = 'eager'
    model = Qwen3_5CompatForCausalLM(cfg).eval()
    ours, theirs = set(model.state_dict()), set(ref['state_dict'])
    assert ours == theirs, f'键名不一致: 多 {sorted(ours - theirs)[:5]} 缺 {sorted(theirs - ours)[:5]}'
    model.load_state_dict(ref['state_dict'], strict=True)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), 'lm_head 未 tie'
    ids, mask = ref['ids'], ref['mask']
    results = {}
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        nomask = model(input_ids=ids[:1])
        for impl in ('eager', 'sdpa'):
            model.config._attn_implementation = impl
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.config._attn_implementation = impl
            results[impl] = model(input_ids=ids, attention_mask=mask).logits
    valid = mask.bool()
    checks = {
        'logits(有效位, eager)': (results['eager'][valid] - ref['logits'][valid]).abs().max(),
        'logits(有效位, sdpa)': (results['sdpa'][valid] - ref['logits'][valid]).abs().max(),
        'logits(无 mask)': (nomask.logits - ref['logits_nomask']).abs().max(),
        'hidden(全部层, 有效位)': max((a[valid] - b[valid]).abs().max() for a, b in zip(out.hidden_states, ref['hidden'])),
        'hidden 层数': torch.tensor(float(len(out.hidden_states) != len(ref['hidden']))),
    }
    ok = True
    for name, v in checks.items():
        v = float(v)
        passed = v < 2e-5
        ok &= passed
        print(f'{"PASS" if passed else "FAIL"}  {name}: max|Δ|={v:.3e}')
    print(f'reference transformers {ref["version"]} vs local {transformers.__version__}')
    print('ALL PASS' if ok else 'SOME FAILED')
    return 0 if ok else 1


# 增量块大小：单 token（走递归内核）、流式一步（<|chunk|> + 8 帧音频）、跨 64 的大块、再回到单 token
INCR_BLOCKS = (5, 1, 1, 9, 1, 70, 9, 1, 1, 3, 50)


def incremental(device='cpu', impls=('eager', 'sdpa'), tol=None):
    """增量缓存对拍：同一序列整段前向 vs 按 INCR_BLOCKS 逐块追加，比较每个位置的 logits 与末层 hidden。

    随机权重、fp32，不需要 transformers 5.x。device='cuda' 时线性注意力走 fla / causal_conv1d 内核。
    容差：CPU（torch 参考实现）2e-4，实测 ~1e-6；CUDA 3e-3 —— fla 的 chunk 内核内部用 TF32，
    **整段前向本身**与 fp32 精确值就差 ~6e-4（logits 量级 ~2），增量与整段的差（~3e-4）小于这个量级。
    """
    tol = tol if tol is not None else (2e-4 if device == 'cpu' else 3e-3)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/
    from streaming_si.model.qwen3_5_compat import Qwen3_5CompatForCausalLM, Qwen3_5TextCompatConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextCompatConfig(**dict(TINY, rope_theta=ROPE['rope_theta'],
                                         partial_rotary_factor=ROPE['partial_rotary_factor']))
    model = Qwen3_5CompatForCausalLM(cfg).eval().to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    total = sum(INCR_BLOCKS)
    ids = torch.randint(0, TINY['vocab_size'], (1, total), generator=torch.Generator().manual_seed(1)).to(device)
    ok = True
    for impl in impls:
        model.config._attn_implementation = impl
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.config._attn_implementation = impl
        with torch.no_grad():
            full = model(input_ids=ids, output_hidden_states=True)
            cache, pos, logits, hidden = None, 0, [], []
            for n in INCR_BLOCKS:
                out = model(input_ids=ids[:, pos:pos + n], past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
                cache = out.past_key_values
                logits.append(out.logits)
                hidden.append(out.hidden_states[-1])
                pos += n
        d_logit = float((torch.cat(logits, 1) - full.logits).abs().max())
        d_hidden = float((torch.cat(hidden, 1) - full.hidden_states[-1]).abs().max())
        seen = cache.get_seq_length()
        passed = d_logit < tol and d_hidden < tol and seen == total
        ok &= passed
        print(f'{"PASS" if passed else "FAIL"}  incremental[{device}/{impl}] blocks={INCR_BLOCKS}: '
              f'logits max|Δ|={d_logit:.3e}  hidden max|Δ|={d_hidden:.3e}  cache_len={seen}/{total}')
    return ok


STATIC_BLOCKS = (19, 1, 1, 9, 1, 10, 1, 1, 70, 10, 9, 1, 3, 1)   # 含 > 16 的长块（chunk 内核路径）


def static_runner(device='cpu', graph_lens=(), tol=None):
    """定长缓存增量前向（`qwen3_5_static`）vs 整段前向：逐块喂入，比较每个位置的末层 hidden 与 logits。

    CUDA 上另跑一遍「录图」版本，要求与不录图版本逐位相同（同一套内核、同一组缓冲）。
    跑两句（中间 reset），验证状态清零后复用正确。
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/
    from streaming_si.model.qwen3_5_compat import Qwen3_5CompatForCausalLM, Qwen3_5TextCompatConfig
    from streaming_si.model.qwen3_5_static import Qwen3_5StaticRunner
    torch.manual_seed(0)
    cfg = Qwen3_5TextCompatConfig(**dict(TINY, rope_theta=ROPE['rope_theta'],
                                         partial_rotary_factor=ROPE['partial_rotary_factor']))
    cfg._attn_implementation = 'sdpa'
    model = Qwen3_5CompatForCausalLM(cfg).eval().to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    total = sum(STATIC_BLOCKS)
    tol = tol if tol is not None else (2e-4 if device == 'cpu' else 3e-3)
    runs = {'eager': Qwen3_5StaticRunner(model.model, max_len=256, graph_lens=())}
    if graph_lens:
        r = Qwen3_5StaticRunner(model.model, max_len=256, graph_lens=graph_lens)
        r.prepare()
        runs['graph'] = r
    ok, got = True, {}
    for sentence in range(2):
        ids = torch.randint(0, TINY['vocab_size'], (1, total),
                            generator=torch.Generator().manual_seed(10 + sentence)).to(device)
        with torch.inference_mode():
            emb = model.model.embed_tokens(ids)
            ref = model.model(inputs_embeds=emb).last_hidden_state
            for name, r in runs.items():
                r.reset()
                pos, hs = 0, []
                for n in STATIC_BLOCKS:
                    hs.append(r(emb[:, pos:pos + n]).clone())
                    pos += n
                h = torch.cat(hs, 1)
                got[name] = h
                d_h = float((h - ref).abs().max())
                d_l = float((model.lm_head(h) - model.lm_head(ref)).abs().max())
                passed = d_h < tol and d_l < tol and r.pos == total
                ok &= passed
                print(f'{"PASS" if passed else "FAIL"}  static[{device}/{name}] 句 {sentence}: '
                      f'hidden max|Δ|={d_h:.3e}  logits max|Δ|={d_l:.3e}')
        if 'graph' in got:
            d = float((got['graph'] - got['eager']).abs().max())
            ok &= d == 0.0
            print(f'{"PASS" if d == 0.0 else "FAIL"}  static[{device}] 句 {sentence}: 录图 vs 不录图 max|Δ|={d:.3e}')
    return ok


def test_static_runner_matches_full_cpu():
    assert static_runner('cpu')


def test_static_runner_cuda_graph():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('无 CUDA')
    assert static_runner('cuda', graph_lens=(1, 9, 10))


def test_incremental_cache_matches_full_cpu():
    assert incremental('cpu')


def test_incremental_cache_matches_full_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('无 CUDA')
    assert incremental('cuda')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'incremental':
        dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
        sys.exit(0 if incremental(dev) else 1)
    if mode == 'static':
        dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
        sys.exit(0 if static_runner(dev, graph_lens=(1, 9, 10) if dev == 'cuda' else ()) else 1)
    path = sys.argv[2]
    sys.exit(reference(path) if mode == 'reference' else compare(path))
