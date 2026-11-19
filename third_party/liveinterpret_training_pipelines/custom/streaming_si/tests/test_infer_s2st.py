"""infer_s2st.py 与 speech_decoder.py 的 CPU 单测（tiny 随机权重）。

    python -m pytest custom/streaming_si/tests/test_infer_s2st.py -q
    SI_SMOKE_DIR=/tmp/si_smoke python -m pytest custom/streaming_si/tests/test_infer_s2st.py -q   # 另跑 CLI 端到端

核心断言：推理拼出的 Thinker / Talker 输入与训练布局逐位一致——
  * Talker 条件 = 训练 shift 后 X / answer_mask 取出的条件（译文位置，不含 eos）；
  * Talker 自回归（KV cache）逐步 argmax = 同一序列 teacher forcing 整段前向的 argmax；
  * Thinker KV cache 生成 = 每步整段重算（Qwen3.5 兼容层走后者）。
flow / HiFT 本身需要 CosyVoice 官方仓库与权重，不在单测内。
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.inference import infer_s2st  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.model.speech_decoder import (SPEECH_DECODER_DIR, bundle_speech_decoder,  # noqa: E402
                                         resolve_speech_decoder)
from streaming_si.tests.test_streaming_si import AUDIO_ID, tiny_config  # noqa: E402
from streaming_si.tests.test_variant_4b import tiny_4b_config  # noqa: E402

SMOKE = os.environ.get('SI_SMOKE_DIR', '/tmp/si_smoke')


def _inputs(n_frames=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = [AUDIO_ID] * n_frames + torch.randint(10, 400, (4,), generator=g).tolist()
    text = torch.randint(10, 400, (5,), generator=g).tolist()
    mel = torch.randn(128, n_frames * 8, generator=g)
    return prompt, text, mel


@pytest.fixture(scope='module', params=['qwen3', 'qwen3_5'])
def model(request):
    torch.manual_seed(0)
    cfg = tiny_config() if request.param == 'qwen3' else tiny_4b_config()
    return StreamingSIForConditionalGeneration(cfg).eval()


def test_talker_condition_matches_training_layout(model):
    """训练：ids = prompt + 译文 + eos，X = ids[:-1]、am = answer_mask[:-1]（只覆盖译文，不含 eos）。"""
    prompt, text, mel = _inputs()
    eos = int(model.config.eos_token_id) % int(model.config.vocab_size)
    ids = prompt + text + [eos]
    am = [0] * len(prompt) + [1] * (len(text) + 1)
    x = torch.tensor([ids[:-1]])
    am_x = torch.tensor([am[:-1]])
    with torch.no_grad():
        out = model.forward_thinker(x, audio_inputs=mel[None], audio_lens=torch.tensor([mel.size(-1)]))
        ref, lens = model.build_talker_condition(out, x, am_x)
    got = infer_s2st.talker_condition(model, prompt, text, mel)
    assert int(lens[0]) == len(text) and got.shape == ref.shape
    assert torch.equal(got, ref)


def test_thinker_cache_matches_full_recompute():
    torch.manual_seed(0)
    m = StreamingSIForConditionalGeneration(tiny_config()).eval()
    prompt, _, mel = _inputs(seed=1)
    kw = dict(eos_id=-1, max_new_tokens=6)                       # eos 不可达：固定生成 6 个
    cached, stop = infer_s2st.generate_text(m, prompt, mel, **kw)
    full, _ = infer_s2st.generate_text(m, prompt, mel, thinker_cache=False, **kw)   # 每步整段重算
    assert stop == 'max_new_tokens' and len(cached) == 6
    assert cached == full


def test_thinker_stops_at_eos(model):
    prompt, _, mel = _inputs(seed=2)
    first, _ = infer_s2st.generate_text(model, prompt, mel, eos_id=-1, max_new_tokens=1)
    toks, stop = infer_s2st.generate_text(model, prompt, mel, eos_id=first[0], max_new_tokens=5)
    assert toks == [] and stop == 'eos'


def test_talker_ar_matches_teacher_forcing(model, monkeypatch):
    """KV cache 逐步生成与训练布局 [sos]+cond+[task]+speech 的整段前向逐位一致（argmax 采样）。"""
    talker = model.talker
    monkeypatch.setattr(talker, '_ras_sampling', lambda scores, *a, **k: int(scores.argmax()))
    prompt, text, mel = _inputs(seed=3)
    cond = infer_s2st.talker_condition(model, prompt, text, mel)
    tokens, stop = infer_s2st.generate_speech(model, cond, max_speech_tokens=6)
    assert stop in ('eos', 'segment_cap') and tokens
    assert all(0 <= t < model.config.speech_token_size for t in tokens)
    with torch.no_grad():
        logits, target = talker(cond, torch.tensor([cond.size(1)]),
                                speech_tokens=torch.tensor([tokens]), speech_lens=torch.tensor([len(tokens)]))
    logp = logits[0].float().log_softmax(-1)
    pos = (target[0] != -100).nonzero().flatten()
    masked = logp[pos].clone()
    eos_col = masked[:, talker.eos_token].clone()
    masked[:, model.config.speech_token_size:] = -float('inf')    # 与 finalize 分支同一掩码
    masked[:, talker.eos_token] = eos_col
    pred = masked.argmax(-1).tolist()
    assert pred[:len(tokens)] == tokens
    if stop == 'eos':
        assert pred[len(tokens)] == talker.eos_token


def _fake_cosy(d, with_campplus=True):
    os.makedirs(d)
    for name in ('flow.pt', 'hift.pt', 'cosyvoice3.yaml') + (('campplus.onnx',) if with_campplus else ()):
        with open(os.path.join(d, name), 'wb') as f:
            f.write(name.encode())
    return d


def test_bundle_and_resolve_speech_decoder(tmp_path):
    cosy = _fake_cosy(str(tmp_path / 'Fun-CosyVoice3-0.5B'))
    model_dir = str(tmp_path / 'model')
    os.makedirs(model_dir)
    with pytest.raises(FileNotFoundError, match='找不到 flow/HiFT'):
        resolve_speech_decoder(model_dir)

    dec = bundle_speech_decoder(cosy, model_dir)
    assert dec == os.path.join(model_dir, SPEECH_DECODER_DIR)
    manifest = json.load(open(os.path.join(dec, 'manifest.json')))
    assert manifest['config_yaml'] == 'cosyvoice3.yaml' and manifest['has_campplus']
    assert sorted(os.listdir(dec)) == ['campplus.onnx', 'cosyvoice3.yaml', 'flow.pt', 'hift.pt', 'manifest.json']
    assert resolve_speech_decoder(model_dir) == (dec, 'cosyvoice3.yaml')
    assert resolve_speech_decoder(str(tmp_path), explicit=cosy) == (cosy, 'cosyvoice3.yaml')

    # S4 adapter 目录只存 adapter：回到 s4_metadata.json 的 base_model_dir 找
    adapter = str(tmp_path / 'adapter')
    os.makedirs(adapter)
    json.dump({'base_model_dir': model_dir}, open(os.path.join(adapter, 's4_metadata.json'), 'w'))
    assert resolve_speech_decoder(adapter)[0] == dec

    bad = str(tmp_path / 'bad')
    os.makedirs(bad)
    with pytest.raises(FileNotFoundError, match='缺'):
        bundle_speech_decoder(bad, model_dir)


@pytest.mark.skipif(not os.path.isdir(os.path.join(SMOKE, 'model')), reason=f'需要冒烟夹具 {SMOKE}')
@pytest.mark.parametrize('name', ['model', 'model_4b'])
def test_cli_end_to_end_without_vocoder(tmp_path, name):
    import soundfile as sf
    model_dir = os.path.join(SMOKE, name)
    if not os.path.isdir(model_dir):
        pytest.skip(f'缺 {model_dir}')
    wav = str(tmp_path / 'src.wav')
    sf.write(wav, (np.random.RandomState(0).randn(16000) * 0.05).astype(np.float32), 16000)
    jsonl = str(tmp_path / 'eval.jsonl')
    with open(jsonl, 'w') as f:
        for i, lang in enumerate(['zh', 'en']):
            f.write(json.dumps({'id': f'u{i}', 'audio': wav, 'tgt_lang': lang, 'reference': 'x'}) + '\n')
        f.write(json.dumps({'id': 'missing', 'audio': str(tmp_path / 'nope.wav'), 'tgt_lang': 'zh'}) + '\n')
    out = str(tmp_path / 'out')
    infer_s2st.main(['--model_dir', model_dir, '--input_jsonl', jsonl, '--out_dir', out, '--no_vocoder',
                     '--device', 'cpu', '--dtype', 'float32', '--max_new_tokens', '4', '--max_speech_tokens', '5'])
    rows = [json.loads(line) for line in open(os.path.join(out, 'results.jsonl'))]
    summary = json.load(open(os.path.join(out, 'summary.json')))
    assert [r['id'] for r in rows] == ['u0', 'u1']
    assert summary['completed'] == 2 and summary['failed'] == 1 and summary['failures'][0]['id'] == 'missing'
    for r in rows:
        assert r['status'] in ('ok', 'empty_text', 'empty_speech') and 'wav' not in r
        assert len(r['text_token_ids']) <= 4 and r['n_speech_tokens'] <= 5


# ---------------------------------------------------------------------------
# talker 静态缓存档位（talker_cache_len）
# ---------------------------------------------------------------------------
def test_talker_cache_ladder():
    f = infer_s2st.talker_cache_ladder
    assert f(None) == (2048,)                        # 缺省 = 旧行为，不改变任何现有结果
    assert f('auto') == (512, 1024, 2048)
    assert f(1024) == (1024, 2048)
    assert f(700) == (1024, 2048)                    # 取不小于它的档位
    assert f(4096) == (2048,)                        # 超过最大档：只剩兜底档


def test_generate_speech_escalates_on_cache_overflow(model, monkeypatch):
    """512 档溢出 -> 丢弃这次生成、恢复随机数状态、用 1024 档重来；两次看到的随机数必须相同。"""
    prompt, text, mel = _inputs(seed=5)
    cond = infer_s2st.talker_condition(model, prompt, text, mel)
    seen = []

    def fake_stream(state, **kw):
        seen.append((None, float(torch.rand(1))))
        if len(seen) == 1:
            torch.rand(7)                                      # 失败的那次消耗了随机数
            raise RuntimeError('talker 序列超过 StaticCache 上限 512（已有 511，追加 1）')
        yield 11
        yield 22
        state.last_stop_reason = 'eos'

    used = []
    real_state = infer_s2st.TalkerStreamState
    monkeypatch.setattr(infer_s2st, 'TalkerStreamState',
                        lambda talker, **kw: (used.append(kw['max_len']), real_state(talker, **kw))[1])
    monkeypatch.setattr(model.talker, 'stream_generate_s4', fake_stream)
    toks, stop = infer_s2st.generate_speech(model, cond, cache_len='auto')
    assert toks == [11, 22] and stop == 'eos'
    assert used == [512, 1024]                                 # 先 512，溢出后 1024
    assert seen[0][1] == seen[1][1]                            # 随机数状态已恢复


def test_generate_speech_other_errors_not_swallowed(model, monkeypatch):
    prompt, text, mel = _inputs(seed=6)
    cond = infer_s2st.talker_condition(model, prompt, text, mel)

    def boom(state, **kw):
        raise RuntimeError('some other failure')
        yield

    monkeypatch.setattr(model.talker, 'stream_generate_s4', boom)
    with pytest.raises(RuntimeError, match='some other failure'):
        infer_s2st.generate_speech(model, cond, cache_len='auto')


def test_decode_graph_cached_per_length(model):
    """不同缓存长度各自一张图（以前请求更小的长度会复用已有的大图）。"""
    if not torch.cuda.is_available():
        pytest.skip('CUDA Graph 需要 GPU')
    t = model.talker
    a, b = t.decode_graph(512), t.decode_graph(1024)
    assert a is not b and a.max_len == 512 and b.max_len == 1024
    assert t.decode_graph(512) is a
