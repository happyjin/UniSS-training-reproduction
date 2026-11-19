"""S4 swift 接线 vs 源项目的逐项对拍（采样计划 / 样本编码 / collate / loss / macro-F1）。

需要：
  SI_SRC_ROOT          源 streaming-si-0.6b 仓库（import 其 dataset / trainer 作为参照）
  SI_QWEN3_TOKENIZER   原始 Qwen3-0.6B tokenizer 目录（源侧用 build_qwen_tokenizer 现场构造）
  SI_SMOKE_DIR         make_smoke_fixture.py 产出目录（默认 /tmp/si_smoke，需含 model/ 与 s4/）

    SI_SRC_ROOT=... SI_QWEN3_TOKENIZER=... python -m pytest custom/streaming_si/tests/test_s4_pipeline.py -q
"""
import ast
import importlib.util
import os
import random
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))              # custom/streaming_si/tests
CUSTOM_DIR = os.path.dirname(os.path.dirname(HERE))           # custom/
sys.path.insert(0, CUSTOM_DIR)

SRC = os.environ.get('SI_SRC_ROOT', '')
QWEN = os.environ.get('SI_QWEN3_TOKENIZER', '')
SMOKE = os.environ.get('SI_SMOKE_DIR', '/tmp/si_smoke')
pytestmark = pytest.mark.skipif(
    not (os.path.isdir(SRC) and os.path.isdir(QWEN) and os.path.isdir(os.path.join(SMOKE, 's4'))),
    reason='需要 SI_SRC_ROOT / SI_QWEN3_TOKENIZER 与 S4 冒烟夹具')

S4 = os.path.join(SMOKE, 's4')
BASE_ENV = {
    'SI_S4_DATA': ','.join(os.path.join(S4, 'si_s4', 'records', d, 'chunk_640ms', 'train', 'records')
                           for d in ('en2zh', 'zh2en')),
    'SI_S4_DATA_ROOT': os.path.join(S4, 'si_s4', 'audio'),
    'SI_S4_VAL_DIR': os.path.join(S4, 'val'),
    'SI_S4_INDEX_DIR': os.path.join(S4, 'index'),
    'SI_S4_SEED': '1337', 'SI_MAX_LENGTH': '640', 'SI_MAX_AUDIO_SEC': '20', 'SI_MAX_SPEECH_LEN': '750',
}
VARIANTS = {
    'stratified_m1': {},
    'uniform_m_sample': {'SI_S4_STRATIFIED': '0', 'SI_S4_LATENCY_MULTIPLIER': 'sample'},
    'prefix_drop': {'SI_S4_P_DROP': '0.5', 'SI_S4_PREFIX_CONTEXT_TURNS': '4', 'SI_S4_TRAIN_SCOPE': 'thinker',
                    'SI_SPEECH_LOSS_WEIGHT': '0'},
    'paired_control': {'SI_S4_P_DROP': '0.5', 'SI_S4_PREFIX_CONTEXT_TURNS': '4', 'SI_S4_PAIRED_CONTROL': '1',
                       'SI_S4_TRAIN_SCOPE': 'thinker', 'SI_SPEECH_LOSS_WEIGHT': '0'},
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in list(os.environ):
        if k.startswith('SI_S4_') or k in ('SI_SPEECH_LOSS_WEIGHT',):
            monkeypatch.delenv(k, raising=False)
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    yield monkeypatch


def _set(monkeypatch, variant):
    for k, v in VARIANTS[variant].items():
        monkeypatch.setenv(k, v)


def _src_modules():
    """源仓库的顶层包 `dataset` 与本仓库 `custom/dataset.py` 同名：导入时临时让源仓库优先，
    随后把源 `dataset.*` 移出 sys.modules 并恢复本仓库的同名模块。源 `model` 包（本仓库无同名
    模块）保留在 sys.modules：源 S4StreamingDataset.__init__ 内有延迟的 `from model.base_model import`。"""
    if 'src_mods' in _CACHE:
        return _CACHE['src_mods']
    saved_path = list(sys.path)
    shadow = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'dataset' or k.startswith('dataset.')}
    sys.path[:] = [SRC] + [p for p in sys.path if os.path.abspath(p or '.') != CUSTOM_DIR]
    try:
        from dataset.s4_context_views import S4ViewIndex as SrcView
        from dataset.s4_streaming_dataset import S4StreamingDataset as SrcDS
        from dataset.s4_streaming_dataset import s4_collate_fn as src_collate
        from dataset import si_sampler as src_sampler
        from model.base_model import build_qwen_tokenizer
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == 'dataset' or k.startswith('dataset.')]:
            sys.modules[f'_src_{k}'] = sys.modules.pop(k)
        sys.modules.update(shadow)
    _CACHE['src_mods'] = (SrcDS, SrcView, src_collate, src_sampler, build_qwen_tokenizer)
    return _CACHE['src_mods']


def _src_function(relpath, name):
    """从源文件抽出单个函数定义执行（避免 import 整个训练脚本及其 peft/分布式依赖）。"""
    path = os.path.join(SRC, relpath)
    text = open(path, encoding='utf-8').read()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'torch': torch}
    exec(compile(ast.Module([fn], []), path, 'exec'), ns)
    return ns[name]


def _load_src_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CACHE = {}


def _tokenizers():
    if 'tok' not in _CACHE:
        from transformers import AutoTokenizer
        from streaming_si.data.s4_pipeline import add_s4_stream_tokens
        ours = AutoTokenizer.from_pretrained(os.path.join(SMOKE, 'model'))
        add_s4_stream_tokens(ours, capacity=151936)
        build = _src_modules()[4]
        _CACHE['tok'] = ours, build(QWEN, add_stream_tokens=True, embedding_capacity=151936)
    return _CACHE['tok']


def _mel():
    if 'mel' not in _CACHE:
        from streaming_si.model.base_model import VoxtralAudioProcessor
        from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
        _CACHE['mel'] = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(os.path.join(SMOKE, 'model')))
    return _CACHE['mel']


def _src_train_ds(cfg, weights=None, thinker=None):
    """按源 train_s4_stream.main 的参数构造（world_size=1，与 swift 全局一份数据对应）。"""
    SrcDS = _src_modules()[0]
    from streaming_si.data.s4_pipeline import read_val_meta
    exclude = set()
    for m in read_val_meta(cfg['val_dir']):
        exclude |= set(m['keys'])
    ds = SrcDS(cfg['data'], _tokenizers()[1], data_root=cfg['data_root'], audio_processor=_mel(),
               max_length=cfg['max_length'], max_audio_sec=cfg['max_audio_sec'],
               max_speech_len=cfg['max_speech_len'], rank=0, world_size=1, use_src_end=True,
               strict_text=False, align_ranks=True, exclude_keys=exclude,
               latency_multiplier=cfg['latency_multiplier'], m_max=cfg['m_max'], m_schedule=cfg['m_schedule'],
               seed=cfg['seed'], prefix_context_turns=cfg['prefix_context_turns'])
    if weights is not None:
        ds.configure_context_view_pools(weights, thinker)
    return ds


def _src_plan(cfg, ds):
    """源 main() 第 5 节：weights_for_dataset ->（p_drop>0）static eligibility。"""
    import numpy as np
    sampler = _src_modules()[3]
    w = np.ones(len(ds))
    if cfg['stratified']:
        w, _, _ = sampler.weights_for_dataset(ds, cfg['index_dir'], min_source_ms=cfg['min_source_ms'],
                                              max_repeat=cfg['max_repeat'], logger=lambda *_: None)
    thinker = np.zeros(0, dtype=np.int64)
    if cfg['p_drop'] > 0:
        static = sampler.static_context_eligibility_for_dataset(
            ds, cfg['index_dir'], thinker_ratio=cfg['p_drop'], talker_ratio=0.0,
            prefix_context_turns=cfg['prefix_context_turns'],
            min_source_ms=max(int(cfg['min_source_ms']), int(ds.min_audio_sec * 1000)),
            max_source_ms=int(cfg['max_audio_sec'] * 1000), max_speech_tokens=cfg['max_speech_len'])
        w = np.asarray(w, dtype=np.float64).copy()
        w[~static.valid_mask] = 0.0
        thinker = np.flatnonzero(static.thinker_mask & (w > 0))
    return w, thinker


TENSOR_KEYS = ('mel', 'audio_seg_lens', 'speech_tokens', 'text_seg_lens', 'speech_seg_lens')
VIEW_KEYS = ('record_index', 'scheduled_record_index', 'designated_view_kind', 'view_kind', 'draw_seed',
             'live_turn', 'retained_start_turn', 'replacement_count', 'audio_feature_offset',
             'speech_loss_weight')


def _assert_item_equal(ours, ref):
    x, y, a = ours['input_ids'][:-1], ours['labels'][1:], ours['answer_mask'][:-1]
    assert torch.equal(x, ref['input_ids']) and torch.equal(y, ref['labels']) and torch.equal(a, ref['answer_mask'])
    for k in TENSOR_KEYS:
        assert torch.equal(ours[k], ref[k]), k
    assert ours['record_id'] == ref['record_id'] and ours['speech_len'] == ref['speech_len']
    for k in VIEW_KEYS:
        assert (k in ours) == (k in ref), k
        if k in ref:
            assert ours[k] == ref[k], k


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_sampling_plan_matches_source(_env, variant):
    import numpy as np
    from streaming_si.data.s4_pipeline import build_sampling_plan, s4_env
    _set(_env, variant)
    cfg = s4_env()
    plan = build_sampling_plan(cfg)
    w, thinker = _src_plan(cfg, _src_train_ds(cfg))
    assert np.array_equal(plan['weights'], w)
    assert np.array_equal(plan['thinker_eligible'], thinker)
    if cfg['p_drop'] > 0:
        assert len(thinker) > 0, '夹具应包含可 crop 的长样本'


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_encoded_rows_match_source_dataset(_env, variant):
    from streaming_si.data.s4_pipeline import S4Encoder, build_sampling_plan, build_train_rows, s4_env
    SrcView = _src_modules()[1]
    _set(_env, variant)
    cfg = s4_env()
    rows = build_train_rows(cfg)
    plan = build_sampling_plan(cfg)
    src = _src_train_ds(cfg, *(
        (plan['weights'], plan['thinker_eligible']) if plan['positive'] else ()))
    enc = S4Encoder(_tokenizers()[0], _mel())
    kinds = set()
    for i in range(len(rows['s4_index'])):
        row = {k: v[i] for k, v in rows.items()}
        ours = enc.encode(row)
        src._m_rng = random.Random(int(row['s4_seed']))
        if row['s4_designated']:
            ref = src[SrcView(row['s4_index'], row['s4_designated'], row['s4_applied'], row['s4_seed'])]
            kinds.add((ref['designated_view_kind'], ref['view_kind']))
        else:
            ref = src[row['s4_index']]
        _assert_item_equal(ours, ref)
    if variant == 'prefix_drop':
        assert ('thinker_crop', 'thinker_crop') in kinds and ('full', 'full') in kinds
    if variant == 'paired_control':
        assert ('thinker_crop', 'full') in kinds
    if variant == 'uniform_m_sample':
        assert len(enc.train_ds().m_stats) > 1, 'm=sample 应抽到多个延迟倍数'
    assert enc.train_ds().skip_stats.get('val_holdout', 0) > 0, 'val 源文本组应被剔除'


def test_val_rows_match_source(_env):
    from streaming_si.data.s4_pipeline import S4Encoder, build_val_rows, read_val_meta, s4_env
    SrcDS = _src_modules()[0]
    _env.setenv('SI_S4_VAL_LIMIT', '1')
    cfg = s4_env()
    rows = build_val_rows(cfg)
    metas = {m['direction']: m for m in read_val_meta(cfg['val_dir'])}
    assert sorted(set(rows['s4_split'])) == ['val:en2zh', 'val:zh2en'] and len(rows['s4_index']) == 2
    enc = S4Encoder(_tokenizers()[0], _mel())
    for split, idx in zip(rows['s4_split'], rows['s4_index']):
        m = metas[split[4:]]
        ref_ds = SrcDS(m['jsonl'], _tokenizers()[1], data_root=m['audio_root'] or cfg['data_root'],
                       audio_processor=_mel(), max_length=640, max_audio_sec=20.0, max_speech_len=750,
                       use_src_end=True)
        row = {'si_stage': 's4', 's4_split': split, 's4_index': idx, 's4_designated': '', 's4_applied': '',
               's4_seed': 0, 's4_plan': ''}
        _assert_item_equal(enc.encode(row), ref_ds[idx])


def _batch(variant, _env, n=4):
    from streaming_si.data.s4_pipeline import S4Encoder, build_train_rows, collate_s4, s4_env
    _set(_env, variant)
    rows = build_train_rows(s4_env())
    enc = S4Encoder(_tokenizers()[0], _mel())
    items = [enc.encode({k: v[i] for k, v in rows.items()}) for i in range(n)]
    return items, collate_s4([dict(it) for it in items])


@pytest.mark.parametrize('variant', ['stratified_m1', 'prefix_drop'])
def test_collate_then_model_shift_equals_source_collate(_env, variant):
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration as M
    src_collate = _src_modules()[2]
    items, batch = _batch(variant, _env)
    ref = src_collate([{**it, 'input_ids': it['input_ids'][:-1], 'labels': it['labels'][1:],
                        'answer_mask': it['answer_mask'][:-1]} for it in items])
    x, y, a, att = M._shifted_views(batch['input_ids'], batch['labels'], batch['answer_mask'], None)
    assert att is None and 'attention_mask' not in batch        # 与源一致：不传 attention_mask
    assert torch.equal(y, ref['labels']) and torch.equal(a, ref['answer_mask'])
    for i, it in enumerate(items):                              # 较短样本在 x 末尾多一个 eos（无监督位）
        n = it['input_ids'].numel() - 1
        assert torch.equal(x[i, :n], ref['input_ids'][i, :n])
        assert bool((y[i, n:] == -100).all())
    for k, v in ref.items():
        if k in ('input_ids', 'labels', 'answer_mask'):
            continue
        got = batch[k]
        if torch.is_tensor(v):
            assert torch.equal(got, v), k
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            assert all(torch.equal(g, r) for g, r in zip(got, v)), k
        else:
            assert got == v, k


@pytest.mark.parametrize('variant,speech_w', [('stratified_m1', 1.0), ('prefix_drop', 0.0), ('prefix_drop', 1.0)])
def test_loss_matches_source_formula(_env, variant, speech_w):
    """wrapper.forward 的 loss == 源 train_epoch：text_w·split_losses + speech_w·weighted_speech_ce + aux。"""
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration as M
    from streaming_si.model.s4_objective import finalize_decision_metrics
    from streaming_si.data.s4_pipeline import s4_stream_token_ids
    items, batch = _batch(variant, _env)
    torch.manual_seed(0)
    model = M.from_pretrained(os.path.join(SMOKE, 'model')).eval()
    ids = s4_stream_token_ids(_tokenizers()[0])
    model.config.s4_decision_weight, model.config.s4_stream_token_ids = 1.0, ids
    model.config.speech_loss_weight = speech_w
    fwd = {k: v for k, v in batch.items() if k not in ('labels', 'speech_loss_weights')}
    with torch.no_grad():
        out = model(**batch)
        x, y, a, _ = M._shifted_views(batch['input_ids'], batch['labels'], batch['answer_mask'], None)
        y = y.contiguous()               # 源 collate 产出的 labels 本就连续（源 split_losses 用 .view）
        res = super(M, model).forward(
            x, audio_inputs=fwd['audio_inputs'], audio_lens=fwd['audio_lens'],
            answer_mask=a if speech_w else None, speech_tokens=fwd['speech_tokens'],
            speech_token_lens=fwd['speech_token_lens'], audio_seg_lens=fwd['audio_seg_lens'],
            audio_feature_offsets=fwd.get('audio_feature_offsets'),
            text_seg_lens=fwd['text_seg_lens'], speech_seg_lens=fwd['speech_seg_lens'])
    split_losses = _src_function('trainer/train_s4_stream.py', 'split_losses')
    src_losses = _load_src_module('trainer/s4_losses.py', 'src_s4_losses')
    dec_ids = torch.tensor(ids[:2])
    total, _, _, dec_hit, dec_n, _, bnd_hit, bnd_n = split_losses(
        res.logits, y, dec_ids, torch.nn.CrossEntropyLoss(reduction='none'), 1.0, ids[2])
    speech = res.aux_loss * 0
    if res.speech_logits is not None:
        speech = src_losses.weighted_speech_ce(res.speech_logits, res.speech_target,
                                               batch.get('speech_loss_weights'))
    ref = 1.0 * total + speech_w * speech + res.aux_loss
    assert torch.allclose(out.loss, ref, atol=1e-6), (float(out.loss), float(ref))
    assert abs(float(out.metrics['dec_acc'].mean()) - float(dec_hit / dec_n)) < 1e-6
    assert abs(float(out.metrics['bnd_acc'].mean()) - float(bnd_hit / bnd_n)) < 1e-6
    # macro-F1：逐 token 均值还原 == 源 macro_f1_from_cm(计数)
    macro_f1 = _src_function('trainer/train_s4_stream.py', 'macro_f1_from_cm')
    flat_y, pred = y.reshape(-1), res.logits.argmax(-1).reshape(-1)
    counts = [[0, 0, 0], [0, 0, 0]]
    for t, p in zip(flat_y.tolist(), pred.tolist()):
        if t in ids[:2]:
            r = ids.index(t)
            counts[r][0 if p == ids[0] else 1 if p == ids[1] else 2] += 1
    logs = {k: float(v.mean()) for k, v in out.metrics.items() if k.startswith('dec_cm_')}
    finalize_decision_metrics(logs)
    assert abs(logs['dec_macro_f1'] - macro_f1(counts)['dec_macro_f1']) < 1e-6
    assert not any(k.startswith('dec_cm_') for k in logs)
