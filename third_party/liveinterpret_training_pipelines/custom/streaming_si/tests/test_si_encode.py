"""si_encode 与源项目 dataset 的逐 token 对拍。

    SI_SRC_ROOT=/path/to/streaming-si-0.6b SI_QWEN3_TOKENIZER=/path/to/Qwen3-0.6B \
        python -m pytest custom/streaming_si/tests/test_si_encode.py -q

两个环境变量缺一即 skip：对拍需要源 dataset 类（参考实现）与真实 Qwen3 tokenizer
（中英混排、特殊 token 的切分只有真词表能暴露问题）。数据为临时生成的迷你 parquet/tar。
"""
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.base_model import build_qwen_tokenizer  # noqa: E402
from streaming_si.data.si_encode import collate_si, encode_s1, encode_st  # noqa: E402

SRC_ROOT = os.environ.get('SI_SRC_ROOT')
TOKENIZER = os.environ.get('SI_QWEN3_TOKENIZER')
pytestmark = pytest.mark.skipif(
    not (SRC_ROOT and TOKENIZER and os.path.isdir(SRC_ROOT) and os.path.isdir(TOKENIZER)),
    reason='需要 SI_SRC_ROOT（源 streaming-si-0.6b）与 SI_QWEN3_TOKENIZER（Qwen3-0.6B tokenizer）')

ROWS = [
    dict(src='Hello everyone, welcome.', tgt='大家好，欢迎。', src_lang='eng', tgt_lang='cmn', sec=1.3,
         fsq=[[5, 17, 6000, 0, 42, 42, 1]]),
    dict(src='今天我们讨论流式同传。', tgt="Today we discuss simultaneous interpretation.", src_lang='cmn',
         tgt_lang='eng', sec=2.05, fsq=[3, 3, 9, 6560, 0]),
    dict(src='短', tgt='Short.', src_lang='cmn', tgt_lang='eng', sec=0.74, fsq=[[1, 2]]),
]


class _FakeMel:
    """确定性 mel 前端：源 dataset 与 si_encode 必须拿到同一份 mel。"""

    def __call__(self, wav, sr=16000):
        n = int(len(wav)) // 160
        g = torch.Generator().manual_seed(n)
        return torch.randn(128, n, generator=g)


def _write_p2(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    fsq_dir, wav_dir = tmp_path / 'p2_fsq_v3', tmp_path / 'p2_source_wav'
    fsq_dir.mkdir()
    wav_dir.mkdir()
    tar_path = wav_dir / 'train-00000.tar'
    refs = []
    with tarfile.open(tar_path, 'w') as tf:
        for i, r in enumerate(ROWS):
            buf = io.BytesIO()
            sf.write(buf, np.random.RandomState(i).randn(int(r['sec'] * 16000)).astype('float32') * 0.1,
                     16000, format='FLAC')
            data = buf.getvalue()
            info = tarfile.TarInfo(f'{i}.flac')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            refs.append(f'train-00000.tar::{m.offset_data}:{m.size}')
    table = pa.table({
        'id': [str(i) for i in range(len(ROWS))],
        'source_transcript': [r['src'] for r in ROWS],
        'translation': [r['tgt'] for r in ROWS],
        'src_lang': [r['src_lang'] for r in ROWS],
        'tgt_lang': [r['tgt_lang'] for r in ROWS],
        'source_wav': refs,
        'target_fsq': [json.loads(json.dumps(r['fsq'])) if isinstance(r['fsq'][0], list)
                       else [r['fsq']] for r in ROWS],
    })
    pq.write_table(table, fsq_dir / 'fsq-00000.parquet')
    return str(fsq_dir / 'fsq-00000.parquet')


def _import_source(*names):
    """按文件路径加载源项目 `dataset/<name>.py`。

    源 `dataset/` 无 `__init__.py`（命名空间包），而普通模块 `custom/dataset.py`
    （依赖 swift，pytest rootdir 会把 custom/ 放进 sys.path）在导入解析中优先于命名空间包，
    改 sys.path 顺序无效。故手工构造一个指向源目录的 `dataset` 包对象，再按依赖顺序
    exec 子模块（s3 内部 `from dataset.s2_st_dataset import ...`），结束后恢复 sys.modules。
    """
    import importlib.util
    import types
    src_dir = os.path.join(SRC_ROOT, 'dataset')
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == 'dataset' or k.startswith('dataset.')}
    pkg = types.ModuleType('dataset')
    pkg.__path__ = [src_dir]
    sys.modules['dataset'] = pkg
    loaded = []
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f'dataset.{name}', os.path.join(src_dir, f'{name}.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f'dataset.{name}'] = mod
            spec.loader.exec_module(mod)
            loaded.append(mod)
        return loaded
    finally:
        for k in [k for k in sys.modules if k == 'dataset' or k.startswith('dataset.')]:
            sys.modules.pop(k)
        sys.modules.update(saved)


@pytest.fixture(scope='module')
def src_modules():
    s2, s3, s1 = _import_source('s2_st_dataset', 's3_s2st_dataset', 's1_talker_dataset')
    return s1.S1TalkerDataset, s2.S2STDataset, s3.S3S2STDataset, s3.s3_collate_fn


@pytest.fixture(scope='module')
def tokenizer():
    return build_qwen_tokenizer(TOKENIZER)


def _shift(enc):
    ids, labels = enc['input_ids'], enc['labels']
    out = [torch.tensor(ids[:-1]), torch.tensor(labels[1:])]
    if 'answer_mask' in enc:
        out.append(torch.tensor(enc['answer_mask'][:-1]))
    return out


@pytest.mark.parametrize('max_length', [512, 20])
def test_s2_matches_source(src_modules, tokenizer, tmp_path, max_length):
    _, S2STDataset, _, _ = src_modules
    ds = S2STDataset(_write_p2(tmp_path), tokenizer, audio_processor=_FakeMel(), max_length=max_length)
    for i in range(len(ds)):
        x, y, mel, mel_len = ds[i]
        wav = ds._read_wav(ds.table['source_wav'][i].as_py())
        r = ROWS[i]
        enc = encode_st(tokenizer, _FakeMel()(wav), r['tgt'], r['tgt_lang'], max_length=max_length)
        gx, gy = _shift(enc)
        assert torch.equal(gx, x), (i, gx.tolist(), x.tolist())
        assert torch.equal(gy, y), i
        assert torch.equal(enc['audio_inputs'], mel) and enc['audio_lens'] == mel_len
        assert 'answer_mask' not in enc          # S2 不跑 talker


@pytest.mark.parametrize('max_length', [512, 20])
def test_s3_matches_source(src_modules, tokenizer, tmp_path, max_length):
    _, _, S3S2STDataset, _ = src_modules
    ds = S3S2STDataset(_write_p2(tmp_path), tokenizer, audio_processor=_FakeMel(), max_length=max_length)
    for i in range(len(ds)):
        x, y, am, speech, mel, mel_len = ds[i]
        wav = ds._read_wav(ds.table['source_wav'][i].as_py())
        r = ROWS[i]
        enc = encode_st(tokenizer, _FakeMel()(wav), r['tgt'], r['tgt_lang'], max_length=max_length,
                        speech_tokens=ds.speech_col[i].as_py()[0] if isinstance(
                            ds.speech_col[i].as_py()[0], list) else ds.speech_col[i].as_py(),
                        with_answer_mask=True)
        gx, gy, gam = _shift(enc)
        assert torch.equal(gx, x) and torch.equal(gy, y) and torch.equal(gam, am), i
        assert enc['speech_tokens'] == speech.tolist()
        assert torch.equal(enc['audio_inputs'], mel) and enc['audio_lens'] == mel_len


def _write_s1(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    convs = [
        [{'role': 'user', 'content': '请用自然的语气朗读：你好世界'}, {'role': 'assistant', 'content': '你好世界'}],
        [{'role': 'user', 'content': 'Read aloud: Streaming is hard.'},
         {'role': 'assistant', 'content': 'Streaming is hard.'}],
    ]
    table = pa.table({
        'conversations': [json.dumps(c, ensure_ascii=False) for c in convs],
        'answer_speech_tokens': [[[1, 2, 3, 4]], [[10, 0, 6560]]],
        'ref_speech_tokens': [[[7, 8, 9]], [[11, 12]]],
    })
    p = tmp_path / 's1.parquet'
    pq.write_table(table, p)
    return str(p), convs, table


@pytest.mark.parametrize('strip,max_length,ref_drop', [(False, 512, 0.0), (True, 512, 0.5), (False, 6, 1.0)])
def test_s1_matches_source(src_modules, tokenizer, tmp_path, strip, max_length, ref_drop):
    S1TalkerDataset, _, _, _ = src_modules
    path, convs, table = _write_s1(tmp_path)
    ds = S1TalkerDataset(path, tokenizer, max_length=max_length, strip_prompt_text=strip,
                         ref_drop_ratio=ref_drop)
    for i in range(len(ds)):
        random.seed(100 + i)
        x, y, am, speech, prompt = ds[i]
        random.seed(100 + i)
        enc = encode_s1(tokenizer, convs[i], table['answer_speech_tokens'][i].as_py(),
                        table['ref_speech_tokens'][i].as_py(), max_length=max_length,
                        strip_prompt=strip, ref_drop_ratio=ref_drop)
        gx, gy, gam = _shift(enc)
        assert torch.equal(gx, x) and torch.equal(gy, y) and torch.equal(gam, am), i
        assert enc['speech_tokens'] == speech.tolist()
        assert enc['prompt_speech_tokens'] == prompt.tolist()


def test_s1_scheduled_sampling_matches_source(src_modules, tokenizer, tmp_path):
    S1TalkerDataset, _, _, _ = src_modules
    path, convs, table = _write_s1(tmp_path)
    ds = S1TalkerDataset(path, tokenizer, text_scheduled_sampling=0.5, ref_drop_ratio=1.0)
    for i in range(len(ds)):
        torch.manual_seed(7 + i)
        random.seed(0)
        x, *_ = ds[i]
        torch.manual_seed(7 + i)
        random.seed(0)
        enc = encode_s1(tokenizer, convs[i], table['answer_speech_tokens'][i].as_py(),
                        table['ref_speech_tokens'][i].as_py(), text_scheduled_sampling=0.5,
                        ref_drop_ratio=1.0, text_vocab_size=ds.text_vocab_size)
        assert torch.equal(torch.tensor(enc['input_ids'][:-1]), x), i


def test_collate_matches_source(src_modules, tokenizer, tmp_path):
    _, _, S3S2STDataset, s3_collate_fn = src_modules
    ds = S3S2STDataset(_write_p2(tmp_path), tokenizer, audio_processor=_FakeMel())
    ref = s3_collate_fn([ds[i] for i in range(len(ds))])
    encs = []
    for i in range(len(ds)):
        wav = ds._read_wav(ds.table['source_wav'][i].as_py())
        fsq = ds.speech_col[i].as_py()
        encs.append(encode_st(tokenizer, _FakeMel()(wav), ROWS[i]['tgt'], ROWS[i]['tgt_lang'],
                              speech_tokens=fsq[0] if isinstance(fsq[0], list) else fsq,
                              with_answer_mask=True))
    got = collate_si(encs)
    # got 为未 shift 序列；逐行 shift 后再 pad 应与源 collate 一致
    L = ref[0].size(1)
    for i, e in enumerate(encs):
        n = len(e['input_ids']) - 1
        assert torch.equal(got['input_ids'][i, :n], ref[0][i, :n])
        assert torch.equal(got['labels'][i, 1:n + 1], ref[1][i, :n])
        assert bool((ref[1][i, n:L] == -100).all())
    assert torch.equal(got['speech_tokens'], ref[3])
    assert torch.equal(got['speech_token_lens'], ref[4])
    assert torch.equal(got['audio_inputs'], ref[5])
    assert torch.equal(got['audio_lens'], ref[6])
