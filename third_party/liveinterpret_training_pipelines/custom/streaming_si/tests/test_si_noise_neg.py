"""噪声负样本（译文为空 + target_fsq 空）在 S3/S4 路径的回归测试。

覆盖 SI_KEEP_NOISE_NEG 改动的三处行为：
  1. encode_st：空译文 + 空 fsq -> 文本目标只剩 eos、speech_tokens 为空（不再 raise）；
     非空译文 + 空 fsq 仍 raise（真异常）。
  2. collate_si：噪声行与正常行混合 batch，speech_token_lens 含 0，不崩。
  3. StreamingSIPreprocessor.preprocess：开关开时噪声行保留、译文非空仍丢弃；关时恢复旧行为。

跑法（需 torch；preprocess 用例另需 swift）：
    PYTHONPATH=custom pytest custom/streaming_si/tests/test_si_noise_neg.py -q
"""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.data.si_encode import SIEncodeError, collate_si, encode_st  # noqa: E402


class _StubTok:
    """最小 tokenizer：'<|audio_pad|>' 必须单 token，其他文本固定切 2 token。"""
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        if text == '<|audio_pad|>':
            return [7]
        if not text:
            return []
        return [3, 4]


def _mel(sec=2.0):
    return torch.randn(128, int(sec * 160))


def test_noise_row_encode():
    enc = encode_st(_StubTok(), _mel(), '', 'cmn', max_length=768,
                    speech_tokens=[], with_answer_mask=True, drop_overlong=True)
    assert enc['speech_tokens'] == []                       # talker 无监督位（build_inputs 全 ignore）
    assert enc['labels'][-1] == 2                           # 唯一监督位 = eos（抑制信号）
    assert all(t == -100 for t in enc['labels'][:-1])
    assert sum(enc['answer_mask']) == 1 and enc['answer_mask'][-1] == 1
    assert enc['input_ids'][-1] == 2 and 7 in enc['input_ids']


def test_nonempty_text_empty_fsq_still_raises():
    with pytest.raises(SIEncodeError):
        encode_st(_StubTok(), _mel(), '你好', 'cmn', max_length=768,
                  speech_tokens=[], with_answer_mask=True, drop_overlong=True)


def test_normal_row_unchanged():
    enc = encode_st(_StubTok(), _mel(), '你好世界', 'cmn', max_length=768,
                    speech_tokens=[5, 6, 8], with_answer_mask=True, drop_overlong=True)
    assert enc['speech_tokens'] == [5, 6, 8]
    assert sum(enc['answer_mask']) == len(enc['answer_mask']) - enc['answer_mask'].count(0)  # 非空


def test_collate_mixed_batch():
    noise = encode_st(_StubTok(), _mel(), '', 'cmn', max_length=768,
                      speech_tokens=[], with_answer_mask=True, drop_overlong=True)
    normal = encode_st(_StubTok(), _mel(3.0), 'hello', 'eng', max_length=768,
                       speech_tokens=[5, 6, 8], with_answer_mask=True, drop_overlong=True)
    batch = collate_si([noise, normal])
    assert batch['speech_token_lens'].tolist() == [0, 3]
    assert batch['answer_mask'].shape == batch['input_ids'].shape
    assert batch['labels'].shape == batch['input_ids'].shape


def _noise_row():
    # 与 p2_parquet_prep.py 写入的元数据列一致（_overlong 依赖）；dataset_name / id 取真实噪声目录的形态
    return {
        'dataset_name': 'dns5_noise', 'id': 'dns5:noise_fullband/x.wav#0',
        'translation': '', 'source_transcript': '', 'tgt_lang': 'cmn',
        'source_wav': 'train-00000.tar::0:4096', 'target_fsq': [],
        'src_samples': 32000, 'instr_tokens': 12, 'tgt_tokens': 0,
    }


def test_preprocess_gate(monkeypatch):
    pytest.importorskip('swift')
    from streaming_si.integration.register_streaming_si import StreamingSIPreprocessor
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv('SI_DROP_OVERLONG', '1')
        monkeypatch.setenv('SI_KEEP_NOISE_NEG', '1')
        keep = StreamingSIPreprocessor('s3', wav_dir=d)
        out = keep.preprocess(_noise_row())
        assert out is not None                                 # 噪声行保留
        assert out['messages'][1]['content'] == ''
        row = _noise_row()
        row['translation'] = '有译文'
        assert keep.preprocess(row) is None                    # 译文非空 + 空 fsq：真异常，丢弃

        # 保险：译文空 + fsq 空，但不是已登记的噪声源（如翻译失败的真实语音行）-> 丢弃
        speech = _noise_row()
        speech.update(dataset_name='ja2zh_emilia', id='ja:clip123#0')
        assert keep.preprocess(speech) is None
        speech.pop('dataset_name')                             # machine shard 没有 dataset_name 列
        speech['id'] = 'machine_0:abc#0'
        assert keep.preprocess(speech) is None

        # 各噪声源的真实 id / dataset_name 形态都应保留
        for name, rid in (('esc50_siren', 'esc50:4-90014-B-42.wav#0'),
                          ('musan_noise', 'musan_noise:free-sound/n-0000.wav#1'),
                          ('dns5_noise', 'dns5:noise_fullband/--diQGtdz9Q.wav#0')):
            r = _noise_row()
            r.update(dataset_name=name, id=rid)
            assert keep.preprocess(r) is not None, name

        monkeypatch.setenv('SI_KEEP_NOISE_NEG', '0')
        drop = StreamingSIPreprocessor('s3', wav_dir=d)
        assert drop.preprocess(_noise_row()) is None           # 旧行为：空 fsq 全丢

        # SI_NOISE_TAGS 可覆盖登记表
        monkeypatch.setenv('SI_KEEP_NOISE_NEG', '1')
        monkeypatch.setenv('SI_NOISE_TAGS', 'newnoise')
        custom = StreamingSIPreprocessor('s3', wav_dir=d)
        assert custom.preprocess(_noise_row()) is None         # dns5 不再登记
        r = _noise_row()
        r.update(dataset_name='newnoise_x')
        assert custom.preprocess(r) is not None
