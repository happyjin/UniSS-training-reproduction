"""S1 / S2 / S3 的样本编码与 collate（纯函数，不依赖 ms-swift）。

swift 的 Template 子类（`template_streaming_si.py`）只是本模块的薄封装，因此这里的
逻辑可以脱离 swift 单测，并与源项目 dataset 逐 token 对拍（`test_si_encode.py`）。

**shift 约定**：这里产出的是**未 shift 的完整序列**（swift 约定）；shift 在
`StreamingSIForConditionalGeneration.forward` 内部完成，逐位等价于源 dataset 的
`X=ids[:-1] / Y=labels[1:] / answer_mask[:-1]`。

截断口径默认与源项目一致（**截断而不是丢样本**）：
  * S2/S3：文本预算不足时先截音频帧，保留完整译文（`s2_st_dataset.py:181-189`）；
  * S1：文本超长时截指令，保留完整待朗读文本（`s1_talker_dataset.py:177-183`）。

S2/S3 可改为**丢弃超长样本**（`encode_st(drop_overlong=True)`，训练脚本经 `SI_DROP_OVERLONG=1` 打开）：
截音频会让译文对应不上被截掉的语音、截译文会让句子没说完就结束，两者都是错误监督。
丢弃条件见 `st_overlong_reason`；预处理阶段（register_streaming_si.StreamingSIPreprocessor）
用同一个函数预先过滤，编码时再兜底检查一次。
"""
import random
from typing import Any, Dict, List, Optional, Sequence

import torch

IGNORE_INDEX = -100
MEL_PER_ADAPTER_FRAME = 8
LANG_NAME = {'cmn': 'Chinese', 'zho': 'Chinese', 'zh': 'Chinese',
             'eng': 'English', 'en': 'English', 'yue': 'Cantonese',
             'jpn': 'Japanese', 'ja': 'Japanese', 'kor': 'Korean', 'ko': 'Korean'}
# 训练数据只用三字母码（cmn/eng/jpn/kor）；两字母别名供推理/评测使用。缺别名时指令会变成
# 「into ja:」这种训练里从未出现过的形态，模型退回源语言/英文（test_realdata ja/ko 遵循失败的根因）。


class SIEncodeError(ValueError):
    """样本不可用（无音频 / 无语音监督）。源项目的处理是跳到下一条样本；在 swift 里
    由预处理器丢弃该行（`preprocess` 返回 None），这里抛出便于上游定位。"""


class SIOverlongError(SIEncodeError):
    """`drop_overlong=True` 时样本超长（音频 / 译文 / 总长 / speech token）。属预期内的丢弃，
    template 把它转成 swift 的 MaxLengthError（LazyLLMDataset 静默换一条，不打 traceback）。"""


def st_num_frames(n_samples_16k: int) -> int:
    """16k 采样点数 -> audio_pad 帧数：mel 帧 = n // 160（hop 160），adapter 帧 = mel // 8。"""
    return max(1, (int(n_samples_16k) // 160) // MEL_PER_ADAPTER_FRAME)


def st_overlong_reason(*, n_frames: int, n_instr: int, n_tgt: int, max_length: int, max_target_len: int,
                       n_speech: Optional[int] = None, max_speech_len: Optional[int] = None) -> Optional[str]:
    """丢弃模式下的超长判定（不含音频时长，时长由读音频处判定）。返回原因，None 表示保留。

    Args:
        n_tgt: 译文 token 数（**不含** eos、未截断）。
    """
    if n_tgt > max_target_len:
        return f'译文 {n_tgt} token > SI_MAX_TARGET_LEN={max_target_len}'
    total = n_frames + n_instr + n_tgt + 1
    if total > max_length:
        return f'总长 {total}（音频 {n_frames} + 指令 {n_instr} + 译文 {n_tgt + 1}）> max_length={max_length}'
    if n_speech is not None and max_speech_len is not None and n_speech > max_speech_len:
        return f'speech token {n_speech} > SI_MAX_SPEECH_LEN={max_speech_len}'
    return None


def st_instruction(tgt_lang: Optional[str], task: str = 'translate') -> str:
    """与源 `S2STDataset._instruction` 逐字一致。"""
    if task == 'transcribe':
        return '\nTranscribe the speech above:\n'
    tgt = LANG_NAME.get((tgt_lang or '').lower(), tgt_lang or 'English')
    return f'\nTranslate the speech above into {tgt}:\n'


def audio_token_id(tokenizer, audio_token: str = '<|audio_pad|>') -> int:
    ids = tokenizer.encode(audio_token, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f'{audio_token} 不是单 token（得到 {ids}），inject_audio_features 依赖单 marker id')
    return ids[0]


def normalize_fsq(fsq) -> List[int]:
    """target_fsq / answer_speech_tokens 可能是 list 或 [[...]] 嵌套。"""
    if fsq and isinstance(fsq[0], (list, tuple)):
        fsq = fsq[0]
    return [int(t) for t in (fsq or [])]


def encode_st(tokenizer, mel: torch.Tensor, target_text: str, tgt_lang: Optional[str], *,
              max_length: int = 512, max_target_len: int = 256, task: str = 'translate',
              speech_tokens: Optional[Sequence[int]] = None, max_speech_len: int = 750,
              with_answer_mask: bool = False, drop_overlong: bool = False) -> Dict[str, Any]:
    """S2（with_answer_mask=False）/ S3（True + speech_tokens）编码。

    Args:
        mel: (128, T) log-mel（由 Voxtral 前端提取，调用方负责）。
        drop_overlong: True 时超长样本抛 `SIOverlongError`（不截断），判定见 `st_overlong_reason`；
            False 为源项目口径（截音频 / 译文 / speech token）。
    """
    if mel.dim() != 2:
        raise ValueError(f'mel 需为 (128, T)，收到 {tuple(mel.shape)}')
    n_frames = max(1, mel.size(-1) // MEL_PER_ADAPTER_FRAME)
    instr_ids = tokenizer.encode(st_instruction(tgt_lang, task), add_special_tokens=False)
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if drop_overlong:
        reason = st_overlong_reason(
            n_frames=n_frames, n_instr=len(instr_ids), n_tgt=len(tgt_ids), max_length=max_length,
            max_target_len=max_target_len,
            n_speech=len(speech_tokens or []) if with_answer_mask else None, max_speech_len=max_speech_len)
        if reason:
            raise SIOverlongError(reason)
    tgt_ids = tgt_ids[:max_target_len]
    eos = tokenizer.eos_token_id
    if eos is not None:
        tgt_ids = tgt_ids + [eos]

    budget = max_length - len(instr_ids) - len(tgt_ids)
    if budget < 1:
        tgt_ids = tgt_ids[:max(1, max_length - len(instr_ids) - 1)]
        budget = max(1, max_length - len(instr_ids) - len(tgt_ids))
    if n_frames > budget:
        n_frames = budget
        mel = mel[..., :n_frames * MEL_PER_ADAPTER_FRAME]

    a_id = audio_token_id(tokenizer)
    input_ids = [a_id] * n_frames + instr_ids + tgt_ids
    labels = [IGNORE_INDEX] * (n_frames + len(instr_ids)) + list(tgt_ids)
    out = {'input_ids': input_ids, 'labels': labels,
           'audio_inputs': mel.to(torch.float32), 'audio_lens': int(mel.size(-1))}
    if with_answer_mask:
        out['answer_mask'] = [0] * (n_frames + len(instr_ids)) + [1] * len(tgt_ids)
        codes = [int(t) for t in (speech_tokens or [])][:max_speech_len]
        if not codes and str(target_text or '').strip():
            raise SIEncodeError('S3 样本无 target_fsq（talker 学不到东西）')
        # 译文也为空 = 噪声负样本：文本目标只剩 eos（教 thinker 在噪声下立即终止），
        # speech 监督留空 —— collate_si 按 lens=0 组 batch，talker 的 build_inputs
        # 对零长度行整行 ignore，compute_si_losses 无有效位时给可微零，均无需特判。
        out['speech_tokens'] = codes
    return out


def strip_prompt_text(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """与源 `S1TalkerDataset._strip_prompt_text` 一致：去掉 user 末尾与 assistant 重复的原文。"""
    text = next((t['content'] for t in conversations if t.get('role') == 'assistant'), None)
    if not text:
        return conversations
    for t in conversations:
        if (t.get('role') == 'user' and isinstance(t.get('content'), str)
                and t['content'].endswith(text) and len(t['content']) > len(text)):
            t['content'] = t['content'][:-len(text)]
    return conversations


def encode_s1(tokenizer, conversations: List[Dict[str, Any]], answer_speech_tokens,
              ref_speech_tokens=None, *, max_length: int = 512, max_speech_len: int = 750,
              max_prompt_speech_len: int = 125, ref_drop_ratio: float = 0.5,
              strip_prompt: bool = False, text_scheduled_sampling: float = 0.0,
              text_vocab_size: int = 151936, rng: Optional[random.Random] = None,
              ) -> Dict[str, Any]:
    """S1（TTS：文本 -> FSQ）编码。

    `rng` 只控制 ref 音色 prompt 的随机丢弃；text_scheduled_sampling 用 torch 全局 RNG，
    两者与源实现的随机源一致（`random.random()` / `torch.rand`），便于同种子复现。
    """
    rng = rng or random
    conversations = [dict(t) for t in conversations]
    if strip_prompt:
        conversations = strip_prompt_text(conversations)
    asst_count = sum(1 for t in conversations if t.get('role') == 'assistant')
    if asst_count < 1:
        raise SIEncodeError('S1 样本没有 assistant turn')

    # 逐行对应源 `S1TalkerDataset._speech_tokens`
    speech: List[int] = []
    if answer_speech_tokens:
        tokens = answer_speech_tokens
        turn = tokens[asst_count - 1] if asst_count <= len(tokens) else tokens[-1]
        if turn and isinstance(turn[0], (list, tuple)):
            turn = turn[0]
        speech = [int(t) for t in (turn or [])][:max_speech_len]
    if not speech:
        raise SIEncodeError('S1 样本无 answer_speech_tokens')

    prompt: List[int] = []
    if ref_speech_tokens and rng.random() >= ref_drop_ratio:
        flat = ref_speech_tokens[0] if isinstance(ref_speech_tokens[0], (list, tuple)) else ref_speech_tokens
        prompt = [int(t) for t in flat][-max_prompt_speech_len:]

    instr = next((t['content'] for t in conversations if t.get('role') == 'user'), '')
    text = next((t['content'] for t in conversations if t.get('role') == 'assistant'), '')
    instr_ids = tokenizer.encode(instr, add_special_tokens=False)
    tgt_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        tgt_ids = tgt_ids + [tokenizer.eos_token_id]
    budget = max_length - len(tgt_ids)
    if budget < 1:
        tgt_ids = tgt_ids[:max_length - 1]
        budget = 1
    instr_ids = instr_ids[:budget]

    input_ids = instr_ids + tgt_ids
    labels = [IGNORE_INDEX] * len(instr_ids) + list(tgt_ids)
    answer_mask = [0] * len(instr_ids) + [1] * len(tgt_ids)
    if text_scheduled_sampling > 0:
        # 源实现在 shift 后的 X 上加噪，噪声位置 = text_labels!=-100（即 labels[1:] 有效位）。
        # 这里在未 shift 序列上等价地作用于 X 的下标 [0, L-1)。
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        noise = (y != IGNORE_INDEX) & (torch.rand(x.shape) < text_scheduled_sampling)
        x = torch.where(noise, torch.randint(0, text_vocab_size, x.shape), x)
        input_ids = x.tolist() + input_ids[-1:]
    return {'input_ids': input_ids, 'labels': labels, 'answer_mask': answer_mask,
            'speech_tokens': speech, 'prompt_speech_tokens': prompt}


def _pad_1d(seqs: Sequence[Sequence[int]], value: int) -> torch.Tensor:
    L = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max(L, 1)), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, :len(s)] = torch.as_tensor(list(s), dtype=torch.long)
    return out


def _pad_with_lens(seqs: Sequence[Sequence[int]]):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return _pad_1d(seqs, 0), lens


def collate_si(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """把 encode_* 的输出右 pad 成 batch（与源 s1/s2st/s3 collate 语义一致）。

    注意 input_ids 用 0 pad（源 collate 如此）；pad 位由 attention_mask=0 屏蔽、labels=-100。
    speech / prompt 轴一律附真实长度：talker 按 lens 裁剪，0 是合法 FSQ 码字，不能靠值判 pad。
    """
    res = {
        'input_ids': _pad_1d([b['input_ids'] for b in batch], pad_token_id),
        'labels': _pad_1d([b['labels'] for b in batch], IGNORE_INDEX),
        'attention_mask': _pad_1d([[1] * len(b['input_ids']) for b in batch], 0),
    }
    if any('answer_mask' in b for b in batch):
        res['answer_mask'] = _pad_1d([b.get('answer_mask') or [] for b in batch], 0)
    if any('speech_tokens' in b for b in batch):
        res['speech_tokens'], res['speech_token_lens'] = _pad_with_lens(
            [b.get('speech_tokens') or [] for b in batch])
    if any(b.get('prompt_speech_tokens') for b in batch):
        res['prompt_speech_tokens'], res['prompt_speech_token_lens'] = _pad_with_lens(
            [b.get('prompt_speech_tokens') or [] for b in batch])
    if any(b.get('audio_inputs') is not None for b in batch):
        mels = [b['audio_inputs'] for b in batch]
        T = max(m.size(-1) for m in mels)
        mel = mels[0].new_zeros(len(mels), mels[0].size(0), T)
        for i, m in enumerate(mels):
            mel[i, :, :m.size(-1)] = m
        res['audio_inputs'] = mel
        res['audio_lens'] = torch.tensor([int(b['audio_lens']) for b in batch], dtype=torch.long)
    return res
