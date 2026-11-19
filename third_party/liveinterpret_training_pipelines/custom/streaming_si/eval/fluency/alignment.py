"""停顿 -> 译文字符位置的对齐。

思路（无需强制对齐工具 / ASR）：
    1. 把译文切成"计时单元"：一个汉字、一个英文单词、一个数字串、一个标点、一段空白；
    2. 给每个单元一个朗读耗时权重（汉字 1.0，英文按 3.5 字母≈1 汉字，标点 0.3，空白 0）；
    3. 把 VAD 得到的"语音总时长"按权重摊派到每个单元上，得到每个单元的 [t_start, t_end]；
    4. 把停顿的音频绝对时间折算成"语音累计时间"（扣除之前的停顿）；
    5. 该时间落在哪个单元，停顿就发生在该单元的前面或后面 -> char_index。

这样得到的是一个工程近似值。若有词级强制对齐（ASR / MFA）结果，可用
`align_pauses_with_word_timestamps` 直接替换，其余流程不变。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence

from config import (
    WEIGHT_CJK,
    WEIGHT_LATIN_PER_CHAR,
    WEIGHT_PUNCT,
    WEIGHT_SPACE,
)
from schema import PAUSE_INTERNAL, SpeechSegment, Unit

# 英文单词 / 数字串 / 空白 / 其它单字符（汉字、标点等逐字切分）
_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)?|\s+|[^\sA-Za-z\d]",
    re.UNICODE,
)
_LATIN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")


def tokenize(text: str) -> List[Unit]:
    """把译文切成计时单元列表。"""
    units: List[Unit] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        units.append(Unit(text=s, start=m.start(), end=m.end(), weight=_weight(s)))
    return units


def _weight(token: str) -> float:
    if token.isspace():
        return WEIGHT_SPACE
    if _LATIN_RE.fullmatch(token) or _DIGIT_RE.fullmatch(token):
        return max(WEIGHT_CJK, len(token) * WEIGHT_LATIN_PER_CHAR)
    # 先按 Unicode 类别判标点：全角标点（，。！？等 U+FF0C 之类）码位也在 CJK
    # 区间（>0x2E80）内，若先判码位会被误当成汉字拿到 1.0 的权重。
    if unicodedata.category(token[0]).startswith(("P", "S")):
        return WEIGHT_PUNCT
    return WEIGHT_CJK


def assign_speech_time(units: Sequence[Unit], total_speech_s: float) -> None:
    """按权重把语音总时长摊派到每个单元上，填充 t_start / t_end。"""
    total_w = sum(u.weight for u in units)
    if total_w <= 0 or total_speech_s <= 0:
        return
    per_unit = total_speech_s / total_w
    acc = 0.0
    for u in units:
        u.t_start = acc
        acc += u.weight * per_unit
        u.t_end = acc


def audio_time_to_speech_time(t: float, segments: Sequence[SpeechSegment]) -> float:
    """把音频绝对时间折算成"已说出的语音累计时间"（不含停顿）。"""
    acc = 0.0
    for s in segments:
        if t <= s.start:
            return acc
        if t <= s.end:
            return acc + (t - s.start)
        acc += s.duration
    return acc


def locate_char_index(speech_time: float, units: Sequence[Unit]) -> int:
    """语音累计时间落在哪个单元 -> 停顿发生在该单元之后（或之前）的字符下标。"""
    if not units:
        return 0

    timed = [u for u in units if u.weight > 0]
    if not timed:
        return len(units[-1].text) + units[-1].start

    if speech_time <= timed[0].t_start:
        return timed[0].start

    last = timed[-1]
    if speech_time >= last.t_end:
        return last.end

    for u in timed:
        if u.t_start <= speech_time <= u.t_end:
            mid = (u.t_start + u.t_end) / 2.0
            # 落在单元前半段 => 停顿在该单元之前；后半段 => 之后
            return u.end if speech_time >= mid else u.start

    # 理论上不会走到这里：兜底返回最后一个单元的结束位置
    return last.end


def align_pauses(
    pauses: List,
    units: Sequence[Unit],
    segments: Sequence[SpeechSegment],
    total_speech_s: float,
) -> None:
    """原地为每个 internal 停顿填 speech_time / progress / char_index。"""
    for p in pauses:
        if p.kind != PAUSE_INTERNAL:
            continue
        p.speech_time = audio_time_to_speech_time(p.start, segments)
        p.progress = (p.speech_time / total_speech_s) if total_speech_s > 0 else 0.0
        p.char_index = locate_char_index(p.speech_time, units)


def align_pauses_with_word_timestamps(pauses: List, word_timestamps) -> None:
    """可选替换方案：若已有词级时间戳 [(word, start, end), ...]，用真实对齐覆盖。

    word_timestamps 需按时间升序，start/end 为音频绝对秒。
    """
    for p in pauses:
        if p.kind != PAUSE_INTERNAL:
            continue
        hit: Optional[tuple] = None
        for w in word_timestamps:
            if w[2] <= p.start:
                hit = w
            else:
                break
        if hit is not None:
            p.extra["aligned_word"] = hit[0]
            p.speech_time = hit[2]
            p.char_index = hit[3] if len(hit) > 3 else p.char_index
