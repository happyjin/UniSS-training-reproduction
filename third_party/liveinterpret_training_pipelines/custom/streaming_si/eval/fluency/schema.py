"""通用数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------- 边界类型 ----------------
WORD_INTERNAL = "word_internal"          # 词内部，如「现｜代化」
TIGHT_STRUCTURE = "tight_structure"      # 紧密语法结构内部，如动宾「把｜报告交给他」
PHRASE_BOUNDARY = "phrase_boundary"      # 短语 / 分句边界
SENTENCE_BOUNDARY = "sentence_boundary"  # 完整句边界

BOUNDARY_TYPES = [WORD_INTERNAL, TIGHT_STRUCTURE, PHRASE_BOUNDARY, SENTENCE_BOUNDARY]

BOUNDARY_LABELS_ZH = {
    WORD_INTERNAL: "词内部",
    TIGHT_STRUCTURE: "紧密结构内部",
    PHRASE_BOUNDARY: "短语/分句边界",
    SENTENCE_BOUNDARY: "完整句边界",
}

# 停顿所在位置：语句内部（参与评分）/ 音频开头 / 音频结尾
PAUSE_INTERNAL = "internal"
PAUSE_LEADING = "leading"
PAUSE_TRAILING = "trailing"


@dataclass
class SpeechSegment:
    """一段被 VAD 判定为语音的连续区间。"""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Unit:
    """译文切分出的最小计时单元（一个汉字 / 一个英文单词 / 一个标点 / 一段空白）。"""
    text: str
    start: int            # 在原文中的起始字符下标
    end: int              # 结束下标（不含）
    weight: float = 0.0   # 朗读耗时权重
    t_start: float = 0.0  # 该单元对应的"语音累计时间"起点（秒，已剔除停顿）
    t_end: float = 0.0    # 终点

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Unit({self.text!r},{self.start}-{self.end},w={self.weight:.2f})"


@dataclass
class Pause:
    """一处停顿。"""
    idx: int
    start: float                      # 音频绝对时间（秒）
    end: float
    kind: str = PAUSE_INTERNAL        # internal / leading / trailing
    speech_time: float = 0.0          # 折算到"语音累计时间"轴上的位置
    progress: float = 0.0             # 在语音中的相对进度 0~1
    char_index: int = -1              # 停顿落在译文第几个字符之后（-1 表示未对齐）
    boundary_type: Optional[str] = None
    boundary_reason: str = ""
    tolerance: float = 0.0            # 该位置允许的停顿时长（秒）
    excess: float = 0.0               # 超额时间 = max(0, 时长 - 容忍)
    scored: bool = True               # 是否参与评分
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def context(self, text: str, span: int = 8) -> str:
        """返回带 〔停〕 标记的上下文片段，便于人工复核。"""
        i = self.char_index
        if i < 0:
            return "〔停〕(未对齐)"
        i = max(0, min(i, len(text)))
        return f"{text[max(0, i - span):i]}〔停〕{text[i:i + span]}"

    def to_dict(self, text: str = "") -> dict:
        return {
            "idx": self.idx,
            "kind": self.kind,
            "start_s": round(self.start, 3),
            "end_s": round(self.end, 3),
            "duration_s": round(self.duration, 3),
            "speech_time_s": round(self.speech_time, 3),
            "progress": round(self.progress, 4),
            "char_index": self.char_index,
            "context": self.context(text) if text else "",
            "boundary_type": self.boundary_type,
            "boundary_type_zh": BOUNDARY_LABELS_ZH.get(self.boundary_type or "", ""),
            "boundary_reason": self.boundary_reason,
            "classification_source": self.extra.get("classification_source", "rule"),
            **{k: self.extra[k] for k in ('align_distance_s', 'alignment_ambiguous', 'alignment_candidates',
                                          'alignment_method', 'alignment_legacy_index', 'alignment_suspects',
                                          'aligned_left', 'aligned_right') if k in self.extra},
            "tolerance_s": round(self.tolerance, 3),
            "excess_s": round(self.excess, 3),
            "scored": self.scored,
        }
