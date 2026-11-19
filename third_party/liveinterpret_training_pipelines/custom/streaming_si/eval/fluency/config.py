"""全局配置：LiteLLM 网关、VAD 参数、四档容忍时间。

所有阈值都可以通过环境变量覆盖，方便做试听校准。
"""

from __future__ import annotations

import os

# ---------------- LiteLLM 网关（沿用本项目既有调用方式） ----------------
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "https://litellm.ai.levelinfinite.com/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY") or os.getenv("SPEECH_LLM_API_KEY") or ""
# 边界分类任务量小，默认用一个便宜且稳定的模型
LITELLM_MODEL = os.getenv("LITELLM_FLUENCY_MODEL", "gpt-6-sol")
LITELLM_TIMEOUT = int(os.getenv("LITELLM_TIMEOUT", "120"))


# ---------------- 音频声道选择 ----------------
# dual.wav 这类双声道文件常是 [源音, 译音]，评流畅度必须选对声道。
# 取值："mix" 混合所有声道；"0"/"1" 指定声道下标（0=左, 1=右）
AUDIO_CHANNEL = os.getenv("AUDIO_CHANNEL", "mix")

# ---------------- Silero VAD ----------------
VAD_SAMPLING_RATE = 16000      # Silero VAD 只支持 8000 / 16000
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "100"))   # 短于此的静音不切开
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))     # 短于此的语音段丢弃
VAD_SPEECH_PAD_MS = 0                                              # 不额外留白，保证停顿时长真实

# 小于该值的间隙不认为是"停顿"（避免把爆破音间隙算进来）
MIN_PAUSE_S = float(os.getenv("MIN_PAUSE_S", "0.10"))

# 仅在报告中额外列出"超长停顿"清单，供人工复核用。
# 不影响评分：所有计分停顿（含超过该值的）一律统一按 excess = 时长 - 容忍 计算。
LONG_PAUSE_S = float(os.getenv("LONG_PAUSE_S", "2.00"))


# ---------------- 四档容忍时间（秒）----------------
# 语义联系越紧，允许的停顿越短。以下为起点值，需用真实语料试听校准。
def _f(env: str, default: float) -> float:
    return float(os.getenv(env, default))


TOLERANCE_S = {
    "word_internal": _f("TOL_WORD_INTERNAL", 0.15),        # 词内部，如「现｜代化」
    "tight_structure": _f("TOL_TIGHT_STRUCTURE", 0.30),    # 紧密结构内部，如「把｜报告交给他」
    "phrase_boundary": _f("TOL_PHRASE_BOUNDARY", 0.50),    # 短语 / 分句边界
    "sentence_boundary": _f("TOL_SENTENCE_BOUNDARY", 1.00),# 完整句边界
}

# 兜底：无法判定边界类型时使用的档位
DEFAULT_BOUNDARY = "tight_structure"

# ---------------- 字数统计 ----------------
# True  = 统计所有非空白字符（含标点）
# False = 只统计汉字 / 字母 / 数字，排除标点
COUNT_PUNCTUATION = os.getenv("COUNT_PUNCTUATION", "1") not in ("0", "false", "False")

# ---------------- 对齐权重 ----------------
# 每个"单元"朗读耗时权重，用于把语音时长摊派到译文字符上
WEIGHT_CJK = 1.0          # 一个汉字 / 全角字符
WEIGHT_PUNCT = 0.3        # 一个标点
WEIGHT_SPACE = 0.0        # 空白不占时
WEIGHT_LATIN_PER_CHAR = 1.0 / 3.5   # 英文按约 3.5 字母 ≈ 1 个汉字的时长

# ---------------- LLM 上下文窗口 ----------------
CONTEXT_SPAN = 20         # 送给 LLM 的停顿左右上下文字符数
