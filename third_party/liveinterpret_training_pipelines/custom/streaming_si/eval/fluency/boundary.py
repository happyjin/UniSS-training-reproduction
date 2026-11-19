"""边界类型判定：LLM 为主，规则兜底。

四档（语义联系由紧到松）：
    word_internal     词内部          「现｜代化」
    tight_structure   紧密结构内部     「把｜报告交给他」
    phrase_boundary   短语/分句边界    逗号处、并列短语之间
    sentence_boundary 完整句边界       「实验结束。｜接下来」
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from config import CONTEXT_SPAN, DEFAULT_BOUNDARY
from schema import (
    BOUNDARY_LABELS_ZH,
    BOUNDARY_TYPES,
    PHRASE_BOUNDARY,
    SENTENCE_BOUNDARY,
    TIGHT_STRUCTURE,
    WORD_INTERNAL,
    Pause,
)

log = logging.getLogger(__name__)

# 规则兜底用的标点集
_SENT_END = set("。！？!?；;")
_PHRASE_END = set("，,、：:）)】」』”’…—")

SYSTEM_PROMPT = (
    "你是语音流畅度评测专家，擅长判断口语/口译录音中停顿发生的句法位置。"
    "你只输出 JSON，不输出任何解释性文字。"
)

USER_TEMPLATE = """下面是译文全文：

{text}

语音中检测到 {n} 处停顿，位置用 ｜ 标出（只给出停顿附近的上下文）：

{items}

请逐一判断每处停顿所在的边界类型，label 只能是以下四个之一：

- word_internal：停顿打断了一个词的中间，听众会觉得"卡壳"，例如「现｜代化」「import｜ant」
- tight_structure：停顿在紧密语法结构内部（动宾、介宾、数量名、定语与中心词、助词与实词等），
  例如「把｜报告交给他」「一个｜重要的决定」
- phrase_boundary：停顿在短语或分句边界（逗号处、并列短语之间、主语与谓语之间、状语与主句之间等），
  例如「他说，｜明天再来」「一方面…｜另一方面…」
- sentence_boundary：停顿在完整句子结束之后（句号、问号、叹号之后另起一句），
  例如「实验结束。｜接下来」

只输出如下 JSON 数组，不要任何其它内容：
[{{"id": 1, "label": "tight_structure", "reason": "动宾之间"}}, ...]
"""


def _context(text: str, idx: int, span: int = CONTEXT_SPAN) -> str:
    i = max(0, min(idx, len(text)))
    return f"{text[max(0, i - span):i]}｜{text[i:i + span]}"


def rule_classify(text: str, idx: int) -> str:
    """无 LLM 时的启发式兜底：只看标点。"""
    if idx <= 0:
        return SENTENCE_BOUNDARY
    prev = text[idx - 1]
    if prev in _SENT_END:
        return SENTENCE_BOUNDARY
    if prev in _PHRASE_END:
        return PHRASE_BOUNDARY
    return TIGHT_STRUCTURE


_LANG_HINT = {
    "ja": "译文语言为日文：判断词内部/紧密结构时按日文词与文节（助词附着于前述成分，不单独成停顿位）。",
    "ko": "译文语言为韩文：判断词内部/紧密结构时按韩文空格分词（助词附着于前述成分，不单独成停顿位）。",
}


def classify_with_llm(
    text: str,
    pauses: Sequence[Pause],
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    verbose: bool = False,
    language: Optional[str] = None,
) -> bool:
    """用 LiteLLM 批量判定边界类型，原地写入 pause.boundary_type / boundary_reason。

    返回 True 表示成功，False 表示调用失败（调用方应回退到规则判定）。
    """
    targets = [p for p in pauses if p.scored and p.kind == "internal"]
    if not targets:
        return True

    items = "\n".join(
        f'{i}. {_context(text, p.char_index)}' for i, p in enumerate(targets, 1)
    )
    lang_hint = _LANG_HINT.get(language or "", "")
    user_content = USER_TEMPLATE.format(text=text, n=len(targets), items=items)
    if lang_hint:
        user_content = lang_hint + "\n\n" + user_content
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content,
        },
    ]

    try:
        from llm_client import chat_json

        data, request_id = chat_json(messages, model=model, api_base=api_base)
        if verbose:
            log.info("LLM 边界分类完成，request_id=%s", request_id)
    except Exception as exc:  # noqa: BLE001 - 任何失败都回退到规则
        log.warning("LLM 边界分类失败，回退到规则判定：%s", exc)
        return False

    if not isinstance(data, list):
        log.warning("LLM 返回格式非数组，回退到规则判定")
        return False

    by_id = {}
    for item in data:
        if isinstance(item, dict):
            try:
                by_id[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue

    hit = 0
    for i, p in enumerate(targets, 1):
        item = by_id.get(i)
        label = (item or {}).get("label", "")
        if label in BOUNDARY_TYPES:
            p.boundary_type = label
            p.boundary_reason = str((item or {}).get("reason", ""))[:80]
            p.extra["classification_source"] = "llm"
            hit += 1
        else:
            p.boundary_type = rule_classify(text, p.char_index)
            p.boundary_reason = "规则兜底（LLM 未给出有效标签）"
            p.extra["classification_source"] = "rule"

    log.info("LLM 成功判定 %d/%d 处停顿", hit, len(targets))
    return hit > 0


def classify(text: str, pauses: Sequence[Pause], use_llm: bool = True,
             model: Optional[str] = None, api_base: Optional[str] = None,
             verbose: bool = False, language: Optional[str] = None) -> str:
    """统一入口。返回实际使用的判定方式：'llm' 或 'rule'。"""
    if use_llm and classify_with_llm(text, pauses, model=model,
                                     api_base=api_base, verbose=verbose,
                                     language=language):
        return "llm"

    for p in pauses:
        if p.scored and p.kind == "internal":
            if not p.boundary_type:
                p.boundary_type = rule_classify(text, p.char_index)
                p.boundary_reason = "规则判定（看前一个字符是否为标点）"
    return "rule"
