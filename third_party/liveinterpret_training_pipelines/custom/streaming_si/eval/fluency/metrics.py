"""容忍时间分配、超额时间计算与指标汇总。

核心指标：
    异常停顿负担 = Σ超额停顿秒数 / 译文字数 × 100     （单位：秒 / 100 字，越低越好）
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Sequence

from config import COUNT_PUNCTUATION, DEFAULT_BOUNDARY, LONG_PAUSE_S, TOLERANCE_S
from schema import (
    BOUNDARY_LABELS_ZH,
    BOUNDARY_TYPES,
    PAUSE_INTERNAL,
    Pause,
    SpeechSegment,
)


def is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S")


def count_chars(text: str, include_punct: bool = COUNT_PUNCTUATION) -> int:
    """统计译文"字数"：默认数所有非空白字符，可配置是否排除标点。"""
    if include_punct:
        return sum(1 for ch in text if not ch.isspace())
    return sum(1 for ch in text if not ch.isspace() and not is_punctuation(ch))


def assign_tolerance_and_excess(
    pauses: Sequence[Pause],
    text: str = "",
    tolerance: Dict[str, float] = None,
) -> None:
    """根据边界类型填写 tolerance 与 excess（原地修改）。"""
    tol = dict(TOLERANCE_S)
    if tolerance:
        tol.update(tolerance)

    for p in pauses:
        if not p.scored or p.kind != PAUSE_INTERNAL:
            p.tolerance = 0.0
            p.excess = 0.0
            continue
        btype = p.boundary_type or DEFAULT_BOUNDARY
        p.tolerance = float(tol.get(btype, tol[DEFAULT_BOUNDARY]))
        p.excess = max(0.0, p.duration - p.tolerance)


def summarize(
    text: str,
    pauses: Sequence[Pause],
    segments: Sequence[SpeechSegment],
    duration_s: float,
    audio_path: str = "",
    method: str = "rule",
    tolerance: Dict[str, float] = None,
) -> Dict:
    """汇总所有指标，返回可直接 JSON 序列化的字典。"""
    tol = dict(TOLERANCE_S)
    if tolerance:
        tol.update(tolerance)

    scored = [p for p in pauses if p.scored and p.kind == PAUSE_INTERNAL]
    total_excess = sum(p.excess for p in scored)
    n_chars = count_chars(text)

    speech_s = sum(s.duration for s in segments)
    burden = (total_excess / n_chars * 100.0) if n_chars else 0.0

    by_type: Dict[str, Dict] = {
        t: {"count": 0, "excess_s": 0.0, "total_s": 0.0,
            "label_zh": BOUNDARY_LABELS_ZH[t], "tolerance_s": tol[t]}
        for t in BOUNDARY_TYPES
    }
    for p in scored:
        b = p.boundary_type or DEFAULT_BOUNDARY
        if b not in by_type:
            b = DEFAULT_BOUNDARY
        by_type[b]["count"] += 1
        by_type[b]["excess_s"] += p.excess
        by_type[b]["total_s"] += p.duration

    worst = max(scored, key=lambda p: p.excess, default=None)

    return {
        "audio_path": audio_path,
        "transcript": text,
        "duration_s": round(duration_s, 3),
        "speech_s": round(speech_s, 3),
        "speech_ratio": round(speech_s / duration_s, 4) if duration_s else 0.0,
        "transcript_chars": n_chars,
        "speaking_rate_cps": round(n_chars / speech_s, 3) if speech_s else 0.0,
        "boundary_method": method,
        "tolerance_table": {k: tol[k] for k in BOUNDARY_TYPES},
        "pause_total": len(pauses),
        "pause_scored": len(scored),
        "total_pause_s": round(sum(p.duration for p in scored), 3),
        "total_excess_s": round(total_excess, 3),
        "abnormal_pause_burden": round(burden, 4),
        "long_pauses": [
            {"start_s": round(p.start, 3), "duration_s": round(p.duration, 3),
             "context": p.context(text)}
            for p in scored if p.duration >= LONG_PAUSE_S
        ],
        "worst_pause": (
            {
                "start_s": round(worst.start, 3),
                "duration_s": round(worst.duration, 3),
                "boundary_type": worst.boundary_type,
                "tolerance_s": round(worst.tolerance, 3),
                "excess_s": round(worst.excess, 3),
                "context": worst.context(text),
            }
            if worst else None
        ),
        "by_boundary_type": {
            k: {
                "label_zh": v["label_zh"],
                "count": v["count"],
                "tolerance_s": round(v["tolerance_s"], 3),
                "total_s": round(v["total_s"], 3),
                "excess_s": round(v["excess_s"], 3),
            }
            for k, v in by_type.items()
        },
        "pauses": [p.to_dict(text) for p in pauses],
    }
