"""单条样本的评测主流程（单条 / 批量共用）。

    VAD 找停顿 -> 对齐译文 -> 判边界类型 -> 分配容忍时间 -> 算超额 -> 汇总

对齐来源：
    mfa       SageMaker 强制对齐服务（中文字符级 / 英文词级）
    timeline  用 pipeline 自带的 timeline.json
    uniform   均匀权重摊派（旧方案）
"""

from __future__ import annotations

import sys
from typing import Optional

import alignment
import audio_vad
import boundary
import metrics
import timeline_align
from config import MIN_PAUSE_S


def evaluate(
    audio_path: str,
    text: str = "",
    use_llm: bool = True,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    tolerance: Optional[dict] = None,
    channel: str = "mix",
    align_source: str = "auto",
    timeline_dir: Optional[str] = None,
    verbose: bool = False,
    language: Optional[str] = None,
    alignment_cache_dir: Optional[str] = None,
    whisper_words: Optional[dict] = None,
) -> dict:
    """对一条（音频, 译文）跑完整评测，返回报告字典。

    align_source:
        "auto"     有 timeline 且时基校验通过就用 timeline，否则回退 uniform
        "timeline" 强制用 timeline，不可用则报错
        "uniform"  强制用均匀摊派（旧行为）
    """
    # VAD 总是要跑：既提供总时长 / 语音段，也用于估计 timeline 的时钟偏移
    info = audio_vad.analyze_audio(audio_path, channel=channel)

    gt = None
    if align_source in ("auto", "timeline"):
        gt = timeline_align.build(audio_path, info, MIN_PAUSE_S, timeline_dir)
        if gt is None and align_source == "timeline":
            raise ValueError("未找到可用的 timeline.json（或时基校验未通过）")
        if gt is not None and verbose:
            print(f"[对齐] timeline 偏移 {gt['offset_s']:+.2f}s  校验 IoU {gt['iou']:.2f}  "
                  f"实际文本 {len(gt['text'])} 字", file=sys.stderr)

    if gt is not None:
        # ---- ground truth 路径 ----
        text_out = gt["text"]
        segments, pauses = gt["segments"], gt["pauses"]
        duration_s = gt["duration_s"]
        src, iou, off = "timeline", gt["iou"], gt["offset_s"]
        if verbose:
            print(f"[VAD] {audio_path} 语音段 {len(segments)} 停顿 {len(pauses)}", file=sys.stderr)
    else:
        # ---- 兜底：均匀权重摊派 ----
        text_out = (text or "").strip()
        if not text_out:
            raise ValueError("译文为空，且没有可用的 timeline.json")
        segments, pauses = info["segments"], info["pauses"]
        duration_s = info["duration_s"]
        src, iou, off = "uniform", None, None
        if verbose:
            print(f"[VAD] {audio_path} 时长 {duration_s:.2f}s 语音 {info['speech_s']:.2f}s "
                  f"语音段 {len(segments)} 停顿 {len(pauses)}", file=sys.stderr)
        if not segments:
            raise ValueError("VAD 未检测到任何语音")

        # 2) 对齐到译文字符位置
        def align_uniform():
            units = alignment.tokenize(text_out)
            alignment.assign_speech_time(units, info["speech_s"])
            alignment.align_pauses(pauses, units, segments, info["speech_s"])

        if align_source == 'mfa':
            import mfa_alignment
            if any(p.scored and p.kind == 'internal' for p in pauses):
                try:
                    aligned = mfa_alignment.align(audio_path, text_out, language, info, alignment_cache_dir)
                    mfa_alignment.locate_pauses(pauses, aligned['units'], segments, info['speech_s'], text_out)
                    src = 'mfa'
                except ValueError as exc:
                    align_uniform()
                    src = 'uniform_fallback'
                    aligned = dict(language=language, model_version='service_unavailable',
                                   resolution='approximate', fallback_reason=str(exc)[:200])
            else:
                aligned = dict(language=language, model_version='not_called', resolution='not_required')
                src = 'not_required'
        elif align_source == 'whisper':
            # ja/ko：MFA 服务不支持，用预提取的 faster-whisper 词级时间戳。
            # 评分文本改用 Whisper 自己的转写（与时间戳自洽）。
            import mfa_alignment
            if not (whisper_words or {}).get('words'):
                raise ValueError('缺少 whisper 词级时间戳（whisper_words）')
            text_out = whisper_words['text'].strip()
            if not text_out:
                raise ValueError('whisper 转写为空')
            units = whisper_words['words']
            if any(p.scored and p.kind == 'internal' for p in pauses):
                if len(units) < 2:
                    raise ValueError('whisper 对齐少于 2 个计时单元，无法定位句中停顿')
                mfa_alignment.locate_pauses(pauses, units, segments, info['speech_s'], text_out)
                src = 'whisper'
                aligned = dict(language=whisper_words.get('language'),
                               model_version='faster-whisper-large-v3-int8', resolution='word')
            else:
                src = 'not_required'
                aligned = dict(language=whisper_words.get('language'),
                               model_version='faster-whisper-large-v3-int8', resolution='not_required')
        else:
            align_uniform()

    # 3) 边界类型
    method = boundary.classify(text_out, pauses, use_llm=use_llm,
                               model=model, api_base=api_base, verbose=verbose,
                               language=language)

    # 4) + 5) 容忍 / 超额 / 汇总
    metrics.assign_tolerance_and_excess(pauses, text_out, tolerance)
    rep = metrics.summarize(
        text=text_out,
        pauses=pauses,
        segments=segments,
        duration_s=duration_s,
        audio_path=audio_path,
        method=method,
        tolerance=tolerance,
    )
    rep["channel"] = channel
    rep["n_segments"] = len(segments)
    rep["align_source"] = src
    rep["align_iou"] = iou
    rep["align_offset_s"] = off
    if src in ('mfa', 'not_required', 'uniform_fallback', 'whisper'):
        rep['alignment'] = {k: v for k, v in aligned.items() if k != 'units'}
        rep['alignment']['ambiguous_pauses'] = sum(p.extra.get('alignment_ambiguous', False) for p in pauses if p.scored)
    if text and text.strip() != text_out:
        rep["reference_text"] = text.strip()   # 人工参考译文，仅作对照
    return rep
