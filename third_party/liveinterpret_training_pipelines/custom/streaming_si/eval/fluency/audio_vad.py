"""音频加载 + Silero VAD 语音段检测 + 停顿提取。

对外主要接口：
    load_audio_16k(path)                 -> (torch.Tensor, 总时长秒)
    detect_speech_segments(wav, ...)     -> list[SpeechSegment]
    extract_pauses(segments, total, ...) -> list[Pause]
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import soundfile as sf
import soxr
import torch

from config import (
    MIN_PAUSE_S,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SAMPLING_RATE,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
)
from schema import PAUSE_INTERNAL, PAUSE_LEADING, PAUSE_TRAILING, Pause, SpeechSegment

_MODEL = None


def get_vad_model():
    """懒加载 Silero VAD（权重打包在 pip 包内，不联网）。"""
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(1)
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def load_audio_16k(
    path: str,
    target_sr: int = VAD_SAMPLING_RATE,
    channel: str = "mix",
) -> Tuple[torch.Tensor, float]:
    """读取任意采样率 / 声道数的音频，重采样为 16k 单声道 float32。

    channel:
        "mix"  -> 所有声道取平均（默认）
        "0"/"1"-> 只取该声道（dual.wav 常用 "1" 取译音）

    返回 (wav_tensor, 总时长秒)。
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    if data.shape[1] > 1 and channel not in ("mix", "", None):
        idx = int(channel)
        if idx >= data.shape[1]:
            raise ValueError(f"{path} 只有 {data.shape[1]} 个声道，无法取 channel={idx}")
        data = data[:, idx]
    else:
        data = data.mean(axis=1)            # 多声道取平均

    if sr != target_sr:
        data = soxr.resample(np.asarray(data, dtype=np.float32), sr, target_sr)
    wav = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    return wav, len(wav) / float(target_sr)


def detect_speech_segments(
    wav: torch.Tensor,
    threshold: float = VAD_THRESHOLD,
    min_silence_ms: int = VAD_MIN_SILENCE_MS,
    min_speech_ms: int = VAD_MIN_SPEECH_MS,
    sampling_rate: int = VAD_SAMPLING_RATE,
) -> List[SpeechSegment]:
    """用 Silero VAD 找出所有语音段（秒）。"""
    from silero_vad import get_speech_timestamps

    model = get_vad_model()
    raw = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        return_seconds=True,
    )
    return [SpeechSegment(start=float(d["start"]), end=float(d["end"])) for d in raw]


def extract_pauses(
    segments: List[SpeechSegment],
    total_duration: float,
    min_pause_s: float = MIN_PAUSE_S,
) -> List[Pause]:
    """由语音段推导停顿。

    - 音频开头的静音、结尾的静音单独标记为 leading / trailing，不参与评分；
    - 语音段之间的间隙即为停顿；
    - 短于 min_pause_s 的间隙忽略（多半是爆破音 / 呼吸的极短间隙）。
    """
    pauses: List[Pause] = []
    idx = 0

    if not segments:
        # 全程静音
        pauses.append(Pause(idx=idx, start=0.0, end=total_duration,
                            kind=PAUSE_LEADING, scored=False))
        return pauses

    # 1) 前置静音
    if segments[0].start > min_pause_s:
        pauses.append(Pause(idx=idx, start=0.0, end=segments[0].start,
                            kind=PAUSE_LEADING, scored=False))
        idx += 1

    # 2) 段间停顿
    for prev, cur in zip(segments, segments[1:]):
        gap = cur.start - prev.end
        if gap >= min_pause_s:
            pauses.append(Pause(idx=idx, start=prev.end, end=cur.start,
                                kind=PAUSE_INTERNAL, scored=True))
            idx += 1

    # 3) 尾部静音
    tail = total_duration - segments[-1].end
    if tail > min_pause_s:
        pauses.append(Pause(idx=idx, start=segments[-1].end, end=total_duration,
                            kind=PAUSE_TRAILING, scored=False))
        idx += 1

    return pauses


def analyze_audio(path: str, channel: str = "mix") -> dict:
    """一步到位：加载 -> VAD -> 停顿，返回中间结果字典。"""
    wav, total = load_audio_16k(path, channel=channel)
    segments = detect_speech_segments(wav)
    pauses = extract_pauses(segments, total)
    return {
        "audio_path": path,
        "channel": channel,
        "duration_s": total,
        "segments": segments,
        "pauses": pauses,
        "speech_s": sum(s.duration for s in segments),
    }


def probe_channels(path: str) -> dict:
    """返回声道信息 + 每个声道的 VAD 统计量，用于判断哪个声道是译音。"""
    info = sf.info(path)
    out = {
        "channels": info.channels,
        "samplerate": info.samplerate,
        "duration_s": round(info.duration, 3),
        "per_channel": [],
    }
    if info.channels < 2:
        return out
    for c in range(info.channels):
        wav, _ = load_audio_16k(path, channel=str(c))
        segs = detect_speech_segments(wav)
        out["per_channel"].append({
            "channel": c,
            "speech_s": round(sum(s.duration for s in segs), 3),
            "n_segments": len(segs),
            "onset_s": round(segs[0].start, 3) if segs else None,
        })
    return out
