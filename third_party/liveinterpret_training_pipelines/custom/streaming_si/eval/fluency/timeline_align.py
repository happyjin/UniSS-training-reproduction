"""用 S2ST pipeline 自带的 timeline.json 做 ground-truth 对齐。

timeline.json 的 segments 记录了每个 chunk 的：
    text            该 chunk 实际发射的文本
    first_pcm_sec   该 chunk 语音 PCM 的起始时间
    dur_sec         该 chunk 语音时长

由此可以直接得到：
    - 实际说出的文本（不是参考译文）
    - 每个字的精确时间区间
    - 停顿的精确位置与时长（chunk 之间的间隙）

⚠ 已知坑：first_pcm_sec 的时钟与 dual.wav 的时间轴差一个「逐条不同」的常数偏移
  （实测范围 ±2.8s，中位数 -0.45s）。因此必须先用 VAD 语音段与 timeline chunk
  做互相关估计偏移，校正后才能使用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from io_utils import key_from_audio
from schema import (PAUSE_INTERNAL, PAUSE_LEADING, PAUSE_TRAILING, Pause,
                    SpeechSegment)

RES = 0.01            # 掩码分辨率（秒）
SEARCH_RANGE = 3.0    # 偏移搜索范围 ±3s
SEARCH_STEP = 0.05    # 偏移搜索步长
MIN_IOU = 0.30        # 校正后 IoU 低于此值认为不可信，回退到均匀摊派


@dataclass
class Chunk:
    """timeline 中的一个发射 chunk。"""
    text: str
    start: float
    end: float
    char_start: int   # 在拼接后的实际文本中的起始下标
    char_end: int     # 结束下标（不含）

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_timeline(audio_path: str, timeline_dir: Optional[str] = None) -> Optional[str]:
    """由音频路径推断 timeline.json 路径。

    目录约定：<...>/<name>/audio/<key>_dual.wav
              <...>/<name>/results/<key>_timeline.json
    """
    key = key_from_audio(audio_path)
    if timeline_dir:
        cand = Path(timeline_dir) / f"{key}_timeline.json"
    else:
        p = Path(audio_path).resolve()
        cand = p.parent.parent / "results" / f"{key}_timeline.json"
    return str(cand) if cand.exists() else None


def _mask(intervals: Sequence[Tuple[float, float]], total: float,
          res: float = RES) -> set:
    """把时间区间集合离散成 10ms 网格的掩码集合。"""
    m: set = set()
    for a, b in intervals:
        i = max(0, int(a / res))
        while i * res < b and i * res < total:
            m.add(i)
            i += 1
    return m


def load_chunks(timeline_path: str, offset: float = 0.0) -> Tuple[List[Chunk], float]:
    """读取 timeline，按时间排序并拼出实际文本，返回 (chunks, dual_sec)。"""
    with open(timeline_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    segs = [s for s in d.get("segments", []) if s.get("text")]
    segs.sort(key=lambda s: s.get("first_pcm_sec", 0.0))

    chunks: List[Chunk] = []
    pos = 0
    for s in segs:
        st = float(s.get("first_pcm_sec", 0.0)) + offset
        en = st + float(s.get("dur_sec", 0.0))
        t = s["text"]
        chunks.append(Chunk(text=t, start=st, end=en,
                            char_start=pos, char_end=pos + len(t)))
        pos += len(t)
    return chunks, float(d.get("dual_sec") or 0.0)


def estimate_offset(chunks: Sequence[Chunk],
                    vad_segments: Sequence[SpeechSegment],
                    total: float,
                    search: float = SEARCH_RANGE,
                    step: float = SEARCH_STEP) -> Tuple[float, float]:
    """用互相关估计 timeline 时钟相对音频时间轴的常数偏移。

    返回 (offset, iou)。offset 为加在 first_pcm_sec 上的修正量。
    """
    vm = _mask([(s.start, s.end) for s in vad_segments], total)
    if not vm:
        return 0.0, 0.0

    base = [(c.start, c.end) for c in chunks]
    best_iou, best_off = -1.0, 0.0
    off = -search
    while off <= search + 1e-9:
        tm = _mask([(a + off, b + off) for a, b in base], total)
        inter = len(tm & vm)
        union = len(tm | vm) or 1
        iou = inter / union
        if iou > best_iou:
            best_iou, best_off = iou, round(off, 3)
        off += step
    return best_off, best_iou


def pauses_from_chunks(chunks: Sequence[Chunk], total_duration: float,
                       min_pause_s: float) -> List[Pause]:
    """由 chunk 间隙直接得到停顿（ground truth，不依赖 VAD）。"""
    pauses: List[Pause] = []
    idx = 0
    if not chunks:
        return pauses

    # 首个 chunk 之前：同传 EVS 延迟，不计分
    if chunks[0].start > min_pause_s:
        pauses.append(Pause(idx=idx, start=0.0, end=chunks[0].start,
                            kind=PAUSE_LEADING, scored=False))
        idx += 1

    # chunk 之间
    spoken = 0.0
    for i, (prev, cur) in enumerate(zip(chunks, chunks[1:])):
        spoken += prev.duration
        gap = cur.start - prev.end
        if gap >= min_pause_s:
            p = Pause(idx=idx, start=prev.end, end=cur.start,
                      kind=PAUSE_INTERNAL, scored=True)
            p.char_index = prev.char_end       # 精确落点：前一个 chunk 最后一个字之后
            p.speech_time = spoken
            pauses.append(p)
            idx += 1
    spoken += chunks[-1].duration

    for p in pauses:
        p.progress = (p.speech_time / spoken) if spoken > 0 else 0.0

    # 末尾静音
    tail = total_duration - chunks[-1].end
    if tail > min_pause_s:
        pauses.append(Pause(idx=idx, start=chunks[-1].end, end=total_duration,
                            kind=PAUSE_TRAILING, scored=False))
    return pauses


def build(audio_path: str,
          vad_info: dict,
          min_pause_s: float,
          timeline_dir: Optional[str] = None) -> Optional[dict]:
    """组装 ground-truth 对齐结果；不可用则返回 None（调用方回退）。"""
    tl_path = find_timeline(audio_path, timeline_dir)
    if not tl_path:
        return None

    raw_chunks, dual_sec = load_chunks(tl_path, offset=0.0)
    if not raw_chunks:
        return None

    total = vad_info.get("duration_s") or dual_sec
    offset, iou = estimate_offset(raw_chunks, vad_info.get("segments", []), total)
    if iou < MIN_IOU:
        return None     # 时间轴对不上，不可信

    chunks, _ = load_chunks(tl_path, offset=offset)
    text = "".join(c.text for c in chunks)
    pauses = pauses_from_chunks(chunks, total, min_pause_s)
    segments = [SpeechSegment(start=c.start, end=c.end) for c in chunks]

    return {
        "source": "timeline",
        "timeline_path": tl_path,
        "offset_s": offset,
        "iou": round(iou, 3),
        "text": text,
        "chunks": chunks,
        "duration_s": total,
        "speech_s": sum(c.duration for c in chunks),
        "segments": segments,
        "pauses": pauses,
    }
