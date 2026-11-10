"""Talker-side fixes of the text->speech gap plan (reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md §5),
used by si_infer.py only when their flags are set (defaults leave si_infer bit-identical).

B2 (--rerun-forced N): a block whose codes reach the SIGuard hard limit was cut off by the guard, not closed by the
   talker ("forced": WER 54-66 % on dev). Such a block is sampled again with another seed, at most N times; the
   first sample that closes on its own replaces it.
B3 (--src-mask-own-ms PAD): in a multi-speaker window every speaker's stream used to read the same mixed window
   audio. With B3 a block reads the window audio heard so far with every sample outside this stream's own segments
   set to zero. The segments are the system's own (start_s..end_s of the stream's blocks written by this block's
   write time, widened by PAD ms), so the talker sees no reference annotation and nothing written later.
"""
from __future__ import annotations

import numpy as np

SR_PER_MS = 16   # 16 kHz


def forced_limit(expected: int) -> int:
    """si_infer's rule for a guard-forced block: codes >= max(2.2 expected, expected + 100)."""
    return max(int(2.2 * expected), expected + 100)


def is_forced(n_codes: int, expected: int) -> bool:
    return n_codes >= forced_limit(expected)


def own_spans_ms(blocks: list[dict], write_ms: int, pad_ms: int) -> list[tuple[int, int]]:
    """The stream's own segments known at write_ms: every block written by then (write_ms <= this one's), its
    start_s..end_s in ms (swapped when the model wrote them reversed), widened by pad_ms on each side."""
    out = []
    for b in blocks:
        if b["write_ms"] > write_ms:
            continue
        s, e = sorted((float(b["start_s"]), float(b["end_s"])))
        out.append((int(round(s * 1000)) - pad_ms, int(round(e * 1000)) + pad_ms))
    return out


def own_speaker_heard(heard: np.ndarray, spans_ms: list[tuple[int, int]]) -> np.ndarray:
    """heard (16 kHz, the window up to the block's write time) with every sample outside spans_ms set to 0; same
    length, so the GLM token count per second (and holdback / glm_end) is unchanged."""
    keep = np.zeros(len(heard), dtype=bool)
    for s, e in spans_ms:
        a, b = max(0, s * SR_PER_MS), min(len(heard), e * SR_PER_MS)
        if b > a:
            keep[a:b] = True
    out = heard.copy()
    out[~keep] = 0.0
    return out
