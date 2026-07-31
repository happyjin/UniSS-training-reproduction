"""Pure helpers for formal Stage-A A4--A8 supervision.

The v1 pilot used proportional source/target boundaries.  This module only
accepts timestamped words and explicit bilingual links.  It deliberately has
no model dependencies so the alignment contract can be tested independently
from Qwen3-ForcedAligner and the multilingual word aligner used by the data
workers.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


FORMAL_ALIGNMENT_KIND = "forced_word_time_neural_bilingual_support_v2"
FORMAL_SAFE_LABEL_KIND = "oracle_bilingual_support_future_monotonic_v2"

_NEGATION = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "不",
    "没",
    "没有",
    "未",
    "无",
    "别",
}
_PUNCTUATION = set(",.!?;:，。！？；：、")
_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "了",
    "的",
    "地",
    "得",
    "着",
    "在",
    "和",
    "与",
    "及",
}


def normalize_language(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key in {"cmn", "zh", "zh-cn", "chinese", "mandarin"}:
        return "zh"
    if key in {"eng", "en", "en-us", "en-gb", "english"}:
        return "en"
    raise ValueError(f"unsupported formal Stage-A language: {value!r}")


def normalize_words(
    values: Iterable[Mapping[str, object]], *, duration_ms: int
) -> list[dict[str, object]]:
    """Validate and normalize forced-alignment word/character timestamps."""

    result: list[dict[str, object]] = []
    previous_start = 0
    previous_end = 0
    for index, value in enumerate(values):
        text = str(value.get("text", value.get("word", ""))).strip()
        if not text:
            continue
        start = int(round(float(value.get("start_ms", 0))))
        end = int(round(float(value.get("end_ms", start))))
        confidence_value = value.get("confidence")
        confidence = None if confidence_value is None else float(confidence_value)
        start = min(max(start, previous_start), max(0, duration_ms))
        end = min(max(end, start + 1, previous_end), max(1, duration_ms))
        if start >= duration_ms:
            break
        result.append(
            {
                "index": len(result),
                "text": text,
                "start_ms": start,
                "end_ms": end,
                "confidence": confidence,
                "aligner_index": index,
            }
        )
        previous_start = start
        previous_end = end
    if not result:
        raise ValueError("forced alignment returned no usable words")
    return result


def alignment_coverage(words: Sequence[Mapping[str, object]], text: str, language: str) -> float:
    aligned = "".join(str(value["text"]) for value in words)
    if normalize_language(language) == "en":
        canonical = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    else:
        canonical = lambda value: re.sub(r"\s|[，。！？；：、,.!?;:]", "", value.lower())
    reference = canonical(text)
    hypothesis = canonical(aligned)
    if not reference:
        return 1.0
    # Character multiset recall is robust to tokenizer differences while still
    # catching empty, truncated, or unrelated forced-alignment output.
    remaining: dict[str, int] = defaultdict(int)
    for char in hypothesis:
        remaining[char] += 1
    matches = 0
    for char in reference:
        if remaining[char] > 0:
            remaining[char] -= 1
            matches += 1
    return matches / len(reference)


def merge_alignment_segments(
    words: Sequence[Mapping[str, object]], segments: Sequence[str]
) -> list[dict[str, object]]:
    """Merge character-level forced timestamps into lexical segments.

    Qwen3 ForcedAligner returns CJK characters.  A8 must not split lexical
    units such as ``解决方案`` at arbitrary character boundaries, so a Chinese
    segmenter supplies the desired units while this function preserves the
    forced start/end times.  Unmatched punctuation is skipped rather than
    consuming the next aligned character.
    """

    if not words:
        return []
    result: list[dict[str, object]] = []
    cursor = 0
    for raw_segment in segments:
        segment = re.sub(r"\s+", "", str(raw_segment))
        if not segment:
            continue
        start_cursor = cursor
        consumed: list[Mapping[str, object]] = []
        combined = ""
        while cursor < len(words):
            token = re.sub(r"\s+", "", str(words[cursor].get("text", words[cursor].get("word", ""))))
            candidate = combined + token
            if not segment.casefold().startswith(candidate.casefold()):
                break
            consumed.append(words[cursor])
            combined = candidate
            cursor += 1
            if combined.casefold() == segment.casefold():
                result.append(
                    {
                        "text": segment,
                        "start_ms": int(consumed[0].get("start_ms", 0)),
                        "end_ms": int(consumed[-1].get("end_ms", 0)),
                        "confidence": min(
                            (
                                float(value["confidence"])
                                for value in consumed
                                if value.get("confidence") is not None
                            ),
                            default=None,
                        ),
                        "children": [dict(value) for value in consumed],
                    }
                )
                break
        else:
            combined = ""
        if combined.casefold() != segment.casefold():
            cursor = start_cursor
            # Punctuation can be absent from the aligner output.  For a lexical
            # mismatch, preserve the original next token instead of inventing
            # a timestamp.
            if segment not in _PUNCTUATION and cursor < len(words):
                result.append(dict(words[cursor]))
                cursor += 1
    result.extend(dict(value) for value in words[cursor:])
    return result


def build_support_alignment(
    source_words: Sequence[Mapping[str, object]],
    target_words: Sequence[Mapping[str, object]],
    links: Iterable[Mapping[str, object]],
    *,
    minimum_confidence: float = 0.35,
) -> list[dict[str, object]]:
    """Map every target word to the latest source evidence it requires.

    The returned support times are a monotonic envelope.  This is necessary
    for irreversible target-order playback: a later target word cannot be
    committed before an earlier unresolved target word.
    """

    by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    for raw in links:
        source_index = int(raw["source_index"])
        target_index = int(raw["target_index"])
        confidence = float(raw.get("confidence", 1.0))
        if not 0 <= source_index < len(source_words):
            raise IndexError(f"source alignment index out of range: {source_index}")
        if not 0 <= target_index < len(target_words):
            raise IndexError(f"target alignment index out of range: {target_index}")
        if confidence >= minimum_confidence:
            by_target[target_index].append(
                {
                    "source_index": source_index,
                    "confidence": confidence,
                    "method": str(raw.get("method", "neural_mutual_nearest")),
                }
            )

    result: list[dict[str, object]] = []
    prefix_support = 0
    for target_index, target in enumerate(target_words):
        accepted = by_target.get(target_index, [])
        if accepted:
            raw_support = max(int(source_words[value["source_index"]]["end_ms"]) for value in accepted)
            confidence = min(float(value["confidence"]) for value in accepted)
            uncertain = False
            reason = "aligned_source_evidence"
        else:
            # Unaligned function words inherit the already available prefix
            # support.  Punctuation and closed-class function words do not
            # independently carry translation evidence; unaligned content
            # words remain uncertain and block an irreversible commit.
            raw_support = prefix_support
            confidence = 0.0
            inherited_function = str(target["text"]).strip().lower() in _FUNCTION_WORDS or str(
                target["text"]
            ) in _PUNCTUATION
            uncertain = not inherited_function
            reason = "inherited_function_word_support" if inherited_function else "unaligned_target_word"
        prefix_support = max(prefix_support, raw_support)
        token = str(target["text"])
        result.append(
            {
                "target_index": target_index,
                "target_text": token,
                "raw_support_end_ms": raw_support,
                "support_end_ms": prefix_support,
                "alignment_confidence": confidence,
                "uncertain": uncertain,
                "uncertainty_reason": reason if uncertain else None,
                "negation_or_entity_risk": _has_commit_risk(token),
                "source_links": accepted,
            }
        )
    return result


def _has_commit_risk(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _NEGATION:
        return True
    if any(char.isdigit() for char in text):
        return True
    # Capitalized English tokens are a conservative named-entity proxy.
    return len(text) > 1 and text[0].isupper() and any(char.isalpha() for char in text[1:])


def _semantic_boundary(time_ms: int, duration_ms: int, semantic_count: int, *, ceil: bool) -> int:
    if duration_ms <= 0 or semantic_count <= 0:
        return 0
    value = time_ms * semantic_count / duration_ms
    boundary = math.ceil(value) if ceil else math.floor(value)
    return min(max(int(boundary), 0), semantic_count)


def _render_words(words: Sequence[Mapping[str, object]], language: str) -> str:
    values = [str(value["text"]) for value in words]
    if normalize_language(language) == "zh":
        return "".join(values)
    rendered = " ".join(values)
    for punct in _PUNCTUATION:
        rendered = rendered.replace(f" {punct}", punct)
    return rendered


def _balanced_semantic_spans(length: int, minimum: int, maximum: int) -> list[tuple[int, int]]:
    chunks = max(1, math.ceil(length / maximum))
    while chunks > 1 and length // chunks < minimum:
        chunks -= 1
    return [(length * index // chunks, length * (index + 1) // chunks) for index in range(chunks)]


def build_micro_write_supervision(
    target_words: Sequence[Mapping[str, object]],
    support: Sequence[Mapping[str, object]],
    *,
    language: str,
    target_duration_ms: int,
    target_semantic_count: int,
    minimum_semantic: int = 8,
    maximum_semantic: int = 16,
    hard_maximum_semantic: int = 24,
    tick_ms: int = 160,
    safety_margin_ms: int = 80,
) -> list[dict[str, object]]:
    """Create compact A7/A8 safe-commit and Micro-WRITE supervision."""

    if len(target_words) != len(support):
        raise ValueError("target word and support lengths differ")
    if target_semantic_count <= 0:
        raise ValueError("target semantic sequence is empty")
    if not 0 < minimum_semantic <= maximum_semantic <= hard_maximum_semantic:
        raise ValueError("invalid Micro-WRITE semantic limits")

    target_words = [
        {**dict(word), "parent_target_index": index, "parent_is_final": True}
        for index, word in enumerate(target_words)
    ]
    support = [dict(value) for value in support]

    word_ends = [
        _semantic_boundary(int(word["end_ms"]), target_duration_ms, target_semantic_count, ceil=True)
        for word in target_words
    ]
    word_ends[-1] = target_semantic_count
    word_ends = [max(value, word_ends[index - 1] if index else 0) for index, value in enumerate(word_ends)]

    events: list[dict[str, object]] = []
    word_start = 0
    semantic_start = 0
    while word_start < len(target_words):
        word_end = word_start
        selected_end = semantic_start
        oversize_word = False
        while word_end < len(target_words):
            candidate_end = max(selected_end, word_ends[word_end])
            candidate_size = candidate_end - semantic_start
            if candidate_size > hard_maximum_semantic and word_end > word_start:
                if selected_end - semantic_start < minimum_semantic:
                    selected_end = candidate_end
                    oversize_word = True
                    word_end += 1
                break
            selected_end = candidate_end
            if candidate_size > hard_maximum_semantic:
                oversize_word = True
                word_end += 1
                break
            at_boundary = str(target_words[word_end]["text"]) in _PUNCTUATION
            word_end += 1
            if candidate_size >= minimum_semantic and (candidate_size >= maximum_semantic or at_boundary):
                break
        if word_end == word_start:
            word_end += 1
            selected_end = max(selected_end, word_ends[word_start])
        if word_end == len(target_words):
            selected_end = target_semantic_count
        selected_end = max(selected_end, semantic_start + 1)

        word_slice = target_words[word_start:word_end]
        support_slice = support[word_start:word_end]
        support_end = max(int(value["support_end_ms"]) for value in support_slice)
        uncertain = any(bool(value["uncertain"]) for value in support_slice)
        risky = any(bool(value["negation_or_entity_risk"]) for value in support_slice)
        safe_after = support_end + safety_margin_ms
        earliest_safe = int(math.ceil(safe_after / tick_ms) * tick_ms)
        events.append(
            {
                "micro_write_index": len(events),
                "text": _render_words(word_slice, language),
                "target_word_start": word_start,
                "target_word_end": word_end,
                "semantic_start": semantic_start,
                "semantic_end": selected_end,
                "semantic_count": selected_end - semantic_start,
                "support_end_ms": support_end,
                "safety_margin_ms": safety_margin_ms,
                "earliest_safe_ms": earliest_safe,
                "safe_label_kind": FORMAL_SAFE_LABEL_KIND,
                "safe_if_source_ms_gte": earliest_safe,
                "future_monotonic_support": True,
                "uncertain_alignment": uncertain,
                "negation_or_entity_risk": risky,
                "natural_boundary": bool(word_slice[-1].get("parent_is_final"))
                or str(word_slice[-1]["text"]) in _PUNCTUATION,
                "oversize_word": oversize_word,
                "final_flush": word_end == len(target_words),
            }
        )
        word_start = word_end
        semantic_start = selected_end

    # A single long syllable may exceed the hard semantic bound.  Preserve its
    # text as one lexical transaction while streaming the acoustic continuation
    # in bounded semantic subchunks; continuation events carry no duplicate
    # text and keep the same support evidence.
    bounded: list[dict[str, object]] = []
    for event in events:
        count = int(event["semantic_count"])
        if count <= hard_maximum_semantic:
            bounded.append(event)
            continue
        spans = _balanced_semantic_spans(count, minimum_semantic, maximum_semantic)
        for span_index, (start, end) in enumerate(spans):
            child = dict(event)
            child["semantic_start"] = int(event["semantic_start"]) + start
            child["semantic_end"] = int(event["semantic_start"]) + end
            child["semantic_count"] = end - start
            child["text"] = str(event["text"]) if span_index == 0 else ""
            child["semantic_continuation"] = span_index > 0
            child["split_oversize_parent"] = True
            child["oversize_word"] = False
            child["natural_boundary"] = bool(event["natural_boundary"]) and span_index == len(spans) - 1
            child["final_flush"] = bool(event["final_flush"]) and span_index == len(spans) - 1
            bounded.append(child)
    events = bounded
    for index, event in enumerate(events):
        event["micro_write_index"] = index

    if events[0]["semantic_start"] != 0 or events[-1]["semantic_end"] != target_semantic_count:
        raise AssertionError("Micro-WRITE semantic coverage is incomplete")
    for left, right in zip(events, events[1:]):
        if left["semantic_end"] != right["semantic_start"]:
            raise AssertionError("Micro-WRITE semantic coverage has a gap or overlap")
    return events


def safe_label(event: Mapping[str, object], source_ms: int) -> int:
    """Materialize a compact A7 event at a sampled source prefix."""

    if bool(event.get("uncertain_alignment")):
        return 0
    threshold = int(event["safe_if_source_ms_gte"])
    return int(source_ms >= threshold)
