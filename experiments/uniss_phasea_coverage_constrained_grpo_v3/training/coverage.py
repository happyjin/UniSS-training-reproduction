"""Auditable target-language coverage metrics for long event rollouts.

The previous event reward treated a model that translated roughly one third of
the reference as "complete enough" whenever it retained the weak historical
baseline.  This module measures what fraction of the frozen teacher target has
actually been produced, in monotonic order, and what fraction has reached a
healthy TTS emission.  It intentionally uses only fields already present in
the immutable episode protocol and runtime event records.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def tokens(text: str, language: str) -> list[str]:
    """Tokenize conservatively without downloading an external tokenizer."""

    text = str(text).lower()
    if str(language).startswith("cmn") or str(language).startswith("zh"):
        return [* _CJK.findall(text), *_WORD.findall(text)]
    return _WORD.findall(text)


def monotonic_match_fraction(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    matcher = SequenceMatcher(a=list(reference), b=list(hypothesis), autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return unit(matched / len(reference))


def length_score(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    ratio = len(hypothesis) / max(1, len(reference))
    return math.exp(-abs(math.log(max(1e-4, ratio))))


def repeated_ngram_fraction(sequence: Sequence[str], n: int = 3) -> float:
    if len(sequence) < n:
        return 0.0
    grams = [tuple(sequence[index : index + n]) for index in range(len(sequence) - n + 1)]
    return unit(1.0 - len(set(grams)) / len(grams))


def language_purity(text: str, target_language: str) -> float:
    """A deliberately light target-language check; names/punctuation are allowed."""

    text = str(text)
    cjk = len(_CJK.findall(text))
    latin = len(_WORD.findall(text))
    total = cjk + latin
    if total == 0:
        return 1.0
    if str(target_language).startswith("cmn") or str(target_language).startswith("zh"):
        # Chinese may contain names and numbers in Latin script, but it should
        # not become predominantly English.
        return unit((cjk + 0.25 * latin) / total)
    # English outputs should not contain un-translated CJK spans.
    return unit(latin / total)


@dataclass(frozen=True)
class CoverageAudit:
    teacher_tokens: int
    generated_tokens: int
    spoken_tokens: int
    target_coverage: float
    spoken_target_coverage: float
    length_score: float
    language_purity: float
    repetition_fraction: float
    eos_pending_items: int
    event_progress: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _emitted_text(events: Iterable[dict[str, object]]) -> str:
    values: list[str] = []
    for event in events:
        for emission in event.get("tts_emissions", []):
            if bool(emission.get("acknowledged", False)):
                values.append(str(emission.get("text", "")))
    return " ".join(values)


def audit_episode(
    *,
    teacher_translation: str,
    generated_translation: str,
    target_language: str,
    events: list[dict[str, object]],
    eos_pending_items: int,
) -> CoverageAudit:
    """Annotate event rows in place with monotonically accumulated coverage."""

    reference = tokens(teacher_translation, target_language)
    generated = tokens(generated_translation, target_language)
    spoken = tokens(_emitted_text(events), target_language)
    progress: list[dict[str, float]] = []
    last_target = 0.0
    last_spoken = 0.0
    emitted_prefix: list[str] = []
    for event in events:
        committed = tokens(str(event.get("mt_committed", "")), target_language)
        target = monotonic_match_fraction(reference, committed)
        for emission in event.get("tts_emissions", []):
            if bool(emission.get("acknowledged", False)):
                emitted_prefix.extend(tokens(str(emission.get("text", "")), target_language))
        spoken_target = monotonic_match_fraction(reference, emitted_prefix)
        item = {
            "target_coverage": target,
            "target_coverage_delta": max(0.0, target - last_target),
            "spoken_target_coverage": spoken_target,
            "spoken_target_coverage_delta": max(0.0, spoken_target - last_spoken),
            "empty_write": float(
                str(event.get("executed_action", "WAIT")) == "WRITE"
                and not str(event.get("mt_new_commit", "")).strip()
            ),
            "language_leak": 1.0 - language_purity(
                str(event.get("mt_new_commit", "")), target_language
            ) if str(event.get("mt_new_commit", "")).strip() else 0.0,
        }
        event["coverage"] = item
        progress.append(item)
        last_target = target
        last_spoken = spoken_target
    return CoverageAudit(
        teacher_tokens=len(reference),
        generated_tokens=len(generated),
        spoken_tokens=len(spoken),
        target_coverage=monotonic_match_fraction(reference, generated),
        spoken_target_coverage=monotonic_match_fraction(reference, spoken),
        length_score=length_score(reference, generated),
        language_purity=language_purity(generated_translation, target_language),
        repetition_fraction=repeated_ngram_fraction(generated),
        eos_pending_items=int(eos_pending_items),
        event_progress=tuple(progress),
    )


__all__ = ["CoverageAudit", "audit_episode", "language_purity", "tokens"]
