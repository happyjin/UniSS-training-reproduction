"""Shared text and CTC utilities for the isolated Stage01 pipeline."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import sentencepiece as spm


LANG_ALIASES = {"eng": "eng", "en": "eng", "cmn": "cmn", "zh": "cmn"}


def canonical_lang(value: str) -> str:
    try:
        return LANG_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported language: {value!r}") from exc


def normalize_text(text: str, language: str) -> str:
    language = canonical_lang(language)
    text = unicodedata.normalize("NFKC", str(text)).strip()
    if language == "eng":
        text = text.casefold()
        return re.sub(r"\s+", " ", text)
    return re.sub(r"\s+", "", text)


def ctc_minimum_frames(token_ids: list[int]) -> int:
    """Minimum CTC path length, including blanks between repeated labels."""

    if not token_ids:
        return 0
    repeats = sum(left == right for left, right in zip(token_ids, token_ids[1:]))
    return len(token_ids) + repeats


def deterministic_split(record_id: str, valid_basis_points: int = 100) -> str:
    if not 0 <= valid_basis_points < 10_000:
        raise ValueError("valid_basis_points must be in [0, 10000)")
    digest = hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little") % 10_000
    return "valid" if bucket < valid_basis_points else "train"


def load_processor(path: str | Path) -> spm.SentencePieceProcessor:
    processor = spm.SentencePieceProcessor()
    if not processor.load(str(path)):
        raise RuntimeError(f"failed to load SentencePiece model: {path}")
    return processor

