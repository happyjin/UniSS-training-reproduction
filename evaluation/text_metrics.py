"""Paper-aligned text normalization and corpus BLEU aggregation."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evaluation.io_utils import iter_jsonl, write_json
from training.constants_uniss import normalize_language


def remove_punctuation(text: str, *, preserve_apostrophe: bool = False) -> str:
    output: list[str] = []
    for character in text:
        if preserve_apostrophe and character in {"'", "’"}:
            output.append("'")
        elif unicodedata.category(character).startswith("P"):
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


def normalize_english(text: str) -> str:
    value = remove_punctuation(unicodedata.normalize("NFKC", text).lower(), preserve_apostrophe=True)
    return " ".join(value.split())


def opencc_simplifier() -> Callable[[str], str]:
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise RuntimeError(
            "Chinese paper-aligned BLEU requires opencc-python-reimplemented; "
            "install the evaluation requirements first"
        ) from exc
    converter = OpenCC("t2s")
    return converter.convert


def normalize_chinese(text: str, *, simplify: Callable[[str], str] | None = None) -> str:
    converter = simplify or opencc_simplifier()
    value = converter(unicodedata.normalize("NFKC", text))
    value = remove_punctuation(value)
    characters = [character for character in value if not character.isspace()]
    return " ".join(characters)


def normalize_for_bleu(text: str, language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "eng":
        return normalize_english(text)
    return normalize_chinese(text)


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str], *, language: str) -> dict[str, object]:
    if len(hypotheses) != len(references):
        raise ValueError("hypothesis/reference counts differ")
    if not hypotheses:
        raise ValueError("cannot compute corpus BLEU on an empty group")
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("Install sacrebleu before computing BLEU") from exc

    normalized_language = normalize_language(language)
    normalized_hypotheses = [normalize_for_bleu(text, normalized_language) for text in hypotheses]
    normalized_references = [normalize_for_bleu(text, normalized_language) for text in references]
    tokenize = "zh" if normalized_language == "cmn" else "13a"
    score = sacrebleu.corpus_bleu(normalized_hypotheses, [normalized_references], tokenize=tokenize)
    return {
        "score": float(score.score),
        "counts": list(score.counts),
        "totals": list(score.totals),
        "precisions": [float(value) for value in score.precisions],
        "bp": float(score.bp),
        "sys_len": int(score.sys_len),
        "ref_len": int(score.ref_len),
        "tokenizer": tokenize,
        "sample_count": len(hypotheses),
    }


def compute_grouped_bleu(
    rows: Sequence[Mapping[str, object]],
    *,
    hypothesis_field: str,
    reference_field: str,
) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    skipped: list[dict[str, object]] = []
    for row in rows:
        hypothesis = row.get(hypothesis_field)
        reference = row.get(reference_field)
        if not isinstance(hypothesis, str) or not hypothesis.strip() or not isinstance(reference, str) or not reference.strip():
            skipped.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "missing_text"})
            continue
        key = (str(row.get("mode", "unknown")), str(row.get("src_lang")), str(row.get("tgt_lang")))
        groups[key].append(row)

    output_groups: dict[str, object] = {}
    for (mode, src_lang, tgt_lang), group_rows in sorted(groups.items()):
        hypotheses = [str(row[hypothesis_field]) for row in group_rows]
        references = [str(row[reference_field]) for row in group_rows]
        output_groups[f"{mode}:{src_lang}->{tgt_lang}"] = corpus_bleu(
            hypotheses,
            references,
            language=tgt_lang,
        )
    return {
        "hypothesis_field": hypothesis_field,
        "reference_field": reference_field,
        "groups": output_groups,
        "input_count": len(rows),
        "scored_count": sum(int(group["sample_count"]) for group in output_groups.values()),  # type: ignore[index,union-attr]
        "skipped_count": len(skipped),
        "skipped": skipped,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis-field", default="generated_translation")
    parser.add_argument("--reference-field", default="translation_ref")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = list(iter_jsonl(args.input))
    report = compute_grouped_bleu(
        rows,
        hypothesis_field=args.hypothesis_field,
        reference_field=args.reference_field,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
