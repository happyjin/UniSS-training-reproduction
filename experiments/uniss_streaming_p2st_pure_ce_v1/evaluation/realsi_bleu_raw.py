#!/usr/bin/env python3
"""ASR-BLEU on raw text, beside the repository's normalized protocol.

``evaluation/text_metrics.compute_grouped_bleu`` applies this lineage's own
normalization before scoring -- NFKC, lowercase and punctuation stripping for
English; OpenCC traditional-to-simplified, punctuation stripping and character
splitting for Chinese -- which is more aggressive than plain sacreBLEU and on
this repository's data is worth a few points.

The paper states neither its sacreBLEU version nor any normalization.  So this
scores the same hypotheses and references *raw*, with only sacreBLEU's own
tokenizer (``zh`` for Chinese, ``13a`` for English).  Both numbers go in the
report: the gap between them is the measured size of the protocol ambiguity,
and quoting one without the other would hide it.

Empty hypotheses are scored, never skipped, for the same reason
``--score-empty-hypotheses`` is mandatory on the other path -- dropping them
would reward an arm for failing to speak.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sacrebleu

TOKENIZER = {"cmn": "zh", "eng": "13a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hypothesis-field", default="asr_text")
    parser.add_argument("--reference-field", default="translation_ref")
    args = parser.parse_args()

    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    missing_reference = 0
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reference = row.get(args.reference_field)
            if not isinstance(reference, str) or not reference.strip():
                missing_reference += 1
                continue
            hypothesis = row.get(args.hypothesis_field)
            hypothesis = hypothesis if isinstance(hypothesis, str) else ""
            key = (
                str(row.get("mode", "unknown")),
                str(row.get("src_lang")),
                str(row.get("tgt_lang")),
            )
            groups[key].append((hypothesis, reference))

    report: dict[str, object] = {
        "protocol": "raw_sacrebleu_no_normalization",
        "sacrebleu": sacrebleu.__version__,
        "missing_reference_rows": missing_reference,
        "groups": {},
    }
    for (mode, src_lang, tgt_lang), pairs in sorted(groups.items()):
        tokenize = TOKENIZER.get(tgt_lang, "13a")
        hypotheses = [pair[0] for pair in pairs]
        references = [pair[1] for pair in pairs]
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        report["groups"][f"{mode}:{src_lang}->{tgt_lang}"] = {  # type: ignore[index]
            "samples": len(pairs),
            "empty_hypotheses": sum(1 for text in hypotheses if not text.strip()),
            "tokenize": tokenize,
            "bleu": bleu.score,
            "chrf": chrf.score,
            "tokenizer": tokenize,
            "brevity_penalty": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for key, row in sorted(report["groups"].items()):  # type: ignore[union-attr]
        print(
            "%-42s n=%-4d BLEU %6.2f  chrF %6.2f  empty %d"
            % (key, row["samples"], row["bleu"], row["chrf"], row["empty_hypotheses"])
        )
    print("wrote", args.output)


if __name__ == "__main__":
    main()
