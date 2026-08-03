#!/usr/bin/env python3
"""Train dedicated English and Chinese SentencePiece CTC tokenizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eng-vocab-size", type=int, default=8_000)
    parser.add_argument("--cmn-vocab-size", type=int, default=8_000)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--input-sentence-size", type=int, default=2_000_000)
    return parser.parse_args()


def train_one(
    language: str,
    corpus_files: list[Path],
    output_dir: Path,
    vocab_size: int,
    num_threads: int,
    input_sentence_size: int,
) -> dict[str, object]:
    if not corpus_files:
        raise FileNotFoundError(f"no tokenizer corpus parts found for {language}")
    prefix = output_dir / f"ctc_{language}"
    model_type = "bpe" if language == "eng" else "unigram"
    spm.SentencePieceTrainer.train(
        input=",".join(str(path) for path in corpus_files),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        # English has a tiny alphabet.  Chinese uses 0.9995 so rare historical
        # or corrupted glyphs do not force the vocabulary above the requested
        # CTC-head size; they are represented by <unk> and counted in Stage01.
        character_coverage=1.0 if language == "eng" else 0.9995,
        normalization_rule_name="identity",
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=num_threads,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return {
        "language": language,
        "model_type": model_type,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": processor.vocab_size(),
        "model": str(prefix.resolve()) + ".model",
        "vocab": str(prefix.resolve()) + ".vocab",
        "blank_id_for_ctc_head": processor.vocab_size(),
        "corpus_parts": len(corpus_files),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for language, vocab_size in (
        ("eng", args.eng_vocab_size),
        ("cmn", args.cmn_vocab_size),
    ):
        corpus = sorted(args.corpus_dir.glob(f"corpus-{language}-part-*.txt"))
        results.append(
            train_one(
                language,
                corpus,
                args.output_dir,
                vocab_size,
                args.num_threads,
                args.input_sentence_size,
            )
        )
    metadata = {
        "schema_version": "uniss_streamspeech_ctc_tokenizers_v1",
        "normalization": {"eng": "NFKC+casefold+spaces", "cmn": "NFKC+no-spaces"},
        "tokenizers": results,
    }
    (args.output_dir / "tokenizers.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
