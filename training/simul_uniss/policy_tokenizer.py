"""Train and load the compact text tokenizer used by Source/Target CTC heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq
import sentencepiece as spm


class PolicyTokenizer:
    """SentencePiece wrapper reserving CTC blank as model class zero.

    SentencePiece ids are shifted by one for CTC targets, so class 0 is always
    the blank and classes 1..N map to SentencePiece ids 0..N-1.
    """

    blank_id = 0

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def piece_size(self) -> int:
        return int(self.processor.get_piece_size())

    @property
    def ctc_vocab_size(self) -> int:
        return self.piece_size + 1

    def encode(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.processor.encode(text, out_type=int)]

    def encode_ctc(self, text: str) -> list[int]:
        return [token_id + 1 for token_id in self.encode(text)]

    def decode_ctc(self, token_ids: Sequence[int]) -> str:
        piece_ids = [int(token_id) - 1 for token_id in token_ids if int(token_id) > 0]
        return self.processor.decode(piece_ids)


def iter_text(paths: Sequence[Path], limit_records: int | None = None) -> Iterator[str]:
    emitted = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["transcription", "translation"], batch_size=1024):
            for row in batch.to_pylist():
                for field in ("transcription", "translation"):
                    text = str(row[field]).strip()
                    if text:
                        yield text.replace("\n", " ")
                emitted += 1
                if limit_records is not None and emitted >= limit_records:
                    return


def train_policy_tokenizer(
    paths: Sequence[Path],
    output_dir: Path,
    *,
    vocab_size: int = 8192,
    limit_records: int | None = None,
    num_threads: int = 8,
) -> Path:
    if vocab_size < 32:
        raise ValueError("vocab_size must be at least 32")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "policy_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for text in iter_text(paths, limit_records=limit_records):
            handle.write(text + "\n")
    prefix = output_dir / "policy_8k"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
        num_threads=num_threads,
        shuffle_input_sentence=False,
    )
    model_path = prefix.with_suffix(".model")
    tokenizer = PolicyTokenizer(model_path)
    manifest = {
        "schema_version": "simul_uniss_policy_tokenizer_v1",
        "model": str(model_path),
        "requested_vocab_size": vocab_size,
        "piece_size": tokenizer.piece_size,
        "ctc_vocab_size": tokenizer.ctc_vocab_size,
        "ctc_blank_id": tokenizer.blank_id,
        "input_shards": [str(path.resolve()) for path in paths],
        "limit_records": limit_records,
    }
    (output_dir / "policy_tokenizer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = train_policy_tokenizer(
        [Path(value) for value in args.input],
        Path(args.output_dir),
        vocab_size=args.vocab_size,
        limit_records=args.limit_records,
        num_threads=args.num_threads,
    )
    print(json.dumps({"model": str(model_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
