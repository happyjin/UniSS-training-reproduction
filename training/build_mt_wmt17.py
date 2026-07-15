"""Build UniSS Phase 1 MT samples from WMT17-style parallel text.

The UniSS paper mixes 2.3B MT tokens into Phase 1. This script turns already
downloaded parallel text into the same prompt/target JSONL format used by the
speech objectives. It deliberately does not download WMT17 itself; dataset
download location and license handling should be managed outside this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import sample_builders as builders


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def iter_parallel_text(
    source_path: Path,
    target_path: Path,
    src_lang: str,
    tgt_lang: str,
    id_prefix: str = "wmt17",
    limit_pairs: int | None = None,
) -> Iterator[dict[str, object]]:
    emitted = 0
    with source_path.open("r", encoding="utf-8") as src_handle, target_path.open(
        "r", encoding="utf-8"
    ) as tgt_handle:
        for line_no, (source_line, target_line) in enumerate(
            itertools.zip_longest(src_handle, tgt_handle), start=1
        ):
            if source_line is None or target_line is None:
                raise ValueError(
                    f"Parallel files have different line counts at line {line_no}: "
                    f"{source_path} vs {target_path}"
                )
            source_text = normalize_text(source_line)
            target_text = normalize_text(target_line)
            if not source_text or not target_text:
                continue
            yield {
                "id": f"{id_prefix}/{line_no}",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source_text": source_text,
                "target_text": target_text,
            }
            emitted += 1
            if limit_pairs is not None and emitted >= limit_pairs:
                return


def iter_jsonl_pairs(path: Path, limit_pairs: int | None = None) -> Iterator[dict[str, object]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            record = normalize_mt_record(raw, default_id=f"{path.stem}/{line_no}")
            if not record["source_text"] or not record["target_text"]:
                continue
            yield record
            emitted += 1
            if limit_pairs is not None and emitted >= limit_pairs:
                return


def normalize_mt_record(raw: Mapping[str, object], default_id: str) -> dict[str, object]:
    for field in ("source_text", "target_text", "src_lang", "tgt_lang"):
        if field not in raw:
            raise KeyError(f"MT record is missing required field: {field}")
    return {
        "id": str(raw.get("id") or default_id),
        "src_lang": str(raw["src_lang"]),
        "tgt_lang": str(raw["tgt_lang"]),
        "source_text": normalize_text(str(raw["source_text"])),
        "target_text": normalize_text(str(raw["target_text"])),
        "dataset_name": str(raw.get("dataset_name", "wmt17")),
        "split": str(raw.get("split", "train")),
    }


def sample_to_json(sample: builders.TrainingSample, source_record: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": source_record.get("id"),
        "phase": "phase1",
        "task": sample.task,
        "prompt_ids": sample.prompt_ids,
        "target_ids": sample.target_ids,
        "prompt_length": sample.prompt_length,
        "target_length": sample.target_length,
        "segment_spans": sample.segment_spans,
        "src_lang": source_record.get("src_lang"),
        "tgt_lang": source_record.get("tgt_lang"),
        "dataset_name": source_record.get("dataset_name"),
        "split": source_record.get("split"),
    }


def convert_records_to_samples(
    records: Iterable[Mapping[str, object]],
    text_encoder: builders.TextEncoder,
    max_sample_tokens: int | None = None,
) -> Iterator[dict[str, object]]:
    for record in records:
        sample = builders.build_mt_sample(
            src_lang=str(record["src_lang"]),
            tgt_lang=str(record["tgt_lang"]),
            source_text=str(record["source_text"]),
            target_text=str(record["target_text"]),
            text_encoder=text_encoder,
            source_id=str(record.get("id", "")) or None,
        )
        if max_sample_tokens is not None and len(sample.input_ids) > max_sample_tokens:
            continue
        yield sample_to_json(sample, record)


def write_jsonl(samples: Iterable[Mapping[str, object]], output_path: Path) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            task = str(sample["task"])
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            counts[task] += 1
            counts["total"] += 1
    return counts


def load_hf_text_encoder(tokenizer_path: str) -> builders.TextEncoder:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-text", help="Source-language text file, one sentence per line")
    parser.add_argument("--target-text", help="Target-language text file, one sentence per line")
    parser.add_argument("--input-jsonl", help="JSONL with source_text/target_text/src_lang/tgt_lang")
    parser.add_argument("--src-lang", help="Source language for --source-text mode")
    parser.add_argument("--tgt-lang", help="Target language for --target-text mode")
    parser.add_argument("--id-prefix", default="wmt17")
    parser.add_argument("--tokenizer", required=True, help="HF/UniSS tokenizer path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--max-sample-tokens", type=int, default=None)
    return parser.parse_args()


def _iter_records_from_args(args: argparse.Namespace) -> Iterator[dict[str, object]]:
    if args.input_jsonl:
        if args.source_text or args.target_text:
            raise ValueError("--input-jsonl cannot be combined with --source-text/--target-text")
        return iter_jsonl_pairs(Path(args.input_jsonl), limit_pairs=args.limit_pairs)

    missing = [
        name
        for name, value in {
            "--source-text": args.source_text,
            "--target-text": args.target_text,
            "--src-lang": args.src_lang,
            "--tgt-lang": args.tgt_lang,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Parallel text mode requires: {', '.join(missing)}")
    return iter_parallel_text(
        Path(args.source_text),
        Path(args.target_text),
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        id_prefix=args.id_prefix,
        limit_pairs=args.limit_pairs,
    )


def main() -> None:
    args = parse_args()
    text_encoder = load_hf_text_encoder(args.tokenizer)
    records = _iter_records_from_args(args)
    samples = convert_records_to_samples(
        records,
        text_encoder=text_encoder,
        max_sample_tokens=args.max_sample_tokens,
    )
    counts = write_jsonl(samples, Path(args.output))
    print(json.dumps({"output": args.output, "counts": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
