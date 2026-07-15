"""Convert UniST parquet rows into UniSS S2ST training samples.

Phase 2 emits Quality, Performance, and Direct S2ST samples. Phase 3 emits only
Quality and Performance samples, matching the paper's final refinement stage.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq

from training import sample_builders as builders


REQUIRED_COLUMNS = {
    "id",
    "transcription",
    "translation",
    "source_glm",
    "target_bicodec",
    "bicodec_global",
    "src_lang",
    "tgt_lang",
}


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(path) for path in glob.glob(pattern)]
        if matched:
            paths.extend(sorted(matched))
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    unique_paths = sorted(dict.fromkeys(paths))
    if not unique_paths:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return unique_paths


def _coerce_token_list(value: object, field_name: str) -> list[int]:
    if value is None:
        raise ValueError(f"{field_name} is missing")
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise TypeError(f"{field_name} item must be int, got {type(item).__name__}")
        tokens.append(item)
    return tokens


def normalize_unist_record(raw: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(raw.keys()))
    if missing:
        raise KeyError(f"UniST row is missing required columns: {missing}")

    record = {
        "id": str(raw["id"]),
        "transcription": str(raw["transcription"]),
        "translation": str(raw["translation"]),
        "source_glm": _coerce_token_list(raw["source_glm"], "source_glm"),
        "target_bicodec": _coerce_token_list(raw["target_bicodec"], "target_bicodec"),
        "bicodec_global": _coerce_token_list(raw["bicodec_global"], "bicodec_global"),
        "src_lang": str(raw["src_lang"]),
        "tgt_lang": str(raw["tgt_lang"]),
    }

    if "source_bicodec" in raw and raw["source_bicodec"] is not None:
        record["source_bicodec"] = _coerce_token_list(raw["source_bicodec"], "source_bicodec")
    if "dataset_name" in raw:
        record["dataset_name"] = str(raw["dataset_name"])
    if "split" in raw:
        record["split"] = str(raw["split"])
    if "duration_ratio" in raw and raw["duration_ratio"] is not None:
        record["duration_ratio"] = float(raw["duration_ratio"])

    return record


def iter_unist_records(paths: Sequence[Path], limit_records: int | None = None) -> Iterator[dict[str, object]]:
    emitted = 0
    for path in paths:
        table = pq.read_table(path)
        for raw in table.to_pylist():
            yield normalize_unist_record(raw)
            emitted += 1
            if limit_records is not None and emitted >= limit_records:
                return


def sample_to_json(sample: builders.TrainingSample, phase: str, source_record: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": source_record.get("id"),
        "phase": phase,
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
        "duration_ratio": source_record.get("duration_ratio"),
    }


def convert_records_to_samples(
    records: Iterable[Mapping[str, object]],
    text_encoder: builders.TextEncoder,
    phase: str,
) -> Iterator[dict[str, object]]:
    if phase not in {"phase2", "phase3"}:
        raise ValueError("phase must be 'phase2' or 'phase3'")

    include_direct = phase == "phase2"
    for record in records:
        samples = builders.build_s2st_samples_from_record(
            record, text_encoder=text_encoder, include_direct=include_direct
        )
        for sample in samples:
            yield sample_to_json(sample, phase, record)


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
    parser.add_argument("--input", nargs="+", required=True, help="Parquet files or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--phase", required=True, choices=["phase2", "phase3"])
    parser.add_argument("--tokenizer", required=True, help="HF/UniSS tokenizer path")
    parser.add_argument("--limit-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_input_paths(args.input)
    text_encoder = load_hf_text_encoder(args.tokenizer)
    records = iter_unist_records(paths, limit_records=args.limit_records)
    samples = convert_records_to_samples(records, text_encoder=text_encoder, phase=args.phase)
    counts = write_jsonl(samples, Path(args.output))
    print(json.dumps({"output": args.output, "counts": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
