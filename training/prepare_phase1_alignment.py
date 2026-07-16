"""Build UniSS Phase 1 speech-text alignment samples.

The paper's first stage uses ASR, TTS, S2TT, and MT objectives. This script
implements the speech-side objectives from tokenized UniST-style parquet files:
ASR, S2TT, and TTS. For UniST-only bring-up runs, it can also emit an MT proxy
from the same row's transcription/translation pair.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import sample_builders as builders


SPEECH_TASKS = ("asr", "s2tt", "tts")
PHASE1_TASKS = (*SPEECH_TASKS, "mt")
REQUIRED_COLUMNS = {
    "id",
    "transcription",
    "translation",
    "source_glm",
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


def normalize_alignment_record(raw: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(raw.keys()))
    if missing:
        raise KeyError(f"Phase 1 row is missing required columns: {missing}")

    record: dict[str, object] = {
        "id": str(raw["id"]),
        "transcription": str(raw["transcription"]),
        "translation": str(raw["translation"]),
        "source_glm": _coerce_token_list(raw["source_glm"], "source_glm"),
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


def iter_alignment_records(paths: Sequence[Path], limit_records: int | None = None) -> Iterator[dict[str, object]]:
    emitted = 0
    for path in paths:
        table = pq.read_table(path)
        for raw in table.to_pylist():
            yield normalize_alignment_record(raw)
            emitted += 1
            if limit_records is not None and emitted >= limit_records:
                return


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
        "duration_ratio": source_record.get("duration_ratio"),
    }


def build_task_samples(
    record: Mapping[str, object],
    text_encoder: builders.TextEncoder,
    tasks: Sequence[str],
) -> list[builders.TrainingSample]:
    selected = set(tasks)
    unknown = selected - set(PHASE1_TASKS)
    if unknown:
        raise ValueError(f"Unsupported Phase 1 speech tasks: {sorted(unknown)}")

    source_id = str(record.get("id", "")) or None
    samples: list[builders.TrainingSample] = []
    if "asr" in selected:
        samples.append(
            builders.build_asr_sample(
                source_glm=record["source_glm"],  # type: ignore[arg-type]
                bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
                src_lang=str(record["src_lang"]),
                transcription=str(record["transcription"]),
                text_encoder=text_encoder,
                source_id=source_id,
            )
        )
    if "s2tt" in selected:
        samples.append(
            builders.build_s2tt_sample(
                source_glm=record["source_glm"],  # type: ignore[arg-type]
                bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
                tgt_lang=str(record["tgt_lang"]),
                translation=str(record["translation"]),
                text_encoder=text_encoder,
                source_id=source_id,
            )
        )
    if "tts" in selected:
        if "source_bicodec" not in record:
            raise KeyError("TTS Phase 1 sample requires source_bicodec")
        samples.append(
            builders.build_tts_sample(
                bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
                src_lang=str(record["src_lang"]),
                transcription=str(record["transcription"]),
                source_bicodec=record["source_bicodec"],  # type: ignore[arg-type]
                text_encoder=text_encoder,
                source_id=source_id,
            )
        )
    if "mt" in selected:
        samples.append(
            builders.build_mt_sample(
                src_lang=str(record["src_lang"]),
                tgt_lang=str(record["tgt_lang"]),
                source_text=str(record["transcription"]),
                target_text=str(record["translation"]),
                text_encoder=text_encoder,
                source_id=source_id,
            )
        )
    return samples


def convert_records_to_samples(
    records: Iterable[Mapping[str, object]],
    text_encoder: builders.TextEncoder,
    tasks: Sequence[str] = SPEECH_TASKS,
) -> Iterator[dict[str, object]]:
    for record in records:
        for sample in build_task_samples(record, text_encoder=text_encoder, tasks=tasks):
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
    parser.add_argument("--input", nargs="+", required=True, help="UniST-style parquet files or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--tokenizer", required=True, help="HF/UniSS tokenizer path")
    parser.add_argument("--tasks", nargs="+", default=list(SPEECH_TASKS), choices=PHASE1_TASKS)
    parser.add_argument(
        "--include-mt-proxy",
        action="store_true",
        help="Also emit an MT proxy sample from UniST transcription -> translation.",
    )
    parser.add_argument("--limit-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_input_paths(args.input)
    text_encoder = load_hf_text_encoder(args.tokenizer)
    tasks = list(args.tasks)
    if args.include_mt_proxy and "mt" not in tasks:
        tasks.append("mt")
    records = iter_alignment_records(paths, limit_records=args.limit_records)
    samples = convert_records_to_samples(records, text_encoder=text_encoder, tasks=tasks)
    counts = write_jsonl(samples, Path(args.output))
    print(json.dumps({"output": args.output, "counts": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
