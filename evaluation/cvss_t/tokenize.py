"""Tokenize canonical CVSS-T pairs into UniSS-compatible direction parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import select_shard, validate_shard


GLOBAL_TOKEN_COUNT = 32


def coerce_tokens(value: object, *, field_name: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()  # type: ignore[union-attr]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a token list, got {type(value).__name__}")
    tokens = [int(token) for token in value]
    if not tokens:
        raise ValueError(f"{field_name} cannot be empty")
    return tokens


def split_bicodec(value: object, *, field_name: str) -> tuple[list[int], list[int]]:
    tokens = coerce_tokens(value, field_name=field_name)
    if len(tokens) <= GLOBAL_TOKEN_COUNT:
        raise ValueError(f"{field_name} must contain 32 global tokens and non-empty semantic tokens")
    return tokens[:GLOBAL_TOKEN_COUNT], tokens[GLOBAL_TOKEN_COUNT:]


def build_direction_records(
    pair: Mapping[str, object],
    *,
    pair_index: int,
    zh_glm: object,
    zh_bicodec: object,
    en_glm: object,
    en_bicodec: object,
    tokenizer_model: str,
) -> tuple[dict[str, object], dict[str, object]]:
    zh_glm_values = coerce_tokens(zh_glm, field_name="zh_glm")
    en_glm_values = coerce_tokens(en_glm, field_name="en_glm")
    zh_global, zh_semantic = split_bicodec(zh_bicodec, field_name="zh_bicodec")
    en_global, en_semantic = split_bicodec(en_bicodec, field_name="en_bicodec")
    sample_id = str(pair["id"])
    source_zh_path = str(pair["source_zh_audio_path"])
    target_en_path = str(pair["target_en_audio_path"])
    source_zh_text = str(pair["source_zh_text"])
    target_en_text = str(pair["target_en_text"])
    source_duration = float(pair["source_zh_duration_seconds"])
    target_duration = float(pair["target_en_duration_seconds"])
    if source_duration <= 0 or target_duration <= 0:
        raise ValueError(f"Invalid CVSS duration for {sample_id}")

    common = {
        "id": sample_id,
        "pair_id": sample_id,
        "pair_index": pair_index,
        "dataset_name": "CVSS-T",
        "split": "test",
        "tokenizer_model": tokenizer_model,
        "source_zh_audio_sha256": pair.get("source_zh_audio_sha256"),
        "target_en_audio_sha256": pair.get("target_en_audio_sha256"),
    }
    zh_en = {
        **common,
        "direction": "cmn->eng",
        "src_lang": "cmn",
        "tgt_lang": "eng",
        "transcription": source_zh_text,
        "translation": target_en_text,
        "source_glm": zh_glm_values,
        "source_bicodec": zh_semantic,
        "target_bicodec": en_semantic,
        "bicodec_global": zh_global,
        "source_audio_path": source_zh_path,
        "reference_audio_path": target_en_path,
        "source_audio_duration_seconds": source_duration,
        "reference_audio_duration_seconds": target_duration,
        "duration_ratio": target_duration / source_duration,
        "synthetic_source": False,
        "synthetic_reference": True,
    }
    en_zh = {
        **common,
        "direction": "eng->cmn",
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "transcription": target_en_text,
        "translation": source_zh_text,
        "source_glm": en_glm_values,
        "source_bicodec": en_semantic,
        "target_bicodec": zh_semantic,
        "bicodec_global": en_global,
        "source_audio_path": target_en_path,
        "reference_audio_path": source_zh_path,
        "source_audio_duration_seconds": target_duration,
        "reference_audio_duration_seconds": source_duration,
        "duration_ratio": source_duration / target_duration,
        "synthetic_source": True,
        "synthetic_reference": False,
    }
    return zh_en, en_zh


def write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite tokenized parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def tokenize_shard(args: argparse.Namespace) -> dict[str, object]:
    from uniss import UniSSTokenizer

    validate_shard(num_shards=args.num_shards, shard_index=args.shard_index)
    pair_manifest = Path(args.pair_manifest)
    output_dir = Path(args.output_dir)
    suffix = f"part_{args.shard_index:03d}-of-{args.num_shards:03d}.parquet"
    zh_en_path = output_dir / "parts" / "zh_en" / suffix
    en_zh_path = output_dir / "parts" / "en_zh" / suffix
    summary_path = output_dir / "parts" / "summaries" / suffix.replace(".parquet", ".json")
    if zh_en_path.exists() or en_zh_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite tokenization shard {args.shard_index}: {output_dir}")

    all_rows = list(iter_jsonl(pair_manifest))
    selected = list(
        select_shard(
            all_rows,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    )
    if args.limit_pairs > 0:
        selected = selected[: args.limit_pairs]
    tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=args.device)
    zh_en_rows: list[dict[str, object]] = []
    en_zh_rows: list[dict[str, object]] = []
    for local_index, pair in enumerate(selected):
        pair_index = args.shard_index + local_index * args.num_shards
        zh_glm, zh_bicodec = tokenizer.tokenize(str(pair["source_zh_audio_path"]))
        en_glm, en_bicodec = tokenizer.tokenize(str(pair["target_en_audio_path"]))
        zh_en, en_zh = build_direction_records(
            pair,
            pair_index=pair_index,
            zh_glm=zh_glm,
            zh_bicodec=zh_bicodec,
            en_glm=en_glm,
            en_bicodec=en_bicodec,
            tokenizer_model=str(Path(args.speech_tokenizer).resolve()),
        )
        zh_en_rows.append(zh_en)
        en_zh_rows.append(en_zh)

    write_parquet_atomic(zh_en_path, zh_en_rows)
    write_parquet_atomic(en_zh_path, en_zh_rows)
    summary = {
        "pair_manifest": str(pair_manifest.resolve()),
        "speech_tokenizer": str(Path(args.speech_tokenizer).resolve()),
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "pair_count": len(selected),
        "zh_en_path": str(zh_en_path.resolve()),
        "en_zh_path": str(en_zh_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speech-tokenizer", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(tokenize_shard(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
