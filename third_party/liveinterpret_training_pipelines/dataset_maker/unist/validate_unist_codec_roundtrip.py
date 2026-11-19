#!/usr/bin/env python3
"""GPU validation for stratified UniST BiCodec/Qwen3 conversion samples."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch


SAMPLE_RATE = 16_000
BICODEC_HOP = 320
QWEN_SAMPLE_RATE = 24_000
SOURCE_REPLAY_MIN_CORRELATION = 0.995
SOURCE_REPLAY_MAX_RMSE = 0.005
REQUIRED_RAW_COLUMNS = (
    "id", "transcription", "translation", "source_bicodec",
    "target_bicodec", "bicodec_global", "dataset_name", "src_lang",
    "tgt_lang", "split",
)


def parse_args() -> argparse.Namespace:
    workspace = Path("/opt/dlami/nvme/zeyangsong")
    repo = workspace / "projects/liveinterpret_training_pipelines"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=Path,
        default=workspace / "data/unist/validation/unist_qwen3_s2s_v2_20260730/roundtrip_samples.jsonl")
    parser.add_argument(
        "--report-dir", type=Path,
        default=workspace / "data/unist/validation/unist_qwen3_s2s_v2_20260730")
    parser.add_argument(
        "--output-root", type=Path,
        default=workspace / "data/unist/converted/unist_qwen3_s2s_v2")
    parser.add_argument("--spark-repo", type=Path, default=workspace / "projects/Spark-TTS")
    parser.add_argument(
        "--bicodec-model", type=Path,
        default=workspace / "projects/Spark-TTS-0.5B/BiCodec")
    parser.add_argument(
        "--qwen-tokenizer", type=Path,
        default=repo / "models/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def normalize_lang(value: Any) -> str:
    value = str(value or "").strip().lower().replace("_", "-")
    if value in {"eng", "english", "en"} or value.startswith("en-"):
        return "en"
    if value in {"cmn", "zho", "chinese", "zh"} or value.startswith("zh-"):
        return "zh"
    raise ValueError(f"unsupported language: {value!r}")


def safe_name(value: Any, max_length: int = 70) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value[:max_length].strip("_") or "sample"


def load_converter(repo: Path):
    path = repo / "dataset_maker/unist/convert_unist_bicodec_to_qwen3.py"
    spec = importlib.util.spec_from_file_location("unist_converter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audio_stats(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(wav).all())
    rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if wav.size and finite else math.nan
    return {
        "sample_rate": int(sample_rate),
        "num_samples": int(wav.size),
        "duration_seconds": float(wav.size / sample_rate),
        "finite": finite,
        "rms": rms,
        "peak": float(np.max(np.abs(wav))) if wav.size and finite else math.nan,
        "near_silence": bool(finite and rms < 1e-4),
        "clipped_fraction": float(np.mean(np.abs(wav) >= 0.999)) if wav.size and finite else math.nan,
    }


def waveform_comparison(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    length = min(len(reference), len(predicted))
    if length <= 0:
        return {"aligned_samples": 0, "rmse": math.nan, "correlation": math.nan}
    ref = np.asarray(reference[:length], dtype=np.float64)
    pred = np.asarray(predicted[:length], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.square(ref - pred))))
    if np.std(ref) == 0 or np.std(pred) == 0:
        correlation = math.nan
    else:
        correlation = float(np.corrcoef(ref, pred)[0, 1])
    return {"aligned_samples": length, "rmse": rmse, "correlation": correlation}


def token_comparison(reference: list[list[int]], predicted: list[list[int]]) -> dict[str, Any]:
    reference_shape = [len(reference), len(reference[0]) if reference else 0]
    predicted_shape = [len(predicted), len(predicted[0]) if predicted else 0]
    if reference_shape != predicted_shape or not reference:
        return {
            "reference_shape": reference_shape,
            "predicted_shape": predicted_shape,
            "matching_tokens": 0,
            "total_tokens": sum(len(frame) for frame in reference),
            "match_rate": 0.0,
        }
    total = sum(len(frame) for frame in reference)
    matching = sum(
        int(left == right)
        for reference_frame, predicted_frame in zip(reference, predicted)
        for left, right in zip(reference_frame, predicted_frame)
    )
    return {
        "reference_shape": reference_shape,
        "predicted_shape": predicted_shape,
        "matching_tokens": matching,
        "total_tokens": total,
        "match_rate": matching / total,
    }


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle]


def file_assignment(samples: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[str(row["source_parquet_path"])].append(row)
    assigned_files = {
        path for index, path in enumerate(sorted(grouped)) if index % world_size == rank
    }
    return [row for row in samples if str(row["source_parquet_path"]) in assigned_files]


def choose_listening_ids(samples: list[dict[str, Any]]) -> set[str]:
    chosen: list[dict[str, Any]] = []
    used_datasets: dict[str, set[str]] = defaultdict(set)
    bins = ("lt1s", "1to3s", "3to8s", "ge8s")

    def duration_bin(row: dict[str, Any]) -> str:
        duration = int(row["target_duration_ms"])
        if duration < 1_000:
            return "lt1s"
        if duration < 3_000:
            return "1to3s"
        if duration < 8_000:
            return "3to8s"
        return "ge8s"

    for direction in ("en-zh", "zh-en"):
        rows = [
            row for row in samples
            if f"{row['source_lang']}-{row['target_lang']}" == direction
        ]
        for wanted_bin in bins:
            candidates = [row for row in rows if duration_bin(row) == wanted_bin]
            candidates.sort(key=lambda row: (
                row["dataset_name"] in used_datasets[direction],
                row["selection_hash"],
            ))
            if candidates:
                choice = candidates[0]
                chosen.append(choice)
                used_datasets[direction].add(str(choice["dataset_name"]))
        remaining = sorted(rows, key=lambda row: (
            row["dataset_name"] in used_datasets[direction], row["selection_hash"]))
        for row in remaining:
            if len([x for x in chosen if f"{x['source_lang']}-{x['target_lang']}" == direction]) >= 5:
                break
            if row["record_id"] not in {x["record_id"] for x in chosen}:
                chosen.append(row)
                used_datasets[direction].add(str(row["dataset_name"]))
    return {row["record_id"] for row in chosen[:10]}


def read_selected_contexts(
    parquet_path: Path, candidates: list[dict[str, Any]], converter,
) -> dict[int, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Rebuild the converter's sorted valid 8192-row block for each sample."""
    table = pq.read_table(parquet_path, columns=list(REQUIRED_RAW_COLUMNS))
    contexts: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        source_index = int(candidate["provenance"]["source_row_index"])
        by_block[(source_index // 8192) * 8192].append(candidate)
    for block_start, block_candidates in by_block.items():
        block = table.slice(block_start, min(8192, len(table) - block_start)).to_pylist()
        for local_index, row in enumerate(block):
            row["_source_row_index"] = block_start + local_index
            row["_reject_reason"] = converter.validate_tokens(
                row, parquet_path, block_start + local_index)
        valid_block = [row for row in block if not row["_reject_reason"]]
        valid_block.sort(key=lambda row: (
            len(row["target_bicodec"]), len(row["source_bicodec"])))
        row_by_index = {int(row["_source_row_index"]): row for row in valid_block}
        for candidate in block_candidates:
            source_index = int(candidate["provenance"]["source_row_index"])
            if source_index not in row_by_index:
                raise ValueError(f"selected manifest row became rejected: {parquet_path}:{source_index}")
            contexts[source_index] = (row_by_index[source_index], valid_block)
    return contexts


@torch.inference_mode()
def decode_bicodec_exact_chunk(
    model, block: list[dict[str, Any]], source_index: int,
    field: str, device: torch.device, group_rows: int = 64,
) -> np.ndarray:
    """Decode the exact same-length chunk used by the production converter."""
    target_row = next(row for row in block if row["_source_row_index"] == source_index)
    token_length = len(target_row[field])
    items = [
        (candidate_field, row)
        for row in block
        for candidate_field in ("source_bicodec", "target_bicodec")
        if len(row[candidate_field]) == token_length
    ]
    target_position = next(
        index for index, (candidate_field, row) in enumerate(items)
        if candidate_field == field and row["_source_row_index"] == source_index)
    chunk_start = (target_position // group_rows) * group_rows
    chunk = items[chunk_start:chunk_start + group_rows]
    semantic = torch.as_tensor(
        [row[candidate_field] for candidate_field, row in chunk],
        dtype=torch.long, device=device)
    global_tokens = torch.as_tensor(
        [row["bicodec_global"] for _, row in chunk],
        dtype=torch.long, device=device).unsqueeze(1)
    decoded = model.detokenize(semantic, global_tokens).float()
    if decoded.ndim == 3 and decoded.shape[1] == 1:
        decoded = decoded[:, 0]
    local_index = target_position - chunk_start
    wav = decoded[local_index, : token_length * BICODEC_HOP]
    wav = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(wav).all():
        raise ValueError("BiCodec emitted non-finite audio")
    return np.clip(wav, -1, 1)


@torch.inference_mode()
def production_outputs_for_candidate(
    model, qwen, converter, block: list[dict[str, Any]],
    source_index: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Replay the complete production BiCodec block and selected Qwen batch."""
    source_arrays, target_arrays = converter.decode_bicodec_block_exact(
        model, block, device, 64)
    index_by_source_row = {
        int(row["_source_row_index"]): index for index, row in enumerate(block)
    }
    qwen_batch = next(
        batch for batch in converter.split_batches(block, 64, 512)
        if any(row["_source_row_index"] == source_index for row in batch))
    arrays = [
        target_arrays[index_by_source_row[int(row["_source_row_index"])]]
        for row in qwen_batch
    ]
    target_wavs, target_lengths = converter.pad_numpy_waveforms(arrays, device)
    batch_codes = converter.qwen_encode_batch_direct(qwen, target_wavs, target_lengths)
    candidate_position = next(
        index for index, row in enumerate(qwen_batch)
        if row["_source_row_index"] == source_index)
    block_position = index_by_source_row[source_index]
    return (
        source_arrays[block_position], target_arrays[block_position],
        batch_codes[candidate_position],
    )


def validate_one(
    candidate: dict[str, Any], raw: dict[str, Any], block: list[dict[str, Any]], bicodec, qwen,
    converter, device: torch.device, wds_root: Path, production_batch_check: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from custom.wds_audio import read_wds_audio_bytes

    errors = []
    if raw["id"] != candidate["id"]:
        errors.append("raw_id_mismatch")
    if str(raw["transcription"]).strip() != str(candidate["source_text"]).strip():
        errors.append("raw_source_text_mismatch")
    if str(raw["translation"]).strip() != str(candidate["target_text"]).strip():
        errors.append("raw_target_text_mismatch")
    if normalize_lang(raw["src_lang"]) != candidate["source_lang"]:
        errors.append("raw_source_lang_mismatch")
    if normalize_lang(raw["tgt_lang"]) != candidate["target_lang"]:
        errors.append("raw_target_lang_mismatch")
    if len(raw["source_bicodec"]) != candidate["provenance"]["source_bicodec_len"]:
        errors.append("raw_source_bicodec_len_mismatch")
    if len(raw["target_bicodec"]) != candidate["provenance"]["target_bicodec_len"]:
        errors.append("raw_target_bicodec_len_mismatch")
    if len(raw["bicodec_global"]) != 32:
        errors.append("raw_global_len")

    source_index = int(candidate["provenance"]["source_row_index"])
    production_codes = None
    if production_batch_check:
        source_bicodec_wav, target_bicodec_wav, production_codes = (
            production_outputs_for_candidate(
                bicodec, qwen, converter, block, source_index, device))
    else:
        source_bicodec_wav = decode_bicodec_exact_chunk(
            bicodec, block, source_index, "source_bicodec", device)
        target_bicodec_wav = decode_bicodec_exact_chunk(
            bicodec, block, source_index, "target_bicodec", device)

    payload = read_wds_audio_bytes(candidate["audios"][0], root=str(wds_root))
    source_wds_wav, source_wds_sr = sf.read(io.BytesIO(payload), dtype="float32")
    source_wds_wav = np.asarray(source_wds_wav, dtype=np.float32).reshape(-1)
    if source_wds_sr != SAMPLE_RATE:
        errors.append("source_wds_sample_rate")
    if len(source_wds_wav) != len(raw["source_bicodec"]) * BICODEC_HOP:
        errors.append("source_wds_duration")
    source_comparison = waveform_comparison(source_wds_wav, source_bicodec_wav)
    if (not math.isfinite(source_comparison["rmse"])
            or not math.isfinite(source_comparison["correlation"])
            or source_comparison["rmse"] > SOURCE_REPLAY_MAX_RMSE
            or source_comparison["correlation"] < SOURCE_REPLAY_MIN_CORRELATION):
        errors.append("source_wds_bicodec_mismatch")

    target_tensor = torch.from_numpy(target_bicodec_wav).to(device)[None, :]
    direct_codes = converter.qwen_encode_batch_direct(
        qwen, target_tensor, [len(target_bicodec_wav)])[0]
    wrapper_codes = converter.qwen_encode_batch(
        qwen, target_tensor, [len(target_bicodec_wav)])[0]
    manifest_codes = candidate["target_codec_codes"]
    individual_direct_matches_manifest = direct_codes == manifest_codes
    individual_wrapper_matches_manifest = wrapper_codes == manifest_codes
    direct_matches_wrapper = direct_codes == wrapper_codes
    if not direct_matches_wrapper:
        errors.append("direct_wrapper_mismatch")
    production_batch_matches_manifest = None
    production_manifest_token_comparison = None
    if production_batch_check:
        assert production_codes is not None
        production_batch_matches_manifest = production_codes == manifest_codes
        production_manifest_token_comparison = token_comparison(
            manifest_codes, production_codes)

    decoded_wavs, decoded_sr = qwen.decode({
        "audio_codes": [torch.tensor(manifest_codes, dtype=torch.long)]
    })
    qwen_wav = np.asarray(decoded_wavs[0], dtype=np.float32).reshape(-1)
    if not np.isfinite(qwen_wav).all() or not qwen_wav.size:
        errors.append("qwen_decode_invalid")
    duration_delta = abs(len(qwen_wav) / decoded_sr - len(target_bicodec_wav) / SAMPLE_RATE)
    if duration_delta > 0.12:
        errors.append("qwen_duration_delta")

    # Compare at 16 kHz for a diagnostic waveform metric. Listening remains the
    # primary perceptual check because codec reconstruction is not phase-exact.
    qwen_16k = converter.AF.resample(
        torch.from_numpy(qwen_wav).to(device)[None, :], decoded_sr, SAMPLE_RATE
    ).squeeze(0).cpu().numpy()
    result = {
        "sample_index": candidate["sample_index"],
        "record_id": candidate["record_id"],
        "id": candidate["id"],
        "dataset_name": candidate["dataset_name"],
        "split": candidate["split"],
        "direction": f"{candidate['source_lang']}-{candidate['target_lang']}",
        "source_text": candidate["source_text"],
        "target_text": candidate["target_text"],
        "source_parquet_path": candidate["source_parquet_path"],
        "source_row_index": candidate["provenance"]["source_row_index"],
        "source_bicodec_tokens": len(raw["source_bicodec"]),
        "target_bicodec_tokens": len(raw["target_bicodec"]),
        "qwen_codec_shape": [len(manifest_codes), len(manifest_codes[0])],
        "individual_direct_matches_manifest": individual_direct_matches_manifest,
        "individual_wrapper_matches_manifest": individual_wrapper_matches_manifest,
        "individual_manifest_token_comparison": token_comparison(
            manifest_codes, direct_codes),
        "direct_matches_wrapper": direct_matches_wrapper,
        "production_batch_checked": production_batch_check,
        "production_batch_matches_manifest": production_batch_matches_manifest,
        "production_manifest_token_comparison": production_manifest_token_comparison,
        "source_wds_audio": audio_stats(source_wds_wav, source_wds_sr),
        "source_bicodec_audio": audio_stats(source_bicodec_wav, SAMPLE_RATE),
        "source_wds_vs_bicodec": source_comparison,
        "target_bicodec_audio": audio_stats(target_bicodec_wav, SAMPLE_RATE),
        "target_qwen_roundtrip_audio": audio_stats(qwen_wav, decoded_sr),
        "target_roundtrip_duration_delta_seconds": duration_delta,
        "target_roundtrip_comparison_16k": waveform_comparison(target_bicodec_wav, qwen_16k),
        "errors": errors,
    }
    audios = {
        "source_wds": source_wds_wav,
        "target_bicodec": target_bicodec_wav,
        "target_qwen_roundtrip": qwen_wav,
    }
    return result, audios


def finalize(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    gpu_dir = args.report_dir / "gpu"
    paths = sorted(gpu_dir.glob("rank*.jsonl"))
    results = []
    for path in paths:
        with path.open("rb") as handle:
            results.extend(orjson.loads(line) for line in handle)
    results.sort(key=lambda row: row["sample_index"])
    # Rank files are append-only evidence. Reclassify the source replay check
    # here so a report can be finalized with the thresholds embedded in this
    # validator even when an earlier rank run used a stricter diagnostic cutoff.
    for row in results:
        comparison = row["source_wds_vs_bicodec"]
        source_replay_failed = (
            not math.isfinite(comparison["rmse"])
            or not math.isfinite(comparison["correlation"])
            or comparison["rmse"] > SOURCE_REPLAY_MAX_RMSE
            or comparison["correlation"] < SOURCE_REPLAY_MIN_CORRELATION
        )
        row["errors"] = [
            error for error in row["errors"]
            if error != "source_wds_bicodec_mismatch"
        ]
        if source_replay_failed:
            row["errors"].append("source_wds_bicodec_mismatch")
    error_counts: Counter[str] = Counter(
        error for row in results for error in row["errors"])
    summary = {
        "schema_version": "unist-codec-roundtrip-validation-v1",
        "samples_expected": len(samples),
        "samples_validated": len(results),
        "directions": dict(Counter(row["direction"] for row in results)),
        "datasets": dict(sorted(Counter(row["dataset_name"] for row in results).items())),
        "individual_direct_matches_manifest": sum(
            row["individual_direct_matches_manifest"] for row in results),
        "individual_wrapper_matches_manifest": sum(
            row["individual_wrapper_matches_manifest"] for row in results),
        "direct_matches_wrapper": sum(row["direct_matches_wrapper"] for row in results),
        "production_batch_checks": sum(row["production_batch_checked"] for row in results),
        "production_batch_matches_manifest": sum(
            row["production_batch_matches_manifest"] is True for row in results),
        "individual_manifest_token_match_rate_mean": float(np.mean([
            row["individual_manifest_token_comparison"]["match_rate"] for row in results
        ])) if results else None,
        "individual_manifest_token_match_rate_min": min((
            row["individual_manifest_token_comparison"]["match_rate"] for row in results
        ), default=None),
        "production_manifest_token_match_rate_mean": float(np.mean([
            row["production_manifest_token_comparison"]["match_rate"]
            for row in results if row["production_manifest_token_comparison"] is not None
        ])) if any(row["production_manifest_token_comparison"] is not None for row in results) else None,
        "production_manifest_token_match_rate_min": min((
            row["production_manifest_token_comparison"]["match_rate"]
            for row in results if row["production_manifest_token_comparison"] is not None
        ), default=None),
        "source_wds_near_silence": sum(row["source_wds_audio"]["near_silence"] for row in results),
        "target_bicodec_near_silence": sum(row["target_bicodec_audio"]["near_silence"] for row in results),
        "target_qwen_near_silence": sum(row["target_qwen_roundtrip_audio"]["near_silence"] for row in results),
        "max_roundtrip_duration_delta_seconds": max(
            (row["target_roundtrip_duration_delta_seconds"] for row in results), default=None),
        "target_roundtrip_correlation_mean": float(np.nanmean([
            row["target_roundtrip_comparison_16k"]["correlation"] for row in results
        ])) if results else None,
        "target_roundtrip_correlation_min": float(np.nanmin([
            row["target_roundtrip_comparison_16k"]["correlation"] for row in results
        ])) if results else None,
        "source_wds_vs_bicodec_rmse_max": max(
            (row["source_wds_vs_bicodec"]["rmse"] for row in results), default=None),
        "source_wds_vs_bicodec_correlation_min": min(
            (row["source_wds_vs_bicodec"]["correlation"] for row in results), default=None),
        "source_replay_thresholds": {
            "rmse_max": SOURCE_REPLAY_MAX_RMSE,
            "correlation_min": SOURCE_REPLAY_MIN_CORRELATION,
        },
        "error_counts": dict(error_counts),
        "passed": len(results) == len(samples) and not error_counts,
    }
    (args.report_dir / "gpu_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    listening_root = args.report_dir / "listening_10"
    listening_records = []
    for metadata_path in sorted(listening_root.glob("*/metadata.json")):
        listening_records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    lines = [
        "# UniST 转换验证：10 条人工试听",
        "",
        "每个子目录包含 `source_wds.wav`、`target_bicodec.wav` 和",
        "`target_qwen_roundtrip.wav`。前两路分别验证正式 WDS source 与原始 target",
        "BiCodec，第三路验证 manifest Qwen3 codec 的解码效果。",
        "",
        "| # | direction | dataset | source text | target text | source s | target s | qwen s |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(listening_records, key=lambda item: item["listening_index"]):
        source = str(row["source_text"]).replace("|", "\\|").replace("\n", " ")
        target = str(row["target_text"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['listening_index']:02d} | {row['direction']} | {row['dataset_name']} | "
            f"{source} | {target} | {row['source_wds_audio']['duration_seconds']:.2f} | "
            f"{row['target_bicodec_audio']['duration_seconds']:.2f} | "
            f"{row['target_qwen_roundtrip_audio']['duration_seconds']:.2f} |"
        )
    listening_root.mkdir(parents=True, exist_ok=True)
    (listening_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    samples = load_samples(args.samples)
    if args.finalize_only:
        finalize(args, samples)
        return
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"invalid rank/world-size: {args.rank}/{args.world_size}")
    assigned = file_assignment(samples, args.rank, args.world_size)
    if args.max_samples:
        assigned = assigned[:args.max_samples]
    listening_ids = choose_listening_ids(samples)
    listening_order = {
        record_id: index for index, record_id in enumerate(sorted(listening_ids), 1)
    }

    sys.path.insert(0, str(args.spark_repo.resolve()))
    from sparktts.models.bicodec import BiCodec
    from qwen_tts import Qwen3TTSTokenizer

    device = torch.device(args.device)
    bicodec = BiCodec.load_from_checkpoint(args.bicodec_model.resolve()).to(device).eval()
    qwen = Qwen3TTSTokenizer.from_pretrained(
        str(args.qwen_tokenizer.resolve()), device_map=str(device))
    converter = load_converter(repo)
    wds_root = args.output_root / "speech"
    gpu_dir = args.report_dir / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    output_path = gpu_dir / f"rank{args.rank:02d}.jsonl"
    listening_root = args.report_dir / "listening_10"
    started = time.time()
    processed = 0

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assigned:
        by_file[str(row["source_parquet_path"])].append(row)
    with output_path.open("wb") as output_handle:
        for parquet_name, candidates in sorted(by_file.items()):
            selected_contexts = read_selected_contexts(
                Path(parquet_name), candidates, converter)
            for candidate in candidates:
                source_index = int(candidate["provenance"]["source_row_index"])
                raw, block = selected_contexts[source_index]
                result, audios = validate_one(
                    candidate, raw, block, bicodec, qwen,
                    converter, device, wds_root,
                    candidate["record_id"] in listening_ids)
                output_handle.write(orjson.dumps(result, option=orjson.OPT_APPEND_NEWLINE))
                output_handle.flush()
                processed += 1
                if candidate["record_id"] in listening_ids:
                    listen_index = listening_order[candidate["record_id"]]
                    sample_dir = listening_root / (
                        f"{listen_index:02d}_{result['direction']}_"
                        f"{safe_name(result['dataset_name'])}_{safe_name(result['id'])}")
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(sample_dir / "source_wds.wav", audios["source_wds"], SAMPLE_RATE, subtype="PCM_16")
                    sf.write(sample_dir / "target_bicodec.wav", audios["target_bicodec"], SAMPLE_RATE, subtype="PCM_16")
                    sf.write(
                        sample_dir / "target_qwen_roundtrip.wav",
                        audios["target_qwen_roundtrip"], QWEN_SAMPLE_RATE, subtype="PCM_16")
                    metadata = dict(result)
                    metadata["listening_index"] = listen_index
                    (sample_dir / "metadata.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[rank{args.rank}] {processed}/{len(assigned)} {result['record_id']} "
                    f"errors={result['errors']}", flush=True)
            del selected_contexts
            gc.collect()
    print(
        f"[rank{args.rank}] COMPLETE samples={processed} elapsed={time.time()-started:.1f}s",
        flush=True)


if __name__ == "__main__":
    main()
