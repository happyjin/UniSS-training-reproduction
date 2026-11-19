#!/usr/bin/env python3
"""Build a non-destructive, quality-filtered S2S training manifest.

The current ms-swift JSONL export intentionally contains every curated row but
does not retain its UTMOS/ECAPA columns.  This pipeline first builds a reusable
SQLite score index from the curated parquet files, joins scores by
``manifest.record_id == curated.id``, then applies the confirmed 01--05 data
quality rules and the previous near-30-second generation-cap rule.

Accepted JSON lines are copied byte-for-byte into a new manifest directory.
Rejected rows are represented by small diagnostic records; the original JSONL
and WDS tar shards are never modified or deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson
import pyarrow.parquet as pq
import soundfile as sf


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom.wds_audio import is_wds_uri, read_wds_audio_bytes  # noqa: E402


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "s2s_cleaning_20260728_v1.json"
OUTPUT_SCHEMA = "liveinterpret.s2s_cleaning_output.v1"
QUALITY_INDEX_SCHEMA = "liveinterpret.s2s_quality_index.v1"


@dataclass(frozen=True)
class QualityScores:
    utmos: float | None
    ecapa: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--curated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-index", required=True,
                        help="Reusable SQLite path built from curated parquet scores")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--wds-audio-root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lookup-batch-size", type=int, default=512)
    parser.add_argument(
        "--audio-tail-mode", choices=("all", "off"), default="all",
        help="all decodes every row surviving quality/static filters; off is for smoke tests only")
    parser.add_argument("--max-rows-per-part", type=int, default=0,
                        help="Debug/sample run; zero processes every row")
    parser.add_argument("--rebuild-quality-index", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "liveinterpret.s2s_cleaning_config.v1":
        raise ValueError(f"Unsupported cleaning config schema: {config.get('schema_version')!r}")
    policy = config["quality"].get("missing_score_policy")
    if policy not in {"reject", "error", "keep"}:
        raise ValueError("quality.missing_score_policy must be reject, error, or keep")
    return config


def _source_signature(curated_dir: Path) -> tuple[str, list[Path]]:
    parts = sorted(curated_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet under {curated_dir}")
    digest = hashlib.sha256()
    for path in parts:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest(), parts


def _read_index_metadata(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT key, value FROM metadata"))
    finally:
        connection.close()


def build_quality_index(curated_dir: Path, index_path: Path, *, rebuild: bool = False) -> int:
    """Create/validate an atomic reusable ``record_id -> scores`` SQLite index."""
    signature, parts = _source_signature(curated_dir)
    if index_path.exists() and not rebuild:
        metadata = _read_index_metadata(index_path)
        if metadata.get("schema_version") != QUALITY_INDEX_SCHEMA:
            raise RuntimeError(f"Quality index has an incompatible schema: {index_path}")
        if metadata.get("source_signature") != signature:
            raise RuntimeError(
                f"Quality index does not match curated parquet files: {index_path}; "
                "use --rebuild-quality-index")
        return int(metadata["record_count"])

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    total = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE quality (record_id TEXT PRIMARY KEY, utmos REAL, ecapa REAL) WITHOUT ROWID")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID")
        for part_number, path in enumerate(parts, start=1):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=100_000,
                columns=["id", "target_audio_utmos_score", "target_audio_ecapa_similarity"],
            ):
                values = batch.to_pydict()
                rows = list(zip(
                    values["id"], values["target_audio_utmos_score"],
                    values["target_audio_ecapa_similarity"], strict=True))
                connection.executemany(
                    "INSERT INTO quality(record_id, utmos, ecapa) VALUES (?, ?, ?)", rows)
                total += len(rows)
            connection.commit()
            print(f"[quality-index] {part_number}/{len(parts)} {path.name}: total={total}", flush=True)
        metadata = {
            "schema_version": QUALITY_INDEX_SCHEMA,
            "source_dir": str(curated_dir.resolve()),
            "source_signature": signature,
            "record_count": str(total),
            "created_at": _utc_now(),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    os.replace(temporary, index_path)
    return total


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _lookup_scores(connection: sqlite3.Connection, ids: list[str]) -> dict[str, QualityScores]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"SELECT record_id, utmos, ecapa FROM quality WHERE record_id IN ({placeholders})", ids)
    return {
        str(record_id): QualityScores(_finite_float(utmos), _finite_float(ecapa))
        for record_id, utmos, ecapa in rows
    }


def _text_units(text: str, target_lang: str) -> tuple[int, str]:
    if target_lang.lower().startswith("zh"):
        return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)), "cjk_chars"
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)), "words"


def _modal_codec_metrics(raw_codes: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(raw_codes, list) or not raw_codes:
        return {"codec_frames": 0, "codec_groups": 0}, "invalid_target_codec_shape"
    first = raw_codes[0]
    if not isinstance(first, list) or not first:
        return {"codec_frames": len(raw_codes), "codec_groups": 0}, "invalid_target_codec_shape"
    frames, groups = len(raw_codes), len(first)
    if any(not isinstance(frame, list) or len(frame) != groups for frame in raw_codes):
        return {"codec_frames": frames, "codec_groups": groups}, "invalid_target_codec_shape"
    metrics: dict[str, Any] = {"codec_frames": int(frames), "codec_groups": int(groups)}
    # The confirmed collapse rule only applies at >=100 frames.  Avoid 16
    # np.unique calls for each of the >1M ordinary short clips.
    if frames < 100:
        return metrics, None
    try:
        codes = np.asarray(raw_codes, dtype=np.int32)
    except (TypeError, ValueError) as exc:
        return metrics, f"codec_array_error:{type(exc).__name__}"
    if frames > 1:
        metrics["main_repeat_ratio"] = float(np.mean(codes[1:, 0] == codes[:-1, 0]))
        metrics["full_frame_repeat_ratio"] = float(
            np.mean(np.all(codes[1:] == codes[:-1], axis=1)))
    else:
        metrics["main_repeat_ratio"] = 0.0
        metrics["full_frame_repeat_ratio"] = 0.0
    metrics["residual_unique_mean"] = (
        float(np.mean([np.unique(codes[:, group]).size for group in range(1, groups)]))
        if groups > 1 else 0.0)
    return metrics, None


def _quality_reasons(
    scores: QualityScores | None, config: dict[str, Any], record_id: str,
) -> tuple[list[str], dict[str, Any]]:
    quality = config["quality"]
    metrics = {
        "target_audio_utmos_score": scores.utmos if scores else None,
        "target_audio_ecapa_similarity": scores.ecapa if scores else None,
    }
    missing = scores is None or scores.utmos is None or scores.ecapa is None
    if missing:
        policy = quality["missing_score_policy"]
        if policy == "error":
            raise RuntimeError(f"Missing MOS/ECAPA score for {record_id}")
        return (["missing_mos_or_ecapa"] if policy == "reject" else []), metrics
    reasons = []
    if scores.utmos < float(quality["min_utmos"]):
        reasons.append("low_utmos")
    if scores.ecapa < float(quality["min_ecapa_similarity"]):
        reasons.append("low_ecapa_similarity")
    return reasons, metrics


def evaluate_static_rules(row: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Evaluate structural plus confirmed non-decoding rules for one row."""
    reasons: list[str] = []
    structural = config["structural"]
    source_duration = int(row.get("source_duration_ms") or 0)
    target_duration = int(row.get("target_duration_ms") or 0)
    metrics, codec_error = _modal_codec_metrics(row.get("target_codec_codes") or [])
    metrics.update({
        "source_duration_ms": source_duration,
        "target_duration_ms": target_duration,
        "target_source_duration_ratio": (
            target_duration / source_duration if source_duration > 0 else None),
    })
    if source_duration < int(structural["min_source_duration_ms"]):
        reasons.append("source_too_short")
    if target_duration < int(structural["min_target_duration_ms"]):
        reasons.append("target_too_short")
    if codec_error:
        reasons.append(codec_error)
        return reasons, metrics
    frames = int(metrics["codec_frames"])
    groups = int(metrics["codec_groups"])
    if groups != int(structural["expected_codec_groups"]):
        reasons.append("unexpected_codec_group_count")
    expected_frames = math.ceil(target_duration / float(structural["codec_frame_ms"]))
    if abs(frames - expected_frames) > int(structural["codec_duration_tolerance_frames"]):
        reasons.append("codec_duration_mismatch")

    rules = config["rules"]
    collapse = rules["01_shorter_silence_collapse"]
    if (
        collapse["enabled"]
        and frames >= int(collapse["min_codec_frames"])
        and metrics.get("main_repeat_ratio", 0.0) >= float(collapse["min_main_repeat_ratio"])
        and metrics["full_frame_repeat_ratio"] >= float(collapse["min_full_frame_repeat_ratio"])
        and metrics["residual_unique_mean"] <= float(collapse["max_residual_unique_mean"])
    ):
        reasons.append("01_shorter_silence_collapse")

    text = str(row.get("target_text") or "")
    target_lang = str(row.get("target_lang") or "")
    units, unit_name = _text_units(text, target_lang)
    metrics.update({"target_text_units": units, "target_text_unit_name": unit_name})
    short = rules["03_very_short_text_long_audio"]
    short_limit = (
        int(short["max_chinese_characters"]) if target_lang.lower().startswith("zh")
        else int(short["max_english_words"]))
    if (
        short["enabled"] and units <= short_limit
        and target_duration >= int(short["min_target_duration_ms"])
    ):
        reasons.append("03_very_short_text_long_audio")

    ratio = metrics["target_source_duration_ratio"]
    longer = rules["04_target_much_longer_than_source"]
    shorter = rules["05_target_much_shorter_than_source"]
    if ratio is not None and longer["enabled"] and ratio >= float(
        longer["min_target_source_duration_ratio"]
    ):
        reasons.append("04_target_much_longer_than_source")
    if ratio is not None and shorter["enabled"] and ratio <= float(
        shorter["max_target_source_duration_ratio"]
    ):
        reasons.append("05_target_much_shorter_than_source")

    cap = rules["previous_30s_generation_cap"]
    if cap["enabled"] and (
        target_duration >= int(cap["min_target_duration_ms"])
        or frames >= int(cap["min_codec_frames"])
    ):
        reasons.append("previous_30s_generation_cap")
    return reasons, metrics


def _decode_audio(uri: str, wds_audio_root: str | None) -> tuple[np.ndarray, int]:
    source: Any = uri
    if is_wds_uri(uri):
        if not wds_audio_root:
            raise ValueError("--wds-audio-root is required to check wds:// target audio")
        source = io.BytesIO(read_wds_audio_bytes(uri, root=wds_audio_root))
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform.reshape(-1), int(sample_rate)


def evaluate_audio_rules(
    row: dict[str, Any], config: dict[str, Any], wds_audio_root: str | None,
) -> tuple[list[str], dict[str, Any]]:
    rule = config["rules"]["02_long_trailing_silence"]
    if not rule["enabled"]:
        return [], {}
    waveform, sample_rate = _decode_audio(str(row.get("target_audio") or ""), wds_audio_root)
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError("decoded target audio is empty")
    frame_hz = int(rule["frame_hz"])
    frame_width = max(1, round(sample_rate / frame_hz))
    rms = []
    for begin in range(0, waveform.size, frame_width):
        frame = waveform[begin:begin + frame_width].astype(np.float64)
        rms.append(float(np.sqrt(np.mean(frame * frame))))
    low = np.asarray(rms) < float(rule["low_energy_rms"])
    trailing_frames = 0
    for value in low[::-1]:
        if not value:
            break
        trailing_frames += 1
    trailing_seconds = trailing_frames / frame_hz
    low_fraction = float(low.mean())
    metrics = {
        "decoded_sample_rate": sample_rate,
        "decoded_duration_seconds": waveform.size / sample_rate,
        "decoded_audio_rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        "decoded_low_energy_fraction": low_fraction,
        "decoded_trailing_low_energy_seconds": trailing_seconds,
    }
    reasons = []
    if low_fraction >= float(rule["near_silence_fraction"]):
        reasons.append("decoded_near_silence")
    if trailing_seconds >= float(rule["min_trailing_silence_seconds"]):
        reasons.append("02_long_trailing_silence")
    return reasons, metrics


def _chunks(handle: Iterable[bytes], size: int, max_rows: int) -> Iterable[list[tuple[int, bytes]]]:
    chunk: list[tuple[int, bytes]] = []
    for line_number, line in enumerate(handle, start=1):
        if max_rows and line_number > max_rows:
            break
        chunk.append((line_number, line))
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _reject_record(
    row: dict[str, Any], part: str, line_number: int, reasons: list[str], metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "manifest_part": part,
        "manifest_line": line_number,
        "primary_reason": reasons[0],
        "reasons": reasons,
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
        "source_audio": (row.get("audios") or [None])[0],
        "target_audio": row.get("target_audio"),
        "target_text": row.get("target_text"),
        **metrics,
    }


def _clean_part(job: tuple[str, str, str, str, dict[str, Any], str, int, int]) -> dict[str, Any]:
    (part_raw, output_raw, index_raw, audio_mode, config, wds_root,
     lookup_batch_size, max_rows) = job
    part = Path(part_raw)
    output_dir = Path(output_raw)
    accepted_path = output_dir / part.name
    rejected_path = output_dir / "rejects" / part.name.replace("manifest-", "reject-")
    accepted_tmp = accepted_path.with_suffix(accepted_path.suffix + f".tmp.{os.getpid()}")
    rejected_tmp = rejected_path.with_suffix(rejected_path.suffix + f".tmp.{os.getpid()}")
    connection = sqlite3.connect(f"file:{index_raw}?mode=ro&immutable=1", uri=True)
    counts: Counter[str] = Counter()
    all_reason_counts: Counter[str] = Counter()
    direction_accepted: Counter[str] = Counter()
    direction_rejected: Counter[str] = Counter()
    try:
        with part.open("rb") as source, accepted_tmp.open("wb") as accepted, rejected_tmp.open("wb") as rejected:
            for chunk in _chunks(source, lookup_batch_size, max_rows):
                parsed = [(line_number, line, orjson.loads(line)) for line_number, line in chunk]
                ids = [str(row.get("record_id") or "") for _, _, row in parsed]
                score_map = _lookup_scores(connection, ids)
                for line_number, raw_line, row in parsed:
                    record_id = str(row.get("record_id") or "")
                    direction = f"{row.get('source_lang')}->{row.get('target_lang')}"
                    reasons, metrics = _quality_reasons(score_map.get(record_id), config, record_id)
                    if not reasons:
                        static_reasons, static_metrics = evaluate_static_rules(row, config)
                        reasons.extend(static_reasons)
                        metrics.update(static_metrics)
                    if not reasons and audio_mode == "all":
                        try:
                            audio_reasons, audio_metrics = evaluate_audio_rules(row, config, wds_root or None)
                            reasons.extend(audio_reasons)
                            metrics.update(audio_metrics)
                        except Exception as exc:
                            reasons.append("target_audio_decode_error")
                            metrics["target_audio_decode_error"] = f"{type(exc).__name__}: {exc}"

                    counts["processed"] += 1
                    if reasons:
                        counts["rejected"] += 1
                        counts[f"primary:{reasons[0]}"] += 1
                        direction_rejected[direction] += 1
                        all_reason_counts.update(set(reasons))
                        rejected.write(orjson.dumps(
                            _reject_record(row, part.name, line_number, reasons, metrics),
                            option=orjson.OPT_APPEND_NEWLINE))
                    else:
                        counts["accepted"] += 1
                        direction_accepted[direction] += 1
                        accepted.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        os.replace(accepted_tmp, accepted_path)
        os.replace(rejected_tmp, rejected_path)
    except Exception:
        accepted_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    if counts["accepted"] + counts["rejected"] != counts["processed"]:
        raise AssertionError(f"Count invariant failed for {part.name}: {counts}")
    return {
        "part": part.name,
        "counts": dict(counts),
        "all_reason_counts": dict(all_reason_counts),
        "direction_accepted": dict(direction_accepted),
        "direction_rejected": dict(direction_rejected),
        "accepted_bytes": accepted_path.stat().st_size,
        "rejected_bytes": rejected_path.stat().st_size,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    all_reasons: Counter[str] = Counter()
    accepted_directions: Counter[str] = Counter()
    rejected_directions: Counter[str] = Counter()
    for result in results:
        totals.update(result["counts"])
        all_reasons.update(result["all_reason_counts"])
        accepted_directions.update(result["direction_accepted"])
        rejected_directions.update(result["direction_rejected"])
    return {
        "counts": dict(sorted(totals.items())),
        "all_reason_counts": dict(sorted(all_reasons.items())),
        "accepted_direction_counts": dict(sorted(accepted_directions.items())),
        "rejected_direction_counts": dict(sorted(rejected_directions.items())),
        "accepted_bytes": sum(x["accepted_bytes"] for x in results),
        "rejected_bytes": sum(x["rejected_bytes"] for x in results),
    }


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir).resolve()
    curated_dir = Path(args.curated_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    quality_index = Path(args.quality_index).resolve()
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    if output_dir == manifest_dir:
        raise ValueError("Output directory must differ from the source manifest directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    parts = sorted(manifest_dir.glob("manifest-*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"No manifest-*.jsonl under {manifest_dir}")
    source_commit_path = manifest_dir / "_COMMITTED.json"
    if not source_commit_path.is_file():
        raise FileNotFoundError(f"Source manifest commit marker is missing: {source_commit_path}")
    source_commit = json.loads(source_commit_path.read_text(encoding="utf-8"))
    expected_source_records = int(source_commit["record_count"])
    expected_source_parts = int(source_commit["part_count"])
    if len(parts) != expected_source_parts:
        raise RuntimeError(
            f"Source part count mismatch: marker={expected_source_parts}, files={len(parts)}")
    if args.audio_tail_mode == "all" and not args.wds_audio_root:
        raise ValueError("--wds-audio-root is required with --audio-tail-mode=all")

    quality_count = build_quality_index(
        curated_dir, quality_index, rebuild=args.rebuild_quality_index)
    staging_dir = output_dir.with_name(output_dir.name + f".building.{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    (staging_dir / "rejects").mkdir()
    (staging_dir / "part_summaries").mkdir()
    shutil.copy2(config_path, staging_dir / "cleaning_config.json")
    jobs = [(
        str(part), str(staging_dir), str(quality_index), args.audio_tail_mode,
        config, str(Path(args.wds_audio_root).resolve()) if args.wds_audio_root else "",
        args.lookup_batch_size, args.max_rows_per_part,
    ) for part in parts]
    results = []
    with mp.Pool(processes=max(1, min(args.workers, len(jobs)))) as pool:
        for result in pool.imap_unordered(_clean_part, jobs):
            results.append(result)
            (staging_dir / "part_summaries" / result["part"].replace(".jsonl", ".json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"[clean] {result['part']}: accepted={result['counts'].get('accepted', 0)} "
                f"rejected={result['counts'].get('rejected', 0)}",
                flush=True)
    results.sort(key=lambda item: item["part"])
    aggregate = _aggregate(results)
    sampled = bool(args.max_rows_per_part)
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "sampled" if sampled else "committed",
        "created_at": _utc_now(),
        "source_manifest_dir": str(manifest_dir),
        "curated_dir": str(curated_dir),
        "quality_index": str(quality_index),
        "quality_index_records": quality_count,
        "config": str(config_path),
        "audio_tail_mode": args.audio_tail_mode,
        "wds_audio_root": str(Path(args.wds_audio_root).resolve()) if args.wds_audio_root else None,
        "part_count": len(results),
        "max_rows_per_part": args.max_rows_per_part,
        **aggregate,
    }
    if summary["counts"].get("processed", 0) != (
        summary["counts"].get("accepted", 0) + summary["counts"].get("rejected", 0)
    ):
        raise AssertionError("Global accepted + rejected != processed")
    if not sampled and summary["counts"].get("processed", 0) != expected_source_records:
        raise AssertionError(
            f"Processed {summary['counts'].get('processed', 0)} rows but source marker "
            f"declares {expected_source_records}")
    summary.update({
        "manifest_version": output_dir.name,
        "record_count": summary["counts"].get("accepted", 0),
        "source_record_count": summary["counts"].get("processed", 0),
        "total_bytes": summary["accepted_bytes"],
    })
    summary_path = staging_dir / ("_SAMPLE_RUN.json" if sampled else "_COMMITTED.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (staging_dir / "_REJECT_SUMMARY.json").write_text(
        json.dumps({
            "schema_version": OUTPUT_SCHEMA,
            "primary_reason_counts": {
                key.removeprefix("primary:"): value
                for key, value in summary["counts"].items() if key.startswith("primary:")
            },
            "all_reason_counts": summary["all_reason_counts"],
            "rejected_direction_counts": summary["rejected_direction_counts"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(staging_dir, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
