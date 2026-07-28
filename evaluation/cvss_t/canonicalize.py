"""Create immutable 16 kHz mono PCM16 CVSS-T evaluation waveforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from evaluation.io_utils import iter_jsonl, write_json, write_jsonl


CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SUBTYPE = "PCM_16"


@dataclass(frozen=True)
class CanonicalTask:
    index: int
    sample_id: str
    source_path: str
    target_path: str
    source_output: str
    target_output: str
    source_text: str
    target_text: str
    resume: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_metadata(path: Path) -> dict[str, object]:
    info = sf.info(path)
    if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
        raise ValueError(f"Invalid audio metadata: {path}: {info}")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.frames / info.samplerate),
        "subtype": str(info.subtype),
    }


def is_valid_canonical(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = audio_metadata(path)
    except Exception:
        return False
    return (
        metadata["sample_rate"] == CANONICAL_SAMPLE_RATE
        and metadata["channels"] == CANONICAL_CHANNELS
        and metadata["subtype"] == CANONICAL_SUBTYPE
    )


def convert_audio(input_path: Path, output_path: Path, *, resume: bool) -> dict[str, object]:
    if output_path.exists():
        if resume and is_valid_canonical(output_path):
            metadata = audio_metadata(output_path)
            return {**metadata, "sha256": sha256_file(output_path), "reused": True}
        raise FileExistsError(f"Refusing to overwrite canonical audio: {output_path}")

    waveform, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError(f"Empty or invalid input audio: {input_path}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f"Non-finite samples in input audio: {input_path}")
    if sample_rate != CANONICAL_SAMPLE_RATE:
        factor = gcd(int(sample_rate), CANONICAL_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            CANONICAL_SAMPLE_RATE // factor,
            int(sample_rate) // factor,
            padtype="constant",
        ).astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"Resampling produced invalid audio: {input_path}")
    mono = np.clip(mono, -1.0, 1.0 - (1.0 / 32768.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mono, CANONICAL_SAMPLE_RATE, subtype=CANONICAL_SUBTYPE)
    if not is_valid_canonical(output_path):
        raise ValueError(f"Canonical output validation failed: {output_path}")
    metadata = audio_metadata(output_path)
    return {**metadata, "sha256": sha256_file(output_path), "reused": False}


def _canonicalize_task(task: CanonicalTask) -> dict[str, object]:
    source_input = Path(task.source_path)
    target_input = Path(task.target_path)
    source_output = Path(task.source_output)
    target_output = Path(task.target_output)
    source_input_metadata = audio_metadata(source_input)
    target_input_metadata = audio_metadata(target_input)
    source = convert_audio(source_input, source_output, resume=task.resume)
    target = convert_audio(target_input, target_output, resume=task.resume)
    max_duration_error = 2.0 / CANONICAL_SAMPLE_RATE
    source_duration_error = abs(
        float(source["duration_seconds"]) - float(source_input_metadata["duration_seconds"])
    )
    target_duration_error = abs(
        float(target["duration_seconds"]) - float(target_input_metadata["duration_seconds"])
    )
    if source_duration_error > max_duration_error or target_duration_error > max_duration_error:
        raise ValueError(
            f"Duration drift exceeds {max_duration_error}s for {task.sample_id}: "
            f"source={source_duration_error}, target={target_duration_error}"
        )
    return {
        "index": task.index,
        "id": task.sample_id,
        "source_zh_audio_path": str(source_output.resolve()),
        "target_en_audio_path": str(target_output.resolve()),
        "source_zh_raw_audio_path": str(source_input.resolve()),
        "target_en_raw_audio_path": str(target_input.resolve()),
        "source_zh_text": task.source_text,
        "target_en_text": task.target_text,
        "source_zh_audio_sha256": source["sha256"],
        "target_en_audio_sha256": target["sha256"],
        "source_zh_duration_seconds": source["duration_seconds"],
        "target_en_duration_seconds": target["duration_seconds"],
        "source_zh_raw_duration_seconds": source_input_metadata["duration_seconds"],
        "target_en_raw_duration_seconds": target_input_metadata["duration_seconds"],
        "source_reused": source["reused"],
        "target_reused": target["reused"],
    }


def build_tasks(
    rows: Iterable[Mapping[str, object]],
    *,
    output_root: Path,
    resume: bool,
) -> list[CanonicalTask]:
    tasks: list[CanonicalTask] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate CVSS pair id: {sample_id}")
        seen.add(sample_id)
        source_path = str(row.get("source_zh_audio_path") or "")
        target_path = str(row.get("target_en_audio_path") or "")
        source_text = str(row.get("source_zh_text") or "").strip()
        target_text = str(row.get("target_en_text") or "").strip()
        if not source_path or not target_path or not source_text or not target_text:
            raise ValueError(f"Incomplete CVSS pair row: {sample_id}")
        canonical_name = f"{sample_id}.wav"
        tasks.append(
            CanonicalTask(
                index=index,
                sample_id=sample_id,
                source_path=source_path,
                target_path=target_path,
                source_output=str(output_root / "source_zh" / canonical_name),
                target_output=str(output_root / "target_en" / canonical_name),
                source_text=source_text,
                target_text=target_text,
                resume=resume,
            )
        )
    if len(tasks) != 4897:
        raise ValueError(f"CVSS-T zh/en test must contain 4,897 pairs, found {len(tasks)}")
    return tasks


def build_direction_rows(pair_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    zh_en: list[dict[str, object]] = []
    en_zh: list[dict[str, object]] = []
    for row in pair_rows:
        shared = {
            "id": row["id"],
            "pair_id": row["id"],
            "source_zh_raw_audio_path": row["source_zh_raw_audio_path"],
            "target_en_raw_audio_path": row["target_en_raw_audio_path"],
        }
        zh_en.append(
            {
                **shared,
                "direction": "cmn->eng",
                "src_lang": "cmn",
                "tgt_lang": "eng",
                "source_audio_path": row["source_zh_audio_path"],
                "reference_audio_path": row["target_en_audio_path"],
                "source_text": row["source_zh_text"],
                "translation_ref": row["target_en_text"],
                "synthetic_source": False,
                "synthetic_reference": True,
            }
        )
        en_zh.append(
            {
                **shared,
                "direction": "eng->cmn",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "source_audio_path": row["target_en_audio_path"],
                "reference_audio_path": row["source_zh_audio_path"],
                "source_text": row["target_en_text"],
                "translation_ref": row["source_zh_text"],
                "synthetic_source": True,
                "synthetic_reference": False,
            }
        )
    return zh_en, en_zh


def canonicalize(args: argparse.Namespace) -> dict[str, object]:
    input_manifest = Path(args.input_manifest)
    output_root = Path(args.output_root)
    manifest_dir = output_root / "manifests"
    summary_path = output_root / "canonical_summary.json"
    pair_manifest_path = manifest_dir / "cvss_t_zh_en_test_pairs.jsonl"
    if summary_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to reuse canonical output without --resume: {output_root}")

    tasks = build_tasks(iter_jsonl(input_manifest), output_root=output_root, resume=args.resume)
    if args.workers == 1:
        pair_rows = [_canonicalize_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pair_rows = list(executor.map(_canonicalize_task, tasks, chunksize=args.chunksize))
    pair_rows.sort(key=lambda row: int(row["index"]))
    if len(pair_rows) != len(tasks):
        raise RuntimeError(f"Canonical result count mismatch: {len(pair_rows)} != {len(tasks)}")

    source_hours = math.fsum(float(row["source_zh_duration_seconds"]) for row in pair_rows) / 3600.0
    target_hours = math.fsum(float(row["target_en_duration_seconds"]) for row in pair_rows) / 3600.0
    zh_en_rows, en_zh_rows = build_direction_rows(pair_rows)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pair_manifest_path, pair_rows)
    write_jsonl(manifest_dir / "cvss_t_zh_en_test.jsonl", zh_en_rows)
    write_jsonl(manifest_dir / "cvss_t_en_zh_test.jsonl", en_zh_rows)
    summary = {
        "input_manifest": str(input_manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "pair_count": len(pair_rows),
        "source_zh_count": len(zh_en_rows),
        "target_en_count": len(en_zh_rows),
        "source_zh_hours": source_hours,
        "target_en_hours": target_hours,
        "sample_rate": CANONICAL_SAMPLE_RATE,
        "channels": CANONICAL_CHANNELS,
        "subtype": CANONICAL_SUBTYPE,
        "source_reused_count": sum(bool(row["source_reused"]) for row in pair_rows),
        "target_reused_count": sum(bool(row["target_reused"]) for row in pair_rows),
        "ready_for_tokenization": len(pair_rows) == 4897,
    }
    write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.chunksize < 1:
        parser.error("--workers and --chunksize must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    summary = canonicalize(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
