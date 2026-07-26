"""Build and validate CVSS-T zh/en test manifests against Common Voice v4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import soundfile as sf

from evaluation.io_utils import write_json, write_jsonl


def duration_seconds(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def read_cvss_translation_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number} must have filename and normalized translation")
            rows.append((row[0], row[1]))
    return rows


def read_common_voice_metadata(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        return {str(row["path"]): str(row["sentence"]) for row in reader}


def build_cvss_manifest(
    cvss_root: Path,
    output_dir: Path,
    *,
    common_voice_root: Path | None = None,
) -> dict[str, object]:
    translations = read_cvss_translation_tsv(cvss_root / "test.tsv")
    if len(translations) != 4897:
        raise ValueError(f"CVSS-T zh_en test must contain 4897 rows, found {len(translations)}")

    cv_metadata: dict[str, str] = {}
    clips_dir: Path | None = None
    if common_voice_root is not None:
        clips_dir = common_voice_root / "clips"
        metadata_path = common_voice_root / "test.tsv"
        if not metadata_path.is_file():
            metadata_path = common_voice_root / "validated.tsv"
        if not metadata_path.is_file() or not clips_dir.is_dir():
            raise FileNotFoundError(
                f"Common Voice v4 root must contain clips/ and test.tsv or validated.tsv: {common_voice_root}"
            )
        cv_metadata = read_common_voice_metadata(metadata_path)

    pending_rows: list[dict[str, object]] = []
    missing_target: list[str] = []
    missing_source: list[str] = []
    target_hours = 0.0
    source_hours = 0.0
    for filename, english_text in translations:
        target_path = cvss_root / "test" / f"{filename}.wav"
        if not target_path.is_file():
            missing_target.append(filename)
            continue
        target_duration = duration_seconds(target_path)
        target_hours += target_duration / 3600.0
        source_path = clips_dir / filename if clips_dir is not None else None
        chinese_text = cv_metadata.get(filename)
        source_duration = None
        if source_path is not None:
            if not source_path.is_file():
                missing_source.append(filename)
            else:
                try:
                    source_duration = duration_seconds(source_path)
                    source_hours += source_duration / 3600.0
                except Exception:
                    # MP3 support varies between libsndfile builds; existence is
                    # still validated here and audio decoding is rechecked by
                    # the inference tokenizer.
                    source_duration = None
        pending_rows.append(
            {
                "id": filename,
                "common_voice_filename": filename,
                "source_zh_audio_path": str(source_path.resolve()) if source_path is not None else None,
                "source_zh_text": chinese_text,
                "target_en_audio_path": str(target_path.resolve()),
                "target_en_text": english_text,
                "source_zh_duration_seconds": source_duration,
                "target_en_duration_seconds": target_duration,
                "source_available": bool(source_path is not None and source_path.is_file()),
            }
        )

    if missing_target:
        raise FileNotFoundError(f"Missing {len(missing_target)} CVSS target WAV files")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_path = output_dir / "cvss_t_zh_en_test_pending.jsonl"
    write_jsonl(pending_path, pending_rows)

    ready = common_voice_root is not None and not missing_source and all(row["source_zh_text"] for row in pending_rows)
    if ready:
        zh_en_rows = [
            {
                "id": row["id"],
                "direction": "cmn->eng",
                "source_audio_path": row["source_zh_audio_path"],
                "reference_audio_path": row["target_en_audio_path"],
                "source_text": row["source_zh_text"],
                "translation_ref": row["target_en_text"],
                "src_lang": "cmn",
                "tgt_lang": "eng",
            }
            for row in pending_rows
        ]
        en_zh_rows = [
            {
                "id": row["id"],
                "direction": "eng->cmn",
                "source_audio_path": row["target_en_audio_path"],
                "reference_audio_path": row["source_zh_audio_path"],
                "source_text": row["target_en_text"],
                "translation_ref": row["source_zh_text"],
                "src_lang": "eng",
                "tgt_lang": "cmn",
            }
            for row in pending_rows
        ]
        write_jsonl(output_dir / "cvss_t_zh_en_test.jsonl", zh_en_rows)
        write_jsonl(output_dir / "cvss_t_en_zh_test.jsonl", en_zh_rows)

    summary = {
        "cvss_root": str(cvss_root.resolve()),
        "common_voice_root": str(common_voice_root.resolve()) if common_voice_root else None,
        "pair_count": len(pending_rows),
        "target_wav_count": len(pending_rows),
        "missing_source_count": len(missing_source) if common_voice_root else len(pending_rows),
        "missing_source_text_count": sum(not row["source_zh_text"] for row in pending_rows),
        "target_en_hours": target_hours,
        "source_zh_hours": source_hours if common_voice_root else None,
        "ready_for_bidirectional_evaluation": ready,
        "pending_manifest": str(pending_path),
    }
    write_json(output_dir / "cvss_t_manifest_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvss-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--common-voice-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_cvss_manifest(
        args.cvss_root,
        args.output_dir,
        common_voice_root=args.common_voice_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
