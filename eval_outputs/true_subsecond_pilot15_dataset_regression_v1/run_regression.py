#!/usr/bin/env python3
"""Decode fixed UniST train/dev rows and compare streaming with Phase3 offline."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch

from web_demo.offline_s2st_phase3_v1.config import DemoConfig as OfflineConfig
from web_demo.offline_s2st_phase3_v1.inference_engine import Phase3QualityEngine
from web_demo.true_subsecond_pilot15_streaming_v1.config import (
    DemoConfig as StreamingConfig,
)
from web_demo.true_subsecond_pilot15_streaming_v1.engine import (
    TrueSubsecondStreamingEngine,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SampleSpec:
    split: str
    parquet: str
    row: int
    label: str


SAMPLES = (
    SampleSpec("train", "train-00000.parquet", 0, "train_en_zh_01"),
    SampleSpec("train", "train-00000.parquet", 2, "train_en_zh_02"),
    SampleSpec("train", "train-00002.parquet", 42768, "train_zh_en_01"),
    SampleSpec("train", "train-00002.parquet", 42776, "train_zh_en_02"),
    SampleSpec("dev", "dev-00000.parquet", 6531, "dev_en_zh_01"),
    SampleSpec("dev", "dev-00000.parquet", 6536, "dev_en_zh_02"),
    SampleSpec("dev", "dev-00000.parquet", 0, "dev_zh_en_01"),
    SampleSpec("dev", "dev-00000.parquet", 2, "dev_zh_en_02"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-ms", type=int, choices=(320, 480, 640), default=640)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    return parser.parse_args()


def direction_for(row: dict[str, object]) -> str:
    pair = (str(row["src_lang"]), str(row["tgt_lang"]))
    if pair == ("eng", "cmn"):
        return "英文 → 中文"
    if pair == ("cmn", "eng"):
        return "中文 → 英文"
    raise ValueError(f"unsupported direction {pair}")


def text_similarity(reference: str, hypothesis: str) -> float:
    normalize = lambda value: "".join(str(value).lower().split())
    return SequenceMatcher(None, normalize(reference), normalize(hypothesis)).ratio()


def copy_if_nonempty(source: str | Path, destination: Path) -> str | None:
    path = Path(source)
    if not path.is_file() or path.stat().st_size <= 44:
        return None
    shutil.copy2(path, destination)
    return str(destination.resolve())


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_rows() -> dict[str, dict[str, object]]:
    columns = (
        "id",
        "transcription",
        "translation",
        "source_bicodec",
        "target_bicodec",
        "bicodec_global",
        "src_lang",
        "tgt_lang",
        "dataset_name",
    )
    tables: dict[str, object] = {}
    rows: dict[str, dict[str, object]] = {}
    for spec in SAMPLES:
        if spec.parquet not in tables:
            path = REPO_ROOT / "data/raw/UniST" / spec.parquet
            tables[spec.parquet] = pq.read_table(path, columns=list(columns))
        table = tables[spec.parquet]
        if not 0 <= spec.row < table.num_rows:
            raise IndexError(f"row {spec.row} outside {spec.parquet}")
        rows[spec.label] = table.slice(spec.row, 1).to_pylist()[0]
    return rows


def decode_dataset_audio(engine, row: dict[str, object], sample_dir: Path) -> None:
    assert engine.bicodec is not None
    device = engine.device
    global_tokens = [int(value) for value in row["bicodec_global"]]
    for name, key in (
        ("source.wav", "source_bicodec"),
        ("reference_target.wav", "target_bicodec"),
    ):
        tokens = torch.tensor(
            [*global_tokens, *[int(value) for value in row[key]]],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            waveform = engine.bicodec.decode_tokens_to_audio(tokens)
        sf.write(sample_dir / name, waveform, 16_000, subtype="PCM_16")


def markdown_report(run_dir: Path, records: list[dict[str, object]]) -> None:
    lines = [
        "# UniST train/dev streaming regression",
        "",
        "试听时优先比较每个子目录中的 `source.wav`、"
        "`reference_target.wav` 和 `offline_phase3.wav`。如果没有 "
        "`streaming_translation.wav`，且 `streaming_stereo.wav` 的右声道为空，"
        "表示质量门拒绝了不安全输出。",
        "",
        "| sample | split | direction | streaming text | natural/forced | coverage | gate | offline text similarity |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for item in records:
        stream = item["streaming"]
        offline = item["offline"]
        text = str(stream["translation"]).replace("|", "\\|") or "(empty)"
        lines.append(
            f"| [{item['label']}]({item['label']}/) | {item['split']} | "
            f"{item['direction']} | {text} | "
            f"{stream['natural_writes']}/{stream['forced_writes']} | "
            f"{stream['coverage']:.1%} | "
            f"{'PASS' if stream['quality_passed'] else 'FAIL'} | "
            f"{offline['translation_similarity']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Aggregate",
            "",
            f"- Samples: {len(records)}",
            f"- Streaming quality passed: {sum(bool(x['streaming']['quality_passed']) for x in records)}/{len(records)}",
            f"- Streaming with natural WRITE: {sum(int(x['streaming']['natural_writes']) > 0 for x in records)}/{len(records)}",
            f"- Streaming with playable audio: {sum(float(x['streaming']['translation_seconds']) > 0 for x in records)}/{len(records)}",
            f"- Offline Phase3 with playable audio: {sum(float(x['offline']['output_seconds']) > 0 for x in records)}/{len(records)}",
        )
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = load_rows()

    streaming_config = replace(
        StreamingConfig.from_env(), device=args.device, decision_chunk_ms=args.chunk_ms
    )
    streaming = TrueSubsecondStreamingEngine(streaming_config)
    streaming.load()
    records: list[dict[str, object]] = []
    for spec in SAMPLES:
        row = rows[spec.label]
        sample_dir = run_dir / spec.label
        sample_dir.mkdir()
        decode_dataset_audio(streaming, row, sample_dir)
        final = None
        for update in streaming.stream(
            sample_dir / "source.wav",
            direction=direction_for(row),
            decision_chunk_ms=args.chunk_ms,
        ):
            if update.result is not None:
                final = update.result
        if final is None:
            raise RuntimeError(f"streaming produced no result for {spec.label}")
        streaming_files = {
            "translation_audio": copy_if_nonempty(
                final.translation_path, sample_dir / "streaming_translation.wav"
            ),
            "timeline_audio": copy_if_nonempty(
                final.timeline_path, sample_dir / "streaming_timeline.wav"
            ),
            "stereo_audio": copy_if_nonempty(
                final.stereo_path, sample_dir / "streaming_stereo.wav"
            ),
            "result_json": copy_if_nonempty(
                final.result_path, sample_dir / "streaming_result.json"
            ),
        }
        records.append(
            {
                "label": spec.label,
                "split": spec.split,
                "parquet": spec.parquet,
                "row": spec.row,
                "id": str(row["id"]),
                "dataset_name": str(row["dataset_name"]),
                "direction": direction_for(row),
                "source_transcription": str(row["transcription"]),
                "reference_translation": str(row["translation"]),
                "files": {
                    "source": str((sample_dir / "source.wav").resolve()),
                    "reference_target": str(
                        (sample_dir / "reference_target.wav").resolve()
                    ),
                    **streaming_files,
                },
                "streaming": {
                    "translation": final.committed_translation,
                    "translation_similarity": text_similarity(
                        str(row["translation"]), final.committed_translation
                    ),
                    "translation_seconds": final.translation_duration_seconds,
                    "coverage": final.translation_coverage_ratio,
                    "natural_writes": final.natural_writes,
                    "forced_writes": final.forced_writes,
                    "first_audio_ms": final.first_useful_audio_source_ms,
                    "rtf": final.rtf,
                    "quality_passed": final.quality_passed,
                    "quality_failures": final.quality_failures,
                },
                "offline": {},
            }
        )
        write_json(sample_dir / "metadata.partial.json", records[-1])
        print(
            f"STREAMING {spec.label}: natural={final.natural_writes} "
            f"forced={final.forced_writes} audio={final.translation_duration_seconds:.2f}s "
            f"gate={final.quality_passed}",
            flush=True,
        )

    del streaming
    gc.collect()
    torch.cuda.empty_cache()

    offline_config = replace(OfflineConfig.from_env(), device=args.device)
    offline = Phase3QualityEngine(offline_config)
    for item in records:
        sample_dir = run_dir / str(item["label"])
        result = offline.translate(
            sample_dir / "source.wav",
            direction=str(item["direction"]),
            use_silence_chunking=False,
        )
        copied = copy_if_nonempty(
            result.output_audio_path, sample_dir / "offline_phase3.wav"
        )
        item["files"]["offline_phase3"] = copied
        item["files"]["offline_result_json"] = copy_if_nonempty(
            result.result_json_path, sample_dir / "offline_result.json"
        )
        item["offline"] = {
            "transcription": result.transcription,
            "translation": result.translation,
            "translation_similarity": text_similarity(
                str(item["reference_translation"]), result.translation
            ),
            "output_seconds": result.output_duration_seconds,
            "processing_seconds": result.total_seconds,
            "warnings": result.warnings,
        }
        (sample_dir / "metadata.partial.json").unlink(missing_ok=True)
        write_json(sample_dir / "metadata.json", item)
        print(
            f"OFFLINE {item['label']}: similarity="
            f"{item['offline']['translation_similarity']:.3f} "
            f"audio={result.output_duration_seconds:.2f}s",
            flush=True,
        )

    write_json(run_dir / "summary.json", {"samples": records})
    markdown_report(run_dir, records)
    print(f"RUN_DIR={run_dir}", flush=True)


if __name__ == "__main__":
    main()
