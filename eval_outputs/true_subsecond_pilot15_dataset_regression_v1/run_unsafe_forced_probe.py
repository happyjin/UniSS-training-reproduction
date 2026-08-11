#!/usr/bin/env python3
"""Create explicitly unsafe forced-WRITE audio for dataset debugging only."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from web_demo.true_subsecond_pilot15_streaming_v1.config import DemoConfig
from web_demo.true_subsecond_pilot15_streaming_v1.engine import (
    TrueSubsecondStreamingEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-ms", type=int, default=640, choices=(320, 480, 640))
    return parser.parse_args()


def copy_if_nonempty(source: str | Path, destination: Path) -> str | None:
    path = Path(source)
    if not path.is_file() or path.stat().st_size <= 44:
        return None
    shutil.copy2(path, destination)
    return str(destination.resolve())


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = replace(
        DemoConfig.from_env(),
        device=args.device,
        decision_chunk_ms=args.chunk_ms,
        allow_unsafe_forced_audio=True,
    )
    engine = TrueSubsecondStreamingEngine(config)
    records = []
    for item in summary["samples"]:
        sample_dir = run_dir / item["label"]
        result = None
        for update in engine.stream(
            sample_dir / "source.wav",
            direction=item["direction"],
            decision_chunk_ms=args.chunk_ms,
        ):
            if update.result is not None:
                result = update.result
        if result is None:
            raise RuntimeError(f"no result for {item['label']}")
        probe = {
            "translation": result.committed_translation,
            "translation_seconds": result.translation_duration_seconds,
            "coverage": result.translation_coverage_ratio,
            "natural_writes": result.natural_writes,
            "forced_writes": result.forced_writes,
            "semantic_tokens": result.semantic_tokens,
            "first_audio_ms": result.first_useful_audio_source_ms,
            "rtf": result.rtf,
            "quality_passed": result.quality_passed,
            "quality_failures": result.quality_failures,
            "translation_audio": copy_if_nonempty(
                result.translation_path,
                sample_dir / "unsafe_forced_streaming_translation.wav",
            ),
            "timeline_audio": copy_if_nonempty(
                result.timeline_path,
                sample_dir / "unsafe_forced_streaming_timeline.wav",
            ),
            "stereo_audio": copy_if_nonempty(
                result.stereo_path,
                sample_dir / "unsafe_forced_streaming_stereo.wav",
            ),
            "result_json": copy_if_nonempty(
                result.result_path,
                sample_dir / "unsafe_forced_streaming_result.json",
            ),
        }
        item["unsafe_forced_probe"] = probe
        metadata_path = sample_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(item)
        print(
            f"UNSAFE {item['label']}: text={result.committed_translation!r} "
            f"audio={result.translation_duration_seconds:.2f}s "
            f"coverage={result.translation_coverage_ratio:.1%}",
            flush=True,
        )
    summary_path.write_text(
        json.dumps({"samples": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Unsafe forced-WRITE diagnostic",
        "",
        "这些文件仅用于听取被安全质量门拦截的训练分布外语音，不能视为正常streaming结果。",
        "",
        "| sample | forced text | audio seconds | coverage | failures |",
        "|---|---|---:|---:|---|",
    ]
    for item in records:
        probe = item["unsafe_forced_probe"]
        text = probe["translation"].replace("|", "\\|") or "(empty)"
        lines.append(
            f"| [{item['label']}]({item['label']}/unsafe_forced_streaming_translation.wav) "
            f"| {text} | {probe['translation_seconds']:.2f} | "
            f"{probe['coverage']:.1%} | {', '.join(probe['quality_failures'])} |"
        )
    (run_dir / "UNSAFE_FORCED_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
