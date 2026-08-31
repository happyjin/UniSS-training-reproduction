#!/usr/bin/env python3
"""Pair each gate sample's source audio with its translated speech in stereo.

Left channel is the original source utterance, right channel is the model's
streaming translation, both starting at t=0 on the source timeline.  Listening
in stereo is what makes simultaneity audible: the right channel has to start
speaking while the left channel is still talking.

The gate worker writes mono translation audio only, so this rebuilds the pair
using the same helper the Phase-A cascade uses,
``uniss_phasea_stateful_longepisode_rl_v1/runtime/stateful_cascade.py::write_stereo``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    SAMPLE_RATE,
    write_stereo,
)


SCHEMA = "uniss_e2e_stereo_demo_v1"


def read_mono(path: Path) -> np.ndarray:
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def normalize_peak(waveform: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Scale down only if the channel would clip; never amplify noise."""

    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak <= target or peak == 0.0:
        return waveform, 1.0
    gain = target / peak
    return waveform * gain, gain


def source_index(selection: Path) -> dict[str, dict[str, object]]:
    value = json.loads(selection.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in value["records"]}


def translation_paths(run_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted((run_root / "audio").glob("worker_*/*.wav")):
        output[path.stem] = path
    return output


def build(args: argparse.Namespace) -> dict[str, object]:
    records = source_index(args.selection)
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, run_root in args.runs:
        available = translation_paths(run_root)
        for sample_id in args.sample_ids or sorted(available):
            translation_path = available.get(sample_id)
            record = records.get(sample_id)
            if translation_path is None or record is None:
                rows.append(
                    {
                        "run": label,
                        "sample_id": sample_id,
                        "status": "missing_translation"
                        if translation_path is None
                        else "missing_source",
                    }
                )
                continue
            source = read_mono(Path(str(record["source_audio"])))
            translation = read_mono(translation_path)
            translation, gain = normalize_peak(translation, args.peak_target)
            destination = args.output_dir / f"{sample_id}__{label}__stereo.wav"
            write_stereo(source, translation, destination)
            rows.append(
                {
                    "run": label,
                    "sample_id": sample_id,
                    "status": "written",
                    "stereo": str(destination),
                    "source_audio": str(record["source_audio"]),
                    "translation_audio": str(translation_path),
                    "source_seconds": len(source) / SAMPLE_RATE,
                    "translation_seconds": len(translation) / SAMPLE_RATE,
                    "duration_ratio": (
                        len(translation) / len(source) if len(source) else 0.0
                    ),
                    "translation_peak_gain": gain,
                    "translation_rms": float(
                        np.sqrt(np.mean(np.square(translation))) if len(translation) else 0.0
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        key: rows[-1][key]
                        for key in (
                            "run",
                            "sample_id",
                            "duration_ratio",
                            "translation_peak_gain",
                            "stereo",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {"schema_version": SCHEMA, "peak_target": args.peak_target, "items": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_ROOT")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[], dest="sample_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--peak-target",
        type=float,
        default=0.95,
        help="scale the translation channel down if it would clip",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.runs = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"--run needs LABEL=RUN_ROOT, got {value}")
        label, _, root = value.partition("=")
        args.runs.append((label, Path(root)))
    report = build(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"MANIFEST={args.manifest}", flush=True)


if __name__ == "__main__":
    main()
