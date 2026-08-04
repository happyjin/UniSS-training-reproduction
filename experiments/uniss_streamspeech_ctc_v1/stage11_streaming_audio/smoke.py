#!/usr/bin/env python3
"""Generate playable Stage11 streaming audio for one fixed validation sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = Path(__file__).resolve().parents[1]
for path in (
    ROOT,
    TREE / "stage03_multitask_encoder",
    TREE / "stage04_b2_discrete_bridge",
):
    sys.path.insert(0, str(path))

from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge_data import (
    B2BridgeAudioDataset,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import Stage09Config

from .config import Stage11Config
from .engine import Stage11Engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("eng->cmn", "cmn->eng"), default="eng->cmn")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Stage11 smoke: {output}")
    stage09 = Stage09Config()
    engine = Stage11Engine(stage09, Stage11Config())
    engine.load()
    dataset = B2BridgeAudioDataset(
        stage09.dataset_index, "valid", stage09.source_manifest, stage09.source_offsets
    )
    direction_id = 0 if args.direction == "eng->cmn" else 1
    selected = next(
        index
        for index in range(len(dataset))
        if (0 if str(dataset._target_row(index)["direction"]) == "eng->cmn" else 1)
        == direction_id
    )
    row = dataset[selected]
    request_dir = Stage11Config().output_root / args.run_name
    session = engine.new_session(
        direction=args.direction,
        speaker_tokens=row["phase3_record"]["bicodec_global"],
        request_dir=request_dir,
    )
    updates = []
    waveform = row["waveform"].numpy()
    chunk = 160 * 16
    for start in range(0, len(waveform), chunk):
        end = min(len(waveform), start + chunk)
        for update in session.push(waveform[start:end], final=end == len(waveform)):
            updates.append(update)
    result = updates[-1].result
    if result is None:
        raise RuntimeError("Stage11 smoke did not finalize")
    payload = {
        "schema_version": "uniss_streamspeech_stage11_audio_smoke_v1",
        "research_only": True,
        "id": row["id"],
        "reference_translation": row["phase3_record"]["translation"],
        "result": result.to_dict(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        "# Stage11 streaming audio smoke\n\n"
        "> Research-only end-to-end Phase3 semantic + Streaming BiCodec output.\n\n"
        f"- sample/direction: `{row['id']}` / {args.direction}\n"
        f"- source/target seconds: {result.source_seconds:.2f} / {result.target_seconds:.2f}\n"
        f"- first WRITE: {result.first_write_ms} ms\n"
        f"- first accepted audio NCA/CA: {result.first_audio_nca_ms} / {result.first_audio_ca_ms} ms\n"
        f"- valid/rejected WRITEs: {result.valid_audio_writes} / {result.rejected_writes}\n"
        f"- offline safety fallback: {result.fallback_used} ({result.fallback_reason})\n"
        f"- translation: {result.translation}\n"
        f"- reference: {row['phase3_record']['translation']}\n"
        f"- target WAV: `{result.translation_audio_path}`\n"
        f"- timeline WAV: `{result.timeline_audio_path}`\n"
        f"- stereo WAV: `{result.stereo_audio_path}`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "first_audio_nca_ms": result.first_audio_nca_ms,
                "first_audio_ca_ms": result.first_audio_ca_ms,
                "valid_audio_writes": result.valid_audio_writes,
                "target_seconds": result.target_seconds,
                "stereo": result.stereo_audio_path,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
