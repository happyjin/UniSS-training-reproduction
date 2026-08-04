#!/usr/bin/env python3
"""One-sample GPU smoke for the Stage09 online runtime."""

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

import torch

from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge_data import (
    B2BridgeAudioDataset,
)

from .config import Stage09Config
from .model_loader import load_stage09_bundle
from .runtime import Stage09OnlineRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--direction", choices=("eng->cmn", "cmn->eng"), default="eng->cmn")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Stage09 smoke: {output}")
    config = Stage09Config()
    bundle = load_stage09_bundle(config)
    dataset = B2BridgeAudioDataset(
        config.dataset_index,
        "valid",
        config.source_manifest,
        config.source_offsets,
    )
    direction_id = 0 if args.direction == "eng->cmn" else 1
    selected = None
    for index in range(args.dataset_index, len(dataset)):
        target = dataset._target_row(index)
        value = 0 if str(target["direction"]) == "eng->cmn" else 1
        if value == direction_id:
            selected = index
            break
    if selected is None:
        raise RuntimeError(f"no sample found for {args.direction}")
    row = dataset[selected]
    runtime = Stage09OnlineRuntime(
        bundle,
        direction=args.direction,
        confirmations=config.confirmations,
        lagging_k=config.lagging_k,
    )
    started = time.perf_counter()
    events = [event.to_dict() for event in runtime.replay_waveform(row["waveform"].numpy())]
    elapsed = time.perf_counter() - started
    writes = [event for event in events if event["action"] == "WRITE"]
    payload = {
        "schema_version": "uniss_streamspeech_stage09_online_runtime_smoke_v1",
        "research_only": True,
        "sample_index": selected,
        "id": row["id"],
        "direction": args.direction,
        "source_seconds": len(row["waveform"]) / 16000.0,
        "segment_ms": runtime.segment_ms,
        "right_context_ms": runtime.right_context_ms,
        "events": events,
        "summary": {
            "chunks": len(events),
            "writes": len(writes),
            "first_write_ms": writes[0]["source_end_ms"] if writes else None,
            "committed_translation": runtime.committed_translation,
            "reference_translation": row["phase3_record"]["translation"],
            "source_conflicts": runtime.policy.source.conflict_events,
            "target_conflicts": runtime.policy.target.conflict_events,
            "wall_seconds": elapsed,
            "compute_rtf": elapsed / max(len(row["waveform"]) / 16000.0, 1e-6),
            "peak_memory_mib": torch.cuda.max_memory_allocated(bundle.device) / (1024**2)
            if bundle.device.type == "cuda"
            else 0.0,
        },
        "provenance": bundle.provenance.__dict__,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    summary = payload["summary"]
    args.output_md.write_text(
        "# Stage09 online runtime smoke\n\n"
        "> Research-only: the upstream BLEU hard gate remains unmet.\n\n"
        f"- sample: `{row['id']}` ({args.direction})\n"
        f"- Emformer segment/right context: {runtime.segment_ms}/{runtime.right_context_ms} ms\n"
        f"- chunks / WRITEs: {summary['chunks']} / {summary['writes']}\n"
        f"- first WRITE: {summary['first_write_ms']} ms\n"
        f"- committed CTC translation: {summary['committed_translation']}\n"
        f"- reference: {summary['reference_translation']}\n"
        f"- conflicts source/target: {summary['source_conflicts']} / {summary['target_conflicts']}\n"
        f"- wall compute RTF: {summary['compute_rtf']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
