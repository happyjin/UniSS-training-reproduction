#!/usr/bin/env python3
"""Drive Stage10 from real Stage09 policy events on one validation utterance."""

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
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.model_loader import (
    load_stage09_bundle,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.runtime import (
    Stage09OnlineRuntime,
)

from .adapter import CachedMicroWriteAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("eng->cmn", "cmn->eng"), default="eng->cmn")
    parser.add_argument("--max-write-events", type=int, default=3)
    parser.add_argument("--max-write-tokens", type=int, default=384)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Stage10 smoke: {output}")
    config = Stage09Config()
    bundle = load_stage09_bundle(config)
    dataset = B2BridgeAudioDataset(
        config.dataset_index, "valid", config.source_manifest, config.source_offsets
    )
    direction_id = 0 if args.direction == "eng->cmn" else 1
    selected = next(
        index
        for index in range(len(dataset))
        if (0 if str(dataset._target_row(index)["direction"]) == "eng->cmn" else 1)
        == direction_id
    )
    row = dataset[selected]
    runtime = Stage09OnlineRuntime(bundle, direction=args.direction)
    target_language = "cmn" if direction_id == 0 else "eng"
    adapter = CachedMicroWriteAdapter(
        bundle.qwen,
        bundle.tokenizer,
        bundle.device,
        target_language,
        row["phase3_record"]["bicodec_global"],
        max_write_tokens=args.max_write_tokens,
    )
    records = []
    write_count = 0
    started = time.perf_counter()
    for event in runtime.replay_waveform(row["waveform"].numpy()):
        adapter.append_source(event.qwen_speech_embeddings)
        write = None
        if event.action == "WRITE" and write_count < args.max_write_events:
            write = adapter.generate_write()
            write_count += 1
        else:
            adapter.commit_wait()
        records.append(
            {
                **event.to_dict(),
                "qwen_write": write.__dict__ if write is not None else None,
                "cache_tokens": adapter._cache_length(),
            }
        )
    elapsed = time.perf_counter() - started
    writes = [row["qwen_write"] for row in records if row["qwen_write"] is not None]
    payload = {
        "schema_version": "uniss_streamspeech_stage10_cached_micro_write_smoke_v1",
        "research_only": True,
        "id": row["id"],
        "direction": args.direction,
        "reference_translation": row["phase3_record"]["translation"],
        "translation": adapter.translation,
        "summary": {
            "events": len(records),
            "policy_writes_executed": len(writes),
            "structurally_valid_writes": sum(bool(value["structurally_valid"]) for value in writes),
            "semantic_tokens": sum(len(value["semantic_values"]) for value in writes),
            "cache_tokens": adapter._cache_length(),
            "source_b1_tokens": adapter.source_tokens,
            "first_qwen_token_seconds": writes[0]["first_token_seconds"] if writes else None,
            "wall_seconds": elapsed,
        },
        "events": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    summary = payload["summary"]
    args.output_md.write_text(
        "# Stage10 cached Micro-WRITE smoke\n\n"
        "> Research-only: CTC-triggered true KV-cache inference on the 15-shard line.\n\n"
        f"- sample/direction: `{row['id']}` / {args.direction}\n"
        f"- executed writes: {summary['policy_writes_executed']}\n"
        f"- structurally valid writes: {summary['structurally_valid_writes']}\n"
        f"- semantic tokens: {summary['semantic_tokens']}\n"
        f"- final KV-cache/source B1 tokens: {summary['cache_tokens']} / {summary['source_b1_tokens']}\n"
        f"- first Qwen token wall time: {summary['first_qwen_token_seconds']} s\n"
        f"- generated translation: {adapter.translation}\n"
        f"- reference: {row['phase3_record']['translation']}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
