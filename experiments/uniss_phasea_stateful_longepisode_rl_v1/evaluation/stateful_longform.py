#!/usr/bin/env python3
"""CLI for complete stateful long-audio Phase-A evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    evaluate_stateful_session,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _load_models,
    _route_for_prompt,
)


def load_generate(path: Path):
    spec = importlib.util.spec_from_file_location("uniss_phasea_readonly_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audio-protocol", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--decision-chunk-ms", type=int, default=640)
    parser.add_argument("--acoustic-rollover-ms", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--strict-runtime", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.decision_chunk_ms % 160:
        raise ValueError("decision chunk must be divisible by the 160 ms physical block")
    if args.acoustic_rollover_ms < args.decision_chunk_ms:
        raise ValueError("acoustic rollover must be at least one decision interval")
    protocol = json.loads(args.audio_protocol.read_text(encoding="utf-8"))
    requested = set(str(value) for value in args.sample_id)
    records = [
        value
        for value in protocol["records"]
        if not requested or str(value["sample_id"]) in requested
    ]
    if requested != {str(row["sample_id"]) for row in records} and requested:
        raise ValueError("protocol is missing requested sample IDs")
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]]
    if len(fixed_speaker) != 32:
        raise ValueError("fixed speaker must contain 32 global tokens")

    model, tokenizer, controller, adapter_manifest, objective, codec = _load_models(args)
    base_generate = load_generate(args.strict_runtime)

    def routed_generate(*call_args, **call_kwargs):
        enabled = args.adapter_checkpoint is not None and _route_for_prompt(
            call_kwargs["prompt_ids"]
        )
        with controller.route(enabled):
            return base_generate(*call_args, **call_kwargs)

    args.output.mkdir(parents=True)
    results = []
    try:
        for index, record in enumerate(records):
            sample_id = str(record["sample_id"])
            print(f"run={args.run_id} sample={sample_id}", flush=True)
            row = {
                "id": sample_id,
                "src_lang": str(record["src_lang"]),
                "tgt_lang": str(record["tgt_lang"]),
                "source_audio": str(Path(record["source_audio"]).resolve()),
                "bicodec_global": fixed_speaker,
                "_stage_a_fixed_speaker_global": fixed_speaker,
            }
            value = evaluate_stateful_session(
                row,
                decision_chunk_ms=args.decision_chunk_ms,
                acoustic_rollover_ms=args.acoustic_rollover_ms,
                model=model,
                tokenizer=tokenizer,
                objective=objective,
                bicodec=codec,
                generate_fn=routed_generate,
                output=args.output,
                seed=20260826 + index * 1_000_000,
            )
            results.append(value)
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "first_audio_source_ms": value["first_audio_source_ms"],
                        "audio_writes": value["audio_writes"],
                        "pending_unspoken": value["tts_pending_unspoken_items"],
                        "rtf": value["rtf"],
                        "passed": value["stateful_runtime_passed"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        controller.close()

    payload = {
        "schema_version": "uniss_phasea_stateful_longepisode_runtime_v2",
        "status": "complete",
        "run_id": args.run_id,
        "decision_chunk_ms": args.decision_chunk_ms,
        "acoustic_rollover_ms": args.acoustic_rollover_ms,
        "adapter_manifest": adapter_manifest,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
