#!/usr/bin/env python3
"""Step 3 AR Pareto smoke: Stage11 + lagging-k wait + Λ-KV window.

Import-only use of Stage09/11 (no shipping edits). After each session is created we
set ``runtime.policy.lagging_k`` and wrap the Micro-WRITE adapter with
``LambdaWindowAdapter``. Metrics: text BLEU/chrF + LAAL proxy from event traces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "experiments/uniss_streamspeech_ctc_v1"
for path in (
    str(ROOT),
    str(TREE / "stage03_multitask_encoder"),
    str(TREE / "stage04_b2_discrete_bridge"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import sacrebleu
import torch
import torch.distributed as dist

from evaluation.simultaneous_streaming.stage4_metrics import token_latency_metrics
from experiments.simul_s2st_route_v1.step3_waitk_pareto.lambda_adapter import (
    LambdaWindowAdapter,
)
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge_data import (
    B2BridgeAudioDataset,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import Stage09Config
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.config import Stage11Config
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.engine import Stage11Engine

SCHEMA_VERSION = "simul_s2st_route_v1_step3_ar_pareto_v1"


def init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def shard(indices: list[int], rank: int, world: int) -> list[int]:
    return indices[rank::world]


def run_one(
    engine: Stage11Engine,
    dataset: B2BridgeAudioDataset,
    index: int,
    *,
    lagging_k: int,
    lambda_window: int,
    request_root: Path,
) -> dict[str, object]:
    row = dataset[index]
    direction = str(row["phase3_record"]["direction"])
    request_dir = request_root / f"k{lagging_k}_w{lambda_window}_idx{index}"
    if request_dir.exists():
        # Allow resume of a crashed shard by clearing only this sample dir.
        import shutil

        shutil.rmtree(request_dir)
    session = engine.new_session(
        direction=direction,
        speaker_tokens=row["phase3_record"]["bicodec_global"],
        request_dir=request_dir,
    )
    session.runtime.policy.lagging_k = int(lagging_k)
    if lambda_window > 0:
        session.adapter = LambdaWindowAdapter(session.adapter, window=lambda_window)
    waveform = row["waveform"].numpy()
    chunk = 160 * 16
    result = None
    for start in range(0, len(waveform), chunk):
        end = min(len(waveform), start + chunk)
        for update in session.push(waveform[start:end], final=end == len(waveform)):
            if update.result is not None:
                result = update.result
    if result is None:
        raise RuntimeError(f"no Stage11 result for index={index}")
    reference = str(row["phase3_record"]["translation"])
    hypothesis = str(result.translation)
    tokenize = "zh" if direction == "eng->cmn" else "13a"
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]], tokenize=tokenize).score
    chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]]).score
    # Build a minimal event_trace for LAAL proxy (WRITE events with text ids).
    event_trace = []
    for event in result.events:
        payload = {
            "action": "write" if event.policy_action == "WRITE" else "wait",
            "source_end_ms": event.source_end_ms,
            "source_glm_end": max(1, int(event.source_end_ms / 40.0)),
            "generated_text_ids": [1] * max(1, len(event.qwen_text_delta.split())),
        }
        event_trace.append(payload)
    latency = token_latency_metrics(
        {
            "source_glm_length": max(1, int(result.source_seconds * 25)),
            "reference_target_text_length": max(1, len(reference.split())),
            "source_duration_ms_proxy": result.source_seconds * 1000.0,
            "event_trace": event_trace,
        }
    )
    return {
        "id": row["id"],
        "index": index,
        "direction": direction,
        "lagging_k": lagging_k,
        "lambda_window": lambda_window,
        "text_bleu": float(bleu),
        "chrf": float(chrf),
        "first_write_ms": result.first_write_ms,
        "first_audio_nca_ms": result.first_audio_nca_ms,
        "first_audio_ca_ms": result.first_audio_ca_ms,
        "valid_audio_writes": result.valid_audio_writes,
        "rejected_writes": result.rejected_writes,
        "fallback_used": result.fallback_used,
        "compute_rtf": float(result.wall_seconds / max(result.source_seconds, 1e-9)),
        "laal_glm_tokens_proxy": latency.get("laal_glm_tokens_proxy"),
        "hypothesis": hypothesis,
        "reference": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--lagging-k", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--lambda-window", type=int, nargs="+", default=[0, 256, 1024])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise SystemExit(f"refusing to overwrite {args.output_json}")

    rank, world, local = init_dist()
    device = f"cuda:{local}" if torch.cuda.is_available() else args.device
    stage09 = Stage09Config(device=device)
    engine = Stage11Engine(stage09, Stage11Config())
    engine.load()
    dataset = B2BridgeAudioDataset(
        stage09.dataset_index, "valid", stage09.source_manifest, stage09.source_offsets
    )
    indices = list(range(min(args.max_samples, len(dataset))))
    local_indices = shard(indices, rank, world)
    request_root = (
        ROOT / "eval_outputs/simul_s2st_route_v1" / args.run_name / f"rank{rank:02d}"
    )
    request_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    started = time.time()
    for lagging_k in args.lagging_k:
        for lambda_window in args.lambda_window:
            for index in local_indices:
                rows.append(
                    run_one(
                        engine,
                        dataset,
                        index,
                        lagging_k=lagging_k,
                        lambda_window=lambda_window,
                        request_root=request_root,
                    )
                )
                if rank == 0:
                    print(
                        {
                            "progress": len(rows),
                            "last": {
                                k: rows[-1][k]
                                for k in (
                                    "id",
                                    "lagging_k",
                                    "lambda_window",
                                    "text_bleu",
                                    "first_write_ms",
                                    "fallback_used",
                                )
                            },
                        },
                        flush=True,
                    )

    part_path = args.output_json.with_suffix(f".rank{rank:02d}.json")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_text(json.dumps({"rank": rank, "rows": rows}, indent=2) + "\n")

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for r in range(world):
            path = args.output_json.with_suffix(f".rank{r:02d}.json")
            merged.extend(json.loads(path.read_text())["rows"])
        # Aggregate by (k, window).
        groups: dict[tuple[int, int], list[dict]] = {}
        for row in merged:
            key = (int(row["lagging_k"]), int(row["lambda_window"]))
            groups.setdefault(key, []).append(row)
        summary = []
        for (lagging_k, lambda_window), group in sorted(groups.items()):
            summary.append(
                {
                    "lagging_k": lagging_k,
                    "lambda_window": lambda_window,
                    "samples": len(group),
                    "mean_text_bleu": float(np.mean([g["text_bleu"] for g in group])),
                    "mean_chrf": float(np.mean([g["chrf"] for g in group])),
                    "mean_first_write_ms": float(
                        np.mean(
                            [
                                g["first_write_ms"]
                                for g in group
                                if g["first_write_ms"] is not None
                            ]
                            or [float("nan")]
                        )
                    ),
                    "fallback_rate": float(np.mean([bool(g["fallback_used"]) for g in group])),
                    "mean_compute_rtf": float(np.mean([g["compute_rtf"] for g in group])),
                    "mean_laal_proxy": float(
                        np.nanmean(
                            [
                                g["laal_glm_tokens_proxy"]
                                for g in group
                                if g["laal_glm_tokens_proxy"] is not None
                            ]
                            or [float("nan")]
                        )
                    ),
                }
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_name": args.run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "config": {
                "lagging_k": args.lagging_k,
                "lambda_window": args.lambda_window,
                "max_samples": args.max_samples,
                "world_size": world,
            },
            "summary": summary,
            "rows": merged,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Step 3 AR Pareto (Stage11 + lagging-k + Λ-KV)",
            "",
            f"> `{args.run_name}` · {payload['generated_at']}",
            "",
            "| k | Λ window | Samples | BLEU | chrF | First WRITE ms | Fallback | RTF | LAAL proxy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for block in summary:
            lines.append(
                f"| {block['lagging_k']} | {block['lambda_window']} | {block['samples']} | "
                f"{block['mean_text_bleu']:.2f} | {block['mean_chrf']:.2f} | "
                f"{block['mean_first_write_ms']:.0f} | {block['fallback_rate']*100:.0f}% | "
                f"{block['mean_compute_rtf']:.2f} | {block['mean_laal_proxy']:.2f} |"
            )
        lines.append("")
        args.output_md.write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"wrote": str(args.output_json), "summary": summary}, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
