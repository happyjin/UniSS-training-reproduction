#!/usr/bin/env python3
"""Evaluate Stage05 with real Stage03b causal CTC logits."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE03 = Path(__file__).resolve().parents[1] / "stage03_multitask_encoder"
STAGE02 = Path(__file__).resolve().parents[1] / "stage02_ctc_probe"
for path in (ROOT, STAGE03, STAGE02, Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch

from audio_data import EndpointCTCAudioDataset
from endpoint_runtime import load_endpoint_model, streaming_ctc_paths
from policy import CTCReadWritePolicy


DIRECTIONS = {
    0: ("asr_eng", "nar_s2tt_cmn", "cmn"),
    1: ("asr_cmn", "nar_s2tt_eng", "eng"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--lagging-k", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def unigram_recall(reference: list[int], hypothesis: list[int]) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    left, right = Counter(reference), Counter(hypothesis)
    overlap = sum((left & right).values())
    return overlap / len(reference)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    first_write = [float(row["first_write_ms"]) for row in rows if row["first_write_ms"]]
    recalls = [float(row["committed_unigram_recall"]) for row in rows]
    return {
        "samples": len(rows),
        "write_coverage": len(first_write) / max(1, len(rows)),
        "first_write_ms_mean": statistics.fmean(first_write) if first_write else None,
        "first_write_ms_p50": percentile(first_write, 0.50),
        "first_write_ms_p95": percentile(first_write, 0.95),
        "committed_unigram_recall_mean": statistics.fmean(recalls) if recalls else None,
        "source_conflicts": sum(int(row["source_conflicts"]) for row in rows),
        "target_conflicts": sum(int(row["target_conflicts"]) for row in rows),
        "rollback_events": 0,
    }


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    return f"""# Stage05 real-CTC policy evaluation

This report feeds actual causal Stage03b ASR/NAR-S2TT logits into the isolated
monotonic Stage05 policy. It measures the policy/CTC proxy only; Phase3 text and
BiCodec audio generation are deliberately not claimed here.

| Metric | Value |
| --- | ---: |
| Samples | {summary['samples']} |
| Samples with a WRITE | {summary['write_coverage']:.4f} |
| First WRITE mean (model-frame ms) | {summary['first_write_ms_mean']} |
| First WRITE p50 (model-frame ms) | {summary['first_write_ms_p50']} |
| First WRITE p95 (model-frame ms) | {summary['first_write_ms_p95']} |
| Committed target unigram recall | {summary['committed_unigram_recall_mean']:.4f} |
| Source/target conflict events | {summary['source_conflicts']} / {summary['target_conflicts']} |
| Rollback events | {summary['rollback_events']} |

`model-frame ms` uses the 40 ms encoder frame clock and includes the configured
right-context frames. It excludes wall-clock compute and downstream synthesis.
"""


def main():
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    device = torch.device(args.device)
    processors = {
        lang: spm.SentencePieceProcessor(
            model_file=str(args.tokenizer_dir / f"ctc_{lang}.model")
        )
        for lang in ("eng", "cmn")
    }
    model = load_endpoint_model(
        args.checkpoint,
        eng_vocab_size=processors["eng"].vocab_size(),
        cmn_vocab_size=processors["cmn"].vocab_size(),
        device=device,
    )
    dataset = EndpointCTCAudioDataset(
        args.dataset_index, args.split, args.source_manifest, args.source_offsets
    )
    rows = []
    for index in range(min(args.max_samples, len(dataset))):
        record = dataset[index]
        direction = int(record["direction_id"])
        source_head, target_head, target_lang = DIRECTIONS[direction]
        target_processor = processors[target_lang]
        policy = CTCReadWritePolicy(
            source_blank_id=processors["eng" if direction == 0 else "cmn"].vocab_size(),
            target_blank_id=target_processor.vocab_size(),
            target_language=target_lang,
            target_id_to_piece=target_processor.id_to_piece,
            confirmations=args.confirmations,
            lagging_k=args.lagging_k,
        )
        waveform = record["waveform"].unsqueeze(0).to(device)
        waveform_length = torch.tensor([waveform.shape[1]], device=device)
        first_write_ms = None
        writes = 0
        chunks = 0
        for source_path, target_path, consumed_frames, final in streaming_ctc_paths(
            model,
            waveform,
            waveform_length,
            source_head=source_head,
            target_head=target_head,
        ):
            decision = policy.update(source_path, target_path, final=final)
            chunks += 1
            if decision.action == "WRITE":
                writes += 1
                if first_write_ms is None:
                    first_write_ms = consumed_frames * 40.0
        reference = record["target_token_ids"].tolist()
        rows.append(
            {
                "id": record["id"],
                "direction": "eng->cmn" if direction == 0 else "cmn->eng",
                "chunks": chunks,
                "writes": writes,
                "first_write_ms": first_write_ms,
                "reference_tokens": len(reference),
                "committed_tokens": len(policy.committed_target),
                "committed_unigram_recall": unigram_recall(
                    reference, policy.committed_target
                ),
                "source_conflicts": policy.source.conflict_events,
                "target_conflicts": policy.target.conflict_events,
            }
        )
    payload = {
        "schema_version": "uniss_streamspeech_stage05_real_ctc_policy_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "confirmations": args.confirmations,
        "lagging_k": args.lagging_k,
        "encoder_segment_ms": int(model.config.segment_frames) * 40,
        "encoder_right_context_ms": int(model.config.right_context_frames) * 40,
        "summary": summarize(rows),
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

