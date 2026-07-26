"""Compute UTMOS22 strong scores for generated speech with resume support."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping, Sequence

import torch
import torchaudio

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import load_keys, select_shard
from training.generate_unist_eval_audio import write_jsonl_row


def aggregate_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), str(row["src_lang"]), str(row["tgt_lang"]))].append(float(row["utmos_score"]))
    return {
        "groups": {
            f"{mode}:{src}->{tgt}": {
                "sample_count": len(values),
                "mean": mean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            for (mode, src, tgt), values in sorted(groups.items())
        },
        "scored_count": len(rows),
    }


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    wave, sample_rate = torchaudio.load(path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    return wave, sample_rate


def run_utmos(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_utmos.jsonl"
    completed: set[tuple[str, str]] = load_keys(args.completed_input)
    existing_rows: list[dict[str, object]] = []
    if per_sample_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite UTMOS output: {per_sample_path}")
        existing_rows = list(iter_jsonl(per_sample_path))
        completed.update((str(row["id"]), str(row["mode"])) for row in existing_rows)

    device = torch.device(args.device)
    predictor = torch.hub.load(args.torch_hub_repo, args.model_name, trust_repo=True)
    predictor.to(device).eval()
    scored = list(existing_rows)
    failures = 0
    for row in select_shard(
        iter_jsonl(input_path),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        key = (str(row["id"]), str(row["mode"]))
        if key in completed or not row.get("audio_path") or row.get("error"):
            continue
        path = Path(str(row["audio_path"]))
        if not path.is_absolute():
            path = input_path.parent / path
        try:
            wave, sample_rate = load_audio(path)
            with torch.inference_mode():
                score = float(predictor(wave.to(device), sample_rate).mean().item())
        except Exception as exc:
            failures += 1
            write_jsonl_row(
                output_dir / "utmos_failures.jsonl",
                {"id": row.get("id"), "mode": row.get("mode"), "error": f"{type(exc).__name__}:{exc}"},
            )
            continue
        scored_row = {
            "id": row.get("id"),
            "mode": row.get("mode"),
            "src_lang": row.get("src_lang"),
            "tgt_lang": row.get("tgt_lang"),
            "audio_path": str(path),
            "utmos_score": score,
            "utmos_model": f"{args.torch_hub_repo}:{args.model_name}",
        }
        write_jsonl_row(per_sample_path, scored_row)
        scored.append(scored_row)
        report = {**aggregate_scores(scored), "failure_count": failures}
        write_json(output_dir / "utmos.json", report)
    return {**aggregate_scores(scored), "failure_count": failures}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-hub-repo", default="tarepan/SpeechMOS:v1.2.0")
    parser.add_argument("--model-name", default="utmos22_strong")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completed-input", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_utmos(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
