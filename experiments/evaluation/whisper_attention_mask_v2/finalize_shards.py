"""Atomically merge corrected ASR shards, score BLEU, and verify a run."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import merge_jsonl_by_key
from evaluation.text_metrics import compute_grouped_bleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--output-subdir", default="metrics_whisper_attention_mask_v2"
    )
    return parser.parse_args()


def finalize(run_root: Path, *, num_shards: int, output_subdir: str) -> bool:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = run_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "COMPLETE"
    lock_path = output_dir / ".finalize.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if complete.exists():
            return True
        markers = [
            output_dir / "shards" / f"shard_{index:03d}.COMPLETE"
            for index in range(num_shards)
        ]
        if not all(path.exists() for path in markers):
            return False

        canonical = output_dir / "asr_results_eng.jsonl"
        parts = [
            output_dir / "shards" / f"asr_results_eng.part_{index:03d}.jsonl"
            for index in range(num_shards)
        ]
        merge_report = merge_jsonl_by_key([canonical, *parts], canonical)
        rows = list(iter_jsonl(canonical))
        summary = {
            "transcribed": len(rows),
            "empty": sum(not str(row.get("asr_text") or "").strip() for row in rows),
            "single_item_retries": sum(
                bool(row.get("asr_single_item_retry")) for row in rows
            ),
            "rejected_suspicious": sum(
                bool(row.get("asr_rejected_reason")) for row in rows
            ),
            "direct_single_items": sum(
                row.get("asr_effective_batch_size") == 1
                and not row.get("asr_single_item_retry")
                for row in rows
            ),
            "merge": merge_report,
            "num_shards": num_shards,
        }
        write_json(output_dir / "asr_results_eng.summary.json", summary)
        bleu = compute_grouped_bleu(
            rows,
            hypothesis_field="asr_text",
            reference_field="translation_ref",
            score_empty_hypotheses=True,
        )
        write_json(output_dir / "speech_bleu_eng.json", bleu)

        verification_path = output_dir / "verification.json"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "experiments/evaluation/whisper_attention_mask_v2/verify.py"),
                "--input",
                str(run_root / "results.jsonl"),
                "--asr",
                str(canonical),
                "--output",
                str(verification_path),
            ],
            cwd=repo_root,
            check=True,
        )
        complete.touch()
        print(
            json.dumps(
                {"complete": True, "output_dir": str(output_dir), **summary},
                ensure_ascii=False,
                indent=2,
            )
        )
        return True


def main() -> None:
    args = parse_args()
    ready = finalize(
        args.run_root,
        num_shards=args.num_shards,
        output_subdir=args.output_subdir,
    )
    if not ready:
        print(json.dumps({"complete": False, "reason": "shards_not_ready"}))


if __name__ == "__main__":
    main()
