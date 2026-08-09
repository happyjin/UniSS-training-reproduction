#!/usr/bin/env python3
"""Step 3 smoke: Student-v2 stability wait-k → first-WRITE latency on CVSS-T.

Does not yet run Qwen/NAR generation (Step 2 head is still blank-collapsed).
Sweeps ``k`` and reports mean source_end_ms of the first WRITE decision — the
latency axis of the eventual LAAL–BLEU Pareto. Quality axis lands when AR or a
non-blank NAR head is attached in the same tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from experiments.simul_s2st_route_v1.step3_waitk_pareto.waitk_policy import (
    StabilityWaitKPolicy,
)
from training.simul_uniss.subsecond_v2.validate_stage_b_latent import load_model
from web_demo.stage_b_v2_streaming_stereo_v1.student_frontend import (
    LatentStudentStreamingSession,
)

SCHEMA_VERSION = "simul_s2st_route_v1_step3_first_write_smoke_v1"


def load_pcm(path: Path, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    waveform = torch.tensor(audio, dtype=torch.float32)
    if rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), int(rate), int(sample_rate)
        ).squeeze(0)
    return waveform.numpy()


def first_write_ms(
    model,
    pcm: np.ndarray,
    *,
    k: int,
    threshold: float,
    chunk_ms: int = 160,
) -> dict[str, object]:
    session = LatentStudentStreamingSession(model, synchronize_cuda=True)
    policy = StabilityWaitKPolicy(k=k, threshold=threshold)
    sample_rate = session.sample_rate
    chunk = int(sample_rate * chunk_ms / 1000)
    first_ms = None
    decisions = 0
    for start in range(0, len(pcm), chunk):
        end = min(len(pcm), start + chunk)
        final = end >= len(pcm)
        event = session.feed(pcm[start:end], final=final)
        decision = policy.observe(session.stability_probabilities)
        decisions += 1
        if decision.action == "WRITE" and first_ms is None:
            first_ms = float(event.source_end_ms)
            break
    return {
        "first_write_ms": first_ms,
        "source_duration_ms": len(pcm) / sample_rate * 1000.0,
        "stable_count": policy._stable,
        "tokens": len(session.stability_probabilities),
        "decisions": decisions,
        "wrote": first_ms is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test/"
            "manifests/cvss_t_zh_en_test_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=ROOT
        / "checkpoints/simul_uniss_subsecond_v2/stage_b_v2_prefix80_finetune_100k_v1/best.pt",
    )
    parser.add_argument("--wait-k", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise SystemExit(f"refusing to overwrite {args.output_json} / {args.output_md}")

    records = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= args.max_samples:
                break
    if not records:
        raise SystemExit(f"empty manifest {args.manifest}")

    device = torch.device(args.device)
    model, _checkpoint = load_model(args.student_checkpoint, device, None, None)
    model.eval()

    def audio_path(record: dict) -> Path:
        for key in (
            "source_zh_audio_path",
            "source_en_audio_path",
            "source_audio_path",
            "wav",
        ):
            if key in record and record[key]:
                return Path(str(record[key]))
        raise KeyError(f"no audio path in {record.get('id')}")

    started = time.time()
    by_k = []
    for k in args.wait_k:
        rows = []
        for record in records:
            pcm = load_pcm(audio_path(record), int(model.config.sample_rate))
            row = first_write_ms(model, pcm, k=k, threshold=args.threshold)
            row["id"] = str(record.get("id") or record.get("utt_id") or len(rows))
            rows.append(row)
        wrote = [row["first_write_ms"] for row in rows if row["first_write_ms"] is not None]
        by_k.append(
            {
                "k": k,
                "samples": len(rows),
                "write_rate": len(wrote) / len(rows),
                "mean_first_write_ms": float(np.mean(wrote)) if wrote else None,
                "median_first_write_ms": float(np.median(wrote)) if wrote else None,
                "rows": rows,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config": {
            "manifest": str(args.manifest),
            "student_checkpoint": str(args.student_checkpoint),
            "wait_k": args.wait_k,
            "threshold": args.threshold,
            "note": (
                "Latency-only smoke. Full LAAL–ASR-BLEU Pareto attaches AR "
                "(or a non-blank NAR head) in a follow-up run in this tree."
            ),
        },
        "by_k": by_k,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Step 3 smoke — Student-v2 wait-k first WRITE latency",
        "",
        f"> `{args.run_name}` · {payload['generated_at']}",
        "",
        "Quality (ASR-BLEU) not measured yet — Step 2 NAR head is blank-collapsed; "
        "this smoke only sweeps the latency axis.",
        "",
        "| k | Write rate | Mean first WRITE (ms) | Median (ms) |",
        "|---:|---:|---:|---:|",
    ]
    for block in by_k:
        mean = (
            f"{block['mean_first_write_ms']:.0f}"
            if block["mean_first_write_ms"] is not None
            else "—"
        )
        median = (
            f"{block['median_first_write_ms']:.0f}"
            if block["median_first_write_ms"] is not None
            else "—"
        )
        lines.append(
            f"| {block['k']} | {block['write_rate']*100:.0f}% | {mean} | {median} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(args.output_json), "by_k": [
        {k: block[k] for k in ("k", "write_rate", "mean_first_write_ms", "median_first_write_ms")}
        for block in by_k
    ]}, indent=2))


if __name__ == "__main__":
    main()
