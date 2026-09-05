"""Did the model learn to keep quiet on the ticks that carry nothing?

Step 2 trains a chunk that committed nothing as "emit the terminator now".
Whether that lesson took is not visible in BLEU and not visible in the loss --
the IDLE supervision lands in the ``boundary_eos`` bucket, mixed with every
other terminator in the pool.  This measures it directly.

How
---
The cascade already expresses the read/wait step: ``p2st_cascade`` reads a
strict fixed grid (``range(read_stride - 1, total_blocks, read_stride)``) and
``switch_rule.next_task`` sends a step with ``source_delta <= 0`` back to
``TASK_READ``.  So the model's *prediction* for tick ``j`` is simply "did the
committer grow at step ``j``", which the trace records in ``source_deltas``
and ``target_deltas``.

The *label* for tick ``j`` comes from the same function that built the
training data -- ``uniform_chunk_tasks.chunk_windows`` -- so the two cannot
drift apart: a tick is a gold IDLE tick exactly when the pool would have
supervised it as one.

This needs gold trajectories, so it runs on the in-domain valid set, not on
RealSI, which carries no event alignment.  On RealSI only the *behaviour* is
observable -- what fraction of read steps commit nothing -- and that is
reported by ``--behaviour-only``, without labels, for comparison against the
training prior.

What the numbers mean
---------------------
``idle_recall`` is the one the step-2 gate names: of the ticks that genuinely
carry nothing, how many did the model stay quiet on.  ``idle_precision`` is
its counterweight and the failure mode to watch -- a model that has learned to
terminate immediately always scores perfect recall and destroys precision, and
that is exactly the risk of spelling IDLE with an existing terminator rather
than a token of its own.  Report both or report neither.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    BLOCK_MS,
    P2STCascadeSession,
)
from experiments.uniss_streaming_p2st_traj_v1.data.uniform_chunk_tasks import (
    chunk_windows,
)


def gold_idle_labels(
    trajectory: E2ETrajectory, *, chunk_ms: int
) -> tuple[list[bool], list[bool]]:
    """``(source idle, target idle)`` per tick, from the pool's own binning."""
    source: list[bool] = []
    target: list[bool] = []
    for window in chunk_windows(trajectory, chunk_ms=chunk_ms):
        source.append(
            not any(e.gold_source_delta.strip() for e in window.events)
        )
        target.append(
            not any(e.target_text_delta.strip() for e in window.events)
        )
    return source, target


def _counts(labels: list[bool], predicted: list[bool]) -> dict[str, int]:
    """Confusion counts with IDLE as the positive class."""
    pairs = list(zip(labels, predicted))
    return {
        "true_idle_kept_quiet": sum(1 for a, b in pairs if a and b),
        "true_idle_spoke": sum(1 for a, b in pairs if a and not b),
        "had_content_kept_quiet": sum(1 for a, b in pairs if not a and b),
        "had_content_spoke": sum(1 for a, b in pairs if not a and not b),
    }


def _rates(counts: dict[str, int]) -> dict[str, float | None]:
    tp = counts["true_idle_kept_quiet"]
    fn = counts["true_idle_spoke"]
    fp = counts["had_content_kept_quiet"]
    tn = counts["had_content_spoke"]
    total = tp + fn + fp + tn
    return {
        "idle_recall": tp / (tp + fn) if tp + fn else None,
        "idle_precision": tp / (tp + fp) if tp + fp else None,
        "content_recall": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / total if total else None,
        "predicted_idle_rate": (tp + fp) / total if total else None,
        "label_idle_rate": (tp + fn) / total if total else None,
        "ticks": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidate-hf", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--read-stride",
        type=int,
        default=4,
        help="read steps of 160 ms per tick; 4 is the 640 ms training grid",
    )
    parser.add_argument("--source-holdback", type=int, default=None)
    parser.add_argument("--target-holdback", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tts-text-scope", default="delta", choices=("delta", "prefix")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunk_ms = int(args.read_stride) * BLOCK_MS
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.candidate_hf), local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(args.candidate_hf),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    objective = (
        stage_a_eval.load_objective(
            args.v1_checkpoint, args.whispervq_model, device
        )
        .eval()
        .requires_grad_(False)
    )
    frontend = make_cached_frontend(objective, device)

    trajectories: list[E2ETrajectory] = []
    with args.gold.open() as handle:
        for line in handle:
            trajectories.append(E2ETrajectory.from_mapping(json.loads(line)))
            if len(trajectories) >= args.samples:
                break

    holdback = (
        {}
        if args.source_holdback is None and args.target_holdback is None
        else {
            "source_holdback": args.source_holdback,
            "target_holdback": args.target_holdback,
        }
    )
    rows: list[dict[str, object]] = []
    source_totals: dict[str, int] = {}
    target_totals: dict[str, int] = {}
    misaligned = 0
    for trajectory in trajectories:
        waveform, rate = sf.read(trajectory.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        if int(rate) != 16_000:
            raise ValueError("source audio must be 16 kHz")
        started = time.time()
        session = P2STCascadeSession(
            model=model,
            tokenizer=tokenizer,
            objective=objective,
            frontend=frontend,
            src_lang=trajectory.src_lang,
            tgt_lang=trajectory.tgt_lang,
            speaker_global=trajectory.speaker_global,
            tts_text_scope=args.tts_text_scope,
            read_stride=int(args.read_stride),
            **holdback,
        )
        trace = session.run(waveform)
        elapsed = time.time() - started

        source_labels, target_labels = gold_idle_labels(
            trajectory, chunk_ms=chunk_ms
        )
        source_pred = [int(v) <= 0 for v in trace.source_deltas]
        target_pred = [int(v) <= 0 for v in trace.target_deltas]
        # The cascade appends a final step when the stride does not divide the
        # audio, so the two grids can differ by one tick.  Compare the common
        # prefix and count the mismatch rather than padding it away, because a
        # systematic offset would silently invert every label.
        keep = min(len(source_labels), len(source_pred))
        if len(source_labels) != len(source_pred):
            misaligned += 1
        source_counts = _counts(source_labels[:keep], source_pred[:keep])
        target_counts = _counts(target_labels[:keep], target_pred[:keep])
        for store, counts in (
            (source_totals, source_counts),
            (target_totals, target_counts),
        ):
            for key, value in counts.items():
                store[key] = store.get(key, 0) + value
        rows.append(
            {
                "sample_id": trajectory.sample_id,
                "direction": f"{trajectory.src_lang}->{trajectory.tgt_lang}",
                "source_duration_ms": trajectory.source_duration_ms,
                "gold_ticks": len(source_labels),
                "run_ticks": len(source_pred),
                "compared_ticks": keep,
                "source": {**source_counts, **_rates(source_counts)},
                "target": {**target_counts, **_rates(target_counts)},
                "fragments": len(trace.fragments),
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"  {trajectory.sample_id} ticks={keep} "
            f"asr_idle_recall={_rates(source_counts)['idle_recall']} "
            f"mt_idle_recall={_rates(target_counts)['idle_recall']}",
            flush=True,
        )

    report = {
        "schema_version": "uniss_streaming_p2st_idle_accuracy_v1",
        "candidate_hf": str(args.candidate_hf.resolve()),
        "gold": str(args.gold.resolve()),
        "read_stride": int(args.read_stride),
        "chunk_ms": chunk_ms,
        "source_holdback": args.source_holdback,
        "target_holdback": args.target_holdback,
        "samples": len(rows),
        "misaligned_grids": misaligned,
        "source_idle": {**source_totals, **_rates(source_totals)},
        "target_idle": {**target_totals, **_rates(target_totals)},
        "per_sample_median_seconds": (
            statistics.median(float(r["seconds"]) for r in rows) if rows else None
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    )
    src = report["source_idle"]
    tgt = report["target_idle"]
    print(
        f"\nASR  idle recall={src['idle_recall']} precision={src['idle_precision']} "
        f"predicted={src['predicted_idle_rate']} label={src['label_idle_rate']}"
    )
    print(
        f"MT   idle recall={tgt['idle_recall']} precision={tgt['idle_precision']} "
        f"predicted={tgt['predicted_idle_rate']} label={tgt['label_idle_rate']}"
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
