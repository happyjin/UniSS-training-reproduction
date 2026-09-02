"""Does the block-causal frontend give the same tokens for a prefix?

Why this has to be checked
--------------------------
The prefix-to-prefix ASR task puts only source GLM positions
``0 .. source_glm_end`` in its prompt, while the established
``build_streaming_asr_task`` always passes the whole trajectory.  The training
dataset loads the *full* waveform and truncates on the GLM side
(``glm_lengths``), so the frontend still sees audio that a real session would
not have heard yet.  That is only safe if the frontend is genuinely
block-causal -- if token ``j`` depends on nothing after block ``j``.

``run_cached_frontend`` pushes 160 ms blocks and sets
``is_final=end == len(waveform)``, so a truncated call marks its last block
final where a full call does not.  This module measures both things that
follow from that:

1. **Causality.**  Compare ``full.tokens[:k]`` against ``prefix.tokens`` for a
   prefix cut at each event boundary.  A mismatch would mean the training
   prompt is conditioned on future audio and the task is unlearnable as
   written.
2. **Count equality.**  ``StageAObjective._inject_causal_glm`` hard-raises
   unless the frontend's token count for the row's waveform equals
   ``glm_lengths``, tolerating only a single terminal codec slot.  So a
   prefix-to-prefix ASR sample must cut the *audio* at ``source_pcm_end`` and
   the frontend must then return exactly ``source_glm_end`` tokens.  This is
   the equality that decides whether the task can train at all.

A third number is reported but deliberately not gated: how often the
frontend's codes equal the recorded ``source_glm_delta``.  Those are the
offline GLM-4 tokenizer's codes and the model never consumes them -- the
trainer feeds ``embedding(causal_codes + offset) + bridge_residual`` and uses
``glm_ids`` only to log ``diagnostic/causal_glm_agreement``, which has read
about 0.001 for the whole of this lineage.  Gating on it would be gating on a
field that has no path to a gradient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
    run_cached_frontend,
)


def _tokens(result) -> list[int]:
    value = result.tokens
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.reshape(-1).tolist()]
    return [int(item) for item in np.asarray(value).reshape(-1).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
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

    report: dict[str, object] = {"schema_version": 1, "samples": []}
    causal_ok = causal_total = 0
    audit_ok = audit_total = 0
    never_short = 0
    offsets: dict[int, int] = {}
    for trajectory in trajectories:
        waveform, sample_rate = sf.read(trajectory.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        if int(sample_rate) != 16_000:
            raise ValueError("source audio must be 16 kHz")
        full = _tokens(run_cached_frontend(frontend, waveform))
        recorded = [
            int(value)
            for event in trajectory.events
            for value in event.source_glm_delta
        ]
        entry: dict[str, object] = {
            "sample_id": trajectory.sample_id,
            "duration_ms": trajectory.source_duration_ms,
            "full_tokens": len(full),
            "recorded_tokens": len(recorded),
            "trajectory_source_glm_length": trajectory.source_glm_length,
            "events": [],
        }
        # Diagnostic only: the recorded codes come from the offline GLM-4
        # tokenizer, the model consumes the causal frontend's codes.
        width = min(len(full), len(recorded))
        agree = sum(1 for a, b in zip(full[:width], recorded[:width]) if a == b)
        entry["recorded_count_matches"] = len(recorded) == len(full)
        entry["glm_agreement_diagnostic"] = agree / max(1, width)

        for event in trajectory.events:
            stop = int(event.source_glm_end)
            if stop <= 0 or stop > len(full):
                continue
            cut = int(event.source_pcm_end)
            if cut <= 0 or cut > len(waveform):
                continue
            prefix = _tokens(run_cached_frontend(frontend, waveform[:cut]))
            overlap = min(stop, len(prefix))
            identical = prefix[:overlap] == full[:overlap]
            causal_total += 1
            causal_ok += 1 if identical and len(prefix) >= stop else 0
            # What the trainer's hard check will compare.
            exact = len(prefix) == stop
            off_by_one = abs(len(prefix) - stop) == 1
            audit_total += 1
            audit_ok += 1 if exact else 0
            gap = len(prefix) - stop
            offsets[gap] = offsets.get(gap, 0) + 1
            never_short += 1 if gap >= 0 else 0
            entry["events"].append(  # type: ignore[union-attr]
                {
                    "event_index": event.event_index,
                    "source_glm_end": stop,
                    "source_pcm_end": cut,
                    "prefix_tokens": len(prefix),
                    "count_exact": exact,
                    "count_off_by_one": off_by_one,
                    "compared": overlap,
                    "identical": identical,
                    "first_mismatch": next(
                        (
                            i
                            for i in range(overlap)
                            if prefix[i] != full[i]
                        ),
                        None,
                    ),
                }
            )
        report["samples"].append(entry)  # type: ignore[union-attr]

    report["causality_pass_rate"] = causal_ok / max(1, causal_total)
    report["causality_checks"] = causal_total
    report["count_exact_rate"] = audit_ok / max(1, audit_total)
    report["count_checks"] = audit_total
    report["count_offsets"] = {str(k): v for k, v in sorted(offsets.items())}
    # The design is viable iff both of these hold.  Count-exactness against
    # the trajectory's own ``source_glm_end`` is reported but not gated: the
    # pool records the frontend's count for the cut instead, which makes
    # ``causal_length == length`` true by construction.  What must hold is
    # that the frontend never returns *fewer* tokens than the trajectory
    # promises, because then no cut could satisfy the trainer.
    report["never_short_rate"] = never_short / max(1, audit_total)
    report["verdict"] = (
        "pass"
        if causal_ok == causal_total and never_short == audit_total
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"causality   {causal_ok}/{causal_total} prefixes reproduce the full run")
    print(
        f"count exact {audit_ok}/{audit_total} prefixes return exactly "
        "source_glm_end tokens (the trainer hard-raises otherwise)"
    )
    for entry in report["samples"]:  # type: ignore[union-attr]
        bad = [
            item
            for item in entry["events"]  # type: ignore[index]
            if not item["identical"]
        ]
        print(
            f"  {entry['sample_id']:<28s} full={entry['full_tokens']:>4d} "
            f"recorded={entry['recorded_tokens']:>4d} "
            f"glm_diag={entry['glm_agreement_diagnostic']:.4f} "
            f"events={len(entry['events'])} noncausal={len(bad)} "
            f"count_off={sum(1 for i in entry['events'] if not i['count_exact'])}"  # type: ignore[index]
        )
        off = [
            i
            for i in entry["events"]  # type: ignore[index]
            if not i["count_exact"]
        ]
        if off:
            worst = max(off, key=lambda i: abs(i["prefix_tokens"] - i["source_glm_end"]))
            print(
                f"      worst count gap: event {worst['event_index']} "
                f"{worst['prefix_tokens']} tokens vs source_glm_end "
                f"{worst['source_glm_end']}"
            )
        if bad:
            first = bad[0]
            print(
                f"      first bad event {first['event_index']} at token "
                f"{first['first_mismatch']} of {first['compared']}"
            )
    print(
        "offsets (prefix_tokens - source_glm_end): "
        + ", ".join(f"{k:+d}x{v}" for k, v in sorted(offsets.items()))
    )
    print(
        f"never short {never_short}/{audit_total} -- the pool can always "
        "record a count the trainer accepts"
    )
    print(f"verdict={report['verdict']}  wrote {args.output}")


if __name__ == "__main__":
    main()
