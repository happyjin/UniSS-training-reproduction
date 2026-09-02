"""Does the cascade run, and did the model really not decide anything?

This is the ⑤a acceptance test and it deliberately does not measure quality.
The checkpoint it runs against has never been trained on isolated
prefix-to-prefix sequences, so its translations will be poor; that is expected
and is not the criterion, exactly as the family canary's loss values were not.

What is checked:

* the cascade completes over real audio for every sample;
* the task sequence the session actually ran equals what
  ``switch_rule.rule_trace`` produces from the deltas it observed, so the
  ordering came from the rule and not from anything the model emitted;
* no WAIT_READ or WRITE_GENERATE token appears in any generated output, which
  is the literal form of "the decision was removed";
* every fragment is placed at or after the audio that justified it, so no
  output can be heard before its source;
* terminator statistics, since an unterminated generation is the failure the
  isolated-sequence design exists to remove.
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
    P2STCascadeSession,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.switch_rule import rule_trace
from training import constants_uniss as c

DECISION_TOKENS = (c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidate-hf", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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

    report: dict[str, object] = {
        "schema_version": 1,
        "candidate_hf": str(args.candidate_hf.resolve()),
        "claim_scope": (
            "mechanics only; this checkpoint has never trained on isolated "
            "prefix-to-prefix sequences, so quality is not a criterion here"
        ),
        "samples": [],
    }
    rule_ok = decision_free = order_ok = 0
    for trajectory in trajectories:
        waveform, rate = sf.read(trajectory.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        if int(rate) != 16_000:
            raise ValueError("source audio must be 16 kHz")
        session = P2STCascadeSession(
            model=model,
            tokenizer=tokenizer,
            objective=objective,
            frontend=frontend,
            src_lang=trajectory.src_lang,
            tgt_lang=trajectory.tgt_lang,
            speaker_global=trajectory.speaker_global,
        )
        started = time.time()
        trace = session.run(
            waveform, max_blocks=args.max_blocks or None
        )
        elapsed = time.time() - started

        expected = rule_trace(
            trace.source_deltas, trace.target_deltas, blocks=len(trace.source_deltas)
        )
        matches_rule = trace.task_sequence() == expected
        rule_ok += int(matches_rule)

        # No decision token may appear anywhere the model wrote.
        emitted = set(
            token
            for fragment in trace.fragments
            for token in c.encode_bicodec_semantic(fragment.semantic)
        )
        emitted |= set(session.source_committer.committed)
        emitted |= set(session.target_committer.committed)
        clean = all(token not in emitted for token in DECISION_TOKENS)
        decision_free += int(clean)

        ordered = all(
            fragment.start_ms >= fragment.source_end_ms - 1e-6
            for fragment in trace.fragments
        )
        order_ok += int(ordered)

        terminated = [stage.stopped_on_terminator for stage in trace.stages]
        report["samples"].append(  # type: ignore[union-attr]
            {
                "sample_id": trajectory.sample_id,
                "direction": f"{trajectory.src_lang}->{trajectory.tgt_lang}",
                "source_duration_ms": trajectory.source_duration_ms,
                "blocks": trace.blocks,
                "stages_run": len(trace.stages),
                "task_counts": {
                    task: trace.task_sequence().count(task)
                    for task in ("asr", "mt", "tts")
                },
                "matches_switch_rule": matches_rule,
                "decision_token_free": clean,
                "fragments_never_precede_source": ordered,
                "terminator_rate": (
                    statistics.fmean(1.0 if v else 0.0 for v in terminated)
                    if terminated
                    else 0.0
                ),
                "fragments": len(trace.fragments),
                "semantic_tokens": sum(len(f.semantic) for f in trace.fragments),
                "first_audible_ms": (
                    trace.fragments[0].start_ms if trace.fragments else None
                ),
                "placed_end_ms": (
                    trace.fragments[-1].end_ms if trace.fragments else None
                ),
                "source_hypothesis": trace.source_text[:200],
                "target_hypothesis": trace.target_text[:200],
                "transcription_reference": trajectory.normalized_transcription[:200],
                "translation_reference": trajectory.normalized_translation[:200],
                "wall_seconds": round(elapsed, 2),
            }
        )

    total = len(trajectories)
    report["matches_switch_rule"] = f"{rule_ok}/{total}"
    report["decision_token_free"] = f"{decision_free}/{total}"
    report["fragments_never_precede_source"] = f"{order_ok}/{total}"
    report["verdict"] = (
        "pass"
        if rule_ok == decision_free == order_ok == total and total > 0
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"switch rule reproduced   {rule_ok}/{total}")
    print(f"no decision token        {decision_free}/{total}")
    print(f"nothing heard too early  {order_ok}/{total}")
    for entry in report["samples"]:  # type: ignore[union-attr]
        print(
            f"  {entry['sample_id']:<26s} {entry['direction']} "
            f"blocks={entry['blocks']:>3d} stages={entry['stages_run']:>3d} "
            f"{entry['task_counts']} frags={entry['fragments']:>2d} "
            f"sem={entry['semantic_tokens']:>4d} "
            f"term={entry['terminator_rate']:.2f} "
            f"first_audible={entry['first_audible_ms']} "
            f"wall={entry['wall_seconds']}s"
        )
        print(f"      src: {entry['source_hypothesis'][:90]}")
        print(f"      tgt: {entry['target_hypothesis'][:90]}")
    print(f"verdict={report['verdict']}  wrote {args.output}")


if __name__ == "__main__":
    main()
